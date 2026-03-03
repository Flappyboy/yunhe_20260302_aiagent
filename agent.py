"""
Agent核心模块 - 使用OpenAI Agents SDK实现
"""
import json
import logging
import traceback
from typing import Dict, Any, Optional

from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_tracing_disabled, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from tools import HouseAPITools
from session import SessionManager
from session_logger import session_logger
from config import MAX_HISTORY_MESSAGES, MAX_TOOL_RESULT_CHARS, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# 精简系统提示以节省 token，保留核心规则
SYSTEM_PROMPT = """你是北京租房助手。能力：查房源(区域/价格/户型/装修/朝向/地铁等)、查地标、租房/退租/下架。
区域：北京各区；价格约500-25000元/月。近地铁=800米内，地铁可达=1000米内。bedrooms：一居=1，两居=2，三居=3。

规则：
1. 用户提到的条件必须全部转为API参数：区域→district，户型→bedrooms，预算→max_price，有电梯→elevator="true"，精装→decoration="精装"，近地铁→max_subway_dist=800。
2. 分页：先page=1看total，若total>page_size则继续page=2,3...直到取完，汇总所有页的房源ID。
3. 回复格式：普通对话用自然语言；房源结果必须JSON：{"message":"...", "houses":["HF_4",...]}，无结果时houses为[]。
4. 租房/退租/下架必须调API，并指定listing_platform(链家/安居客/58同城)，默认安居客。地标附近房先查地标拿landmark_id再nearby。
禁止：编造或猜测house_id；关键信息不明时先追问再查。"""


_current_session_id = None


def set_current_session(session_id: str):
    """设置当前会话ID，用于工具日志记录"""
    global _current_session_id
    _current_session_id = session_id


def get_current_session() -> Optional[str]:
    """获取当前会话ID"""
    return _current_session_id


class RentalAgent:
    """租房AI Agent - 使用OpenAI Agents SDK"""
    
    def __init__(self, model_ip: str, api_base_url: str, user_id: str, api_version: str = "v1"):
        """
        初始化Agent
        
        Args:
            model_ip: LLM模型服务IP
            api_base_url: 房源API基础URL
            user_id: 用户工号
            api_version: 模型API版本，v1或v2，默认v1
        """
        logger.info(f"初始化RentalAgent: model_ip={model_ip}, api_base_url={api_base_url}, user_id={user_id}, api_version={api_version}")
        self.model_ip = model_ip
        self.api_base_url = api_base_url
        self.user_id = user_id
        self.api_version = api_version
        self.session_manager = SessionManager()
        
        logger.debug("禁用tracing")
        set_tracing_disabled(True)
        
        logger.debug("创建HouseAPITools实例")
        self.tools_instance = HouseAPITools(api_base_url, user_id)
        
        logger.debug("创建工具函数")
        self._tools = self._create_tools()
        
        logger.info("RentalAgent初始化完成")
    
    def _create_tools(self):
        """创建工具函数列表"""
        tools_instance = self.tools_instance
        
        def _truncate_result(result_str: str) -> str:
            """截断过长工具结果以节省 token"""
            if len(result_str) <= MAX_TOOL_RESULT_CHARS:
                return result_str
            return result_str[:MAX_TOOL_RESULT_CHARS] + f"\n...[已截断，原长{len(result_str)}字符]"

        def _log_tool(tool_name: str, args_dict: dict, result: str):
            """记录工具调用日志"""
            session_id = get_current_session()
            if session_id:
                session_logger.log_tool_request(session_id, tool_name, args_dict)
                session_logger.log_tool_response(session_id, tool_name, result)
        
        @function_tool
        def init_houses() -> str:
            """重置房源数据到初始状态，新会话开始时调用。"""
            result = tools_instance.init_houses()
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("init_houses", {}, result_str)
            return result_str
        
        @function_tool
        def get_landmarks(category: Optional[str] = None, district: Optional[str] = None) -> str:
            """获取地标列表。category: subway/company/landmark；district: 海淀、朝阳等。"""
            args = {"category": category, "district": district}
            result = tools_instance.get_landmarks(category, district)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_landmarks", args, result_str)
            return result_str
        
        @function_tool
        def get_landmark_by_name(name: str) -> str:
            """按名称查地标，返回id/经纬度等，供nearby用。name如西二旗站、国贸。"""
            args = {"name": name}
            result = tools_instance.get_landmark_by_name(name)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_landmark_by_name", args, result_str)
            return result_str
        
        @function_tool
        def search_landmarks(q: str, category: Optional[str] = None, district: Optional[str] = None) -> str:
            """关键词模糊搜地标。q: 关键词；category: subway/company/landmark；district: 行政区。"""
            args = {"q": q, "category": category, "district": district}
            result = tools_instance.search_landmarks(q, category, district)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("search_landmarks", args, result_str)
            return result_str
        
        @function_tool
        def get_landmark_by_id(landmark_id: str) -> str:
            """按地标ID查详情。landmark_id 如 SS_001、LM_002。"""
            args = {"landmark_id": landmark_id}
            result = tools_instance.get_landmark_by_id(landmark_id)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_landmark_by_id", args, result_str)
            return result_str
        
        @function_tool
        def get_landmark_stats() -> str:
            """地标统计：总数、按类别分布。"""
            result = tools_instance.get_landmark_stats()
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_landmark_stats", {}, result_str)
            return result_str
        
        @function_tool
        def get_house_by_id(house_id: str) -> str:
            """按房源ID查单套详情。house_id 如 HF_2001。"""
            args = {"house_id": house_id}
            result = tools_instance.get_house_by_id(house_id)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_house_by_id", args, result_str)
            return result_str
        
        @function_tool
        def get_house_listings(house_id: str) -> str:
            """房源在各平台挂牌记录。house_id 如 HF_2001。"""
            args = {"house_id": house_id}
            result = tools_instance.get_house_listings(house_id)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_house_listings", args, result_str)
            return result_str
        
        @function_tool
        def get_houses_by_community(community: str, listing_platform: Optional[str] = None,
                                     page: int = 1, page_size: int = 10) -> str:
            """按小区名查可租房源。community 如建清园；listing_platform 链家/安居客/58同城；page/page_size 分页。"""
            args = {"community": community, "listing_platform": listing_platform, "page": page, "page_size": page_size}
            result = tools_instance.get_houses_by_community(community, listing_platform, page, page_size)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_houses_by_community", args, result_str)
            return result_str
        
        @function_tool
        def get_houses_by_platform(
            listing_platform: Optional[str] = None,
            district: Optional[str] = None,
            area: Optional[str] = None,
            min_price: Optional[int] = None,
            max_price: Optional[int] = None,
            bedrooms: Optional[str] = None,
            rental_type: Optional[str] = None,
            decoration: Optional[str] = None,
            orientation: Optional[str] = None,
            elevator: Optional[str] = None,
            min_area: Optional[int] = None,
            max_area: Optional[int] = None,
            property_type: Optional[str] = None,
            subway_line: Optional[str] = None,
            max_subway_dist: Optional[int] = None,
            subway_station: Optional[str] = None,
            utilities_type: Optional[str] = None,
            available_from_before: Optional[str] = None,
            commute_to_xierqi_max: Optional[int] = None,
            sort_by: Optional[str] = None,
            sort_order: Optional[str] = None,
            page: int = 1,
            page_size: int = 10
        ) -> str:
            """按条件筛房。返回total和items。须传用户所有条件；total>page_size时继续查page=2,3。district/area/bedrooms(1,2,3)/max_price/elevator/decoration/max_subway_dist等。"""
            args = {
                "listing_platform": listing_platform, "district": district, "area": area,
                "min_price": min_price, "max_price": max_price, "bedrooms": bedrooms,
                "rental_type": rental_type, "decoration": decoration, "orientation": orientation,
                "elevator": elevator, "min_area": min_area, "max_area": max_area,
                "property_type": property_type, "subway_line": subway_line,
                "max_subway_dist": max_subway_dist, "subway_station": subway_station,
                "utilities_type": utilities_type, "available_from_before": available_from_before,
                "commute_to_xierqi_max": commute_to_xierqi_max, "sort_by": sort_by,
                "sort_order": sort_order, "page": page, "page_size": page_size
            }
            result = tools_instance.get_houses_by_platform(
                listing_platform=listing_platform,
                district=district,
                area=area,
                min_price=min_price,
                max_price=max_price,
                bedrooms=bedrooms,
                rental_type=rental_type,
                decoration=decoration,
                orientation=orientation,
                elevator=elevator,
                min_area=min_area,
                max_area=max_area,
                property_type=property_type,
                subway_line=subway_line,
                max_subway_dist=max_subway_dist,
                subway_station=subway_station,
                utilities_type=utilities_type,
                available_from_before=available_from_before,
                commute_to_xierqi_max=commute_to_xierqi_max,
                sort_by=sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size
            )
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_houses_by_platform", args, result_str)
            return result_str
        
        @function_tool
        def get_houses_nearby(landmark_id: str, max_distance: Optional[int] = None,
                               listing_platform: Optional[str] = None,
                               page: int = 1, page_size: int = 10) -> str:
            """地标附近房源，先查地标得landmark_id。max_distance米默认2000；page/page_size分页。"""
            args = {"landmark_id": landmark_id, "max_distance": max_distance, "listing_platform": listing_platform, "page": page, "page_size": page_size}
            result = tools_instance.get_houses_nearby(landmark_id, max_distance, listing_platform, page, page_size)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_houses_nearby", args, result_str)
            return result_str
        
        @function_tool
        def get_nearby_landmarks(community: str, landmark_type: Optional[str] = None,
                                  max_distance_m: Optional[int] = None) -> str:
            """小区周边地标。community小区名；landmark_type: shopping/park；max_distance_m米默认3000。"""
            args = {"community": community, "landmark_type": landmark_type, "max_distance_m": max_distance_m}
            result = tools_instance.get_nearby_landmarks(community, landmark_type, max_distance_m)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_nearby_landmarks", args, result_str)
            return result_str
        
        @function_tool
        def get_house_stats() -> str:
            """房源统计：总套数、状态/区域/户型分布、价格区间。"""
            result = tools_instance.get_house_stats()
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("get_house_stats", {}, result_str)
            return result_str
        
        @function_tool
        def rent_house(house_id: str, listing_platform: str) -> str:
            """租房：设为已租。须调API。house_id如HF_2001；listing_platform链家/安居客/58同城。"""
            args = {"house_id": house_id, "listing_platform": listing_platform}
            result = tools_instance.rent_house(house_id, listing_platform)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("rent_house", args, result_str)
            return result_str
        
        @function_tool
        def terminate_rental(house_id: str, listing_platform: str) -> str:
            """退租：恢复可租。house_id、listing_platform必填。"""
            args = {"house_id": house_id, "listing_platform": listing_platform}
            result = tools_instance.terminate_rental(house_id, listing_platform)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("terminate_rental", args, result_str)
            return result_str
        
        @function_tool
        def take_offline(house_id: str, listing_platform: str) -> str:
            """下架房源。house_id、listing_platform必填。"""
            args = {"house_id": house_id, "listing_platform": listing_platform}
            result = tools_instance.take_offline(house_id, listing_platform)
            result_str = _truncate_result(json.dumps(result, ensure_ascii=False))
            _log_tool("take_offline", args, result_str)
            return result_str
        
        return [
            init_houses,
            get_landmarks,
            get_landmark_by_name,
            search_landmarks,
            get_landmark_by_id,
            get_landmark_stats,
            get_house_by_id,
            get_house_listings,
            get_houses_by_community,
            get_houses_by_platform,
            get_houses_nearby,
            get_nearby_landmarks,
            get_house_stats,
            rent_house,
            terminate_rental,
            take_offline
        ]
    
    def _create_agent(self, session_id: str) -> tuple:
        """创建Agent实例，带有Session-ID header
        
        Returns:
            tuple: (agent, openai_client) - 返回agent和client，以便后续关闭client
        """
        base_url = f"http://{self.model_ip}:8888/{self.api_version}"
        logger.info(f"创建OpenAI客户端: base_url={base_url}, Session-ID={session_id}")
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=60.0,
            default_headers={"Session-ID": session_id}
        )
        
        logger.debug("创建OpenAIChatCompletionsModel")
        model = OpenAIChatCompletionsModel(
            model="default",
            openai_client=openai_client
        )
        
        logger.debug("创建Agent")
        agent = Agent(
            name="租房助手",
            instructions=SYSTEM_PROMPT,
            model=model,
            tools=self._tools,
            model_settings=ModelSettings(max_tokens=LLM_MAX_TOKENS)
        )
        return agent, openai_client
    
    def chat(self, session_id: Optional[str], message: str) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            session_id: 会话ID，为空则创建新会话
            message: 用户消息
            
        Returns:
            包含session_id和response的字典
        """
        import asyncio
        
        logger.info(f"chat开始: session_id={session_id}, message={message[:50]}...")
        
        session_id = self.session_manager.get_or_create_session(session_id)
        logger.debug(f"使用session_id: {session_id}")
        
        set_current_session(session_id)
        
        session_logger.log_user_request(session_id, message)
        
        if not self.session_manager.is_session_initialized(session_id):
            logger.info("初始化房源数据")
            session_logger.log_tool_request(session_id, "init_houses", {})
            init_result = self.tools_instance.init_houses()
            session_logger.log_tool_response(session_id, "init_houses", json.dumps(init_result, ensure_ascii=False))
            self.session_manager.mark_session_initialized(session_id)
        
        history = self.session_manager.get_messages(session_id)
        logger.debug(f"历史消息数量: {len(history)}")
        # 只保留最近 N 条历史，节省 token
        if len(history) > MAX_HISTORY_MESSAGES:
            history = history[-MAX_HISTORY_MESSAGES:]
            logger.debug(f"截断为最近 {MAX_HISTORY_MESSAGES} 条历史")
        
        input_messages = []
        for msg in history:
            if msg.get("role") == "user":
                input_messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "assistant" and msg.get("content"):
                input_messages.append({"role": "assistant", "content": msg["content"]})
        
        input_messages.append({"role": "user", "content": message})
        logger.debug(f"输入消息数量: {len(input_messages)}")
        
        session_logger.log_model_request(session_id, input_messages)
        
        openai_client = None
        try:
            logger.info("开始调用Runner.run")
            
            agent, openai_client = self._create_agent(session_id)
            
            result = self._run_agent_sync(agent, input_messages, session_id)
            
            response_text = result.final_output or "抱歉，我暂时无法处理您的请求"
            logger.info(f"Runner.run完成，响应长度: {len(response_text)}")
            
            session_logger.log_model_response(session_id, response_text)
            
        except Exception as e:
            error_msg = f"处理请求时发生错误: {str(e)}"
            logger.error(error_msg)
            logger.error(f"异常堆栈:\n{traceback.format_exc()}")
            session_logger.log_error(session_id, type(e).__name__, str(e), traceback.format_exc())
            response_text = error_msg
        finally:
            if openai_client is not None:
                try:
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(openai_client.close())
                    finally:
                        loop.close()
                    logger.debug("OpenAI客户端已关闭")
                except Exception as close_err:
                    logger.debug(f"关闭OpenAI客户端时出现异常（可忽略）: {close_err}")
        
        self.session_manager.add_message(session_id, "user", message)
        self.session_manager.add_message(session_id, "assistant", response_text)
        
        session_logger.log_agent_response(session_id, response_text)
        
        logger.info(f"chat完成: session_id={session_id}")
        return {
            "session_id": session_id,
            "response": response_text
        }
    
    def _run_agent_sync(self, agent, input_messages, session_id: str):
        """在新的事件循环中同步运行agent"""
        import asyncio
        
        set_current_session(session_id)
        
        return Runner.run_sync(agent, input=input_messages)
