"""
Agent核心模块 - 使用OpenAI Agents SDK实现
"""
import json
from typing import Dict, Any, Optional

from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, set_tracing_disabled, set_default_openai_client
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from tools import HouseAPITools
from session import SessionManager


SYSTEM_PROMPT = """你是一个专业的北京租房助手，帮助用户查找和管理房源信息。

## 你的能力
1. 查询房源：支持按区域、价格、户型、装修、朝向、地铁距离、通勤时间等多维度筛选
2. 查询地标：地铁站、公司、商圈等
3. 房源操作：租房、退租、下架

## 数据范围
- 覆盖区域：北京（海淀、朝阳、通州、昌平、大兴、房山、西城、丰台、顺义、东城）
- 价格区间：约500-25000元/月
- 支持查询维度：价格、户型、区域、地铁距离、附近地标、可入住日期、西二旗通勤时间等

## 重要概念
- 近地铁：房源到最近地铁站距离800米以内
- 地铁可达：房源到最近地铁站距离1000米以内
- 整租：租整套房子
- 合租：租单间

## 工作流程
1. 理解用户需求，提取关键筛选条件
2. 调用合适的工具查询数据
3. 整理结果，给出清晰的回复

## 回复格式要求
- 普通对话：直接用自然语言回复
- 房源查询结果：必须返回JSON格式，包含message（给用户的说明）和houses（房源ID列表）
  示例：{"message": "为您找到以下符合条件的房源：", "houses": ["HF_4", "HF_6", "HF_277"]}
- 如果查询无结果，也返回JSON：{"message": "抱歉，未找到符合条件的房源，建议放宽筛选条件", "houses": []}

## 注意事项
1. 租房/退租/下架必须调用对应API才算完成，仅回复文字无效
2. 操作时必须指定listing_platform（链家/安居客/58同城），默认使用安居客
3. 查询地标附近房源时，先通过地标接口获取landmark_id，再调用nearby接口
4. 回复要简洁专业，突出关键信息"""


class RentalAgent:
    """租房AI Agent - 使用OpenAI Agents SDK"""
    
    def __init__(self, model_ip: str, api_base_url: str, user_id: str):
        """
        初始化Agent
        
        Args:
            model_ip: LLM模型服务IP
            api_base_url: 房源API基础URL
            user_id: 用户工号
        """
        self.model_ip = model_ip
        self.api_base_url = api_base_url
        self.user_id = user_id
        self.session_manager = SessionManager()
        
        set_tracing_disabled(True)
        
        self.tools_instance = HouseAPITools(api_base_url, user_id)
        
        self._tools = self._create_tools()
        
        self._agent = self._create_agent()
    
    def _create_tools(self):
        """创建工具函数列表"""
        tools_instance = self.tools_instance
        
        @function_tool
        def init_houses() -> str:
            """重置房源数据到初始状态。在新会话开始时调用，确保数据干净。"""
            result = tools_instance.init_houses()
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_landmarks(category: Optional[str] = None, district: Optional[str] = None) -> str:
            """获取地标列表，支持按类别和行政区筛选。用于查地铁站、公司、商圈等地标。
            
            Args:
                category: 地标类别：subway(地铁)/company(公司)/landmark(商圈等)
                district: 行政区，如 海淀、朝阳
            """
            result = tools_instance.get_landmarks(category, district)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_landmark_by_name(name: str) -> str:
            """按名称精确查询地标，如西二旗站、百度。返回地标id、经纬度等，用于后续nearby查房。
            
            Args:
                name: 地标名称，如 西二旗站、国贸
            """
            result = tools_instance.get_landmark_by_name(name)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def search_landmarks(q: str, category: Optional[str] = None, district: Optional[str] = None) -> str:
            """关键词模糊搜索地标，支持按类别和行政区筛选。
            
            Args:
                q: 搜索关键词
                category: 地标类别：subway/company/landmark
                district: 行政区，如 海淀、朝阳
            """
            result = tools_instance.search_landmarks(q, category, district)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_landmark_by_id(landmark_id: str) -> str:
            """按地标ID查询地标详情。
            
            Args:
                landmark_id: 地标ID，如 SS_001、LM_002
            """
            result = tools_instance.get_landmark_by_id(landmark_id)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_landmark_stats() -> str:
            """获取地标统计信息（总数、按类别分布等）。"""
            result = tools_instance.get_landmark_stats()
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_house_by_id(house_id: str) -> str:
            """根据房源ID获取单套房源详情。
            
            Args:
                house_id: 房源ID，如 HF_2001
            """
            result = tools_instance.get_house_by_id(house_id)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_house_listings(house_id: str) -> str:
            """获取房源在链家/安居客/58同城等各平台的全部挂牌记录。
            
            Args:
                house_id: 房源ID，如 HF_2001
            """
            result = tools_instance.get_house_listings(house_id)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_houses_by_community(community: str, listing_platform: Optional[str] = None,
                                     page: int = 1, page_size: int = 10) -> str:
            """按小区名查询该小区下可租房源。用于指代消解、查某小区地铁信息或隐性属性。
            
            Args:
                community: 小区名，如 建清园(南区)、保利锦上(二期)
                listing_platform: 挂牌平台：链家/安居客/58同城，不传则默认安居客
                page: 页码，默认1
                page_size: 每页条数，默认10
            """
            result = tools_instance.get_houses_by_community(community, listing_platform, page, page_size)
            return json.dumps(result, ensure_ascii=False)
        
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
            """按条件筛选可租房源，支持多维度筛选：行政区、商圈、价格、户型、装修、朝向、电梯、面积、地铁距离、通勤时间等。
            
            Args:
                listing_platform: 挂牌平台：链家/安居客/58同城，不传则默认安居客
                district: 行政区，逗号分隔，如 海淀,朝阳
                area: 商圈，逗号分隔，如 西二旗,上地
                min_price: 最低月租金（元）
                max_price: 最高月租金（元）
                bedrooms: 卧室数，逗号分隔，如 1,2
                rental_type: 整租或合租
                decoration: 装修类型：精装/简装/豪华/毛坯/空房
                orientation: 朝向：朝南/朝北/朝东/朝西/南北/东西
                elevator: 是否有电梯：true/false
                min_area: 最小面积（平米）
                max_area: 最大面积（平米）
                property_type: 物业类型，如 住宅
                subway_line: 地铁线路，如 13号线
                max_subway_dist: 最大地铁距离（米），近地铁建议800
                subway_station: 地铁站名，如 车公庄站
                utilities_type: 水电类型，如 民水民电
                available_from_before: 可入住日期上限，YYYY-MM-DD格式
                commute_to_xierqi_max: 到西二旗通勤时间上限（分钟）
                sort_by: 排序字段：price/area/subway
                sort_order: 排序方向：asc/desc
                page: 页码，默认1
                page_size: 每页条数，默认10
            """
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
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_houses_nearby(landmark_id: str, max_distance: Optional[int] = None,
                               listing_platform: Optional[str] = None,
                               page: int = 1, page_size: int = 10) -> str:
            """以地标为圆心，查询在指定距离内的可租房源，返回带直线距离、步行距离、步行时间。需先通过地标接口获得landmark_id。
            
            Args:
                landmark_id: 地标ID或地标名称
                max_distance: 最大直线距离（米），默认2000
                listing_platform: 挂牌平台：链家/安居客/58同城
                page: 页码，默认1
                page_size: 每页条数，默认10
            """
            result = tools_instance.get_houses_nearby(landmark_id, max_distance, listing_platform, page, page_size)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_nearby_landmarks(community: str, landmark_type: Optional[str] = None,
                                  max_distance_m: Optional[int] = None) -> str:
            """查询某小区周边某类地标（商超/公园），按距离排序。用于回答「附近有没有商场/公园」。
            
            Args:
                community: 小区名，用于定位基准点
                landmark_type: 地标类型：shopping(商超)/park(公园)
                max_distance_m: 最大距离（米），默认3000
            """
            result = tools_instance.get_nearby_landmarks(community, landmark_type, max_distance_m)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def get_house_stats() -> str:
            """获取房源统计信息（总套数、按状态/行政区/户型分布、价格区间等）。"""
            result = tools_instance.get_house_stats()
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def rent_house(house_id: str, listing_platform: str) -> str:
            """租房操作，将房源设为已租状态。必须调用此API才算完成租房，仅对话生成[已租]无效。
            
            Args:
                house_id: 房源ID，如 HF_2001
                listing_platform: 挂牌平台（必填）：链家/安居客/58同城
            """
            result = tools_instance.rent_house(house_id, listing_platform)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def terminate_rental(house_id: str, listing_platform: str) -> str:
            """退租操作，将房源恢复为可租状态。
            
            Args:
                house_id: 房源ID，如 HF_2001
                listing_platform: 挂牌平台（必填）：链家/安居客/58同城
            """
            result = tools_instance.terminate_rental(house_id, listing_platform)
            return json.dumps(result, ensure_ascii=False)
        
        @function_tool
        def take_offline(house_id: str, listing_platform: str) -> str:
            """下架操作，将房源设为下架状态。
            
            Args:
                house_id: 房源ID，如 HF_2001
                listing_platform: 挂牌平台（必填）：链家/安居客/58同城
            """
            result = tools_instance.take_offline(house_id, listing_platform)
            return json.dumps(result, ensure_ascii=False)
        
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
    
    def _create_agent(self) -> Agent:
        """创建Agent实例"""
        openai_client = AsyncOpenAI(
            base_url=f"http://{self.model_ip}:8888/v1",
            api_key="not-needed"
        )
        
        model = OpenAIChatCompletionsModel(
            model="default",
            openai_client=openai_client
        )
        
        return Agent(
            name="租房助手",
            instructions=SYSTEM_PROMPT,
            model=model,
            tools=self._tools
        )
    
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
        
        session_id = self.session_manager.get_or_create_session(session_id)
        
        if not self.session_manager.is_session_initialized(session_id):
            self.tools_instance.init_houses()
            self.session_manager.mark_session_initialized(session_id)
        
        history = self.session_manager.get_messages(session_id)
        
        input_messages = []
        for msg in history:
            if msg.get("role") == "user":
                input_messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "assistant" and msg.get("content"):
                input_messages.append({"role": "assistant", "content": msg["content"]})
        
        input_messages.append({"role": "user", "content": message})
        
        try:
            result = asyncio.get_event_loop().run_until_complete(
                Runner.run(self._agent, input=input_messages)
            )
            response_text = result.final_output or "抱歉，我暂时无法处理您的请求"
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    Runner.run(self._agent, input=input_messages)
                )
                response_text = result.final_output or "抱歉，我暂时无法处理您的请求"
            finally:
                loop.close()
        except Exception as e:
            response_text = f"处理请求时发生错误: {str(e)}"
        
        self.session_manager.add_message(session_id, "user", message)
        self.session_manager.add_message(session_id, "assistant", response_text)
        
        return {
            "session_id": session_id,
            "response": response_text
        }
