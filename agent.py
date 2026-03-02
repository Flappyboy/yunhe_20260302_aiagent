"""
Agent核心模块 - 处理用户意图和工具调用
"""
import json
import re
from typing import Dict, Any, List, Optional

from tools import HouseAPITools, TOOLS_DEFINITION
from llm_client import LLMClient
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
    """租房AI Agent"""
    
    def __init__(self, model_ip: str, api_base_url: str, user_id: str):
        """
        初始化Agent
        
        Args:
            model_ip: LLM模型服务IP
            api_base_url: 房源API基础URL
            user_id: 用户工号
        """
        self.llm_client = LLMClient(model_ip)
        self.tools = HouseAPITools(api_base_url, user_id)
        self.session_manager = SessionManager()
        self.user_id = user_id
    
    def _execute_tool(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具调用"""
        tool_methods = {
            "init_houses": lambda args: self.tools.init_houses(),
            "get_landmarks": lambda args: self.tools.get_landmarks(
                args.get("category"), args.get("district")
            ),
            "get_landmark_by_name": lambda args: self.tools.get_landmark_by_name(
                args["name"]
            ),
            "search_landmarks": lambda args: self.tools.search_landmarks(
                args["q"], args.get("category"), args.get("district")
            ),
            "get_landmark_by_id": lambda args: self.tools.get_landmark_by_id(
                args["landmark_id"]
            ),
            "get_landmark_stats": lambda args: self.tools.get_landmark_stats(),
            "get_house_by_id": lambda args: self.tools.get_house_by_id(
                args["house_id"]
            ),
            "get_house_listings": lambda args: self.tools.get_house_listings(
                args["house_id"]
            ),
            "get_houses_by_community": lambda args: self.tools.get_houses_by_community(
                args["community"],
                args.get("listing_platform"),
                args.get("page", 1),
                args.get("page_size", 10)
            ),
            "get_houses_by_platform": lambda args: self.tools.get_houses_by_platform(
                listing_platform=args.get("listing_platform"),
                district=args.get("district"),
                area=args.get("area"),
                min_price=args.get("min_price"),
                max_price=args.get("max_price"),
                bedrooms=args.get("bedrooms"),
                rental_type=args.get("rental_type"),
                decoration=args.get("decoration"),
                orientation=args.get("orientation"),
                elevator=args.get("elevator"),
                min_area=args.get("min_area"),
                max_area=args.get("max_area"),
                property_type=args.get("property_type"),
                subway_line=args.get("subway_line"),
                max_subway_dist=args.get("max_subway_dist"),
                subway_station=args.get("subway_station"),
                utilities_type=args.get("utilities_type"),
                available_from_before=args.get("available_from_before"),
                commute_to_xierqi_max=args.get("commute_to_xierqi_max"),
                sort_by=args.get("sort_by"),
                sort_order=args.get("sort_order"),
                page=args.get("page", 1),
                page_size=args.get("page_size", 10)
            ),
            "get_houses_nearby": lambda args: self.tools.get_houses_nearby(
                args["landmark_id"],
                args.get("max_distance"),
                args.get("listing_platform"),
                args.get("page", 1),
                args.get("page_size", 10)
            ),
            "get_nearby_landmarks": lambda args: self.tools.get_nearby_landmarks(
                args["community"],
                args.get("landmark_type"),
                args.get("max_distance_m")
            ),
            "get_house_stats": lambda args: self.tools.get_house_stats(),
            "rent_house": lambda args: self.tools.rent_house(
                args["house_id"], args["listing_platform"]
            ),
            "terminate_rental": lambda args: self.tools.terminate_rental(
                args["house_id"], args["listing_platform"]
            ),
            "take_offline": lambda args: self.tools.take_offline(
                args["house_id"], args["listing_platform"]
            )
        }
        
        if function_name not in tool_methods:
            return json.dumps({"error": f"未知工具: {function_name}"}, ensure_ascii=False)
        
        try:
            result = tool_methods[function_name](arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)
    
    def _is_house_query_response(self, content: str) -> bool:
        """判断是否是房源查询响应"""
        if not content:
            return False
        try:
            data = json.loads(content)
            return "houses" in data and isinstance(data["houses"], list)
        except (json.JSONDecodeError, TypeError):
            return False
    
    def _format_response(self, content: str, is_house_query: bool = False) -> str:
        """格式化响应内容"""
        if not content:
            return json.dumps({
                "message": "抱歉，我暂时无法处理您的请求",
                "houses": []
            }, ensure_ascii=False) if is_house_query else "抱歉，我暂时无法处理您的请求"
        
        if is_house_query:
            try:
                data = json.loads(content)
                if "houses" in data:
                    return content
            except (json.JSONDecodeError, TypeError):
                pass
            return json.dumps({
                "message": content,
                "houses": []
            }, ensure_ascii=False)
        
        return content
    
    def chat(self, session_id: Optional[str], message: str) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            session_id: 会话ID，为空则创建新会话
            message: 用户消息
            
        Returns:
            包含session_id和response的字典
        """
        session_id = self.session_manager.get_or_create_session(session_id)
        
        if not self.session_manager.is_session_initialized(session_id):
            self.tools.init_houses()
            self.session_manager.mark_session_initialized(session_id)
        
        self.session_manager.add_message(session_id, "user", message)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.session_manager.get_messages(session_id))
        
        max_iterations = 10
        iteration = 0
        final_content = None
        
        while iteration < max_iterations:
            iteration += 1
            
            response = self.llm_client.chat_completion(
                messages=messages,
                session_id=session_id,
                tools=TOOLS_DEFINITION
            )
            
            result = self.llm_client.extract_response(response)
            
            if result["finish_reason"] == "error":
                final_content = result["content"]
                break
            
            if result["tool_calls"]:
                self.session_manager.add_tool_call(session_id, result["tool_calls"])
                messages.append({
                    "role": "assistant",
                    "content": result["content"],
                    "tool_calls": result["tool_calls"]
                })
                
                for tool_call in result["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    tool_result = self._execute_tool(function_name, arguments)
                    
                    self.session_manager.add_tool_result(
                        session_id,
                        tool_call["id"],
                        function_name,
                        tool_result
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": function_name,
                        "content": tool_result
                    })
            else:
                final_content = result["content"]
                break
        
        if final_content is None:
            final_content = "处理超时，请重试"
        
        self.session_manager.add_message(session_id, "assistant", final_content)
        
        return {
            "session_id": session_id,
            "response": final_content
        }
