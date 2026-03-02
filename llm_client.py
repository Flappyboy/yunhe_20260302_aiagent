"""
LLM客户端模块 - 与OpenAI兼容API交互
"""
import requests
import json
from typing import List, Dict, Any, Optional


class LLMClient:
    """LLM客户端，与OpenAI兼容的Chat Completion API交互"""
    
    def __init__(self, model_ip: str, port: int = 8888):
        """
        初始化LLM客户端
        
        Args:
            model_ip: 模型服务IP地址
            port: 端口号，默认8888
        """
        self.base_url = f"http://{model_ip}:{port}"
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
    
    def chat_completion(self, 
                        messages: List[Dict[str, Any]], 
                        session_id: str,
                        tools: Optional[List[Dict]] = None,
                        model: str = "default",
                        temperature: float = 0.7,
                        max_tokens: int = 4096) -> Dict[str, Any]:
        """
        调用Chat Completion API
        
        Args:
            messages: 消息列表
            session_id: 会话ID，放入请求头Session-ID
            tools: 可用工具列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            API响应
        """
        headers = {
            "Content-Type": "application/json",
            "Session-ID": session_id
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"LLM API调用失败: {str(e)}",
                "choices": []
            }
    
    def extract_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        从API响应中提取有用信息
        
        Args:
            api_response: API原始响应
            
        Returns:
            包含content和tool_calls的字典
        """
        if "error" in api_response:
            return {
                "content": api_response["error"],
                "tool_calls": None,
                "finish_reason": "error"
            }
        
        if not api_response.get("choices"):
            return {
                "content": "未获取到有效响应",
                "tool_calls": None,
                "finish_reason": "error"
            }
        
        choice = api_response["choices"][0]
        message = choice.get("message", {})
        
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason", "stop")
        }
