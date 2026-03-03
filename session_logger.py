"""
会话日志管理模块 - 按session记录所有交互内容
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)

LOG_DIR = "logs"


class SessionLogger:
    """会话日志管理器，按session记录所有交互"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._session_logs: Dict[str, list] = {}
        self._file_locks: Dict[str, Lock] = {}
        
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
    
    def _get_file_lock(self, session_id: str) -> Lock:
        """获取session对应的文件锁"""
        if session_id not in self._file_locks:
            self._file_locks[session_id] = Lock()
        return self._file_locks[session_id]
    
    def _get_log_file_path(self, session_id: str) -> str:
        """获取session日志文件路径"""
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(LOG_DIR, f"session_{safe_session_id}.log")
    
    def _write_to_file(self, session_id: str, log_entry: Dict[str, Any]):
        """写入日志到文件"""
        file_path = self._get_log_file_path(session_id)
        lock = self._get_file_lock(session_id)
        
        with lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def _create_log_entry(self, session_id: str, log_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建日志条目"""
        return {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "type": log_type,
            "data": data
        }
    
    def log_user_request(self, session_id: str, message: str):
        """记录用户请求"""
        log_entry = self._create_log_entry(session_id, "USER_REQUEST", {
            "message": message
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] USER_REQUEST: {message[:100]}...")
    
    def log_agent_response(self, session_id: str, response: str):
        """记录Agent响应"""
        log_entry = self._create_log_entry(session_id, "AGENT_RESPONSE", {
            "response": response
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] AGENT_RESPONSE: {response[:100]}...")
    
    def log_model_request(self, session_id: str, messages: list, tools: Optional[list] = None):
        """记录模型请求"""
        log_entry = self._create_log_entry(session_id, "MODEL_REQUEST", {
            "messages": messages,
            "tools_count": len(tools) if tools else 0
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] MODEL_REQUEST: messages_count={len(messages)}")
    
    def log_model_response(self, session_id: str, response: Any, tool_calls: Optional[list] = None):
        """记录模型响应（不含 token 信息）"""
        log_entry = self._create_log_entry(session_id, "MODEL_RESPONSE", {
            "content": str(response)[:500] if response else None,
            "tool_calls": tool_calls
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] MODEL_RESPONSE: content_length={len(str(response)) if response else 0}, tool_calls={len(tool_calls) if tool_calls else 0}")

    def log_model_usage(self, session_id: str, usage: Optional[Dict[str, Any]]):
        """记录本次调用的 token 消耗"""
        if not usage:
            return
        log_entry = self._create_log_entry(session_id, "MODEL_USAGE", {
            "usage": usage
        })
        self._write_to_file(session_id, log_entry)
        # 尽量简洁，只打出总 token
        total = usage.get("total_tokens") or usage.get("total") or usage
        logger.info(f"[{session_id}] MODEL_USAGE: {total}")
    
    def log_tool_request(self, session_id: str, tool_name: str, arguments: Dict[str, Any]):
        """记录工具调用请求"""
        log_entry = self._create_log_entry(session_id, "TOOL_REQUEST", {
            "tool_name": tool_name,
            "arguments": arguments
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] TOOL_REQUEST: {tool_name}({json.dumps(arguments, ensure_ascii=False)[:200]})")
    
    def log_tool_response(self, session_id: str, tool_name: str, result: Any):
        """记录工具调用响应"""
        result_str = str(result) if result else ""
        log_entry = self._create_log_entry(session_id, "TOOL_RESPONSE", {
            "tool_name": tool_name,
            "result": result_str[:2000],
            "result_length": len(result_str)
        })
        self._write_to_file(session_id, log_entry)
        logger.info(f"[{session_id}] TOOL_RESPONSE: {tool_name} -> {result_str[:200]}...")
    
    def log_error(self, session_id: str, error_type: str, error_message: str, stack_trace: Optional[str] = None):
        """记录错误"""
        log_entry = self._create_log_entry(session_id, "ERROR", {
            "error_type": error_type,
            "error_message": error_message,
            "stack_trace": stack_trace
        })
        self._write_to_file(session_id, log_entry)
        logger.error(f"[{session_id}] ERROR: {error_type} - {error_message}")


session_logger = SessionLogger()
