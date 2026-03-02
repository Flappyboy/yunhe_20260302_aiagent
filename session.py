"""
会话管理模块 - 管理多轮对话上下文
"""
import uuid
import time
from typing import Dict, List, Any, Optional
from threading import Lock


class SessionManager:
    """会话管理器，管理多轮对话的上下文"""
    
    def __init__(self, session_timeout: int = 3600):
        """
        初始化会话管理器
        
        Args:
            session_timeout: 会话超时时间（秒），默认1小时
        """
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self._session_timeout = session_timeout
    
    def create_session(self) -> str:
        """创建新会话，返回会话ID"""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = {
                "messages": [],
                "created_at": time.time(),
                "last_access": time.time(),
                "initialized": False
            }
        return session_id
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """获取或创建会话"""
        if session_id and self.session_exists(session_id):
            self._update_last_access(session_id)
            return session_id
        return self.create_session()
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._lock:
            return session_id in self._sessions
    
    def is_session_initialized(self, session_id: str) -> bool:
        """检查会话是否已初始化（已调用房源重置接口）"""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id].get("initialized", False)
            return False
    
    def mark_session_initialized(self, session_id: str):
        """标记会话已初始化"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["initialized"] = True
    
    def _update_last_access(self, session_id: str):
        """更新会话最后访问时间"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["last_access"] = time.time()
    
    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """获取会话的消息历史"""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]["messages"].copy()
            return []
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息到会话"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["messages"].append({
                    "role": role,
                    "content": content
                })
                self._sessions[session_id]["last_access"] = time.time()
    
    def add_tool_call(self, session_id: str, tool_calls: List[Dict[str, Any]]):
        """添加工具调用消息"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["messages"].append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
    
    def add_tool_result(self, session_id: str, tool_call_id: str, 
                        function_name: str, result: str):
        """添加工具调用结果"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": result
                })
    
    def clear_session(self, session_id: str):
        """清除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        with self._lock:
            expired = [
                sid for sid, data in self._sessions.items()
                if current_time - data["last_access"] > self._session_timeout
            ]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)
