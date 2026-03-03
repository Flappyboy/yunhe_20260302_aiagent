"""
配置文件
"""

# 模型服务配置
MODEL_PORT = 8888

# 房源API配置
API_PORT = 8080

# Flask服务配置
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 8448
FLASK_DEBUG = False

# 会话配置
SESSION_TIMEOUT = 3600

# LLM配置
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 4096
LLM_MODEL = "default"

# Agent配置（节省 token）
MAX_TOOL_ITERATIONS = 10
# 送入模型的最大历史消息条数（user+assistant 合计），超出只保留最近 N 条
MAX_HISTORY_MESSAGES = 5
# 单次工具返回给模型的最大字符数，超出则截断并注明
MAX_TOOL_RESULT_CHARS = 8000
