"""
Flask应用主入口 - 提供HTTP接口
"""
import logging
import traceback
from flask import Flask, request, jsonify
from agent import RentalAgent

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

API_PORT = 8080
DEFAULT_USER_ID = "l00576045"

agents = {}


def get_agent(model_ip: str, user_id: str = DEFAULT_USER_ID, api_version: str = "v1") -> RentalAgent:
    """获取或创建Agent实例"""
    key = f"{model_ip}_{user_id}_{api_version}"
    if key not in agents:
        logger.info(f"创建新的Agent实例: model_ip={model_ip}, user_id={user_id}, api_version={api_version}")
        api_base_url = f"http://{model_ip}:{API_PORT}"
        agents[key] = RentalAgent(model_ip, api_base_url, user_id, api_version)
    return agents[key]


@app.route('/api/v1/chat', methods=['POST'])
def chat():
    """
    聊天接口
    
    请求体:
    {
        "model_ip": "模型资源接口IP",
        "session_id": "会话ID（可选，用于多轮对话）",
        "message": "用户消息",
        "api_version": "模型API版本，v1或v2（可选，默认v2）"
    }
    
    响应:
    {
        "session_id": "会话ID",
        "response": "回复内容"
    }
    """
    try:
        logger.info("收到聊天请求")
        data = request.get_json()
        logger.debug(f"请求数据: {data}")
        
        if not data:
            logger.warning("请求体为空")
            return jsonify({
                "error": "请求体不能为空"
            }), 400
        
        model_ip = data.get('model_ip')
        if not model_ip:
            logger.warning("缺少model_ip字段")
            return jsonify({
                "error": "model_ip 是必填字段"
            }), 400
        
        message = data.get('message')
        if not message:
            logger.warning("缺少message字段")
            return jsonify({
                "error": "message 是必填字段"
            }), 400
        
        session_id = data.get('session_id')
        api_version = data.get('api_version', 'v1')
        logger.info(f"处理消息: model_ip={model_ip}, session_id={session_id}, api_version={api_version}, message={message[:50]}...")
        
        agent = get_agent(model_ip, DEFAULT_USER_ID, api_version)
        
        logger.info("开始调用agent.chat")
        result = agent.chat(session_id, message)
        logger.info(f"agent.chat返回: session_id={result.get('session_id')}")
        logger.debug(f"响应内容: {result.get('response', '')[:200]}...")
        
        return jsonify(result)
    
    except Exception as e:
        error_msg = f"服务器内部错误: {str(e)}"
        logger.error(f"处理请求时发生异常: {error_msg}")
        logger.error(f"异常堆栈:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    logger.info("启动Flask服务: host=0.0.0.0, port=8448")
    app.run(host='0.0.0.0', port=8448, debug=False)
