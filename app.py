"""
Flask应用主入口 - 提供HTTP接口
"""
from flask import Flask, request, jsonify
from agent import RentalAgent

app = Flask(__name__)

API_PORT = 8080
DEFAULT_USER_ID = "l00576045"

agents = {}


def get_agent(model_ip: str, user_id: str = DEFAULT_USER_ID) -> RentalAgent:
    """获取或创建Agent实例"""
    key = f"{model_ip}_{user_id}"
    if key not in agents:
        api_base_url = f"http://{model_ip}:{API_PORT}"
        agents[key] = RentalAgent(model_ip, api_base_url, user_id)
    return agents[key]


@app.route('/api/v1/chat', methods=['POST'])
def chat():
    """
    聊天接口
    
    请求体:
    {
        "model_ip": "模型资源接口IP",
        "session_id": "会话ID（可选，用于多轮对话）",
        "message": "用户消息"
    }
    
    响应:
    {
        "session_id": "会话ID",
        "response": "回复内容"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "请求体不能为空"
            }), 400
        
        model_ip = data.get('model_ip')
        if not model_ip:
            return jsonify({
                "error": "model_ip 是必填字段"
            }), 400
        
        message = data.get('message')
        if not message:
            return jsonify({
                "error": "message 是必填字段"
            }), 400
        
        session_id = data.get('session_id')
        
        agent = get_agent(model_ip, DEFAULT_USER_ID)
        
        result = agent.chat(session_id, message)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "error": f"服务器内部错误: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    app.run()
