# 租房AI Agent助手

基于Python Flask的租房AI Agent助手，提供智能租房查询和管理服务。

## 功能特性

- 多维度房源查询（区域、价格、户型、装修、朝向、地铁距离等）
- 地标查询（地铁站、公司、商圈）
- 房源操作（租房、退租、下架）
- 多轮对话支持
- 与OpenAI兼容的LLM API集成

## 项目结构

```
├── app.py              # Flask应用主入口
├── agent.py            # Agent核心模块
├── tools.py            # 房源API工具封装
├── llm_client.py       # LLM客户端
├── session.py          # 会话管理
├── config.py           # 配置文件
├── requirements.txt    # 依赖文件
└── README.md           # 说明文档
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python app.py
```

服务将在 `http://0.0.0.0:5000` 启动。

## API接口

### POST /api/v1/chat

聊天接口，用于与租房助手对话。

**请求体：**

```json
{
    "model_ip": "模型资源接口IP",
    "session_id": "会话ID（可选，用于多轮对话）",
    "message": "用户消息",
    "user_id": "用户工号（可选）"
}
```

**响应：**

```json
{
    "session_id": "会话ID",
    "response": "回复内容"
}
```

**回复格式说明：**

- 普通对话：回复自然语言文本
- 房源查询：回复JSON字符串

```json
{
    "message": "为您找到以下符合条件的房源：",
    "houses": ["HF_4", "HF_6", "HF_277"]
}
```

### GET /health

健康检查接口。

## 使用示例

```bash
# 查询房源
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_ip": "192.168.1.100",
    "message": "帮我找海淀区3000元以下的一居室"
  }'

# 多轮对话
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model_ip": "192.168.1.100",
    "session_id": "之前返回的session_id",
    "message": "有近地铁的吗"
  }'
```

## 配置说明

- 模型服务端口：8888
- 房源API端口：8080
- Flask服务端口：5000

## 注意事项

1. 请求头中需要携带正确的用户工号（X-User-ID）
2. 每个新会话会自动调用房源数据重置接口
3. 租房/退租/下架操作必须调用对应API才算完成
