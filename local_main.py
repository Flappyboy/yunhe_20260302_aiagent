"""
测试入口 - 通过HTTP接口调用进行测试
"""
import json
import sys
import requests
from typing import List, Dict, Any


TEST_CASES = [
    {
        "name": "用例1：东城区精装两居查询（无结果）",
        "rounds": [
            {
                "session_id": "EV-43",
                "user_input": "东城区精装两居，租金 5000 以内，离地铁 500 米以内的有吗？",
                "expected": {
                    "message_contains": ["没有"],
                    "expectedHouses": []
                }
            }
        ]
    },
    {
        "name": "用例2：西城区一居室多轮对话",
        "rounds": [
            {
                "session_id": "EV-46",
                "user_input": "西城区离地铁近的一居室有吗？按离地铁从近到远排。",
                "expected": {
                    "message_contains": ["西城", "1", "800", "subway_distance", "asc"],
                    "expectedHouses": ["HF_13"]
                }
            },
            {
                "session_id": "EV-46",
                "user_input": "还有其他的吗？把所有符合条件的都给出来",
                "expected": {
                    "message_contains": ["没有其他的了，只有这一套"],
                    "expectedHouses": ["HF_13"]
                }
            }
        ]
    },
    {
        "name": "用例3：海淀区两居室查询并租房",
        "rounds": [
            {
                "session_id": "EV-45",
                "user_input": "海淀区离地铁近的两居有吗？按离地铁从近到远排一下。",
                "expected": {
                    "message_contains": ["海淀", "2", "800", "subway_distance", "asc"],
                    "expectedHouses": ["HF_906", "HF_1586", "HF_1876", "HF_706", "HF_33"]
                }
            },
            {
                "session_id": "EV-45",
                "user_input": "就租最近的那套吧。",
                "expected": {
                    "message_contains": ["好的"],
                    "expectedHouses": ["HF_906"]
                }
            }
        ]
    }
]


AGENT_BASE_URL = "http://127.0.0.1:8448"
MODEL_IP = "7.225.29.223"


def call_chat_api(model_ip: str, message: str, 
                  session_id: str = None) -> Dict[str, Any]:
    """调用聊天HTTP接口"""
    url = f"{AGENT_BASE_URL}/api/v1/chat"
    
    payload = {
        "model_ip": model_ip,
        "message": message
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "session_id": None, "response": f"HTTP请求失败: {e}"}


def parse_response(response: str) -> Dict[str, Any]:
    """解析响应内容，尝试提取JSON"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"message": response, "houses": []}


def check_message_contains(response: str, expected_keywords: List[str]) -> tuple:
    """检查响应是否包含期望的关键词"""
    missing = []
    for keyword in expected_keywords:
        if keyword.lower() not in response.lower():
            missing.append(keyword)
    return len(missing) == 0, missing


def check_houses(actual_houses: List[str], expected_houses: List[str]) -> tuple:
    """检查房源ID是否匹配"""
    if not expected_houses:
        return len(actual_houses) == 0, actual_houses
    
    missing = [h for h in expected_houses if h not in actual_houses]
    extra = [h for h in actual_houses if h not in expected_houses]
    
    return len(missing) == 0, {"missing": missing, "extra": extra}


def run_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个测试用例"""
    results = {
        "name": test_case["name"],
        "rounds": [],
        "passed": True
    }
    
    session_map = {}
    
    for i, round_data in enumerate(test_case["rounds"]):
        round_result = {
            "round": i + 1,
            "user_input": round_data["user_input"],
            "passed": True,
            "errors": []
        }
        
        original_session_id = round_data.get("session_id")
        if original_session_id and original_session_id in session_map:
            actual_session_id = session_map[original_session_id]
        else:
            actual_session_id = None
        
        print(f"\n  第{i+1}轮对话:")
        print(f"  用户: {round_data['user_input']}")
        
        response = call_chat_api(MODEL_IP, round_data["user_input"], 
                                  actual_session_id)
        
        if "error" in response:
            round_result["passed"] = False
            round_result["errors"].append(f"API调用失败: {response['error']}")
            results["passed"] = False
            print(f"  ❌ API调用失败: {response['error']}")
            results["rounds"].append(round_result)
            continue
        
        if original_session_id and original_session_id not in session_map:
            session_map[original_session_id] = response["session_id"]
        
        response_text = response.get("response", "")
        print(f"  助手: {response_text[:200]}..." if len(response_text) > 200 else f"  助手: {response_text}")
        
        round_result["response"] = response_text
        round_result["session_id"] = response.get("session_id")
        
        expected = round_data.get("expected", {})
        
        if "message_contains" in expected:
            passed, missing = check_message_contains(response_text, expected["message_contains"])
            if not passed:
                round_result["passed"] = False
                round_result["errors"].append(f"缺少关键词: {missing}")
        
        if "expectedHouses" in expected:
            parsed = parse_response(response_text)
            actual_houses = parsed.get("houses", [])
            passed, diff = check_houses(actual_houses, expected["expectedHouses"])
            if not passed:
                round_result["passed"] = False
                if expected["expectedHouses"]:
                    round_result["errors"].append(f"房源不匹配: {diff}")
                else:
                    round_result["errors"].append(f"期望无房源，实际返回: {diff}")
        
        if not round_result["passed"]:
            results["passed"] = False
            print(f"  ❌ 失败: {round_result['errors']}")
        else:
            print(f"  ✓ 通过")
        
        results["rounds"].append(round_result)
    
    return results


def run_all_tests():
    """运行所有测试用例"""
    print("=" * 60)
    print("租房AI Agent HTTP接口测试")
    print("=" * 60)
    print(f"Agent服务地址: {AGENT_BASE_URL}")
    print(f"模型IP: {MODEL_IP}:8888")
    print("=" * 60)
    
    all_results = []
    passed_count = 0
    
    for i, test_case in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"测试用例 {i+1}: {test_case['name']}")
        print("-" * 60)
        
        result = run_test_case(test_case)
        all_results.append(result)
        
        if result["passed"]:
            passed_count += 1
            print(f"\n✓ 用例通过")
        else:
            print(f"\n❌ 用例失败")
    
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    print(f"总用例数: {len(TEST_CASES)}")
    print(f"通过: {passed_count}")
    print(f"失败: {len(TEST_CASES) - passed_count}")
    print("=" * 60)
    
    return all_results


def run_single_test(test_index: int):
    """运行单个测试用例"""
    if test_index < 0 or test_index >= len(TEST_CASES):
        print(f"错误: 测试用例索引 {test_index} 超出范围 [0, {len(TEST_CASES)-1}]")
        return None
    
    test_case = TEST_CASES[test_index]
    print(f"\n运行测试用例 {test_index + 1}: {test_case['name']}")
    print("-" * 60)
    
    return run_test_case(test_case)


def interactive_mode():
    """交互模式，手动输入进行测试"""
    print("=" * 60)
    print("交互测试模式（HTTP接口）")
    print(f"Agent服务地址: {AGENT_BASE_URL}")
    print(f"模型IP: {MODEL_IP}:8888")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'new' 开始新会话")
    print("=" * 60)
    
    session_id = None
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("退出交互模式")
                break
            
            if user_input.lower() == 'new':
                session_id = None
                print("已开始新会话")
                continue
            
            if not user_input:
                continue
            
            response = call_chat_api(MODEL_IP, user_input, session_id)
            
            if "error" in response:
                print(f"\n错误: {response['error']}")
                continue
            
            session_id = response.get("session_id")
            
            print(f"\n助手: {response.get('response', '')}")
            print(f"\n[Session ID: {session_id}]")
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break


def main():
    """主入口"""
    print("=" * 60)
    print("租房AI Agent 测试工具")
    print("=" * 60)
    print(f"Agent服务地址: {AGENT_BASE_URL}")
    print(f"模型IP: {MODEL_IP}:8888")
    print("=" * 60)
    
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if command == "all":
        run_all_tests()
    elif command == "test":
        if len(sys.argv) < 3:
            print("错误: 请指定测试用例索引")
            print("用法: python test_main.py test <index>")
            return
        test_index = int(sys.argv[2])
        run_single_test(test_index)
    elif command == "interactive":
        interactive_mode()
    elif command in ["-h", "--help", "help"]:
        print("")
        print("命令:")
        print("  all          - 运行所有测试用例（默认）")
        print("  test <index> - 运行指定测试用例（索引从0开始）")
        print("  interactive  - 交互模式")
        print("")
        print("示例:")
        print("  python test_main.py")
        print("  python test_main.py all")
        print("  python test_main.py test 0")
        print("  python test_main.py interactive")
    else:
        print(f"未知命令: {command}")
        print("使用 python test_main.py help 查看帮助")


if __name__ == "__main__":
    main()
