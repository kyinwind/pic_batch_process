import requests
import json

url = "http://192.168.1.105:11434/api/generate"
data = {
    "model": "deepseek-r1:8b",
    "prompt": "什么是人工智能？",
    "stream": False,  # 关闭流式输出，一次性获取结果
}

response = requests.post(url, json=data)
result = json.loads(response.text)
print(result["response"])
