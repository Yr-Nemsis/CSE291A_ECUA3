import requests
import json

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "ui-tars-7b",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "the 'Customize Chrome' button at the bottom right corner of the window",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "test.png"},
                },
            ],
        }
    ],
    "temperature": 0.0,
    "max_tokens": 128,
}

resp = requests.post(url, json=payload)
print("status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
