# File name: model_client.py
import requests

response = requests.post("http://127.0.0.1:8000/infer")
french_text = response.json()

print(french_text)