import base64
import requests

with open("/home/faizmk/NFC4_nerd.js/database/sample_document.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "name": "sample.pdf",
    "data": b64
}

r = requests.post("http://127.0.0.1:8000/upload", json=payload)

print("Status:", r.status_code)
print("Raw response:", r.text)
print("JSON:", r.json())
