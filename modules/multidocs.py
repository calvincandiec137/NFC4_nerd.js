import requests
import json
import os
from dotenv import load_dotenv
import chardet

def read_file_with_encoding_detection(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        detected_encoding = result['encoding']

    with open(file_path, 'r', encoding=detected_encoding, errors='ignore') as f:
        return f.read()


load_dotenv()
api_key = os.getenv("OPEN_ROUTER")

# Load and prepare the two documents
def multi(doc1_path, doc2_path):
    doc1 = read_file_with_encoding_detection(doc1_path)
    doc2 = read_file_with_encoding_detection(doc2_path)

    # Initial context shared with model
    base_context = f"""
    You are a comparative analysis assistant. Use the following two documents for all future questions:

    Document 1:
    {doc1}

    ---

    Document 2:
    {doc2}
    """

    # Start chat loop
    chat_history = [
        {"role": "system", "content": base_context}
    ]

    print("Multi-Doc Comparative Chat. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        chat_history.append({"role": "user", "content": user_input})

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "mistralai/mistral-small-3.2-24b-instruct:free",
                "messages": chat_history,
            })
        )

        if response.status_code == 200:
            reply = response.json()["choices"][0]["message"]["content"]
            print(f"\nAssistant: {reply}")
            chat_history.append({"role": "assistant", "content": reply})
        else:
            print("Error:", response.status_code, response.text)
