import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPEN_ROUTER")

# Load and prepare the two documents
def multi(doc1_path, doc2_path):
    with open(f"{doc1_path}", "r") as f:
        doc1 = f.read()

    with open(f"{doc2_path}", "r") as f:
        doc2 = f.read()

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
