import os
import gradio as gr
from mistralai import Mistral

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French cheese?",
        },
    ]
)
print(chat_response.choices[0].message.content)