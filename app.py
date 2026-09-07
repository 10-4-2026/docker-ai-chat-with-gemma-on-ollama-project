import time
import requests

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


app = FastAPI()

templates = Jinja2Templates(directory="templates")

OLLAMA_URL = "http://ollama:11434/api/chat"
MODEL_NAME = "gemma3:1b"

SYSTEM_PROMPT = {
    "role": "system",
    "content": """ You are a AI assistant."""
}

# Memory theo từng user
chat_histories = {}


class ChatRequest(BaseModel):
    user_id: str = "default"
    message: str


def wait_for_ollama():
    """
    Chờ Ollama sẵn sàng.
    """

    while True:
        try:
            response = requests.get(
                "http://ollama:11434",
                timeout=5
            )

            print("✅ Ollama Ready")
            return

        except Exception:
            print("⏳ Waiting for Ollama...")
            time.sleep(5)


@app.on_event("startup")
async def startup_event():
    wait_for_ollama()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/health")
async def health():

    return {
        "status": "ok"
    }


@app.post("/chat")
async def chat(req: ChatRequest):

    if req.user_id not in chat_histories:

        chat_histories[req.user_id] = [
            SYSTEM_PROMPT.copy()
        ]

    history = chat_histories[req.user_id]

    history.append(
        {
            "role": "user",
            "content": req.message
        }
    )

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": history,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()

        answer = response.json()["message"]["content"]

        history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # chỉ giữ 20 message gần nhất
        if len(history) > 20:
            history = [history[0]] + history[-19:]
            chat_histories[req.user_id] = history

        return {
            "answer": answer
        }

    except Exception as ex:

        return {
            "answer": f"Ollama calling error: {str(ex)}"
        }


@app.post("/reset")
async def reset_memory():

    chat_histories.clear()

    return {
        "message": "Memory has been reset"
    }