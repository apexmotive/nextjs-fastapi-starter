import os
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from groq import Groq
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.stream import patch_response_with_headers, stream_text
from .utils.tools import AVAILABLE_TOOLS, TOOL_DEFINITIONS
from vercel import oidc
from vercel.headers import set_headers


load_dotenv(".env.local")

app = FastAPI()


@app.middleware("http")
async def _vercel_set_headers(request: FastAPIRequest, call_next):
    set_headers(dict(request.headers))
    return await call_next(request)


class Request(BaseModel):
    messages: List[ClientMessage]


def get_groq_client():
    """Get Groq client with appropriate API key for local or production."""
    # For local development, use GROQ_API_KEY from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    # If no API key is set, try Vercel OIDC (for production)
    if not api_key:
        try:
            api_key = oidc.get_vercel_oidc_token()
        except Exception:
            raise ValueError(
                "Either GROQ_API_KEY must be set in .env.local for local development, "
                "or the app must be deployed on Vercel for production."
            )
    
    return Groq(api_key=api_key)


@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)

    client = get_groq_client()
    response = StreamingResponse(
        stream_text(client, openai_messages, TOOL_DEFINITIONS, AVAILABLE_TOOLS, protocol),
        media_type="text/event-stream",
    )
    return patch_response_with_headers(response, protocol)
