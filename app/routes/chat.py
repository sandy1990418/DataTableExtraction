from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.services.llm_service import chat

router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat", summary="Send text content; returns download URL if a table is generated")
async def chat_endpoint(body: ChatRequest, settings: Settings = Depends(get_settings)):
    return await chat(body.message, settings)
