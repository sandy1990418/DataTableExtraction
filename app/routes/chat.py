from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.models import ChatRequest
from app.services.llm_service import chat

router = APIRouter()


@router.post("/chat", summary="Send text content; returns download URL if a table is generated")
async def chat_endpoint(body: ChatRequest, settings: Settings = Depends(get_settings)):
    return await chat(body.message, settings)
