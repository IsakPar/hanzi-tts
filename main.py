"""
HanziMaster TTS Service

A lightweight microservice for Chinese text-to-speech using Alibaba's DashScope CosyVoice API.
Optimized for accurate pronunciation of single and double character words.

Uses HTTP API instead of WebSocket for better compatibility with serverless platforms.
"""

import os
import base64
import time
import httpx
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# API Key - read at request time to ensure env var is available
def get_api_key():
    return os.getenv("DASHSCOPE_API_KEY")

# Available voices - using base names without version suffix for compatibility
VOICES = {
    "longxiaochun": {
        "id": "longxiaochun",
        "name": "Xiaochun",
        "gender": "female",
        "description": "Standard Mandarin female, gentle and clear",
        "language": "zh",
    },
    "longwan": {
        "id": "longwan",
        "name": "Wan", 
        "gender": "female",
        "description": "Sweet female voice",
        "language": "zh",
    },
    "longhua": {
        "id": "longhua",
        "name": "Hua",
        "gender": "male",
        "description": "Standard male voice",
        "language": "zh",
    },
    "longshuo": {
        "id": "longshuo",
        "name": "Shuo",
        "gender": "male", 
        "description": "Warm male voice",
        "language": "zh",
    },
    "longyue": {
        "id": "longyue",
        "name": "Yue",
        "gender": "female",
        "description": "Professional female narrator",
        "language": "zh",
    },
    "longjing": {
        "id": "longjing",
        "name": "Jing",
        "gender": "female",
        "description": "Clear female voice",
        "language": "zh",
    },
}

DEFAULT_VOICE = "longxiaochun"
# Model name - can be overridden via env var if needed
MODEL = os.getenv("TTS_MODEL", "cosyvoice")

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="HanziMaster TTS Service",
    description="Chinese text-to-speech using DashScope CosyVoice",
    version="1.0.0",
)

# CORS - allow calls from backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════

class SynthesizeRequest(BaseModel):
    """Request to synthesize speech"""
    text: str
    voice: Optional[str] = DEFAULT_VOICE
    # Optional pinyin hint for disambiguation
    pinyin: Optional[str] = None


class SynthesizeResponse(BaseModel):
    """Response with synthesized audio"""
    audioBase64: str
    format: str = "mp3"
    durationMs: Optional[int] = None
    charactersUsed: int
    voice: str
    latencyMs: int


class VoiceInfo(BaseModel):
    """Voice information"""
    id: str
    key: str
    name: str
    gender: str
    description: str
    language: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    configured: bool
    model: str
    voiceCount: int


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        configured=bool(get_api_key()),
        model=MODEL,
        voiceCount=len(VOICES),
    )


@app.get("/voices", response_model=list[VoiceInfo])
async def get_voices():
    """Get available voices"""
    return [
        VoiceInfo(
            id=v["id"],
            key=key,
            name=v["name"],
            gender=v["gender"],
            description=v["description"],
            language=v["language"],
        )
        for key, v in VOICES.items()
    ]


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from Chinese text using HTTP API.
    
    For single/double character words, you can provide pinyin hint
    to ensure correct pronunciation.
    """
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="DashScope API key not configured")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Get voice
    voice_key = request.voice or DEFAULT_VOICE
    if voice_key not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_key}")
    
    voice_id = VOICES[voice_key]["id"]
    text = request.text.strip()
    
    # For short words with pinyin, add context to help pronunciation
    synthesis_text = text
    if request.pinyin and len(text) <= 2:
        # Format: "谢(xiè)" - helps model with correct tone
        synthesis_text = f"{text}({request.pinyin})"
    
    try:
        start_time = time.time()
        
        # Use DashScope HTTP API - International Edition endpoint
        api_url = os.getenv("DASHSCOPE_API_URL", "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text2audio/text-synthesize")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-Async": "disable",  # Synchronous mode
                },
                json={
                    "model": MODEL,
                    "input": {
                        "text": synthesis_text,
                    },
                    "parameters": {
                        "voice": voice_id,
                        "format": "mp3",
                        "sample_rate": 22050,
                    },
                },
            )
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        if response.status_code != 200:
            error_detail = response.text
            print(f"[TTS] API Error: {response.status_code} - {error_detail}")
            raise HTTPException(status_code=500, detail=f"DashScope API error: {error_detail}")
        
        result = response.json()
        print(f"[TTS] Response keys: {result.keys()}")
        
        # Check for audio in response
        if "output" in result and "audio" in result["output"]:
            # Audio is returned as base64 in the response
            audio_base64 = result["output"]["audio"]
        elif "output" in result and "audio_url" in result["output"]:
            # Audio is returned as URL - need to download
            audio_url = result["output"]["audio_url"]
            async with httpx.AsyncClient() as client:
                audio_response = await client.get(audio_url)
                audio_base64 = base64.b64encode(audio_response.content).decode("utf-8")
        else:
            print(f"[TTS] Unexpected response format: {result}")
            raise HTTPException(status_code=500, detail="Unexpected API response format")
        
        print(f"[TTS] Text: '{text}', Voice: {voice_key}, Latency: {latency_ms}ms")
        
        return SynthesizeResponse(
            audioBase64=audio_base64,
            format="mp3",
            charactersUsed=len(text),
            voice=voice_key,
            latencyMs=latency_ms,
        )
        
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        print(f"[TTS] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.post("/synthesize-batch")
async def synthesize_batch(texts: list[str], voice: Optional[str] = DEFAULT_VOICE):
    """
    Synthesize multiple texts in batch.
    Returns list of audio base64 strings.
    """
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="DashScope API key not configured")
    
    results = []
    for text in texts:
        try:
            result = await synthesize(SynthesizeRequest(text=text, voice=voice))
            results.append({
                "text": text,
                "audioBase64": result.audioBase64,
                "success": True,
            })
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e),
                "success": False,
            })
    
    return {"results": results}


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

