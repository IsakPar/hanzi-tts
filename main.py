"""
HanziMaster TTS Service

A lightweight microservice for Chinese text-to-speech using Alibaba's DashScope CosyVoice API.
Optimized for accurate pronunciation of single and double character words.
"""

import os
import base64
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

# API Key - read at request time to ensure env var is available
def get_api_key():
    return os.getenv("DASHSCOPE_API_KEY")

# Available voices
VOICES = {
    "longxiaochun": {
        "id": "longxiaochun_v2",
        "name": "Xiaochun",
        "gender": "female",
        "description": "Standard Mandarin female, gentle and clear",
        "language": "zh",
    },
    "longxiaobai": {
        "id": "longxiaobai_v2",
        "name": "Xiaobai", 
        "gender": "female",
        "description": "Young energetic female voice",
        "language": "zh",
    },
    "longlaotie": {
        "id": "longlaotie_v2",
        "name": "Laotie",
        "gender": "male",
        "description": "Mature male voice",
        "language": "zh",
    },
    "longshu": {
        "id": "longshu_v2",
        "name": "Shu",
        "gender": "male",
        "description": "Professional male narrator",
        "language": "zh",
    },
    "longshuo": {
        "id": "longshuo_v2",
        "name": "Shuo",
        "gender": "male", 
        "description": "Warm male voice",
        "language": "zh",
    },
    "longjielidou": {
        "id": "longjielidou_v2",
        "name": "Jielidou",
        "gender": "female",
        "description": "Sweet female voice",
        "language": "zh",
    },
    "longxiaoxia": {
        "id": "longxiaoxia_v2",
        "name": "Xiaoxia",
        "gender": "female",
        "description": "Gentle female teacher voice",
        "language": "zh",
    },
}

DEFAULT_VOICE = "longxiaochun"
MODEL = "cosyvoice-v3-plus"

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
    Synthesize speech from Chinese text.
    
    For single/double character words, you can provide pinyin hint
    to ensure correct pronunciation.
    """
    api_key = get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="DashScope API key not configured")
    
    # Set the API key for this request
    dashscope.api_key = api_key
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Get voice
    voice_key = request.voice or DEFAULT_VOICE
    if voice_key not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_key}")
    
    voice_id = VOICES[voice_key]["id"]
    text = request.text.strip()
    
    # For short words with pinyin, we can use SSML or special formatting
    # CosyVoice may support pronunciation hints - test this
    synthesis_text = text
    if request.pinyin and len(text) <= 2:
        # Try adding pinyin context to help with pronunciation
        # Format: "谢(xiè)" - the model might use this as a hint
        synthesis_text = f"{text}({request.pinyin})"
    
    try:
        start_time = time.time()
        
        # Create synthesizer
        synthesizer = SpeechSynthesizer(model=MODEL, voice=voice_id)
        
        # Generate audio
        audio_bytes = synthesizer.call(synthesis_text)
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Get first packet latency for debugging
        first_packet_latency = synthesizer.get_first_package_delay()
        print(f"[TTS] Text: '{text}', Voice: {voice_key}, Latency: {latency_ms}ms, First packet: {first_packet_latency}ms")
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return SynthesizeResponse(
            audioBase64=audio_base64,
            format="mp3",
            charactersUsed=len(text),
            voice=voice_key,
            latencyMs=latency_ms,
        )
        
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
    
    dashscope.api_key = api_key
    
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

