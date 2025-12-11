"""
HanziMaster TTS Service

Azure-based text-to-speech for Chinese words with SSML phoneme control.
Guarantees correct tone pronunciation for single characters.
"""

import os
import base64
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import azure.cognitiveservices.speech as speechsdk

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════

def get_speech_config():
    """Get Azure Speech config from environment."""
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION", "germanywestcentral")
    
    if not key:
        return None
    
    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )
    return config

# Available voices
VOICES = {
    "xiaoxiao": {
        "id": "zh-CN-XiaoxiaoNeural",
        "name": "Xiaoxiao",
        "gender": "female",
        "description": "Young female, natural and clear",
        "language": "zh-CN",
    },
    "xiaoyi": {
        "id": "zh-CN-XiaoyiNeural",
        "name": "Xiaoyi",
        "gender": "female",
        "description": "Warm female voice",
        "language": "zh-CN",
    },
    "yunxi": {
        "id": "zh-CN-YunxiNeural",
        "name": "Yunxi",
        "gender": "male",
        "description": "Young male voice",
        "language": "zh-CN",
    },
    "yunyang": {
        "id": "zh-CN-YunyangNeural",
        "name": "Yunyang",
        "gender": "male",
        "description": "Professional male narrator",
        "language": "zh-CN",
    },
    "xiaomo": {
        "id": "zh-CN-XiaomoNeural",
        "name": "Xiaomo",
        "gender": "female",
        "description": "Gentle female voice",
        "language": "zh-CN",
    },
}

DEFAULT_VOICE = "xiaoxiao"

# ═══════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="HanziMaster TTS Service",
    description="Azure-based Chinese TTS with phoneme control",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    pinyin: Optional[str] = None  # e.g., "xiè" or "xie4"


class SynthesizeResponse(BaseModel):
    """Response with synthesized audio"""
    audioBase64: str
    format: str = "mp3"
    durationMs: Optional[int] = None
    charactersUsed: int
    voice: str
    latencyMs: int
    usedPhoneme: bool = False


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
    provider: str
    region: str
    voiceCount: int


# ═══════════════════════════════════════════════════════════
# PINYIN TO PHONEME CONVERSION
# ═══════════════════════════════════════════════════════════

def pinyin_to_sapi(pinyin: str) -> str:
    """
    Convert pinyin (with tone mark or number) to SAPI phoneme format.
    
    Examples:
        "xiè" -> "xie4"
        "xie4" -> "xie4"  
        "nǐ hǎo" -> "ni3 hao3"
    """
    if not pinyin:
        return ""
    
    # Tone mark to number mapping
    tone_marks = {
        'ā': ('a', '1'), 'á': ('a', '2'), 'ǎ': ('a', '3'), 'à': ('a', '4'),
        'ē': ('e', '1'), 'é': ('e', '2'), 'ě': ('e', '3'), 'è': ('e', '4'),
        'ī': ('i', '1'), 'í': ('i', '2'), 'ǐ': ('i', '3'), 'ì': ('i', '4'),
        'ō': ('o', '1'), 'ó': ('o', '2'), 'ǒ': ('o', '3'), 'ò': ('o', '4'),
        'ū': ('u', '1'), 'ú': ('u', '2'), 'ǔ': ('u', '3'), 'ù': ('u', '4'),
        'ǖ': ('v', '1'), 'ǘ': ('v', '2'), 'ǚ': ('v', '3'), 'ǜ': ('v', '4'),
        'ü': ('v', '5'),  # Neutral ü
    }
    
    result = []
    current_syllable = ""
    tone = ""
    
    for char in pinyin.lower():
        if char in tone_marks:
            base, t = tone_marks[char]
            current_syllable += base
            tone = t
        elif char == ' ':
            if current_syllable:
                result.append(current_syllable + tone)
                current_syllable = ""
                tone = ""
        elif char.isalpha():
            current_syllable += char
        elif char.isdigit():
            tone = char
    
    # Don't forget the last syllable
    if current_syllable:
        result.append(current_syllable + (tone or "5"))  # 5 = neutral tone
    
    return " ".join(result)


def build_ssml(text: str, voice_id: str, pinyin: Optional[str] = None) -> str:
    """
    Build SSML with phoneme hints if pinyin is provided.
    """
    # If we have pinyin, use phoneme tags
    if pinyin:
        sapi_pinyin = pinyin_to_sapi(pinyin)
        content = f'<phoneme alphabet="sapi" ph="{sapi_pinyin}">{text}</phoneme>'
    else:
        content = text
    
    ssml = f'''<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" 
           xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="{voice_id}">
        {content}
    </voice>
</speak>'''
    
    return ssml


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    config = get_speech_config()
    return HealthResponse(
        status="ok",
        configured=config is not None,
        provider="Azure Speech Services",
        region=os.getenv("AZURE_SPEECH_REGION", "germanywestcentral"),
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
    
    For accurate pronunciation, provide pinyin with tone:
    - Tone marks: "xiè", "nǐ hǎo"
    - Tone numbers: "xie4", "ni3 hao3"
    """
    config = get_speech_config()
    if not config:
        raise HTTPException(status_code=500, detail="Azure Speech not configured")
    
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Get voice
    voice_key = request.voice or DEFAULT_VOICE
    if voice_key not in VOICES:
        raise HTTPException(status_code=400, detail=f"Unknown voice: {voice_key}")
    
    voice_id = VOICES[voice_key]["id"]
    text = request.text.strip()
    
    # Build SSML
    ssml = build_ssml(text, voice_id, request.pinyin)
    used_phoneme = bool(request.pinyin)
    
    print(f"[TTS] Text: '{text}', Pinyin: '{request.pinyin}', Voice: {voice_key}")
    print(f"[TTS] SSML: {ssml}")
    
    try:
        start_time = time.time()
        
        # Create synthesizer (outputs to memory, not speakers)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=config, 
            audio_config=None  # No audio output, we want the data
        )
        
        # Synthesize
        result = synthesizer.speak_ssml_async(ssml).get()
        
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            
            print(f"[TTS] Success! Latency: {latency_ms}ms, Audio size: {len(audio_data)} bytes")
            
            return SynthesizeResponse(
                audioBase64=audio_base64,
                format="mp3",
                charactersUsed=len(text),
                voice=voice_key,
                latencyMs=latency_ms,
                usedPhoneme=used_phoneme,
            )
        
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"Synthesis canceled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            print(f"[TTS] Error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        else:
            raise HTTPException(status_code=500, detail=f"Unknown result: {result.reason}")
        
    except Exception as e:
        print(f"[TTS] Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
