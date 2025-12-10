# HanziMaster TTS Service

A lightweight microservice for Chinese text-to-speech using Alibaba's DashScope CosyVoice API.

## Why This Exists

ElevenLabs struggles with single Chinese characters - it guesses tones incorrectly. CosyVoice is a Chinese-native TTS model that handles this much better.

## Features

- 🎤 **7 Chinese voices** including male/female options
- 🎯 **Accurate tones** for single/double character words
- ⚡ **Fast** - typical latency < 500ms
- 💰 **Cheap** - ~10x cheaper than ElevenLabs
- 📦 **MP3 output** - ready to use

## API Endpoints

### `GET /health`
Health check.

### `GET /voices`
List available voices.

### `POST /synthesize`
Synthesize speech.

```json
{
  "text": "谢谢",
  "voice": "longxiaochun",
  "pinyin": "xièxiè"  // optional hint
}
```

Response:
```json
{
  "audioBase64": "...",
  "format": "mp3",
  "charactersUsed": 2,
  "voice": "longxiaochun",
  "latencyMs": 350
}
```

## Available Voices

| Key | Name | Gender | Description |
|-----|------|--------|-------------|
| longxiaochun | Xiaochun | Female | Standard Mandarin, gentle |
| longxiaobai | Xiaobai | Female | Young, energetic |
| longlaotie | Laotie | Male | Mature |
| longshu | Shu | Male | Professional narrator |
| longshuo | Shuo | Male | Warm |
| longjielidou | Jielidou | Female | Sweet |
| longxiaoxia | Xiaoxia | Female | Teacher voice |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DASHSCOPE_API_KEY` | ✅ | API key from DashScope |
| `PORT` | ❌ | Port number (default: 8000) |

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export DASHSCOPE_API_KEY=your-key

# Run
python main.py
```

## Deploy to Sevalla

1. Push to GitHub
2. Create new app in Sevalla
3. Connect to this repo
4. Set `DASHSCOPE_API_KEY` in environment variables
5. Deploy

## Getting a DashScope API Key

1. Go to [DashScope Console](https://dashscope.console.aliyun.com/)
2. Sign up for Alibaba Cloud account
3. Enable DashScope service
4. Create API key

## Cost

CosyVoice pricing: ~$0.02 per 1000 characters (vs ElevenLabs ~$0.30)

