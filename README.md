# HanziMaster TTS Service

Azure-based Chinese text-to-speech with **SSML phoneme control** for accurate tone pronunciation.

## Why Azure?

ElevenLabs guesses tones for single Chinese characters. Azure allows explicit phoneme control:

```xml
<phoneme alphabet="sapi" ph="xie4">谢</phoneme>
```

This guarantees correct pronunciation every time.

## API Endpoints

### `GET /health`
Health check.

### `GET /voices`
List available Chinese voices.

### `POST /synthesize`
Synthesize speech with optional pinyin for tone control.

```json
{
  "text": "谢",
  "voice": "xiaoxiao",
  "pinyin": "xiè"
}
```

Response:
```json
{
  "audioBase64": "...",
  "format": "mp3",
  "charactersUsed": 1,
  "voice": "xiaoxiao",
  "latencyMs": 350,
  "usedPhoneme": true
}
```

## Pinyin Format

Both formats are supported:
- **Tone marks:** `xiè`, `nǐ hǎo`, `zhōngguó`
- **Tone numbers:** `xie4`, `ni3 hao3`, `zhong1 guo2`

## Available Voices

| Key | Name | Gender | Description |
|-----|------|--------|-------------|
| xiaoxiao | Xiaoxiao | Female | Young, natural and clear |
| xiaoyi | Xiaoyi | Female | Warm voice |
| yunxi | Yunxi | Male | Young male |
| yunyang | Yunyang | Male | Professional narrator |
| xiaomo | Xiaomo | Female | Gentle voice |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_SPEECH_KEY` | ✅ | Azure Speech Services API key |
| `AZURE_SPEECH_REGION` | ✅ | Azure region (e.g., `germanywestcentral`) |
| `PORT` | ❌ | Port number (default: 8000) |

## Deploy to Sevalla

1. Push to GitHub
2. Create new app in Sevalla
3. Set environment variables:
   ```
   AZURE_SPEECH_KEY=your-key
   AZURE_SPEECH_REGION=germanywestcentral
   ```
4. Deploy

## Local Development

```bash
pip install -r requirements.txt
export AZURE_SPEECH_KEY=your-key
export AZURE_SPEECH_REGION=germanywestcentral
python main.py
```

## Cost

Azure Neural TTS: ~$16 per 1 million characters
(~20x cheaper than ElevenLabs)
