# kokoro-tts-mcp

Lightweight Node + Hono MCP server using `kokoro-js` (`Kokoro-82M-v1.0-ONNX`, quantized `q8`).

## Authentication (Env Secret)

Set `MCP_API_KEY` (or `API_KEY`) in your Cloud Run environment variables to require authentication on `/mcp` and `/tts`.

When set, pass the secret in any of these ways when connecting:
- **Bearer Token Header:** `Authorization: Bearer <your-secret>`
- **API Key Header:** `x-api-key: <your-secret>`
- **Query Param:** `https://<your-cloud-run-url>/mcp?api_key=<your-secret>`

## Audio Formats Returned

Both the `generate_tts` MCP tool and the `POST /tts` endpoint support `"format": "mp3"` (default) and `"format": "wav"`:
- **Via MCP (`generate_tts`):** Returns the audio directly in the MCP response payload as base64 (`type: "audio"`, `mimeType: "audio/mpeg"` or `"audio/wav"`).
- **Via REST (`POST /tts`):** Returns the raw binary `.mp3` or `.wav` file stream directly with the matching `Content-Type`.

## Run (HTTP + Streamable HTTP MCP on Hono)

```bash
npm start
```

- **MCP Endpoint (Streamable HTTP):** `http://0.0.0.0:8080/mcp`
- **Direct REST TTS Endpoint:** `POST http://0.0.0.0:8080/tts` with JSON `{ "text": "Hello world", "voice": "af_heart", "speed": 1.0, "format": "mp3" }`
- **Health Check:** `GET http://0.0.0.0:8080/health`
