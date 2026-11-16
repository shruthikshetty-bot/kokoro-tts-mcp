# kokoro-tts-mcp

Lightweight Node + Hono MCP server using `kokoro-js` (`Kokoro-82M-v1.0-ONNX`, quantized `q8`).

## Authentication (Env Secret)

Set `MCP_API_KEY` (or `API_KEY`) in your Cloud Run environment variables to require authentication on `/mcp` and `/tts`.

When set, pass the secret in any of these ways when connecting:
- **Bearer Token Header:** `Authorization: Bearer <your-secret>`
- **API Key Header:** `x-api-key: <your-secret>`
- **Query Param:** `https://<your-cloud-run-url>/mcp?api_key=<your-secret>`

## Audio Formats & Temporary Download Links

Both the `generate_tts` MCP tool and the `POST /tts` endpoint support `"format": "mp3"` (default) and `"format": "wav"`:
- **Via MCP (`generate_tts`):** Returns the audio as base64 (`type: "audio"`) plus a temporary `GET /download/<uuid>` URL (stored in RAM for 10 minutes, auto-deleted via `setTimeout`, no auth required on `/download/:id`).
- **Via REST (`POST /tts`):** Returns the raw binary `.mp3` or `.wav` file stream directly with `X-Download-Url` and `X-Download-Id` headers.

## Run (HTTP + Streamable HTTP MCP on Hono)

```bash
npm start
```

- **MCP Endpoint (Streamable HTTP):** `http://0.0.0.0:8080/mcp`
- **Direct REST TTS Endpoint:** `POST http://0.0.0.0:8080/tts` with JSON `{ "text": "Hello world", "voice": "af_heart", "speed": 1.0, "format": "mp3" }`
- **Temporary Download Endpoint:** `GET http://0.0.0.0:8080/download/<uuid>` (auto-expires after 10 minutes)
- **Health Check:** `GET http://0.0.0.0:8080/health`
