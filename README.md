# kokoro-tts-mcp

Lightweight Node + Hono MCP server using `kokoro-js` (`Kokoro-82M-v1.0-ONNX`, quantized `q8`). Ready for direct source deployment on Google Cloud Run.

## Run (HTTP + Streamable HTTP MCP on Hono)

```bash
npm start
```

- **MCP Endpoint (Streamable HTTP):** `http://0.0.0.0:8080/mcp`
- **Direct REST TTS Endpoint:** `POST http://0.0.0.0:8080/tts` with JSON `{ "text": "Hello world", "voice": "af_heart", "speed": 1.0 }`
- **Health Check:** `GET http://0.0.0.0:8080/health`

## Deploy to Google Cloud Run (Direct Source)

No Dockerfile required! Google Cloud Buildpacks automatically detect `package.json` and run `npm start`.

- **Recommended Memory:** `1 GiB` or `2 GiB` (since the quantized ONNX model loads into RAM on first request)
- **Port:** `8080` (default)

## Run (Stdio MCP mode for desktop clients)

```bash
npm run stdio
```
