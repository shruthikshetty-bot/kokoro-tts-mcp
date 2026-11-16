import { mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Mp3Encoder } from "@breezystack/lamejs";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { WebStandardStreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/webStandardStreamableHttp.js";
import { KokoroTTS } from "kokoro-js";
import { z } from "zod";

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const DTYPE = process.env.KOKORO_DTYPE || "q8";
const IS_CLOUD_RUN = Boolean(process.env.K_SERVICE);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR =
  process.env.OUTPUT_DIR ||
  (IS_CLOUD_RUN ? path.join(os.tmpdir(), "kokoro-tts-output") : path.join(__dirname, "output"));

if (process.env.HF_CACHE_DIR || IS_CLOUD_RUN) {
  try {
    const { env: hfEnv } = await import("@huggingface/transformers");
    hfEnv.cacheDir = process.env.HF_CACHE_DIR || path.join(os.tmpdir(), "kokoro-hf-cache");
  } catch (err) {
    console.warn("Could not configure transformers cacheDir:", err);
  }
}

const STATIC_VOICES =
  Object.getOwnPropertyDescriptor(KokoroTTS.prototype, "voices")?.get?.() ?? null;

let ttsPromise = null;
let ttsReady = false;
function getTTS() {
  if (!ttsPromise) {
    console.error(`Loading Kokoro TTS model (${MODEL_ID}, dtype=${DTYPE})...`);
    ttsPromise = KokoroTTS.from_pretrained(MODEL_ID, {
      dtype: DTYPE,
      device: "cpu",
    })
      .then((tts) => {
        ttsReady = true;
        console.error("Kokoro TTS model loaded and ready.");
        return tts;
      })
      .catch((err) => {
        ttsPromise = null;
        ttsReady = false;
        throw err;
      });
  }
  return ttsPromise;
}

function encodeMp3(rawAudio, kbps = 128) {
  const floatSamples = rawAudio.audio;
  const sampleRate = rawAudio.sampling_rate || 24000;
  const int16Samples = new Int16Array(floatSamples.length);

  for (let i = 0; i < floatSamples.length; i++) {
    const s = Math.max(-1, Math.min(1, floatSamples[i]));
    int16Samples[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  const encoder = new Mp3Encoder(1, sampleRate, kbps);
  const chunks = [];
  const blockSize = 1152;

  for (let i = 0; i < int16Samples.length; i += blockSize) {
    const chunk = int16Samples.subarray(i, i + blockSize);
    const mp3buf = encoder.encodeBuffer(chunk);
    if (mp3buf.length > 0) {
      chunks.push(Buffer.from(mp3buf));
    }
  }

  const endBuf = encoder.flush();
  if (endBuf.length > 0) {
    chunks.push(Buffer.from(endBuf));
  }

  return Buffer.concat(chunks);
}

function resolveOutputPath(outputPath, voice, format) {
  const ext = `.${format}`;
  if (!outputPath) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    return path.join(OUTPUT_DIR, `speech-${voice}-${timestamp}${ext}`);
  }
  const hasDir = path.isAbsolute(outputPath) || outputPath.includes("/") || outputPath.includes("\\");
  let target = hasDir ? outputPath : path.join(OUTPUT_DIR, outputPath);
  if (!path.extname(target)) {
    target += ext;
  }
  return target;
}

async function synthesizeAudio({ text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath }) {
  const tts = await getTTS();
  const audio = await tts.generate(text, { voice, speed });

  const isWav =
    format === "wav" || (outputPath && outputPath.toLowerCase().endsWith(".wav") && format !== "mp3");
  const finalFormat = isWav ? "wav" : "mp3";
  const mimeType = finalFormat === "wav" ? "audio/wav" : "audio/mpeg";
  const audioBuffer = finalFormat === "wav" ? Buffer.from(audio.toWav()) : encodeMp3(audio);

  const finalPath = resolveOutputPath(outputPath, voice, finalFormat);
  let savedPath = null;
  try {
    await mkdir(path.dirname(finalPath), { recursive: true });
    await writeFile(finalPath, audioBuffer);
    savedPath = finalPath;
  } catch (err) {
    if (outputPath) {
      throw err;
    }
    console.warn("Skipping disk write for generated audio:", err.message);
  }

  return { audioBuffer, finalFormat, mimeType, savedPath, finalPath };
}

function createMcpServer() {
  const server = new McpServer({
    name: "kokoro-tts-mcp",
    version: "1.0.0",
  });

  server.tool(
    "generate_tts",
    "Convert text into spoken audio (MP3 by default) using lightweight local Kokoro TTS and save to the output folder.",
    {
      text: z.string().min(1).describe("The text to convert to speech"),
      voice: z
        .string()
        .default("af_heart")
        .describe("Voice ID (e.g. af_heart, af_bella, am_michael, bm_george)"),
      speed: z
        .number()
        .min(0.5)
        .max(2.0)
        .default(1.0)
        .describe("Speech speed multiplier (0.5 to 2.0)"),
      format: z
        .enum(["mp3", "wav"])
        .default("mp3")
        .describe("Audio output format (defaults to mp3)"),
      outputPath: z
        .string()
        .optional()
        .describe("Optional filename or path. Defaults to saving inside kokoro-tts-mcp/output/."),
    },
    async ({ text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath }) => {
      const { audioBuffer, finalFormat, mimeType, savedPath } = await synthesizeAudio({
        text,
        voice,
        speed,
        format,
        outputPath,
      });

      const statusText = savedPath
        ? `Generated ${finalFormat.toUpperCase()} speech with voice "${voice}" and saved to ${savedPath}.`
        : `Generated ${finalFormat.toUpperCase()} speech with voice "${voice}".`;

      return {
        content: [
          {
            type: "audio",
            data: audioBuffer.toString("base64"),
            mimeType,
          },
          {
            type: "text",
            text: statusText,
          },
        ],
      };
    }
  );

  server.tool(
    "list_voices",
    "List all available Kokoro TTS voice IDs and details.",
    {},
    async () => {
      const voices = STATIC_VOICES ?? (await getTTS()).voices;
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(voices, null, 2),
          },
        ],
      };
    }
  );

  return server;
}

const app = new Hono();

app.get("/health", (c) => c.json({ ok: true, model: MODEL_ID, ready: ttsReady }));

const corsMiddleware = cors({
  origin: "*",
  allowMethods: ["GET", "POST", "DELETE", "OPTIONS"],
  exposeHeaders: ["mcp-session-id", "X-Output-Path"],
});

// Optional API key / Bearer token auth middleware for /mcp and /tts
const authMiddleware = async (c, next) => {
  const secret = process.env.MCP_API_KEY || process.env.API_KEY;
  if (!secret || c.req.method === "OPTIONS") {
    return next();
  }

  const authHeader = c.req.header("Authorization") || "";
  const bearerToken = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : null;
  const headerKey = c.req.header("x-api-key") || c.req.header("X-API-Key");
  const queryKey = c.req.query("api_key") || c.req.query("key");

  const provided = bearerToken || headerKey || queryKey;
  if (provided !== secret) {
    return c.json({ error: "Unauthorized: Invalid or missing API secret" }, 401);
  }

  return next();
};

app.use("/mcp", corsMiddleware, authMiddleware);
app.use("/tts", corsMiddleware, authMiddleware);

// Direct HTTP endpoint for quick app usage alongside MCP
app.post("/tts", async (c) => {
  let payload;
  try {
    payload = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body" }, 400);
  }

  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return c.json({ error: "Request body must be a JSON object" }, 400);
  }

  const { text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath } = payload;
  if (typeof text !== "string" || !text.trim()) {
    return c.json({ error: "Missing or invalid 'text' field" }, 400);
  }
  if (typeof speed !== "number" || Number.isNaN(speed) || speed < 0.5 || speed > 2.0) {
    return c.json({ error: "'speed' must be a number between 0.5 and 2.0" }, 400);
  }
  if (format !== "mp3" && format !== "wav") {
    return c.json({ error: "'format' must be 'mp3' or 'wav'" }, 400);
  }

  const { audioBuffer, mimeType, savedPath, finalPath } = await synthesizeAudio({
    text,
    voice,
    speed,
    format,
    outputPath,
  });

  const headers = {
    "Content-Type": mimeType,
    "Content-Disposition": `inline; filename="${path.basename(finalPath)}"`,
  };
  if (savedPath) {
    headers["X-Output-Path"] = savedPath;
  }

  return new Response(audioBuffer, { headers });
});

// Streamable HTTP MCP endpoint (stateless Web Standard transport)
app.all("/mcp", async (c) => {
  if (c.req.method === "GET") {
    return c.json(
      {
        jsonrpc: "2.0",
        error: { code: -32000, message: "Method not allowed." },
        id: null,
      },
      405,
      { Allow: "POST, DELETE" }
    );
  }

  const server = createMcpServer();
  const transport = new WebStandardStreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  });

  let cleanedUp = false;
  const cleanup = () => {
    if (cleanedUp) return;
    cleanedUp = true;
    transport.close().catch(() => {});
    server.close().catch(() => {});
  };
  c.env?.outgoing?.once?.("close", cleanup);

  await server.connect(transport);
  try {
    return await transport.handleRequest(c.req.raw);
  } catch (err) {
    cleanup();
    throw err;
  }
});

const port = Number(process.env.PORT ?? 8080);
const hostname = process.env.HOST ?? "0.0.0.0";

// Keep native global Response/Request intact so @huggingface/transformers FileCache works on fresh containers
serve({ fetch: app.fetch, port, hostname, overrideGlobalObjects: false }, () => {
  console.log(`Kokoro TTS MCP server running at http://${hostname}:${port}/mcp`);
  getTTS().catch((err) => {
    console.error("Background Kokoro TTS warmup failed:", err);
  });
});
