import crypto from "node:crypto";
import os from "node:os";
import path from "node:path";
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
const AUDIO_TTL_MS = Number(process.env.AUDIO_TTL_MS ?? 10 * 60 * 1000);
const MAX_STORED_CLIPS = Number(process.env.MAX_STORED_CLIPS ?? 100);
const audioStore = new Map();

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

function sanitizeFilename(outputPath, voice, format) {
  const ext = `.${format}`;
  if (!outputPath) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    return `speech-${voice}-${timestamp}${ext}`;
  }
  const base = path.basename(outputPath).replace(/["\r\n]/g, "").trim();
  const clean = base || `speech-${voice}`;
  const currentExt = path.extname(clean);
  if (!currentExt) return `${clean}${ext}`;
  if (currentExt.toLowerCase() !== ext) {
    return `${clean.slice(0, -currentExt.length)}${ext}`;
  }
  return clean;
}

function storeAudioClip({ buffer, mimeType, filename, format }) {
  if (MAX_STORED_CLIPS > 0 && audioStore.size >= MAX_STORED_CLIPS) {
    const oldestKey = audioStore.keys().next().value;
    if (oldestKey !== undefined) {
      const oldest = audioStore.get(oldestKey);
      if (oldest?.timeout) clearTimeout(oldest.timeout);
      audioStore.delete(oldestKey);
    }
  }

  const id = crypto.randomUUID();
  const expiresAt = Date.now() + AUDIO_TTL_MS;
  const timeout = setTimeout(() => {
    audioStore.delete(id);
  }, AUDIO_TTL_MS);
  timeout.unref?.();

  audioStore.set(id, {
    buffer,
    mimeType,
    filename,
    format,
    createdAt: Date.now(),
    expiresAt,
    timeout,
  });

  return { id, expiresAt };
}

function getBaseUrl(c) {
  const configured = process.env.PUBLIC_URL || process.env.BASE_URL;
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  const forwardedHost = c.req.header("x-forwarded-host") || c.req.header("host");
  const forwardedProto =
    c.req.header("x-forwarded-proto") ||
    (IS_CLOUD_RUN ? "https" : new URL(c.req.url).protocol.replace(":", ""));
  if (forwardedHost) {
    const proto = forwardedProto.split(",")[0].trim();
    const host = forwardedHost.split(",")[0].trim();
    return `${proto}://${host}`;
  }
  return new URL(c.req.url).origin;
}

async function synthesizeAudio({ text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath, baseUrl }) {
  const tts = await getTTS();
  const audio = await tts.generate(text, { voice, speed });

  const isWav = format === "wav";
  const finalFormat = isWav ? "wav" : "mp3";
  const mimeType = finalFormat === "wav" ? "audio/wav" : "audio/mpeg";
  const audioBuffer = finalFormat === "wav" ? Buffer.from(audio.toWav()) : encodeMp3(audio);
  const filename = sanitizeFilename(outputPath, voice, finalFormat);

  const { id, expiresAt } = storeAudioClip({
    buffer: audioBuffer,
    mimeType,
    filename,
    format: finalFormat,
  });

  const downloadPath = `/download/${id}`;
  const downloadUrl = baseUrl ? `${baseUrl}${downloadPath}` : downloadPath;

  return { audioBuffer, finalFormat, mimeType, filename, id, expiresAt, downloadPath, downloadUrl };
}

function createMcpServer({ baseUrl } = {}) {
  const server = new McpServer({
    name: "kokoro-tts-mcp",
    version: "1.0.0",
  });

  server.tool(
    "generate_tts",
    "Convert text into spoken audio (MP3 by default) using Kokoro TTS and return a temporary 10-minute download link.",
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
        .describe("Optional download filename (e.g. greeting.mp3). Defaults to speech-<voice>-<timestamp>.<format>."),
    },
    async ({ text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath }) => {
      const { audioBuffer, finalFormat, mimeType, filename, downloadUrl } = await synthesizeAudio({
        text,
        voice,
        speed,
        format,
        outputPath,
        baseUrl,
      });

      const ttlMinutes = Math.max(1, Math.round(AUDIO_TTL_MS / 60000));
      return {
        content: [
          {
            type: "audio",
            data: audioBuffer.toString("base64"),
            mimeType,
          },
          {
            type: "text",
            text: [
              `Generated ${finalFormat.toUpperCase()} speech with voice "${voice}" (${filename}).`,
              `Download link (expires in ${ttlMinutes} minutes): ${downloadUrl}`,
            ].join("\n"),
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
  exposeHeaders: ["mcp-session-id", "X-Download-Url", "X-Download-Id", "Content-Disposition"],
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
app.use("/download/*", corsMiddleware);

// Unauthenticated temporary download endpoint (unguessable UUID, auto-expires after 10 minutes)
app.get("/download/:id", (c) => {
  const rawId = c.req.param("id");
  const id = rawId.replace(/\.(mp3|wav)$/i, "");
  const clip = audioStore.get(id);

  if (!clip || Date.now() > clip.expiresAt) {
    if (clip) {
      if (clip.timeout) clearTimeout(clip.timeout);
      audioStore.delete(id);
    }
    return c.json({ error: "Audio clip not found or expired" }, 404);
  }

  const disposition = c.req.query("inline") === "1" ? "inline" : "attachment";
  return new Response(clip.buffer, {
    headers: {
      "Content-Type": clip.mimeType,
      "Content-Length": String(clip.buffer.length),
      "Content-Disposition": `${disposition}; filename="${clip.filename}"`,
      "Cache-Control": "private, no-store",
    },
  });
});

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

  const { audioBuffer, mimeType, filename, id, downloadUrl } = await synthesizeAudio({
    text,
    voice,
    speed,
    format,
    outputPath,
    baseUrl: getBaseUrl(c),
  });

  return new Response(audioBuffer, {
    headers: {
      "Content-Type": mimeType,
      "Content-Disposition": `inline; filename="${filename}"`,
      "X-Download-Url": downloadUrl,
      "X-Download-Id": id,
    },
  });
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

  const server = createMcpServer({ baseUrl: getBaseUrl(c) });
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
