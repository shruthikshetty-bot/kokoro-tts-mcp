import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Mp3Encoder } from "@breezystack/lamejs";
import { serve } from "@hono/node-server";
import { Hono } from "hono";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { KokoroTTS } from "kokoro-js";
import { z } from "zod";

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUTPUT_DIR = path.join(__dirname, "output");

let ttsPromise = null;
function getTTS() {
  if (!ttsPromise) {
    console.error("Loading Kokoro TTS model (first run downloads ~86MB)...");
    ttsPromise = KokoroTTS.from_pretrained(MODEL_ID, {
      dtype: "q8",
      device: "cpu",
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
      const tts = await getTTS();
      const audio = await tts.generate(text, { voice, speed });

      const isWav = format === "wav" || (outputPath && outputPath.toLowerCase().endsWith(".wav") && format !== "mp3");
      const finalFormat = isWav ? "wav" : "mp3";
      const mimeType = finalFormat === "wav" ? "audio/wav" : "audio/mpeg";
      const audioBuffer = finalFormat === "wav" ? Buffer.from(audio.toWav()) : encodeMp3(audio);

      const finalPath = resolveOutputPath(outputPath, voice, finalFormat);
      await mkdir(path.dirname(finalPath), { recursive: true });
      await writeFile(finalPath, audioBuffer);

      return {
        content: [
          {
            type: "audio",
            data: audioBuffer.toString("base64"),
            mimeType,
          },
          {
            type: "text",
            text: `Generated ${finalFormat.toUpperCase()} speech with voice "${voice}" and saved to ${finalPath}.`,
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
      const tts = await getTTS();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(tts.voices, null, 2),
          },
        ],
      };
    }
  );

  return server;
}

const app = new Hono();

app.get("/health", (c) => c.json({ ok: true, model: MODEL_ID }));

// Direct HTTP endpoint for quick app usage alongside MCP
app.post("/tts", async (c) => {
  const { text, voice = "af_heart", speed = 1.0, format = "mp3", outputPath } = await c.req.json();
  if (!text) return c.json({ error: "Missing 'text' field" }, 400);
  const tts = await getTTS();
  const audio = await tts.generate(text, { voice, speed });

  const finalFormat = format === "wav" ? "wav" : "mp3";
  const mimeType = finalFormat === "wav" ? "audio/wav" : "audio/mpeg";
  const audioBuffer = finalFormat === "wav" ? Buffer.from(audio.toWav()) : encodeMp3(audio);

  const finalPath = resolveOutputPath(outputPath, voice, finalFormat);
  await mkdir(path.dirname(finalPath), { recursive: true });
  await writeFile(finalPath, audioBuffer);

  return new Response(audioBuffer, {
    headers: {
      "Content-Type": mimeType,
      "Content-Disposition": `inline; filename="${path.basename(finalPath)}"`,
      "X-Output-Path": finalPath,
    },
  });
});

// Streamable HTTP MCP endpoint
app.all("/mcp", async (c) => {
  const server = createMcpServer();
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  });
  await server.connect(transport);

  const req = c.env.incoming;
  const res = c.env.outgoing;
  const body = c.req.method === "POST" ? await c.req.json() : undefined;
  await transport.handleRequest(req, res, body);
  return new Response(null);
});

const port = Number(process.env.PORT ?? 8080);
const hostname = process.env.HOST ?? "0.0.0.0";

serve({ fetch: app.fetch, port, hostname }, () => {
  console.log(`Kokoro TTS MCP server running at http://${hostname}:${port}/mcp`);
});
