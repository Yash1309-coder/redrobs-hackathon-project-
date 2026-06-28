// Generates narration audio (ElevenLabs) + illustrations (OpenAI gpt-image-1),
// measures audio durations with ffprobe, and writes src/remotion/narration.json
// which drives scene timing in Demo.tsx (visuals follow audio = perfect sync).
//
// Run:  node --env-file=.env scripts/generate-assets.mjs
// Needs: ELEVENLABS_API_KEY (audio), OPENAI_API_KEY (images), ffmpeg/ffprobe on PATH.

import { writeFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import path from "node:path";

const FPS = 30;
const PAD_FRAMES = 18; // ~0.6s breathing room after each line
const VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "EXAVITQu4vr4xnSDxMaL"; // Sarah (premade)
const ELEVEN_MODEL = "eleven_multilingual_v2";

const SCENES = [
  {
    id: "title",
    narration:
      "Recruiter-Brain — a candidate ranker that surfaces the top one hundred people out of one hundred thousand. On a CPU, in under five minutes, with no network.",
    imagePrompt:
      "Abstract dark navy technology background, a glowing cyan and violet neural network forming a subtle human brain silhouette, fine particles, cinematic depth, lots of empty negative space in the center, 16:9.",
  },
  {
    id: "problem",
    narration:
      "The real problem isn't search. Over eight thousand candidates carry the right skills, but only seven hundred fifty-five have a job title that actually backs them up. A ten to one wall of keyword stuffers.",
    imagePrompt:
      "Dark navy abstract data background, a vast field of faint translucent resume documents receding into the distance, only a few glowing soft green, the rest dim, moody, negative space in the center, 16:9.",
  },
  {
    id: "rubric",
    narration:
      "Our signature move is a single, human-reviewed rubric. It understands what the role really needs — and drives ranking, honeypot avoidance, and reasoning, all at once.",
    imagePrompt:
      "Abstract glowing holographic blueprint of a structured document floating in dark navy space, cyan and violet neon wireframe lines, central soft glow, clean negative space around it, 16:9.",
  },
  {
    id: "arch",
    narration:
      "All the intelligence happens offline. At rank time there is no model loaded — just a fast weighted score over precomputed features. The five minute budget is never at risk.",
    imagePrompt:
      "Two abstract glowing data pipelines on the left and right of a dark navy scene, streams of light flowing between them, cyan and violet neon, empty space in the middle, cinematic, 16:9.",
  },
  {
    id: "results",
    narration:
      "And we built our own evaluation. N D C G at ten: zero point nine nine six eight. Honeypot rate in the top one hundred: zero percent. Every single candidate, domain corroborated.",
    imagePrompt:
      "Abstract success visualization on dark navy, softly glowing rising line charts and bokeh particles in green and cyan, sense of achievement, generous negative space in the center, 16:9.",
  },
  {
    id: "outro",
    narration:
      "We built a system, not a prompt. And one command reproduces all of it.",
    imagePrompt:
      "Dark cinematic terminal aesthetic, a single glowing command line in cyan and green on deep navy, soft vignette, lots of negative space in the center, 16:9.",
  },
];

const ROOT = path.resolve(import.meta.dirname, "..");
const AUDIO_DIR = path.join(ROOT, "public", "audio");
const IMG_DIR = path.join(ROOT, "public", "img");
const OUT_JSON = path.join(ROOT, "src", "remotion", "narration.json");

const ffprobeDuration = (file) => {
  const out = execFileSync("ffprobe", [
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=nw=1:nk=1",
    file,
  ]);
  return parseFloat(out.toString().trim());
};

async function tts(scene) {
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) throw new Error("ELEVENLABS_API_KEY not set");
  const res = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}?output_format=mp3_44100_128`,
    {
      method: "POST",
      headers: { "xi-api-key": key, "Content-Type": "application/json" },
      body: JSON.stringify({
        text: scene.narration,
        model_id: ELEVEN_MODEL,
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    },
  );
  if (!res.ok) throw new Error(`ElevenLabs ${scene.id}: ${res.status} ${await res.text()}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const file = path.join(AUDIO_DIR, `${scene.id}.mp3`);
  await writeFile(file, buf);
  return file;
}

async function image(scene) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    console.warn(`! OPENAI_API_KEY not set — skipping image for "${scene.id}"`);
    return false;
  }
  const res = await fetch("https://api.openai.com/v1/images/generations", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "gpt-image-1",
      prompt: scene.imagePrompt,
      size: "1536x1024",
      quality: "high",
      n: 1,
    }),
  });
  if (!res.ok) throw new Error(`OpenAI image ${scene.id}: ${res.status} ${await res.text()}`);
  const json = await res.json();
  const b64 = json.data[0].b64_json;
  await writeFile(path.join(IMG_DIR, `${scene.id}.png`), Buffer.from(b64, "base64"));
  return true;
}

async function main() {
  const onlyAudio = process.argv.includes("--audio-only");
  const onlyImages = process.argv.includes("--images-only");
  await mkdir(AUDIO_DIR, { recursive: true });
  await mkdir(IMG_DIR, { recursive: true });

  // Images-only: reuse existing narration.json timings, just fill in images.
  if (onlyImages) {
    const { readFile } = await import("node:fs/promises");
    const manifest = JSON.parse(await readFile(OUT_JSON, "utf8"));
    for (const entry of manifest) {
      const scene = SCENES.find((s) => s.id === entry.id);
      process.stdout.write(`• ${entry.id}: image…`);
      const ok = await image(scene);
      entry.image = ok ? `img/${entry.id}.png` : entry.image;
      console.log(ok ? " ✓" : " (skipped)");
    }
    await writeFile(OUT_JSON, JSON.stringify(manifest, null, 2));
    console.log(`\n✓ images patched into ${OUT_JSON}`);
    return;
  }

  const manifest = [];
  for (const scene of SCENES) {
    process.stdout.write(`• ${scene.id}: audio…`);
    const audioFile = await tts(scene);
    const seconds = ffprobeDuration(audioFile);
    const durationInFrames = Math.ceil(seconds * FPS) + PAD_FRAMES;

    let hasImage = false;
    if (!onlyAudio) {
      process.stdout.write(" image…");
      hasImage = await image(scene);
    }
    console.log(` ${seconds.toFixed(2)}s → ${durationInFrames}f${hasImage ? " +img" : ""}`);

    manifest.push({
      id: scene.id,
      audio: `audio/${scene.id}.mp3`,
      image: hasImage ? `img/${scene.id}.png` : null,
      durationInFrames,
    });
  }

  await writeFile(OUT_JSON, JSON.stringify(manifest, null, 2));
  const total = manifest.reduce((a, s) => a + s.durationInFrames, 0);
  console.log(`\n✓ ${OUT_JSON}  (total ${total}f / ${(total / FPS).toFixed(1)}s)`);
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
