#!/usr/bin/env node
/**
 * Lädt die eigenen Chunks aus data/my-corpus.json in einen EIGENEN Pinecone-Namespace.
 * Kopie von seed-handbook-vectors.mjs, angepasst für Aufgabe M4.3.1:
 *   - liest data/my-corpus.json statt handbook-corpus.json
 *   - Namespace wird beim Aufruf angegeben (workshop-shared ist gesperrt)
 *   - speichert zusätzlich source_file und chunk_index als Metadaten
 *   - lädt in Paketen hoch (funktioniert auch mit vielen Chunks)
 *   - generiert kein Handbuch-PDF
 *
 * Benutzung (im Projektordner):
 *   node seed/seed-my-vectors.mjs DEIN-NAMESPACE
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const CORPUS_FILE = join(ROOT, "data/my-corpus.json");

const INDEX_NAME = process.env.WORKSHOP_PINECONE_INDEX ?? "handbook";
const CLOUD = process.env.WORKSHOP_PINECONE_CLOUD ?? "aws";
const REGION = process.env.WORKSHOP_PINECONE_REGION ?? "us-east-1";
const EMBEDDING_MODEL = "text-embedding-3-small";
const EMBEDDING_DIM = 1536;
const BATCH = 50;

const NAMESPACE = process.argv[2];
if (!NAMESPACE || NAMESPACE.startsWith("-")) {
  console.error("Bitte deinen Namespace angeben, z. B.:\n  node seed/seed-my-vectors.mjs anna-eigene-pdfs");
  process.exit(1);
}
if (NAMESPACE === "workshop-shared") {
  console.error('"workshop-shared" ist der gemeinsame Namespace. Bitte einen eigenen Namen wählen.');
  process.exit(1);
}

function loadEnvLocal() {
  for (const name of [".env.local", ".env"]) {
    const path = join(ROOT, name);
    if (!existsSync(path)) continue;
    for (const line of readFileSync(path, "utf8").split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq === -1) continue;
      const key = trimmed.slice(0, eq).trim();
      let val = trimmed.slice(eq + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      if (process.env[key] === undefined || process.env[key] === "") {
        process.env[key] = val;
      }
    }
  }
}

async function ensureIndex(pc) {
  const existing = await pc.listIndexes();
  const names = existing.indexes?.map((i) => i.name) ?? [];
  if (names.includes(INDEX_NAME)) {
    const desc = await pc.describeIndex(INDEX_NAME);
    if (desc.dimension !== EMBEDDING_DIM) {
      throw new Error(`Index ${INDEX_NAME} has dimension ${desc.dimension}, expected ${EMBEDDING_DIM}`);
    }
    console.log(`Using existing index "${INDEX_NAME}" (${desc.dimension}d)`);
    return;
  }
  console.log(`Creating index "${INDEX_NAME}" (${EMBEDDING_DIM}d, cosine)…`);
  await pc.createIndex({
    name: INDEX_NAME,
    dimension: EMBEDDING_DIM,
    metric: "cosine",
    spec: { serverless: { cloud: CLOUD, region: REGION } },
    waitUntilReady: true,
  });
  console.log(`✓ Index "${INDEX_NAME}" ready`);
}

async function embedBatch(openai, texts) {
  const response = await openai.embeddings.create({ model: EMBEDDING_MODEL, input: texts });
  return response.data.sort((a, b) => a.index - b.index).map((row) => row.embedding);
}

// Neuere Pinecone-Versionen erwarten { records } bzw. { ids }, ältere ein Array.
// So funktioniert es mit beiden.
async function upsertCompat(ns, records) {
  try {
    await ns.upsert({ records });
  } catch (err) {
    try {
      await ns.upsert(records);
    } catch {
      throw err;
    }
  }
}
async function fetchCompat(ns, ids) {
  try {
    return await ns.fetch({ ids });
  } catch (err) {
    try {
      return await ns.fetch(ids);
    } catch {
      throw err;
    }
  }
}

function chunkMetadata(chunk) {
  return {
    doc: chunk.source_file,
    source_file: chunk.source_file,
    chunk_index: chunk.chunk_index,
    page: chunk.page,
    section: chunk.section,
    section_id: chunk.section_id ?? "",
    chunk_text: chunk.chunk_text,
  };
}

async function main() {
  loadEnvLocal();
  const pineconeKey = process.env.PINECONE_API_KEY;
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!pineconeKey) throw new Error("Missing PINECONE_API_KEY — set in .env or .env.local");
  if (!openaiKey) throw new Error("Missing OPENAI_API_KEY — set in .env or .env.local");

  if (!existsSync(CORPUS_FILE)) {
    throw new Error("data/my-corpus.json fehlt. Zuerst scripts/pdf-zu-chunks.mjs ausführen.");
  }
  const corpus = JSON.parse(readFileSync(CORPUS_FILE, "utf8"));
  const chunks = corpus.chunks;
  console.log(`Embedding ${chunks.length} chunks from ${corpus.doc}…`);

  const { Pinecone } = await import("@pinecone-database/pinecone");
  const { default: OpenAI } = await import("openai");
  const pc = new Pinecone({ apiKey: pineconeKey });
  const openai = new OpenAI({ apiKey: openaiKey });

  await ensureIndex(pc);
  const indexHost = (await pc.describeIndex(INDEX_NAME)).host;
  const index = pc.index(INDEX_NAME, indexHost);
  const ns = index.namespace(NAMESPACE);

  console.log(`Upserting ${chunks.length} vectors to ${INDEX_NAME}/${NAMESPACE}…`);
  for (let i = 0; i < chunks.length; i += BATCH) {
    const part = chunks.slice(i, i + BATCH);
    const embeddings = await embedBatch(openai, part.map((c) => c.chunk_text));
    const vectors = part.map((chunk, j) => ({
      id: chunk.id,
      values: embeddings[j],
      metadata: chunkMetadata(chunk),
    }));
    await upsertCompat(ns, vectors);
    console.log(`  … ${Math.min(i + BATCH, chunks.length)}/${chunks.length}`);
  }

  // Prüfen, ob alles angekommen ist
  const missing = [];
  let example = null;
  for (let i = 0; i < chunks.length; i += BATCH) {
    const ids = chunks.slice(i, i + BATCH).map((c) => c.id);
    const records = (await fetchCompat(ns, ids)).records ?? {};
    for (const id of ids) {
      if (!records[id]?.values?.length) missing.push(id);
      else if (!example) example = records[id];
    }
  }
  if (missing.length) throw new Error(`Upsert verification failed: ${missing.join(", ")}`);

  console.log(`\n✓ Synced ${chunks.length} vectors to Pinecone`);
  console.log(`  Index:     ${INDEX_NAME}`);
  console.log(`  Namespace: ${NAMESPACE}`);
  console.log(`  Vektoren:  ${chunks.length}`);

  try {
    const stats = await index.describeIndexStats();
    const count = stats.namespaces?.[NAMESPACE]?.recordCount;
    if (count !== undefined) {
      console.log(`  Laut Pinecone-Statistik: ${count} Vektoren in "${NAMESPACE}"`);
      console.log(`  (Die Statistik kann ein paar Sekunden hinterherhinken.)`);
    }
  } catch {
    // Statistik ist nur eine Zusatzinfo
  }

  if (example) {
    console.log("\nBeispiel-Chunk aus deinem Index (Text + Metadaten):");
    console.log(JSON.stringify({ id: example.id, metadata: example.metadata }, null, 2));
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
