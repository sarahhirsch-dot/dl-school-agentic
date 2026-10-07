#!/usr/bin/env node
/**
 * Eigene RAG-Chain für Aufgabe M4.3.2 — Kopie von example/rag/deterministic-chain.ts,
 * angepasst auf die eigenen PDFs (Namespace aus seed/seed-my-vectors.mjs):
 *   - fragt den eigenen Namespace ab (Standard: sarah-hirsch-dev)
 *   - Zitat mit Quelldatei, Abschnitt und Seite
 *   - klare Ablehnung, wenn die Antwort nicht in den Dokumenten steht
 *   - loggt chunk_ids und Scores
 *
 * Benutzung:
 *   npm run my:rag -- "Wie alt werden Hunde?"
 */

import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { loadEnv } from "../example/shared/load-env.js";

const INDEX_NAME = "handbook";
const EMBEDDING_MODEL = "text-embedding-3-small";
const TOP_K = 3;
const MIN_SCORE = 0.3; // schwächere Treffer gelten als "nicht gefunden"
const REFUSAL = "Dazu habe ich in meinen Dokumenten nichts gefunden.";

const DEFAULT_QUESTION = "Wie alt werden Hunde im Durchschnitt?";

type Hit = {
  id: string;
  score: number;
  text: string;
  source: string;
  section: string;
  page?: number;
};

async function search(question: string, namespace: string): Promise<Hit[]> {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const embedding = (
    await openai.embeddings.create({ model: EMBEDDING_MODEL, input: question })
  ).data[0].embedding;

  const pc = new Pinecone({ apiKey: process.env.PINECONE_API_KEY! });
  const host = (await pc.describeIndex(INDEX_NAME)).host;
  const ns = pc.index(INDEX_NAME, host).namespace(namespace);
  const results = await ns.query({ vector: embedding, topK: TOP_K, includeMetadata: true });

  return (results.matches ?? []).map((m) => {
    const meta = m.metadata ?? {};
    return {
      id: m.id,
      score: m.score ?? 0,
      text: String(meta.chunk_text ?? ""),
      source: String(meta.source_file ?? ""),
      section: String(meta.section ?? ""),
      page: typeof meta.page === "number" ? meta.page : undefined,
    };
  });
}

function formatContext(hits: Hit[]): string {
  return hits
    .map((h, i) => {
      const cite = [h.source, h.section, h.page != null ? `S. ${h.page}` : null]
        .filter(Boolean)
        .join(" · ");
      return `[${i + 1}] ${cite}\n${h.text}`;
    })
    .join("\n\n");
}

async function main(): Promise<void> {
  loadEnv();
  if (!process.env.PINECONE_API_KEY) throw new Error("Missing PINECONE_API_KEY — set in .env");
  if (!process.env.OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY — set in .env");

  const namespace = process.env.MY_PINECONE_NAMESPACE ?? "sarah-hirsch-dev";
  const question = process.argv.slice(2).join(" ").trim() || DEFAULT_QUESTION;

  console.log(`Frage: ${question}\n`);

  const hits = (await search(question, namespace)).filter((h) => h.score >= MIN_SCORE);
  console.log("Treffer (chunk_id · score):");
  for (const h of hits) console.log(`  ${h.id} · ${h.score.toFixed(3)}`);
  console.log();

  if (!hits.length) {
    console.log("--- Antwort ---\n");
    console.log(REFUSAL);
    return;
  }

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });
  const response = await llm.invoke([
    new SystemMessage(
      "Beantworte die Frage ausschließlich mit dem Kontext unten. " +
        "Gib am Ende ein wörtliches Zitat aus dem passenden Abschnitt an, im Format: " +
        'Zitat: "<Satz aus dem Kontext>" (<Datei> · <Abschnitt> · <Seite>). ' +
        `Wenn der Kontext die Antwort nicht enthält, antworte nur: "${REFUSAL}"`,
    ),
    new HumanMessage(`Kontext:\n${formatContext(hits)}\n\nFrage: ${question}`),
  ]);

  console.log("--- Antwort ---\n");
  console.log(
    typeof response.content === "string" ? response.content : JSON.stringify(response.content),
  );
}

main().catch((err: unknown) => {
  console.error(err);
  process.exit(1);
});
