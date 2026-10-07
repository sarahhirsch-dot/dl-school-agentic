#!/usr/bin/env node
/**
 * Zerlegt eigene PDFs in Chunks mit Metadaten (Überschrift, Quelldatei, Chunk-Nummer)
 * und speichert sie im Format von data/handbook-corpus.json.
 *
 * Benutzung (im Projektordner):
 *   node scripts/pdf-zu-chunks.mjs meine-pdfs/datei1.pdf meine-pdfs/datei2.pdf
 *
 * Ergebnis: data/my-corpus.json  (die Original-Datei handbook-corpus.json bleibt unverändert)
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const OUTPUT = join(ROOT, "data/my-corpus.json");

const MAX_CHARS = 1200; // längere Abschnitte werden in mehrere Chunks geteilt
const MIN_CHARS = 80; // winzige Schnipsel (z. B. nur eine Zeile) werden angehängt statt eigener Chunk

let pdfjs;
try {
  pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
} catch {
  console.error(
    "Das Paket zum Lesen von PDFs fehlt. Bitte einmal ausführen:\n  npm install pdfjs-dist@4\n",
  );
  process.exit(1);
}

// ---------- PDF lesen: Zeilen mit Schriftgröße ----------

async function readLines(path) {
  const data = new Uint8Array(readFileSync(path));
  const pdf = await pdfjs.getDocument({ data, verbosity: 0 }).promise;
  const lines = [];

  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const rows = [];
    for (const item of content.items) {
      if (!item.str || !item.str.trim()) continue;
      const [a, b, , , x, y] = item.transform;
      const size = Math.round(Math.hypot(a, b) * 10) / 10;
      let row = rows.find((r) => Math.abs(r.y - y) < 2);
      if (!row) rows.push((row = { y, items: [] }));
      row.items.push({ x, str: item.str, size });
    }
    rows.sort((r1, r2) => r2.y - r1.y);
    for (const row of rows) {
      row.items.sort((i1, i2) => i1.x - i2.x);
      const text = row.items.map((i) => i.str).join(" ").replace(/\s+/g, " ").trim();
      const size = Math.max(...row.items.map((i) => i.size));
      if (text) lines.push({ page: p, text, size });
    }
  }
  return lines;
}

// ---------- Überschriften erkennen ----------

function bodySize(lines) {
  // häufigste Schriftgröße (gewichtet nach Textlänge) = normaler Fließtext
  const count = {};
  for (const l of lines) count[l.size] = (count[l.size] ?? 0) + l.text.length;
  return Number(Object.entries(count).sort((x, y) => y[1] - x[1])[0]?.[0] ?? 10);
}

function isHeading(line, body) {
  const t = line.text;
  if (t.length > 100 || /^\d+$/.test(t)) return false;
  if (/[.,;:]$/.test(t)) return false;
  if (line.size >= body * 1.15) return true;
  // z. B. "3.2 Remote Work" oder "Kapitel 4 ..." in normaler Schriftgröße
  return /^(\d+(\.\d+)*\.?|Kapitel \d+)\s+\p{Lu}/u.test(t) && t.length < 80;
}

function sectionId(heading) {
  const m = heading.match(/^(\d+(\.\d+)*)/);
  return m ? m[1] : "";
}

// ---------- Abschnitte bilden und in Chunks teilen ----------

function buildSections(lines, fallbackTitle) {
  const body = bodySize(lines);
  const sections = [];
  let current = { section: fallbackTitle, page: lines[0]?.page ?? 1, parts: [] };

  for (const line of lines) {
    if (/^\d+$/.test(line.text)) continue; // reine Seitenzahlen weglassen
    if (isHeading(line, body)) {
      if (current.parts.length) sections.push(current);
      current = { section: line.text, page: line.page, parts: [] };
    } else {
      current.parts.push(line);
    }
  }
  if (current.parts.length) sections.push(current);
  return sections;
}

function joinText(parts) {
  // Zeilen zusammenfügen, Silbentrennung am Zeilenende reparieren
  return parts
    .map((p) => p.text)
    .join(" ")
    .replace(/(\p{L})- (\p{Ll})/gu, "$1$2")
    .replace(/\s+/g, " ")
    .trim();
}

function splitLong(text) {
  if (text.length <= MAX_CHARS) return [text];
  const sentences = text.match(/[^.!?]+[.!?]+["»«]?\s*|[^.!?]+$/g) ?? [text];
  const pieces = [];
  let buf = "";
  for (const s of sentences) {
    if (buf && (buf + s).length > MAX_CHARS) {
      pieces.push(buf.trim());
      buf = "";
    }
    buf += s;
  }
  if (buf.trim()) pieces.push(buf.trim());
  return pieces;
}

function slug(s) {
  return s
    .toLowerCase()
    .replace(/ä/g, "ae").replace(/ö/g, "oe").replace(/ü/g, "ue").replace(/ß/g, "ss")
    .replace(/\.pdf$/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

async function chunkPdf(path) {
  const file = basename(path);
  const lines = await readLines(path);
  if (!lines.length) {
    console.warn(
      `⚠ In "${file}" wurde kein Text gefunden. Vermutlich ist es ein eingescanntes Bild-PDF. Bitte ein anderes PDF nehmen.`,
    );
    return [];
  }

  const chunks = [];
  for (const sec of buildSections(lines, file.replace(/\.pdf$/i, ""))) {
    for (const piece of splitLong(joinText(sec.parts))) {
      const prev = chunks[chunks.length - 1];
      if (piece.length < MIN_CHARS && prev && prev.section === sec.section) {
        prev.chunk_text += " " + piece;
        continue;
      }
      chunks.push({ section: sec.section, page: sec.page, chunk_text: piece });
    }
  }

  return chunks.map((c, i) => ({
    id: `${slug(file)}-chunk-${i + 1}`,
    page: c.page,
    section: c.section,
    section_id: sectionId(c.section),
    source_file: file,
    chunk_index: i + 1,
    chunk_text: c.chunk_text,
  }));
}

// ---------- Start ----------

const files = process.argv.slice(2);
if (!files.length) {
  console.error("Bitte die PDFs angeben, z. B.:\n  node scripts/pdf-zu-chunks.mjs meine-pdfs/a.pdf meine-pdfs/b.pdf");
  process.exit(1);
}

const all = [];
for (const f of files) {
  if (!existsSync(f)) {
    console.error(`Datei nicht gefunden: ${f}`);
    process.exit(1);
  }
  const chunks = await chunkPdf(f);
  console.log(`✓ ${basename(f)}: ${chunks.length} Chunks`);
  all.push(...chunks);
}

const corpus = {
  doc: files.map((f) => basename(f)).join(" + "),
  version: new Date().toISOString().slice(0, 10),
  chunks: all,
  golden_questions: [],
};

writeFileSync(OUTPUT, JSON.stringify(corpus, null, 2) + "\n");
console.log(`\nGespeichert: data/my-corpus.json (${all.length} Chunks insgesamt)`);
if (all[0]) {
  console.log("\nBeispiel-Chunk:");
  console.log(JSON.stringify(all[0], null, 2));
}
