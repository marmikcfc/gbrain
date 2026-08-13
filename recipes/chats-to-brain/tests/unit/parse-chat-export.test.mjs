/**
 * parse-chat-export.test.mjs — export-shape invariants.
 *
 * The load-bearing correctness property is branch reconstruction. ChatGPT
 * stores a conversation as a TREE: every edit or regenerate forks a branch,
 * and abandoned branches stay in the export forever. Walking
 * `Object.values(mapping)` — the obvious implementation — interleaves them,
 * producing transcripts that contradict themselves and billing the operator
 * to embed text they never saw. The only correct traversal starts at
 * `current_node` and follows `parent` pointers to the root.
 *
 * The ChatGPT fixture plants an `ABANDONED BRANCH` node specifically so a
 * regression there fails loud rather than silently degrading retrieval.
 *
 * Also pinned: system/tool/hidden turns are dropped, non-text parts are noted
 * rather than fabricated, Claude's structured `content[]` wins over the legacy
 * `text` field, attachments survive, and nothing is dropped without a reason
 * in the skip ledger.
 */

import { describe, expect, it, beforeAll, afterAll } from 'vitest';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPT = join(HERE, '..', '..', 'code', 'parse-chat-export.mjs');
const FIXTURES = join(HERE, '..', 'fixtures');

let outDir;

function run(fixture, extra = []) {
  execFileSync(
    process.execPath,
    [SCRIPT, join(FIXTURES, fixture), '--out', outDir, '--min-messages', '3', '--min-user-chars', '50', ...extra],
    { encoding: 'utf8' },
  );
}

function readPages(provider) {
  const dir = join(outDir, provider);
  return readdirSync(dir)
    .filter((f) => f.endsWith('.md'))
    .map((f) => readFileSync(join(dir, f), 'utf8'));
}

beforeAll(() => {
  outDir = mkdtempSync(join(tmpdir(), 'chats-to-brain-'));
  run('chatgpt.json');
  run('claude.json');
});

afterAll(() => {
  if (outDir) rmSync(outDir, { recursive: true, force: true });
});

describe('chats-to-brain: ChatGPT tree reconstruction', () => {
  it('excludes abandoned regenerate branches', () => {
    const all = readPages('chatgpt').join('\n');
    expect(all).not.toContain('ABANDONED BRANCH');
  });

  it('drops system turns', () => {
    expect(readPages('chatgpt').join('\n')).not.toContain('you are helpful');
  });

  it('drops turns hidden from the conversation', () => {
    // metadata.is_visually_hidden_from_conversation
    const all = readPages('chatgpt').join('\n');
    expect(all).not.toMatch(/^hidden$/m);
  });

  it('notes non-text parts instead of fabricating them', () => {
    expect(readPages('chatgpt').join('\n')).toContain('image_asset_pointer omitted');
  });

  it('keeps the surviving user and assistant turns in order', () => {
    const page = readPages('chatgpt').find((p) => p.includes('Designing a retrieval pipeline'));
    expect(page).toBeDefined();
    const userIdx = page.indexOf('hybrid search only works because RRF');
    const asstIdx = page.indexOf('Reciprocal rank fusion is forgiving');
    expect(userIdx).toBeGreaterThan(-1);
    expect(asstIdx).toBeGreaterThan(userIdx);
  });
});

describe('chats-to-brain: Claude export shapes', () => {
  it('prefers structured content[] over the legacy text field', () => {
    const all = readPages('claude').join('\n');
    expect(all).toContain('knowledge management failed for thirty years');
    expect(all).not.toContain('ignored fallback');
  });

  it('falls back to `text` when content[] is absent', () => {
    expect(readPages('claude').join('\n')).toContain('Older-export shape');
  });

  it('inlines attachment extracted_content', () => {
    expect(readPages('claude').join('\n')).toContain('Attached note body.');
  });
});

describe('chats-to-brain: filtering is auditable', () => {
  it('records every skipped conversation with a reason', () => {
    const ledger = readFileSync(join(outDir, 'chatgpt', '_skipped.tsv'), 'utf8');
    expect(ledger.split('\n')[0]).toBe('title\treason');
    // The one-message "fix this regex" conversation is below --min-messages.
    expect(ledger).toContain('fix this regex');
    expect(ledger).toContain('messages<3');
  });

  it('writes a manifest describing what landed', () => {
    const manifest = JSON.parse(readFileSync(join(outDir, 'chatgpt', '_manifest.json'), 'utf8'));
    expect(manifest.provider).toBe('chatgpt');
    expect(manifest.written.length).toBeGreaterThan(0);
  });

  it('auto-detects provider from JSON shape, not filename', () => {
    // Both fixtures are named for their provider but detection reads
    // `mapping` vs `chat_messages`; the output dirs prove which branch ran.
    expect(readdirSync(outDir).sort()).toEqual(['chatgpt', 'claude']);
  });
});
