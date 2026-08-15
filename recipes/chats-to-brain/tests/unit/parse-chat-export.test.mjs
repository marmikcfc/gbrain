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
import { mkdtempSync, rmSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
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

/**
 * Re-import identity. Providers retitle conversations (auto-titling settles
 * late, users rename threads), and the filename embeds the title — so keying
 * pages on the filename means a retitled conversation lands a SECOND time
 * while the first page stays behind: duplicate transcripts, embeddings, facts
 * and retrieval hits. Identity is the stable source id (`uuid` / `conversation_id`).
 */
describe('chats-to-brain: re-import is stable under retitles', () => {
  let dir;
  const ID = 'cl-retitle-1';

  const exportWith = (name) => JSON.stringify([{
    uuid: ID,
    name,
    created_at: '2025-03-04T09:00:00.000000Z',
    updated_at: '2025-03-04T10:00:00.000000Z',
    chat_messages: Array.from({ length: 6 }, (_, i) => ({
      uuid: `m${i}`,
      sender: i % 2 === 0 ? 'human' : 'assistant',
      created_at: '2025-03-04T09:00:00Z',
      content: [{ type: 'text', text: `Turn ${i}: a long enough message about retrieval to clear the user-char floor.` }],
    })),
  }]);

  function reimport(name, extra = []) {
    const src = join(dir, `export-${Buffer.from(name).toString('hex').slice(0, 8)}.json`);
    writeFileSync(src, exportWith(name), 'utf8');
    return execFileSync(
      process.execPath,
      [SCRIPT, src, '--out', join(dir, 'out'), '--min-messages', '3', '--min-user-chars', '50', ...extra],
      { encoding: 'utf8' },
    );
  }

  const pages = () => readdirSync(join(dir, 'out', 'claude')).filter((f) => f.endsWith('.md'));
  const manifest = () => JSON.parse(readFileSync(join(dir, 'out', 'claude', '_manifest.json'), 'utf8'));

  beforeAll(() => { dir = mkdtempSync(join(tmpdir(), 'chats-retitle-')); });
  afterAll(() => { if (dir) rmSync(dir, { recursive: true, force: true }); });

  it('writes ONE page per conversation id when the provider retitles it', () => {
    reimport('Draft thoughts on retrieval');
    expect(pages()).toEqual(['2025-03-04-draft-thoughts-on-retrieval.md']);

    const out = reimport('Precomputed synthesis at write time');

    // The defect this pins: a second page under the new title's name.
    expect(pages()).toEqual(['2025-03-04-draft-thoughts-on-retrieval.md']);
    expect(manifest().written).toHaveLength(1);
    expect(manifest().written[0].id).toBe(ID);
    expect(manifest().written[0].reused).toBe(true);

    // Never silent: the drift is reported on stdout and in the manifest.
    expect(out).toContain('retitled:');
    expect(manifest().retitled).toHaveLength(1);
    expect(manifest().retitled[0]).toMatchObject({
      id: ID,
      title: 'Precomputed synthesis at write time',
      would_be_slug: '2025-03-04-precomputed-synthesis-at-write-time',
    });

    // The page itself is current even though its filename is not.
    const page = readFileSync(join(dir, 'out', 'claude', pages()[0]), 'utf8');
    expect(page).toContain('title: "Precomputed synthesis at write time"');
    expect(page).toContain(`source_id: ${ID}`);
  });

  it('carries the enrich pass forward instead of overwriting it', () => {
    const path = join(dir, 'out', 'claude', pages()[0]);
    writeFileSync(path, readFileSync(path, 'utf8').replace(/_Not yet synthesized[^\n]*/, 'AGENT SYNTHESIS'), 'utf8');

    reimport('Precomputed synthesis at write time');

    expect(pages()).toHaveLength(1);
    const page = readFileSync(join(dir, 'out', 'claude', pages()[0]), 'utf8');
    expect(page).toContain('AGENT SYNTHESIS');           // above the rule: preserved
    expect(page).toContain('Turn 0: a long enough');      // below the rule: regenerated
  });

  it('--rehome moves the page rather than leaving a second one', () => {
    const out = reimport('Precomputed synthesis at write time', ['--rehome']);

    expect(pages()).toEqual(['2025-03-04-precomputed-synthesis-at-write-time.md']);
    expect(out).toContain('renamed:');
    expect(out).toContain('2025-03-04-draft-thoughts-on-retrieval.md');
    expect(manifest().renamed).toHaveLength(1);
    expect(manifest().renamed[0].id).toBe(ID);
    expect(readFileSync(join(dir, 'out', 'claude', pages()[0]), 'utf8')).toContain('AGENT SYNTHESIS');
  });
});
