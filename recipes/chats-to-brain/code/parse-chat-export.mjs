#!/usr/bin/env bun
/**
 * parse-chat-export.mjs — ChatGPT / Claude data-export → gbrain markdown.
 *
 * The DETERMINISTIC half of the chats-to-brain recipe. Code handles data;
 * the agent handles judgment. This script never calls an LLM, never guesses
 * what matters, and never silently drops a conversation — everything it
 * skips lands in _skipped.tsv with a reason.
 *
 * Supports (auto-detected):
 *   - ChatGPT  conversations.json  — node/parent tree, walked from current_node
 *   - Claude   conversations.json  — flat chat_messages array
 *
 * Zero dependencies. Runs under bun or node >= 18.
 *
 * Usage:
 *   bun parse-chat-export.mjs <export-path> --out ~/brain/sources/chats [flags]
 *
 * Flags:
 *   --out <dir>            Output dir (default: ./chats-out)
 *   --provider <p>         auto | chatgpt | claude   (default: auto)
 *   --min-messages <n>     Skip convos with fewer messages (default: 6)
 *   --min-user-chars <n>   Skip convos where you wrote less (default: 400)
 *   --since <YYYY-MM-DD>   Skip convos created before this date
 *   --exclude-title <re>   Skip titles matching regex (repeatable)
 *   --limit <n>            Stop after writing n conversations
 *   --rehome               On a retitle, MOVE the page to the new title's
 *                          filename instead of keeping the original one.
 *                          Either way identity is the source id, never the name.
 *   --dry-run              Report only; write nothing
 *   --json                 Machine-readable summary on stdout
 */

import {
  readFileSync, writeFileSync, mkdirSync, existsSync, statSync,
  readdirSync, unlinkSync, openSync, readSync, closeSync,
} from 'node:fs';
import { join, resolve, basename } from 'node:path';
import { createHash } from 'node:crypto';

// ─────────────────────────────────────────────────────────── args

function parseArgs(argv) {
  const opts = {
    input: null,
    out: './chats-out',
    provider: 'auto',
    minMessages: 6,
    minUserChars: 400,
    since: null,
    excludeTitle: [],
    limit: Infinity,
    rehome: false,
    dryRun: false,
    json: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--out') opts.out = argv[++i];
    else if (a === '--provider') opts.provider = argv[++i];
    else if (a === '--min-messages') opts.minMessages = Number(argv[++i]);
    else if (a === '--min-user-chars') opts.minUserChars = Number(argv[++i]);
    else if (a === '--since') opts.since = argv[++i];
    else if (a === '--exclude-title') opts.excludeTitle.push(new RegExp(argv[++i], 'i'));
    else if (a === '--limit') opts.limit = Number(argv[++i]);
    else if (a === '--rehome') opts.rehome = true;
    else if (a === '--dry-run') opts.dryRun = true;
    else if (a === '--json') opts.json = true;
    else if (a === '--help' || a === '-h') { console.log(HELP); process.exit(0); }
    else if (a.startsWith('--')) { die(`unknown flag: ${a}`); }
    else if (!opts.input) opts.input = a;
  }
  if (!opts.input) die('missing <export-path>. Try --help.');
  return opts;
}

const HELP = readFileSync(new URL(import.meta.url).pathname, 'utf8')
  .split('\n').slice(1).filter((l) => l.startsWith(' *'))
  .map((l) => l.replace(/^ \*ic?/, '').replace(/^ \* ?/, '')).join('\n');

function die(msg) { console.error(`error: ${msg}`); process.exit(1); }

// ─────────────────────────────────────────────────────────── input

/** Accept either conversations.json directly or the unzipped export folder. */
function locateConversationsFile(input) {
  const p = resolve(input);
  if (!existsSync(p)) die(`path not found: ${p}`);
  if (statSync(p).isDirectory()) {
    const candidate = join(p, 'conversations.json');
    if (!existsSync(candidate)) {
      die(`no conversations.json inside ${p}. Point at the file directly.`);
    }
    return candidate;
  }
  return p;
}

/**
 * Format detection is structural, not filename-based — both providers ship a
 * file literally named conversations.json, so the shape is the only signal.
 * ChatGPT conversations carry a `mapping` object; Claude's carry
 * `chat_messages`. Anything else we refuse rather than half-parse.
 */
function detectProvider(convos) {
  const sample = convos.find((c) => c && typeof c === 'object');
  if (!sample) die('export contains no conversation objects');
  if (sample.mapping) return 'chatgpt';
  if (sample.chat_messages) return 'claude';
  die('unrecognized export shape (no `mapping` or `chat_messages` key)');
}

// ─────────────────────────────────────────────────────────── extraction

/**
 * ChatGPT stores a conversation as a TREE, not a list — every edit or
 * regenerate forks a branch, and `mapping` holds all of them. Reading
 * Object.values(mapping) would interleave abandoned branches into the
 * transcript. Walking parent-pointers up from `current_node` yields exactly
 * the surviving conversation, which is what the user actually saw.
 */
function extractChatGPT(convo) {
  const mapping = convo.mapping ?? {};
  const path = [];
  const seen = new Set();
  let nodeId = convo.current_node;

  while (nodeId && mapping[nodeId] && !seen.has(nodeId)) {
    seen.add(nodeId);                     // cycle guard: malformed exports exist
    path.push(mapping[nodeId]);
    nodeId = mapping[nodeId].parent;
  }
  path.reverse();

  const messages = [];
  for (const node of path) {
    const m = node.message;
    if (!m) continue;

    const role = m.author?.role;
    if (role !== 'user' && role !== 'assistant') continue;      // drop system/tool
    if (m.metadata?.is_visually_hidden_from_conversation) continue;
    if (m.recipient && m.recipient !== 'all') continue;          // tool-directed

    const text = chatGPTContentToText(m.content);
    if (!text.trim()) continue;

    messages.push({
      role: role === 'user' ? 'user' : 'assistant',
      text: text.trim(),
      ts: m.create_time ? new Date(m.create_time * 1000) : null,
    });
  }

  return {
    id: convo.conversation_id ?? convo.id ?? null,
    title: (convo.title ?? '').trim() || 'untitled',
    created: convo.create_time ? new Date(convo.create_time * 1000) : null,
    updated: convo.update_time ? new Date(convo.update_time * 1000) : null,
    messages,
  };
}

/** content_type varies: text, multimodal_text, code, execution_output, thoughts. */
function chatGPTContentToText(content) {
  if (!content) return '';
  if (typeof content === 'string') return content;

  if (Array.isArray(content.parts)) {
    return content.parts
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object') {
          if (typeof part.text === 'string') return part.text;
          // image_asset_pointer and friends: note the gap, don't fabricate.
          if (part.content_type) return `_[${part.content_type} omitted]_`;
        }
        return '';
      })
      .filter(Boolean)
      .join('\n\n');
  }
  if (typeof content.text === 'string') return content.text;
  return '';
}

/** Claude's export is mercifully flat — an ordered chat_messages array. */
function extractClaude(convo) {
  const messages = [];
  for (const m of convo.chat_messages ?? []) {
    const role = m.sender === 'human' ? 'user' : m.sender === 'assistant' ? 'assistant' : null;
    if (!role) continue;

    // Newer exports use content[]; older ones only have `text`. Prefer the
    // structured array so attachments/blocks stay ordered, fall back cleanly.
    let text = '';
    if (Array.isArray(m.content) && m.content.length) {
      text = m.content
        .map((b) => (typeof b?.text === 'string' ? b.text : b?.type ? `_[${b.type} omitted]_` : ''))
        .filter(Boolean)
        .join('\n\n');
    }
    if (!text.trim() && typeof m.text === 'string') text = m.text;
    if (!text.trim()) continue;

    for (const att of m.attachments ?? []) {
      if (att?.extracted_content) {
        text += `\n\n**Attachment — ${att.file_name ?? 'file'}:**\n\n${att.extracted_content}`;
      }
    }

    messages.push({ role, text: text.trim(), ts: safeDate(m.created_at) });
  }

  return {
    id: convo.uuid ?? null,
    title: (convo.name ?? '').trim() || 'untitled',
    created: safeDate(convo.created_at),
    updated: safeDate(convo.updated_at),
    messages,
  };
}

function safeDate(v) {
  if (!v) return null;
  const d = new Date(v);
  return Number.isNaN(d.getTime()) ? null : d;
}

// ─────────────────────────────────────────────────────────── filtering

function evaluate(convo, opts) {
  const userChars = convo.messages
    .filter((m) => m.role === 'user')
    .reduce((n, m) => n + m.text.length, 0);

  if (convo.messages.length === 0) return { keep: false, reason: 'empty', userChars };
  if (convo.messages.length < opts.minMessages) {
    return { keep: false, reason: `messages<${opts.minMessages}`, userChars };
  }
  if (userChars < opts.minUserChars) {
    return { keep: false, reason: `user_chars<${opts.minUserChars}`, userChars };
  }
  if (opts.since && convo.created && convo.created < new Date(opts.since)) {
    return { keep: false, reason: `before ${opts.since}`, userChars };
  }
  for (const re of opts.excludeTitle) {
    if (re.test(convo.title)) return { keep: false, reason: `title~${re.source}`, userChars };
  }
  return { keep: true, reason: null, userChars };
}

// ─────────────────────────────────────────────────────────── rendering

function slugify(title) {
  return (title || 'untitled')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60) || 'untitled';
}

function isoDay(d) { return d ? d.toISOString().slice(0, 10) : 'undated'; }

function sourceUrl(provider, id) {
  if (!id) return null;
  return provider === 'chatgpt'
    ? `https://chatgpt.com/c/${id}`
    : `https://claude.ai/chat/${id}`;
}

/**
 * The compiled-truth block a fresh page starts life with. The enrich pass
 * (agent judgment) rewrites this; a re-import must not stomp on that work,
 * so `readCompiled` lifts whatever is there now and hands it back to
 * `renderPage` unchanged.
 */
const COMPILED_STUB = [
  '## Summary',
  '',
  '_Not yet synthesized — run the enrich pass. Everything below the rule is raw evidence._',
  '',
  '## Open Threads',
  '',
  '_TBD_',
  '',
  '## See Also',
  '',
  '_TBD_',
  '',
].join('\n');

/**
 * Recover the compiled-truth block from a page written by an earlier run.
 * Above the rule is the agent's synthesis; below it is evidence this script
 * regenerates from the export. Re-importing a conversation therefore refreshes
 * the transcript and the frontmatter while leaving the synthesis alone —
 * without this, every re-run would silently delete the enrich pass.
 */
function readCompiled(path) {
  if (!path || !existsSync(path)) return COMPILED_STUB;
  let text;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return COMPILED_STUB;
  }
  const rule = text.match(/\n---\n\n## Transcript\n/);
  if (!rule) return COMPILED_STUB;
  const start = text.search(/^## /m);
  if (start < 0 || start >= rule.index) return COMPILED_STUB;
  return text.slice(start, rule.index);
}

/**
 * gbrain's two-layer page contract: compiled truth ABOVE the horizontal rule
 * (rewritten freely as understanding improves), append-only timeline BELOW it
 * (never rewritten). The raw transcript is evidence, so it belongs below;
 * the summary block above is a stub on a first import and is carried forward
 * verbatim on every re-import.
 */
function renderPage(convo, provider, compiled = COMPILED_STUB) {
  const created = isoDay(convo.created);
  const url = sourceUrl(provider, convo.id);
  const label = provider === 'chatgpt' ? 'ChatGPT' : 'Claude';
  const userChars = convo.messages.filter((m) => m.role === 'user').reduce((n, m) => n + m.text.length, 0);

  const fm = [
    '---',
    `title: ${JSON.stringify(convo.title)}`,
    'kind: conversation',
    `source: ${provider}`,
    convo.id ? `source_id: ${convo.id}` : null,
    url ? `url: ${url}` : null,
    `created: ${created}`,
    `updated: ${isoDay(convo.updated)}`,
    `message_count: ${convo.messages.length}`,
    `user_chars: ${userChars}`,
    `imported_by: chats-to-brain`,
    'tags: [chat-import, ' + provider + ']',
    '---',
  ].filter(Boolean).join('\n');

  const head = [
    '',
    `# ${convo.title}`,
    '',
    `**Source:** ${label}${url ? ` · [open original](${url})` : ''} · `
      + `**Started:** ${created} · **Messages:** ${convo.messages.length}`,
    '',
    compiled,
    '---',
    '',
    '## Transcript',
    '',
  ].join('\n');

  const body = convo.messages
    .map((m) => {
      const who = m.role === 'user' ? 'Me' : label;
      const when = m.ts ? ` · ${isoDay(m.ts)}` : '';
      return `### ${who}${when}\n\n${m.text}\n`;
    })
    .join('\n');

  return `${fm}${head}${body}`;
}

// ─────────────────────────────────────────────────────────── identity

/**
 * A conversation's identity is the provider's stable id — ChatGPT
 * `conversation_id`, Claude `uuid` — and NEVER its filename.
 *
 * Both providers retitle conversations (auto-titling settles late, users
 * rename threads). The filename is `<created-day>-<title-slug>`, so a retitle
 * moves it. Keying on the filename therefore means a retitled conversation
 * imports a SECOND time under a new name while the old page sits there
 * forever: two transcripts, two embeddings, two sets of extracted facts, two
 * retrieval hits that disagree about which is current.
 *
 * So every run starts by rebuilding the id → page index from what is already
 * on disk. `source_id:` in each page's frontmatter is the ground truth — it
 * survives a deleted, stale, or hand-edited manifest, and it is what the
 * duplicate would actually be duplicating. The previous `_manifest.json` is
 * read as well, but only as a hint: a page it lists that no longer exists
 * cannot be duplicated, so disk always wins.
 */

const FRONTMATTER_PROBE_BYTES = 4096;

/** Read just the head of a file — pages are long, frontmatter is not. */
function readHead(path, bytes = FRONTMATTER_PROBE_BYTES) {
  const fd = openSync(path, 'r');
  try {
    const buf = Buffer.alloc(bytes);
    const n = readSync(fd, buf, 0, bytes, 0);
    return buf.subarray(0, n).toString('utf8');
  } finally {
    closeSync(fd);
  }
}

function sourceIdOf(head) {
  const m = head.match(/^source_id:[ \t]*(\S.*?)\s*$/m);
  return m ? m[1] : null;
}

/**
 * @returns {{ byId: Map<string, string[]>, idBySlug: Map<string, string|null> }}
 *   byId     — every existing page path carrying that source id, sorted, so a
 *              duplicate left behind by an older buggy run is visible, not lost.
 *   idBySlug — which slug is already occupied by whom, so a new conversation
 *              never clobbers a page belonging to a different conversation
 *              (including one this run's filters excluded).
 */
function buildPriorIndex(outDir) {
  const byId = new Map();
  const idBySlug = new Map();
  const add = (id, path) => {
    const paths = byId.get(id) ?? [];
    if (!paths.includes(path)) paths.push(path);
    byId.set(id, paths);
  };

  if (!existsSync(outDir)) return { byId, idBySlug };

  // 1. the previous manifest — a hint only, and only for pages still present.
  const manifestPath = join(outDir, '_manifest.json');
  if (existsSync(manifestPath)) {
    try {
      const prior = JSON.parse(readFileSync(manifestPath, 'utf8'));
      for (const w of prior?.written ?? []) {
        if (w?.id && w?.path && existsSync(w.path)) add(String(w.id), w.path);
      }
    } catch {
      // A corrupt manifest must never break an import; the scan below stands
      // on its own.
    }
  }

  // 2. the pages themselves — ground truth.
  for (const f of readdirSync(outDir)) {
    if (!f.endsWith('.md')) continue;
    const path = join(outDir, f);
    let id = null;
    try {
      id = sourceIdOf(readHead(path));
    } catch {
      // unreadable page: still occupies its slug, just anonymously
    }
    idBySlug.set(f.slice(0, -3), id);
    if (id) add(id, path);
  }

  for (const [id, paths] of byId) byId.set(id, paths.sort());
  return { byId, idBySlug };
}

/**
 * Decide where each kept conversation lands, before anything is written.
 *
 * Two properties matter and neither survives a streaming single pass:
 *   - a conversation already imported keeps its page (identity by id), and
 *   - which of two same-slug conversations gets the bare name must not depend
 *     on their order in the export file, which providers do not promise.
 */
function assignPaths(kept, prior, outDir, opts) {
  const entries = kept.map((k) => {
    const id = k.convo.id != null ? String(k.convo.id) : null;
    const priorPaths = (id && prior.byId.get(id)) || [];
    return {
      ...k,
      id,
      priorPaths,
      canonical: priorPaths[0] ?? null,
      desired: `${isoDay(k.convo.created)}-${slugify(k.convo.title)}`,
    };
  });

  const owner = new Map(prior.idBySlug); // slug -> id | null (null = unknown owner)
  const floating = [];

  // Previously imported conversations claim their existing page first.
  for (const e of entries) {
    if (e.canonical && !opts.rehome) {
      e.slug = basename(e.canonical, '.md');
      owner.set(e.slug, e.id);
    } else {
      floating.push(e);
    }
  }

  // Order-independent collision handling: if two conversations in this run
  // want the same slug, BOTH get the hash suffix. Handing the bare slug to
  // whichever appeared first makes the output depend on export ordering.
  const demand = new Map();
  for (const e of floating) demand.set(e.desired, (demand.get(e.desired) ?? 0) + 1);

  for (const e of floating) {
    const own = new Set(e.priorPaths.map((p) => basename(p, '.md')));
    const free = (s) => own.has(s) || !owner.has(s) || owner.get(s) === e.id;
    const h = createHash('sha1')
      .update(e.id ?? `${e.desired}|${e.convo.title}`)
      .digest('hex');

    const cands = demand.get(e.desired) > 1 ? [] : [e.desired];
    cands.push(`${e.desired}-${h.slice(0, 6)}`, `${e.desired}-${h}`);
    let slug = cands.find(free);
    for (let n = 2; !slug; n++) {
      const c = `${e.desired}-${h.slice(0, 6)}-${n}`;
      if (free(c)) slug = c;
    }
    e.slug = slug;
    owner.set(slug, e.id);
  }

  for (const e of entries) e.path = join(outDir, `${e.slug}.md`);
  return entries;
}

// ─────────────────────────────────────────────────────────── main

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const file = locateConversationsFile(opts.input);

  let raw;
  try {
    raw = JSON.parse(readFileSync(file, 'utf8'));
  } catch (e) {
    die(`could not parse ${file}: ${e.message}`);
  }
  const convos = Array.isArray(raw) ? raw : raw.conversations;
  if (!Array.isArray(convos)) die('expected a top-level array of conversations');

  const provider = opts.provider === 'auto' ? detectProvider(convos) : opts.provider;
  const extract = provider === 'chatgpt' ? extractChatGPT : extractClaude;

  const outDir = join(resolve(opts.out), provider);
  if (!opts.dryRun) mkdirSync(outDir, { recursive: true });

  // What a previous run left behind, keyed by stable id — read BEFORE the
  // manifest is overwritten.
  const prior = buildPriorIndex(outDir);

  const skipped = [];
  const kept = [];

  for (const rawConvo of convos) {
    if (kept.length >= opts.limit) {
      skipped.push({ title: rawConvo?.title ?? rawConvo?.name ?? '?', reason: `--limit ${opts.limit}` });
      continue;
    }

    let convo;
    try {
      convo = extract(rawConvo);
    } catch (e) {
      skipped.push({ title: rawConvo?.title ?? rawConvo?.name ?? '?', reason: `parse_error: ${e.message}` });
      continue;
    }

    const verdict = evaluate(convo, opts);
    if (!verdict.keep) {
      skipped.push({ title: convo.title, reason: verdict.reason });
      continue;
    }

    kept.push({ convo, userChars: verdict.userChars });
  }

  const plan = assignPaths(kept, prior, outDir, opts);

  const written = [];
  const retitled = [];   // page kept at its old path; the title moved
  const renamed = [];    // --rehome: page moved to the new title's path
  const duplicates = []; // >1 existing page for one id — damage from older runs

  for (const e of plan) {
    // Carry the enrich pass forward: only the evidence half is regenerated.
    const compiled = readCompiled(e.canonical ?? e.path);
    if (!opts.dryRun) writeFileSync(e.path, renderPage(e.convo, provider, compiled), 'utf8');

    const stale = e.priorPaths.filter((p) => p !== e.path);
    const drifted = e.canonical && e.slug !== e.desired && !e.slug.startsWith(`${e.desired}-`);

    if (stale.length && opts.rehome) {
      // The move is the point of --rehome, so old paths go away rather than
      // linger as duplicates. The report names every one of them.
      for (const p of stale) {
        if (!opts.dryRun) { try { unlinkSync(p); } catch { /* already gone */ } }
      }
      renamed.push({ id: e.id, title: e.convo.title, from: stale, to: e.path });
    } else if (stale.length) {
      // Not ours to delete — a page may carry hand-written synthesis. Report it.
      duplicates.push({ id: e.id, title: e.convo.title, canonical: e.path, extra: stale });
    }

    if (drifted && !opts.rehome) {
      retitled.push({ id: e.id, title: e.convo.title, path: e.path, would_be_slug: e.desired });
    }

    written.push({
      id: e.id,
      slug: e.slug,
      title: e.convo.title,
      messages: e.convo.messages.length,
      userChars: e.userChars,
      path: e.path,
      reused: Boolean(e.canonical),
    });
  }

  const keptIds = new Set(plan.map((e) => e.id).filter(Boolean));
  const untouched = [...prior.byId.keys()].filter((id) => !keptIds.has(id)).length;
  const reused = written.filter((w) => w.reused).length;

  // Nothing is dropped silently: the skip ledger is a first-class artifact.
  if (!opts.dryRun) {
    writeFileSync(
      join(outDir, '_skipped.tsv'),
      ['title\treason', ...skipped.map((s) => `${s.title.replace(/\t/g, ' ')}\t${s.reason}`)].join('\n'),
      'utf8',
    );
    writeFileSync(
      join(outDir, '_manifest.json'),
      JSON.stringify({
        provider,
        source_file: file,
        written,
        skipped_count: skipped.length,
        reused_count: reused,
        retitled,
        renamed,
        duplicate_pages: duplicates,
        prior_pages_untouched: untouched,
      }, null, 2),
      'utf8',
    );
  }

  const summary = {
    provider,
    source_file: file,
    total: convos.length,
    written: written.length,
    reused: reused,
    retitled: retitled.length,
    renamed: renamed.length,
    duplicate_pages: duplicates.length,
    skipped: skipped.length,
    out_dir: outDir,
    dry_run: opts.dryRun,
  };

  if (opts.json) {
    console.log(JSON.stringify({ ...summary, retitled, renamed, duplicate_pages: duplicates, skipped_reasons: tally(skipped) }, null, 2));
  } else {
    console.log(`provider:  ${provider}`);
    console.log(`total:     ${convos.length}`);
    console.log(`written:   ${written.length}${opts.dryRun ? ' (dry run — nothing written)' : ''}`);
    console.log(`reused:    ${reused} (matched an existing page by source id)`);
    console.log(`skipped:   ${skipped.length}`);
    for (const [reason, n] of Object.entries(tally(skipped)).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${String(n).padStart(6)}  ${reason}`);
    }
    for (const r of retitled) {
      console.log(`retitled:  ${basename(r.path)} is now titled ${JSON.stringify(r.title)} `
        + `(filename left alone; --rehome would move it to ${r.would_be_slug}.md)`);
    }
    for (const r of renamed) {
      console.log(`renamed:   ${r.from.map((p) => basename(p)).join(', ')} → ${basename(r.to)} `
        + `— re-run \`gbrain import\`, then prune the old page from the brain`);
    }
    for (const d of duplicates) {
      console.log(`duplicate: source_id ${d.id} also lives at ${d.extra.map((p) => basename(p)).join(', ')} `
        + `— left in place; delete by hand or re-run with --rehome`);
    }
    console.log(`out:       ${outDir}`);
    if (!opts.dryRun) console.log(`ledger:    ${join(outDir, '_skipped.tsv')}`);
  }
}

function tally(rows) {
  const out = {};
  for (const r of rows) out[r.reason] = (out[r.reason] ?? 0) + 1;
  return out;
}

main();
