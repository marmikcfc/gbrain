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
 *   --dry-run              Report only; write nothing
 *   --json                 Machine-readable summary on stdout
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
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
 * gbrain's two-layer page contract: compiled truth ABOVE the horizontal rule
 * (rewritten freely as understanding improves), append-only timeline BELOW it
 * (never rewritten). The raw transcript is evidence, so it belongs below;
 * the summary block above is left as a stub for the agent to fill in.
 */
function renderPage(convo, provider, opts) {
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

  const written = [];
  const skipped = [];
  const usedSlugs = new Set();

  for (const rawConvo of convos) {
    if (written.length >= opts.limit) {
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

    // Collisions are common (many convos titled "untitled"); disambiguate with
    // a short hash of the stable source id rather than a mutable counter.
    let slug = `${isoDay(convo.created)}-${slugify(convo.title)}`;
    if (usedSlugs.has(slug)) {
      const h = createHash('sha1').update(String(convo.id ?? convo.title)).digest('hex').slice(0, 6);
      slug = `${slug}-${h}`;
    }
    usedSlugs.add(slug);

    const path = join(outDir, `${slug}.md`);
    if (!opts.dryRun) writeFileSync(path, renderPage(convo, provider, opts), 'utf8');

    written.push({ slug, title: convo.title, messages: convo.messages.length, userChars: verdict.userChars, path });
  }

  // Nothing is dropped silently: the skip ledger is a first-class artifact.
  if (!opts.dryRun) {
    writeFileSync(
      join(outDir, '_skipped.tsv'),
      ['title\treason', ...skipped.map((s) => `${s.title.replace(/\t/g, ' ')}\t${s.reason}`)].join('\n'),
      'utf8',
    );
    writeFileSync(
      join(outDir, '_manifest.json'),
      JSON.stringify({ provider, source_file: file, written, skipped_count: skipped.length }, null, 2),
      'utf8',
    );
  }

  const summary = {
    provider,
    source_file: file,
    total: convos.length,
    written: written.length,
    skipped: skipped.length,
    out_dir: outDir,
    dry_run: opts.dryRun,
  };

  if (opts.json) {
    console.log(JSON.stringify({ ...summary, skipped_reasons: tally(skipped) }, null, 2));
  } else {
    console.log(`provider:  ${provider}`);
    console.log(`total:     ${convos.length}`);
    console.log(`written:   ${written.length}${opts.dryRun ? ' (dry run — nothing written)' : ''}`);
    console.log(`skipped:   ${skipped.length}`);
    for (const [reason, n] of Object.entries(tally(skipped)).sort((a, b) => b[1] - a[1])) {
      console.log(`  ${String(n).padStart(6)}  ${reason}`);
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
