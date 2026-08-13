---
id: chats-to-brain
name: Chats-to-Brain
version: 0.1.0
description: ChatGPT and Claude data exports become brain pages. Deterministic parser walks the export, filters noise, and writes conversation pages; the agent then promotes the signal into subject directories.
category: sense
requires: []
secrets: []
health_checks:
  - type: file_exists
    path: "$CHAT_EXPORT_PATH/conversations.json"
    label: "Chat export"
setup_time: 15 min
cost_estimate: "$0 to parse. Embedding cost scales with what you keep — filter first."
---

# Chats-to-Brain: Your ChatGPT and Claude History as Brain Pages

Years of conversations with an LLM contain your actual thinking — theses you
worked out loud, frameworks you invented, problems you chewed on for months.
That history is trapped in two vendors' export archives. This recipe frees it.

## IMPORTANT: Instructions for the Agent

**You are the installer.** Follow these steps precisely.

**The core pattern: code for data, LLMs for judgment.**

1. DETERMINISTIC: `parse-chat-export.mjs` walks the export, reconstructs each
   conversation, applies numeric filters, and writes markdown. This never
   fails and never guesses. Branch reconstruction and timestamps are exact.
2. LATENT: you (the agent) read the imported pages and make the judgment
   calls. What is original thinking? Which entities are notable? What gets
   promoted out of `sources/` into a subject directory?

**Do not invert this.** Do not ask the model which conversations to keep —
that costs a fortune and is less reliable than a message-count threshold.
Do not ask the script which ideas matter — it cannot know.

## Architecture

```
  ChatGPT export.zip                Claude export.zip
  └── conversations.json            └── conversations.json
        (node/parent TREE)                (flat chat_messages[])
              │                                  │
              └──────────┬───────────────────────┘
                         ▼
              parse-chat-export.mjs          ← deterministic, zero-dep
                 · auto-detects provider
                 · walks current_node → root  (ChatGPT branches)
                 · drops system/tool/hidden turns
                 · numeric filters + skip ledger
                         │
                         ▼
        sources/chats/<provider>/<date>-<slug>.md
                         │
              gbrain import → embed → extract-conversation-facts
                         │
                         ▼
              AGENT ENRICH PASS  ← judgment lives here
                 originals/ · concepts/ · people/ · companies/
```

## Opinionated Defaults

| Default | Value | Why |
|---|---|---|
| Output directory | `sources/chats/<provider>/` | Filing rules: `sources/` is for bulk data imports. A chat archive is exactly that. |
| `--min-messages` | 6 | "fix this regex" is not knowledge. Short exchanges are overwhelmingly throwaway. |
| `--min-user-chars` | 400 | Filters on **what you wrote**, not total length. A long assistant monologue you replied "ok" to carries no signal of yours. |
| Branch handling | active path only | ChatGPT stores every regenerate as a fork. Importing all branches duplicates content and poisons retrieval. |
| System / tool turns | dropped | Not your thinking, and they wreck entity extraction. |
| Skipped conversations | logged to `_skipped.tsv` | Never drop anything silently. A filter you can't audit is a filter you can't tune. |

The two thresholds are the whole cost-control story. Raise them before you
lower them — you can always re-run with looser filters, but you cannot
un-spend the embedding budget.

## Prerequisites

- A working brain (`gbrain doctor` passes)
- `bun` or node >= 18
- Your export(s), unzipped:
  - **ChatGPT** — Settings → Data Controls → Export data → email link → unzip
  - **Claude** — Settings → Privacy → Export data → email link → unzip

Both arrive as a folder containing `conversations.json`.

## Setup Flow

### Step 1: Survey before you import

Always dry-run first. This is the step that tells you what your archive
actually looks like.

```bash
bun code/parse-chat-export.mjs ~/Downloads/chatgpt-export --dry-run
```

Read the skip tally. If it says `written: 3000`, your thresholds are too
loose — most people's archives are 80–90% noise. Tighten and re-run until
the kept count looks like "conversations I would actually want resurfaced."

### Step 2: Convert

```bash
BRAIN=~/brain

bun code/parse-chat-export.mjs ~/Downloads/chatgpt-export \
  --out $BRAIN/sources/chats \
  --min-messages 8 --min-user-chars 600 --since 2024-01-01

bun code/parse-chat-export.mjs ~/Downloads/claude-export \
  --out $BRAIN/sources/chats \
  --min-messages 8 --min-user-chars 600
```

Provider is auto-detected from the JSON shape and each writes to its own
subdirectory, so both commands can share one `--out`.

### Step 3: Import and index

```bash
gbrain import $BRAIN/sources/chats/ --no-embed
gbrain embed --stale
gbrain extract-conversation-facts       # REQUIRED — see below
gbrain extract links --source db
gbrain stats                            # verify pages and links > 0
```

**Do not skip `extract-conversation-facts`.** Long conversations defeat
chunk-level retrieval: a chunk reading "the answer is 9494" has no topical
anchor, so it is unfindable by any natural query. That command extracts each
claim as a discrete fact row with its own embedding and entity linkage. It is
the difference between an archive you can search and one you merely stored.

### Step 4: The enrich pass (agent judgment)

Now the LLM half. For each imported conversation, working newest-first:

1. Rewrite the `## Summary` block above the horizontal rule. Everything below
   the rule is append-only evidence — never edit it.
2. Extract **original thinking** — your theses, frameworks, observations —
   into `originals/<slug>`. Capture **exact phrasing**; your language is the
   insight. Paraphrasing destroys the thing you were trying to keep.
3. Extract reusable mental models into `concepts/<slug>`.
4. Apply the notability gate to entities before creating `people/` or
   `companies/` pages. A one-off mention is not a person page.
5. Back-link everything (Iron Law). An unlinked mention is a broken brain.

Batch this — do not run it interactively over 500 pages.

### Step 5: Log setup completion

```bash
gbrain put-page sources/chats/README.md   # note provider, date range, filters used
```

Record the exact thresholds you used. Future you re-running this with
different filters needs to know what the first pass already covered.

## Implementation Guide

### Branch reconstruction (ChatGPT only, CRITICAL)

ChatGPT's `mapping` is a tree keyed by node id. Every edit or regenerate
forks a branch, and abandoned branches stay in the export forever. The only
correct traversal is: start at `current_node`, follow `parent` pointers to
the root, reverse.

Iterating `Object.values(mapping)` is the common bug. It interleaves
abandoned branches into the transcript, producing conversations that
contradict themselves and inflating your embedding bill with text you never
saw. The parser also carries a cycle guard, because malformed exports exist.

### Content shape handling

| Provider | Shape | Handling |
|---|---|---|
| ChatGPT | `content.parts[]` | Strings joined; non-text parts noted as `_[type omitted]_`, never fabricated |
| ChatGPT | `content.text` | Used for `code` / `execution_output` types |
| Claude | `content[]` blocks | Preferred — preserves block order |
| Claude | `text` | Fallback for older exports |
| Claude | `attachments[]` | `extracted_content` inlined under the message |

### Deduplication

Slugs are `<created-date>-<title-slug>`. Collisions (many conversations are
titled "untitled") get a 6-char SHA-1 of the stable source id appended —
not a counter, so re-runs stay idempotent.

### What to test after setup

```bash
# 1. no abandoned branches leaked into any page
grep -rl "regenerate" $BRAIN/sources/chats/ | head

# 2. a known conversation is retrievable by topic, not by title
gbrain query "the thing I concluded about <topic>"

# 3. facts landed, not just chunks
gbrain stats | grep -i fact

# 4. the skip ledger matches expectations
head -20 $BRAIN/sources/chats/chatgpt/_skipped.tsv
```

Check #2 is the one that matters. If topical queries miss,
`extract-conversation-facts` did not run or did not finish.

## Cost Estimate

| Item | Cost |
|---|---|
| Parsing | $0 — local, zero-dependency |
| Embedding | ~$0.03–0.30 per 1,000 kept conversations (provider-dependent) |
| `extract-conversation-facts` | LLM-bearing — scales with kept volume |
| Ongoing | $0 — this is a one-time backfill |

The filters are the cost lever. An unfiltered 5,000-conversation archive can
cost 10× a well-filtered 500-conversation one and retrieve worse, because
noise dilutes every neighborhood in the vector space.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `unrecognized export shape` | Vendor changed format, or wrong file | Confirm the JSON has `mapping` (ChatGPT) or `chat_messages` (Claude); force with `--provider` |
| `written: 0` | Filters too tight for a small archive | Lower `--min-messages` / `--min-user-chars`; check `_skipped.tsv` for the reason tally |
| Transcripts contradict themselves | Branch bug — not this parser | Confirm you are on this script; verify no page contains both a message and its regenerated twin |
| Topical queries miss | `extract-conversation-facts` not run | Run it; long conversations are unretrievable by chunk embedding alone |
| Import is enormous / slow | Too permissive a first pass | Re-run with tighter filters into a clean dir; `gbrain import` is idempotent by slug |
| Brain feels noisier after import | Raw pages never got promoted | The enrich pass (Step 4) is not optional — `sources/` is a staging area, not a destination |
