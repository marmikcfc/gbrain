/**
 * ai-sdk LanguageModelV2 implementation that dispatches via the `codex exec`
 * CLI subprocess. Used by the `codex-cli` recipe to route gateway.toolLoop /
 * gateway.chat calls through the Codex CLI's ChatGPT-subscription session
 * instead of the OpenAI SDK + OPENAI_API_KEY (per-token billing).
 *
 * Sibling of `claude-cli-language-model.ts`; the same per-call routing
 * contract holds. The gateway resolves `codex-cli:gpt-5.6-luna` to this
 * recipe, instantiates one object per modelId, and calls doGenerate. A
 * sibling job with `anthropic:claude-sonnet-4-6` keeps routing through the
 * Anthropic SDK in the same worker — no env-var switch, no global state.
 *
 * Tool use: same system-prompt-instructed `<use_tools>` JSON emission the
 * claude-cli adapter uses (shared code in cli-subprocess-protocol.ts), plus a
 * codex-specific guard clause. This was NOT assumed — it was measured, and
 * the first measurement was a FAILURE: with the shared instructions alone,
 * gpt-5.6-luna intermittently ignored the protocol and went looking for the
 * tools with its own shell (32k input tokens, then "I couldn't access the
 * gbrain search tool in this session"). Telling it that its native tools are
 * unavailable for the turn fixed it: 6/6 emissions afterwards, single and
 * parallel, plus a full two-turn call → result → answer loop.
 *
 * WIRE SHAPE — the big divergence from claude-cli:
 *   `claude --print --output-format json` prints ONE envelope with
 *   `is_error` and exits non-zero on failure. `codex exec --json` prints a
 *   JSONL EVENT STREAM and — measured on codex-cli 0.146.1 — EXITS 0 EVEN
 *   WHEN THE TURN FAILED (a bad model id yields `{"type":"error",...}` +
 *   `{"type":"turn.failed",...}` and status 0). So exit code is NOT the
 *   success signal here: `turn.completed` is, and `turn.failed` / a
 *   top-level `error` event is the failure signal. Trusting the exit code
 *   the way the claude-cli adapter does would silently return empty text
 *   for every API-level failure.
 *
 *   `item.completed` items of `type:"error"` are WARNINGS, not failures —
 *   codex emits e.g. "Skill descriptions were shortened to fit the 2%
 *   skills context budget" on a fully successful turn. They are surfaced
 *   through the ai-sdk `warnings` channel instead of being swallowed or
 *   mistaken for a failed turn.
 *
 * Usage accounting is strictly better than claude-cli's: `turn.completed`
 * carries `cached_input_tokens` (a real, material number — 8,960–9,984 of
 * ~15.8k input tokens across probe runs, OpenAI's automatic prefix cache)
 * plus `cache_write_input_tokens` and `reasoning_output_tokens`. Those are
 * threaded into LanguageModelV2Usage so gateway.chat records
 * `cache_read_tokens` from real data rather than a worst-case estimate.
 *
 * Context isolation (same problem claude-cli has, different levers):
 *   The subprocess runs in a dedicated tmpdir (both as spawn cwd and via
 *   `-C`) so codex's AGENTS.md / project-file auto-discovery finds nothing.
 *   `--ignore-user-config` is the load-bearing flag: without it codex boots
 *   every MCP server in ~/.codex/config.toml on every call — a measured
 *   failure (a Supabase OAuth-refresh error appeared on the baseline probe
 *   and vanished with the flag), and the exact recursion / PGLite
 *   single-writer contention hazard `--strict-mcp-config` exists to prevent
 *   on the claude-cli side, since gbrain's own MCP server would be among
 *   them. `--ignore-rules` skips user/project execpolicy `.rules`;
 *   `--ephemeral` keeps a library dispatching thousands of calls from
 *   filling ~/.codex/sessions with rollout files.
 *
 *   Measured cost of the remaining context: 16,322 input tokens for a
 *   one-word reply unisolated, 15,762 isolated. The residue is codex's own
 *   base instructions plus built-in skills, which live under $CODEX_HOME and
 *   load regardless of config. Pointing CODEX_HOME at a scratch dir with a
 *   symlinked auth.json was measured too: it only reached 14,803 tokens (a
 *   ~6% win) while materializing a fresh sqlite/model-cache home per call and
 *   risking the CLI's own token-refresh writes landing in a throwaway
 *   directory — the user's real auth. Not worth it; operators who want it can
 *   still export CODEX_HOME themselves, since the child inherits the env.
 *
 * WHAT COULD NOT BE REPLICATED from the claude-cli adapter (no equivalent
 * flag exists on codex exec 0.146.1 — do not read the absence as an
 * oversight):
 *   - `--system-prompt` (replace the system prompt). Codex's base
 *     instructions are fixed; system messages are rendered into the prompt
 *     body under an explicit header instead. A prompt-body instruction is
 *     weaker than a real system prompt, and it competes with codex's own
 *     agent persona.
 *   - `--tools ""` (disable built-in tools). `codex exec` is an agent with
 *     shell/apply_patch/web-search tools and there is no way to strip them.
 *     `--sandbox read-only` is the closest available bound: the model can
 *     still decide to run a command, but it cannot write or reach the
 *     network through the shell. A turn that goes exploring costs extra
 *     tokens and wall-clock; the empty tmpdir workspace keeps that cheap.
 *   - `--disable-slash-commands`. Skills load from $CODEX_HOME/skills with
 *     no flag to suppress them (measured: `--ignore-user-config` does not).
 *
 * doStream is not implemented; the model declares no streaming, matching
 * claude-cli. Callers (gateway.toolLoop primarily) use doGenerate.
 */
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  SharedV2ProviderMetadata,
} from '@ai-sdk/provider';
// Prompt rendering + the <use_tools> emission protocol are shared with the
// claude-cli adapter; see cli-subprocess-protocol.ts for why they live there.
import {
  buildToolUseInstructions,
  extractToolCalls,
  normalizeModel,
  renderPrompt,
} from './cli-subprocess-protocol.ts';

function codexBin(): string {
  return process.env.GBRAIN_CODEX_CLI_BIN ?? 'codex';
}

const CODEX_CWD = join(tmpdir(), `gbrain-codex-cli-cwd-${process.pid}`);
let cwdEnsured = false;
function ensureCleanCwd(): string {
  if (!cwdEnsured) {
    mkdirSync(CODEX_CWD, { recursive: true });
    cwdEnsured = true;
  }
  return CODEX_CWD;
}

/** Usage block on a `turn.completed` event. All fields are token counts. */
interface CodexUsage {
  input_tokens?: number;
  /** Subset of input_tokens served from OpenAI's automatic prefix cache. */
  cached_input_tokens?: number;
  cache_write_input_tokens?: number;
  output_tokens?: number;
  /** Subset of output_tokens spent on reasoning. */
  reasoning_output_tokens?: number;
}

/** Everything the adapter needs out of one `codex exec --json` event stream. */
export interface CodexRunResult {
  /**
   * The agent's reply. When codex emits several `agent_message` items
   * (preamble + answer), the LAST one wins — the same choice codex itself
   * makes for `-o/--output-last-message`, and the one that keeps a preamble
   * out of the model's "final answer".
   */
  text: string;
  /** `item.completed` error items: advisory, not turn failures. */
  warnings: string[];
  /** Summed across every `turn.completed` in the stream. */
  usage: CodexUsage;
  threadId?: string;
  /** Set when a `turn.failed` or top-level `error` event was seen. */
  failure?: string;
  /** A turn is only successful if codex said so explicitly. */
  sawTurnCompleted: boolean;
}

/**
 * Parse `codex exec --json` JSONL into a CodexRunResult.
 *
 * Exported for direct unit testing: the event stream is the fragile surface
 * of this adapter (codex adds event types between releases), so it is tested
 * as a pure function rather than only through a spawned stub.
 *
 * Unparseable lines are skipped rather than fatal. codex prints progress
 * chatter ("Reading prompt from stdin...") around the JSONL and a future
 * release adding one more human-readable line must not turn every call into
 * a hard error; a stream with no `turn.completed` is caught by the caller.
 */
export function parseCodexJsonl(stdout: string): CodexRunResult {
  const result: CodexRunResult = {
    text: '',
    warnings: [],
    usage: {},
    sawTurnCompleted: false,
  };

  const addUsage = (u: CodexUsage | undefined): void => {
    if (!u) return;
    // Summed, not overwritten: a single `codex exec` can report more than one
    // `turn.completed` (a turn that ran tools then continued). Overwriting
    // would undercount the budget ledger on exactly the expensive calls.
    for (const key of [
      'input_tokens',
      'cached_input_tokens',
      'cache_write_input_tokens',
      'output_tokens',
      'reasoning_output_tokens',
    ] as const) {
      const v = u[key];
      if (typeof v === 'number' && Number.isFinite(v)) {
        result.usage[key] = (result.usage[key] ?? 0) + v;
      }
    }
  };

  for (const line of stdout.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    let ev: Record<string, unknown>;
    try {
      ev = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      continue;
    }
    switch (ev.type) {
      case 'thread.started': {
        if (typeof ev.thread_id === 'string') result.threadId = ev.thread_id;
        break;
      }
      case 'item.completed': {
        const item = ev.item as Record<string, unknown> | undefined;
        if (!item) break;
        if (item.type === 'agent_message' && typeof item.text === 'string') {
          result.text = item.text;
        } else if (item.type === 'error' && typeof item.message === 'string') {
          // Advisory only. Measured: codex emits an error ITEM about skill
          // descriptions being truncated on turns that succeed normally.
          result.warnings.push(item.message);
        }
        break;
      }
      case 'turn.completed': {
        result.sawTurnCompleted = true;
        addUsage(ev.usage as CodexUsage | undefined);
        break;
      }
      // Both failure events keep the FIRST message seen. codex emits the
      // top-level `error` first and then `turn.failed`; the first carries the
      // original cause and later ones are fallout from it.
      case 'turn.failed': {
        if (!result.failure) {
          const err = ev.error as Record<string, unknown> | undefined;
          result.failure = err && typeof err.message === 'string' ? err.message : 'turn failed';
        }
        break;
      }
      case 'error': {
        if (!result.failure) {
          result.failure = typeof ev.message === 'string' ? ev.message : 'codex reported an error';
        }
        break;
      }
      default:
        break;
    }
  }

  return result;
}

/** Argv for one `codex exec` dispatch. Exported so tests can pin the flags. */
export function buildCodexArgs(model: string, cwd: string): string[] {
  const args = [
    'exec',
    '-m', model,
    '--json',
    // MANDATORY: without it codex refuses with "Not inside a trusted
    // directory" and exits before reaching the model — and the tmpdir
    // workspace this adapter spawns in is never a git repo by construction.
    '--skip-git-repo-check',
    // Explicit workspace root. Belt-and-braces with the spawn cwd: if a
    // caller ever spawns this adapter with an inherited cwd, `-C` still
    // pins codex's project-file discovery to the empty tmpdir.
    '-C', cwd,
    // Read-only sandbox: `codex exec` is an agent that can run shell
    // commands and there is no flag to remove its tools. This is the
    // strongest available bound on a chat-shaped call.
    '--sandbox', 'read-only',
    // Do not persist rollout/session files; this adapter can be called
    // thousands of times per sync.
    '--ephemeral',
    // Do not load user execpolicy `.rules`.
    '--ignore-rules',
  ];
  // Skipping ~/.codex/config.toml is what keeps every call from booting the
  // user's MCP servers (gbrain's own MCP among them → recursion + PGLite
  // single-writer contention) and from inheriting hooks/notify commands.
  // Auth is unaffected: codex still reads $CODEX_HOME/auth.json. The escape
  // hatch exists for deployments whose config.toml carries a required custom
  // model provider — it re-enables the MCP boot, so it is opt-in only.
  if (process.env.GBRAIN_CODEX_CLI_ALLOW_USER_CONFIG !== '1') {
    args.push('--ignore-user-config');
  }
  return args;
}

/**
 * Spawn `codex exec` and return the parsed event stream. Aborts propagate to
 * SIGTERM on the child.
 */
function runCodex(prompt: string, model: string, signal?: AbortSignal): Promise<CodexRunResult> {
  return new Promise((resolve, reject) => {
    const cwd = ensureCleanCwd();
    const args = buildCodexArgs(model, cwd);
    // Prompt goes over stdin, never argv. Rendered transcripts in a subagent
    // loop routinely exceed ARG_MAX (~1MB on macOS); `codex exec` with no
    // prompt argument reads instructions from stdin to EOF, so this path has
    // no size ceiling.
    const env = { ...process.env };
    // Env scrub, same intent as the claude-cli adapter's ANTHROPIC_* scrub:
    // this recipe exists so calls bill to a ChatGPT subscription. Measured on
    // 0.146.1 that a stored chatgpt auth.json already wins over an env key,
    // so this is defense against a future precedence change, not a fix for
    // today's behavior.
    delete env.OPENAI_API_KEY;
    delete env.OPENAI_BASE_URL;
    delete env.OPENAI_API_BASE;

    const child = spawn(codexBin(), args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd,
      env,
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', chunk => { stdout += String(chunk); });
    child.stderr.on('data', chunk => { stderr += String(chunk); });

    const onAbort = () => {
      child.kill('SIGTERM');
      reject(new Error('codex-cli adapter aborted'));
    };
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    child.on('error', err => {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`codex-cli spawn failed: ${err instanceof Error ? err.message : String(err)}`));
    });

    child.on('close', code => {
      if (signal) signal.removeEventListener('abort', onAbort);
      const parsed = parseCodexJsonl(stdout);
      // Failure detection is event-driven, NOT exit-code-driven: codex 0.146.1
      // exits 0 on a failed turn (verified with an unsupported model id). A
      // non-zero exit is still fatal — it means codex never got as far as the
      // event stream (bad flag, missing auth).
      if (parsed.failure) {
        reject(new Error(`codex-cli turn failed: ${parsed.failure}`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`codex-cli exited ${code}: ${stderr.trim() || stdout.trim().slice(0, 500)}`));
        return;
      }
      if (!parsed.sawTurnCompleted) {
        // No turn.completed and no explicit failure: a truncated/killed stream.
        // Returning parsed.text here would hand the caller a half-finished
        // answer that looks complete.
        reject(new Error(
          `codex-cli stream ended without turn.completed\n--- raw ---\n${stdout.slice(0, 500)}`,
        ));
        return;
      }
      resolve(parsed);
    });

    // stdin error handler: if the binary does not exist (ENOENT) or the child
    // dies before draining stdin, write/end can emit an unhandled 'error'
    // (EPIPE) that would crash the worker. The spawn-level 'error' / non-zero
    // 'close' handlers above already surface the real failure.
    child.stdin.on('error', () => { /* surfaced via child 'error'/'close' */ });
    try {
      child.stdin.write(prompt);
      child.stdin.end();
    } catch (e) {
      if (signal) signal.removeEventListener('abort', onAbort);
      reject(new Error(`codex-cli stdin write failed (is the codex binary installed?): ${e instanceof Error ? e.message : String(e)}`));
    }
  });
}

/**
 * Settings the ai-sdk exposes that `codex exec` has no way to honor. Reported
 * as warnings rather than silently dropped, so a caller that sets
 * `temperature: 0` for determinism learns it did nothing instead of trusting
 * a knob that was never wired.
 */
const UNSUPPORTED_SETTINGS = [
  'temperature',
  'topP',
  'topK',
  'maxOutputTokens',
  'stopSequences',
  'seed',
  'presencePenalty',
  'frequencyPenalty',
  'responseFormat',
] as const;

/**
 * Warn-once ledger, keyed by warning text.
 *
 * The AI SDK LOGS every warning a model returns, on every call. gateway.chat
 * always passes `maxTokens`, and codex emits its "Skill descriptions were
 * shortened" advisory on nearly every turn — so returning these unconditionally
 * printed two identical stderr lines per call, which during a sync means
 * thousands of copies of the same sentence. The information is in the first
 * occurrence; repeats are noise that trains users to ignore the channel.
 */
const _warnedOnce = new Set<string>();
function firstTimeOnly<T extends { type: string }>(warnings: T[], keyOf: (w: T) => string): T[] {
  return warnings.filter(w => {
    const key = keyOf(w);
    if (_warnedOnce.has(key)) return false;
    _warnedOnce.add(key);
    return true;
  });
}

/** Test seam: clear the warn-once ledger so warning assertions are order-free. */
export function __resetCodexWarnOnceForTests(): void {
  _warnedOnce.clear();
}

export class CodexCliLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly provider = 'codex-cli';
  readonly modelId: string;
  readonly supportedUrls = {};

  constructor(modelId: string) {
    this.modelId = normalizeModel(modelId);
  }

  async doGenerate(options: LanguageModelV2CallOptions): Promise<{
    content: LanguageModelV2Content[];
    finishReason: 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';
    usage: {
      inputTokens: number | undefined;
      outputTokens: number | undefined;
      totalTokens: number | undefined;
      reasoningTokens?: number;
      cachedInputTokens?: number;
    };
    providerMetadata?: SharedV2ProviderMetadata;
    warnings: LanguageModelV2CallWarning[];
  }> {
    const { systemText, userPrompt } = renderPrompt(options.prompt);
    const toolInstructions = buildToolUseInstructions(options.tools);
    const preamble = [
      systemText,
      // Codex-only guard, measured into existence: `codex exec` is an agent
      // with its own shell/apply_patch/web tools and no flag disables them
      // (unlike claude-cli's `--tools ""`). Given only the shared <use_tools>
      // instructions, gpt-5.6-luna sometimes ignored the protocol and went
      // exploring with the shell instead — burning 32k input tokens and
      // answering "I couldn't access the gbrain search tool in this session"
      // rather than emitting a tool call. Naming its native tools as
      // unavailable for this turn is what makes the protocol win.
      toolInstructions
        ? 'You are being used as a text-only reasoning endpoint. Your own shell, file-editing, ' +
          'and web tools are NOT available for this turn and must not be used or referred to — ' +
          'do not try to find or run the tools below yourself. The tool-use protocol described ' +
          'next is the ONLY way for you to take an action; if you need a tool, emit the block.'
        : '',
      toolInstructions,
    ].filter(s => s.length > 0).join('\n');

    // No --system-prompt on codex exec: the system text is folded into the
    // prompt body under an explicit header so the model can still tell
    // operator instructions from conversation. Header is only emitted when
    // there is system text, so a plain chat call sends a clean prompt.
    const prompt = preamble
      ? `# Instructions\n\n${preamble}\n\n# Conversation\n\n${userPrompt}`
      : userPrompt;

    const result = await runCodex(prompt, this.modelId, options.abortSignal);
    const { toolCalls, beforeText, afterText } = extractToolCalls(result.text, 'toolu_codex_cli_');

    const content: LanguageModelV2Content[] = [];
    if (beforeText) content.push({ type: 'text', text: beforeText });
    for (const call of toolCalls) {
      content.push({
        type: 'tool-call',
        toolCallId: call.id,
        toolName: call.name,
        input: call.input,
      });
    }
    if (afterText) content.push({ type: 'text', text: afterText });
    if (content.length === 0) {
      // Empty response — still hand the caller a well-formed content array.
      content.push({ type: 'text', text: result.text ?? '' });
    }

    const warnings = firstTimeOnly<LanguageModelV2CallWarning>(
      [
        ...UNSUPPORTED_SETTINGS.filter(s => options[s] !== undefined).map(
          (setting): LanguageModelV2CallWarning => ({
            type: 'unsupported-setting',
            setting,
            details: 'codex exec exposes no flag for this setting',
          }),
        ),
        ...result.warnings.map((message): LanguageModelV2CallWarning => ({ type: 'other', message })),
      ],
      w => (w.type === 'unsupported-setting' ? `setting:${String(w.setting)}` : `msg:${(w as { message: string }).message}`),
    );

    const inputTokens = result.usage.input_tokens;
    const outputTokens = result.usage.output_tokens;

    return {
      content,
      finishReason: toolCalls.length > 0 ? 'tool-calls' : 'stop',
      usage: {
        // `input_tokens` is INCLUSIVE of `cached_input_tokens` (codex derives
        // its non-cached figure by subtraction), which matches the ai-sdk
        // convention where cachedInputTokens is a subset of inputTokens — so
        // both pass through unmodified. gateway.chat reads cachedInputTokens
        // into `cache_read_tokens`, giving the budget ledger real cache-hit
        // data instead of claude-cli's worst-case zero.
        inputTokens,
        outputTokens,
        totalTokens:
          inputTokens !== undefined && outputTokens !== undefined
            ? inputTokens + outputTokens
            : undefined,
        ...(result.usage.reasoning_output_tokens !== undefined
          ? { reasoningTokens: result.usage.reasoning_output_tokens }
          : {}),
        ...(result.usage.cached_input_tokens !== undefined
          ? { cachedInputTokens: result.usage.cached_input_tokens }
          : {}),
      },
      // cache_write is not part of LanguageModelV2Usage; it rides in provider
      // metadata so an operator debugging cache behavior can still see it.
      // The thread id makes a call traceable to a codex session on disk when
      // GBRAIN_CODEX_CLI_ALLOW_USER_CONFIG / non-ephemeral runs keep them.
      providerMetadata: {
        'codex-cli': {
          ...(result.usage.cache_write_input_tokens !== undefined
            ? { cacheWriteInputTokens: result.usage.cache_write_input_tokens }
            : {}),
          ...(result.threadId ? { threadId: result.threadId } : {}),
        },
      },
      warnings,
    };
  }

  async doStream(): Promise<never> {
    throw new Error(
      'codex-cli LanguageModel does not support streaming. Use doGenerate or set ' +
      'the model on a non-streaming chat surface (gateway.toolLoop is non-streaming).',
    );
  }
}
