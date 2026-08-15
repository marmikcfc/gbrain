/**
 * Tests for the codex-cli LanguageModelV2 implementation that the
 * `codex-cli` recipe instantiates.
 *
 * Two layers, because `codex exec --json` has a fundamentally different wire
 * shape from `claude --print --output-format json` (a JSONL EVENT STREAM, not
 * one envelope):
 *
 *  1. `parseCodexJsonl` is tested as a pure function. The event stream is the
 *     surface most likely to drift between codex releases, and a pure test
 *     pins the semantics (last agent_message wins, error ITEMS are warnings,
 *     turn.failed is the failure signal, usage sums across turns) without
 *     paying for a subprocess.
 *  2. A POSIX shell stub at GBRAIN_CODEX_CLI_BIN emits scripted JSONL so the
 *     doGenerate surface is exercised end to end: argv/cwd isolation, stdin
 *     prompt delivery, env scrub, abort, and the exit-0-on-failure trap.
 *
 * No codex installation or ChatGPT subscription required.
 *
 * Env isolation: GBRAIN_CODEX_CLI_BIN is set per-test via withEnv(), NOT in
 * beforeAll. The provider reads the env var at spawn time, so withEnv's
 * save/restore in try/finally is sufficient.
 */
import { describe, test, expect, beforeAll, afterAll } from 'bun:test';
import { writeFileSync, readFileSync, chmodSync, mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { LanguageModelV2CallOptions } from '@ai-sdk/provider';
import { withEnv } from '../helpers/with-env.ts';

const stubDir = join(tmpdir(), `codex-cli-recipe-stub-${process.pid}`);
const stubBin = join(stubDir, 'codex');
const stubResponsePath = join(stubDir, 'codex_events.jsonl');
const stdinLog = join(stubDir, 'stdin.log');
const argvLog = join(stubDir, 'argv.log');
const cwdLog = join(stubDir, 'cwd.log');
const envLog = join(stubDir, 'env.log');

/** The default stub: records stdin/argv/cwd/env, then replays staged JSONL. */
function recordingStub(): string {
  return [
    '#!/bin/sh',
    `printf "%s\\n" "$@" > "${argvLog}"`,
    `pwd > "${cwdLog}"`,
    `printf "key=%s\\nbase=%s\\n" "\${OPENAI_API_KEY:-UNSET}" "\${OPENAI_BASE_URL:-UNSET}" > "${envLog}"`,
    `cat > "${stdinLog}"`,
    `cat "${stubResponsePath}"`,
  ].join('\n');
}

function installStub(body: string): void {
  writeFileSync(stubBin, body);
  chmodSync(stubBin, 0o755);
}

beforeAll(() => {
  mkdirSync(stubDir, { recursive: true });
  installStub(recordingStub());
});

afterAll(() => {
  rmSync(stubDir, { recursive: true, force: true });
});

function withStubEnv<T>(fn: () => T | Promise<T>): Promise<T> {
  return withEnv({ GBRAIN_CODEX_CLI_BIN: stubBin }, fn);
}

function stageEvents(events: Array<Record<string, unknown>>): void {
  writeFileSync(stubResponsePath, events.map(e => JSON.stringify(e)).join('\n') + '\n');
}

/** A successful single-turn stream, matching real `codex exec --json` output. */
function successEvents(text: string, usage: Record<string, number> = {}): Array<Record<string, unknown>> {
  return [
    { type: 'thread.started', thread_id: 'thread-test-1' },
    { type: 'turn.started' },
    { type: 'item.completed', item: { id: 'item_0', type: 'agent_message', text } },
    {
      type: 'turn.completed',
      usage: {
        input_tokens: 12258,
        cached_input_tokens: 9984,
        cache_write_input_tokens: 0,
        output_tokens: 5,
        reasoning_output_tokens: 0,
        ...usage,
      },
    },
  ];
}

function userMessage(text: string): LanguageModelV2CallOptions['prompt'][number] {
  return { role: 'user', content: [{ type: 'text', text }] };
}

describe('codex-cli recipe registration', () => {
  test('getRecipe returns a chat-only Recipe with the documented models', async () => {
    const { getRecipe } = await import('../../src/core/ai/recipes/index.ts');
    const recipe = getRecipe('codex-cli');
    expect(recipe).toBeDefined();
    expect(recipe!.id).toBe('codex-cli');
    expect(recipe!.implementation).toBe('codex-cli');
    // The CLI owns auth (ChatGPT session in ~/.codex/auth.json); nothing for
    // the gateway to forward, exactly like claude-cli.
    expect(recipe!.auth_env!.required).toEqual([]);
    expect(recipe!.touchpoints.chat).toBeDefined();
    expect(recipe!.touchpoints.chat!.models).toEqual([
      'gpt-5.6-luna',
      'gpt-5.6-terra',
      'gpt-5.6-sol',
    ]);
    expect(recipe!.touchpoints.chat!.supports_tools).toBe(true);
    expect(recipe!.touchpoints.chat!.supports_subagent_loop).toBe(true);
    // Codex reports cache hits but exposes no cache_control plane, so the
    // gateway must not try to place breakpoints.
    expect(recipe!.touchpoints.chat!.supports_prompt_cache).toBe(false);
    expect(recipe!.touchpoints.chat!.max_context_tokens).toBe(272000);
    // Chat-only: codex has no embedding surface at all.
    expect(recipe!.touchpoints.embedding).toBeUndefined();
    expect(recipe!.touchpoints.expansion).toBeUndefined();
  });

  test('tier aliases resolve to full slugs', async () => {
    const { getRecipe } = await import('../../src/core/ai/recipes/index.ts');
    const recipe = getRecipe('codex-cli');
    expect(recipe!.aliases!['luna']).toBe('gpt-5.6-luna');
    expect(recipe!.aliases!['terra']).toBe('gpt-5.6-terra');
    expect(recipe!.aliases!['sol']).toBe('gpt-5.6-sol');
  });

  test('gateway treats codex-cli as a subprocess provider for probe budgeting', async () => {
    // A cold `codex exec` measured ~7s — past the 5s network probe budget, so
    // without this classification `gbrain models` reports a working route as
    // unreachable.
    const { probeTimeoutMsFor } = await import('../../src/commands/models.ts');
    expect(probeTimeoutMsFor('codex-cli:gpt-5.6-luna')).toBe(
      probeTimeoutMsFor('claude-cli:claude-sonnet-4-6'),
    );
    expect(probeTimeoutMsFor('codex-cli:gpt-5.6-luna')).toBeGreaterThan(
      probeTimeoutMsFor('openai:gpt-5.2'),
    );
  });
});

describe('codex-cli parseCodexJsonl', () => {
  test('extracts the agent message, usage, and thread id', async () => {
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(successEvents('OK').map(e => JSON.stringify(e)).join('\n'));
    expect(parsed.text).toBe('OK');
    expect(parsed.sawTurnCompleted).toBe(true);
    expect(parsed.threadId).toBe('thread-test-1');
    expect(parsed.usage.input_tokens).toBe(12258);
    expect(parsed.usage.cached_input_tokens).toBe(9984);
    expect(parsed.failure).toBeUndefined();
  });

  test('error ITEMS are warnings, not turn failures', async () => {
    // Real codex behavior: a fully successful turn emitted an error item
    // saying "Skill descriptions were shortened...". Treating every error
    // item as a failure would reject every call on this machine.
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const stream = [
      { type: 'thread.started', thread_id: 't' },
      { type: 'turn.started' },
      { type: 'item.completed', item: { id: 'item_0', type: 'error', message: 'Skill descriptions were shortened' } },
      { type: 'item.completed', item: { id: 'item_1', type: 'agent_message', text: 'OK' } },
      { type: 'turn.completed', usage: { input_tokens: 1, output_tokens: 2 } },
    ];
    const parsed = parseCodexJsonl(stream.map(e => JSON.stringify(e)).join('\n'));
    expect(parsed.failure).toBeUndefined();
    expect(parsed.text).toBe('OK');
    expect(parsed.warnings).toEqual(['Skill descriptions were shortened']);
  });

  test('turn.failed is the failure signal', async () => {
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(
      [
        JSON.stringify({ type: 'turn.started' }),
        JSON.stringify({ type: 'turn.failed', error: { message: 'model not supported' } }),
      ].join('\n'),
    );
    expect(parsed.failure).toBe('model not supported');
    expect(parsed.sawTurnCompleted).toBe(false);
  });

  test('top-level error events keep the FIRST message (the original cause)', async () => {
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(
      [
        JSON.stringify({ type: 'error', message: 'status 400 invalid_request_error' }),
        JSON.stringify({ type: 'turn.failed', error: { message: 'fallout' } }),
      ].join('\n'),
    );
    expect(parsed.failure).toBe('status 400 invalid_request_error');
  });

  test('the LAST agent_message wins (preamble then answer)', async () => {
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(
      [
        JSON.stringify({ type: 'item.completed', item: { type: 'agent_message', text: 'let me look' } }),
        JSON.stringify({ type: 'item.completed', item: { type: 'agent_message', text: 'the answer is 42' } }),
        JSON.stringify({ type: 'turn.completed', usage: { input_tokens: 1, output_tokens: 1 } }),
      ].join('\n'),
    );
    expect(parsed.text).toBe('the answer is 42');
  });

  test('usage sums across multiple turn.completed events', async () => {
    // A single `codex exec` can report more than one completed turn; the
    // budget ledger must see the total, not the last one.
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(
      [
        JSON.stringify({ type: 'turn.completed', usage: { input_tokens: 100, cached_input_tokens: 40, output_tokens: 10, reasoning_output_tokens: 4 } }),
        JSON.stringify({ type: 'turn.completed', usage: { input_tokens: 200, cached_input_tokens: 60, output_tokens: 20, reasoning_output_tokens: 6 } }),
      ].join('\n'),
    );
    expect(parsed.usage.input_tokens).toBe(300);
    expect(parsed.usage.cached_input_tokens).toBe(100);
    expect(parsed.usage.output_tokens).toBe(30);
    expect(parsed.usage.reasoning_output_tokens).toBe(10);
  });

  test('non-JSON chatter around the stream is skipped, not fatal', async () => {
    // codex prints human-readable progress lines ("Reading prompt from
    // stdin..."); one more of them in a future release must not break parsing.
    const { parseCodexJsonl } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const parsed = parseCodexJsonl(
      [
        'Reading prompt from stdin...',
        JSON.stringify({ type: 'item.completed', item: { type: 'agent_message', text: 'OK' } }),
        '{ this is not json',
        JSON.stringify({ type: 'turn.completed', usage: { input_tokens: 1, output_tokens: 1 } }),
        '',
      ].join('\n'),
    );
    expect(parsed.text).toBe('OK');
    expect(parsed.sawTurnCompleted).toBe(true);
  });
});

describe('codex-cli buildCodexArgs', () => {
  test('carries the mandatory + isolation flags', async () => {
    const { buildCodexArgs } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const args = buildCodexArgs('gpt-5.6-luna', '/tmp/codex-cwd');
    expect(args[0]).toBe('exec');
    // Without --skip-git-repo-check codex refuses ("Not inside a trusted
    // directory") before reaching the model — the tmpdir cwd is never a repo.
    expect(args).toContain('--skip-git-repo-check');
    expect(args).toContain('--json');
    expect(args).toContain('-m');
    expect(args).toContain('gpt-5.6-luna');
    expect(args).toContain('-C');
    expect(args).toContain('/tmp/codex-cwd');
    expect(args).toContain('--sandbox');
    expect(args).toContain('read-only');
    expect(args).toContain('--ephemeral');
    expect(args).toContain('--ignore-rules');
    // The load-bearing one: skips ~/.codex/config.toml, so no MCP servers
    // boot (gbrain's own MCP among them → recursion + PGLite lock contention).
    expect(args).toContain('--ignore-user-config');
    // The prompt is never argv — a rendered transcript can exceed ARG_MAX.
    expect(args.some(a => a.startsWith('#') || a.includes('Conversation'))).toBe(false);
  });

  test('GBRAIN_CODEX_CLI_ALLOW_USER_CONFIG=1 opts back into the user config', async () => {
    const { buildCodexArgs } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    await withEnv({ GBRAIN_CODEX_CLI_ALLOW_USER_CONFIG: '1' }, () => {
      expect(buildCodexArgs('gpt-5.6-luna', '/tmp/x')).not.toContain('--ignore-user-config');
    });
    // Restored: the safe default comes back outside the escape hatch.
    expect(buildCodexArgs('gpt-5.6-luna', '/tmp/x')).toContain('--ignore-user-config');
  });
});

describe('codex-cli LanguageModel — text-only round trip', () => {
  test('returns text plus real cached/reasoning usage and provider metadata', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('hello world', { output_tokens: 34, reasoning_output_tokens: 12, cache_write_input_tokens: 7 }));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({ prompt: [userMessage('hi')] } as LanguageModelV2CallOptions);

      expect(result.finishReason).toBe('stop');
      expect(result.content).toEqual([{ type: 'text', text: 'hello world' }]);
      expect(result.usage.inputTokens).toBe(12258);
      expect(result.usage.outputTokens).toBe(34);
      expect(result.usage.totalTokens).toBe(12292);
      // The genuine advantage over claude-cli: a real cache-read number that
      // gateway.chat maps onto `cache_read_tokens`.
      expect(result.usage.cachedInputTokens).toBe(9984);
      expect(result.usage.reasoningTokens).toBe(12);
      expect(result.providerMetadata!['codex-cli']).toMatchObject({
        cacheWriteInputTokens: 7,
        threadId: 'thread-test-1',
      });
    });
  });

  test('strips provider prefixes from the model id', async () => {
    const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    expect(new CodexCliLanguageModel('codex-cli:gpt-5.6-luna').modelId).toBe('gpt-5.6-luna');
  });

  test('warns for settings codex exec cannot honor instead of dropping them silently', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel, __resetCodexWarnOnceForTests } =
        await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      __resetCodexWarnOnceForTests();
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({
        prompt: [userMessage('hi')],
        temperature: 0,
        maxOutputTokens: 128,
      } as LanguageModelV2CallOptions);

      const settings = result.warnings
        .filter(w => w.type === 'unsupported-setting')
        .map(w => String((w as { setting: unknown }).setting));
      expect(settings).toContain('temperature');
      expect(settings).toContain('maxOutputTokens');
    });
  });

  test('surfaces codex error ITEMS as warnings on an otherwise successful call', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents([
        { type: 'thread.started', thread_id: 't' },
        { type: 'turn.started' },
        { type: 'item.completed', item: { type: 'error', message: 'Skill descriptions were shortened' } },
        { type: 'item.completed', item: { type: 'agent_message', text: 'OK' } },
        { type: 'turn.completed', usage: { input_tokens: 5, output_tokens: 1 } },
      ]);
      const { CodexCliLanguageModel, __resetCodexWarnOnceForTests } =
        await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      __resetCodexWarnOnceForTests();
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({ prompt: [userMessage('hi')] } as LanguageModelV2CallOptions);
      expect(result.finishReason).toBe('stop');
      expect(result.warnings).toContainEqual({ type: 'other', message: 'Skill descriptions were shortened' });
    });
  });

  test('each distinct warning is returned once per process, not once per call', async () => {
    // The AI SDK LOGS every returned warning. gateway.chat always sets
    // maxTokens and codex emits its skills advisory on nearly every turn, so
    // returning them unconditionally printed the same two lines on every call
    // of a sync — thousands of copies of one sentence.
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents([
        { type: 'turn.started' },
        { type: 'item.completed', item: { type: 'error', message: 'Skill descriptions were shortened' } },
        { type: 'item.completed', item: { type: 'agent_message', text: 'OK' } },
        { type: 'turn.completed', usage: { input_tokens: 5, output_tokens: 1 } },
      ]);
      const { CodexCliLanguageModel, __resetCodexWarnOnceForTests } =
        await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      __resetCodexWarnOnceForTests();
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const opts = { prompt: [userMessage('hi')], maxOutputTokens: 64 } as LanguageModelV2CallOptions;

      const first = await model.doGenerate(opts);
      const second = await model.doGenerate(opts);
      expect(first.warnings.length).toBe(2);
      expect(second.warnings).toEqual([]);
    });
  });
});

describe('codex-cli LanguageModel — tool use', () => {
  test('parses a <use_tools> block into tool-call content', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(
        successEvents(
          [
            'I will look up the pattern first.',
            '<use_tools>',
            '[{"id": "call_01ABC", "name": "search", "input": {"query": "n+1 query"}}]',
            '</use_tools>',
          ].join('\n'),
        ),
      );
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({
        prompt: [userMessage('find n+1 queries')],
        tools: [
          {
            type: 'function',
            name: 'search',
            description: 'Search the brain',
            inputSchema: { type: 'object', properties: { query: { type: 'string' } } },
          },
        ],
      } as LanguageModelV2CallOptions);

      expect(result.finishReason).toBe('tool-calls');
      expect(result.content[0]).toMatchObject({ type: 'text', text: 'I will look up the pattern first.' });
      expect(result.content[1]).toMatchObject({
        type: 'tool-call',
        toolCallId: 'call_01ABC',
        toolName: 'search',
        input: '{"query":"n+1 query"}',
      });
    });
  });

  test('parses multiple parallel tool calls in a single block', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(
        successEvents(
          [
            '<use_tools>',
            '[',
            '  {"id": "call_A", "name": "search", "input": {"query": "foo"}},',
            '  {"id": "call_B", "name": "get_page", "input": {"slug": "areas/x"}}',
            ']',
            '</use_tools>',
          ].join('\n'),
        ),
      );
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({
        prompt: [userMessage('multi')],
        tools: [
          { type: 'function', name: 'search', description: 's', inputSchema: { type: 'object', properties: {} } },
          { type: 'function', name: 'get_page', description: 'g', inputSchema: { type: 'object', properties: {} } },
        ],
      } as LanguageModelV2CallOptions);

      const calls = result.content.filter(c => c.type === 'tool-call');
      expect(calls.map(c => (c as { toolName: string }).toolName)).toEqual(['search', 'get_page']);
      expect(result.finishReason).toBe('tool-calls');
    });
  });

  test('synthesizes a codex-tagged id when the model omits it', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('<use_tools>\n[{"name": "search", "input": {"q": "x"}}]\n</use_tools>'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({
        prompt: [userMessage('no id')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      const call = result.content.find(c => c.type === 'tool-call') as { toolCallId: string } | undefined;
      // Prefixed per-CLI so ids stay traceable in a mixed-provider transcript.
      expect(call!.toolCallId).toMatch(/^toolu_codex_cli_/);
    });
  });

  test('prompt tells codex its own shell/web tools are unavailable for the turn', async () => {
    // Regression guard for a measured failure: with the shared <use_tools>
    // instructions alone, gpt-5.6-luna sometimes ignored the protocol and went
    // hunting for the tools with its own shell (32k input tokens, then "I
    // couldn't access the gbrain search tool"). `codex exec` has no way to
    // strip its native tools, so the prompt has to do it.
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await model.doGenerate({
        prompt: [userMessage('find something')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);

      const stdin = readFileSync(stdinLog, 'utf8');
      expect(stdin).toContain('shell, file-editing, and web tools are NOT available');
      expect(stdin).toContain('## Tool Use Protocol');
    });
  });

  test('the tools-unavailable guard is absent on a plain chat call', async () => {
    // No tools registered → no protocol noise, and no instruction telling the
    // model it has no tools when nothing asked it to use any.
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await model.doGenerate({ prompt: [userMessage('just chat')] } as LanguageModelV2CallOptions);
      expect(readFileSync(stdinLog, 'utf8')).not.toContain('NOT available');
    });
  });

  test('falls back to text when the block is malformed', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('<use_tools>\nnot valid json\n</use_tools>'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const result = await model.doGenerate({
        prompt: [userMessage('malformed')],
        tools: [{ type: 'function', name: 'search', description: '', inputSchema: { type: 'object', properties: {} } }],
      } as LanguageModelV2CallOptions);
      expect(result.content.filter(c => c.type === 'tool-call')).toHaveLength(0);
      expect(result.finishReason).toBe('stop');
    });
  });
});

describe('codex-cli LanguageModel — context isolation', () => {
  test('spawns in the dedicated tmpdir and passes it via -C with the isolation flags', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await model.doGenerate({
        prompt: [{ role: 'system', content: 'You are a gbrain subagent.' }, userMessage('hi')],
      } as LanguageModelV2CallOptions);

      const argv = readFileSync(argvLog, 'utf8').split('\n').filter(Boolean);
      const cwd = readFileSync(cwdLog, 'utf8').trim();
      expect(argv).toContain('--skip-git-repo-check');
      expect(argv).toContain('--ignore-user-config');
      expect(argv).toContain('--ephemeral');
      expect(argv).toContain('read-only');
      expect(cwd).toMatch(/gbrain-codex-cli-cwd-\d+$/);
      // -C points at the same empty dir the process was spawned in, so
      // codex's AGENTS.md discovery has nothing to find either way.
      expect(argv[argv.indexOf('-C') + 1]).toMatch(/gbrain-codex-cli-cwd-\d+$/);
    });
  });

  test('delivers the prompt over stdin with system text under an Instructions header', async () => {
    // codex exec has NO --system-prompt equivalent; system messages have to
    // ride in the prompt body. stdin (not argv) because a rendered transcript
    // can exceed ARG_MAX.
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await model.doGenerate({
        prompt: [{ role: 'system', content: 'You are a gbrain subagent.' }, userMessage('hi there')],
      } as LanguageModelV2CallOptions);

      const stdin = readFileSync(stdinLog, 'utf8');
      expect(stdin).toContain('# Instructions');
      expect(stdin).toContain('You are a gbrain subagent.');
      expect(stdin).toContain('# Conversation');
      expect(stdin).toContain('User: hi there');
    });
  });

  test('omits the Instructions header when there is no system text or tool protocol', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents(successEvents('ok'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await model.doGenerate({ prompt: [userMessage('plain question')] } as LanguageModelV2CallOptions);
      const stdin = readFileSync(stdinLog, 'utf8');
      expect(stdin).not.toContain('# Instructions');
      expect(stdin).toBe('User: plain question');
    });
  });

  test('scrubs OPENAI_* credentials from the child env (subscription-only billing)', async () => {
    await withStubEnv(async () => {
      await withEnv(
        { OPENAI_API_KEY: 'sk-should-never-leak', OPENAI_BASE_URL: 'https://proxy.should.never.leak' },
        async () => {
          installStub(recordingStub());
          stageEvents(successEvents('ok'));
          const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
          const model = new CodexCliLanguageModel('gpt-5.6-luna');
          await model.doGenerate({ prompt: [userMessage('hi')] } as LanguageModelV2CallOptions);
          const seen = readFileSync(envLog, 'utf8');
          expect(seen).toContain('key=UNSET');
          expect(seen).toContain('base=UNSET');
        },
      );
    });
  });
});

describe('codex-cli LanguageModel — failure modes', () => {
  test('rejects on turn.failed EVEN THOUGH codex exits 0', async () => {
    // The trap this adapter exists to avoid: codex-cli 0.146.1 exits 0 after
    // a failed turn (measured with an unsupported model id). An exit-code-only
    // check would return empty text and look like a successful empty answer.
    await withStubEnv(async () => {
      installStub(['#!/bin/sh', 'cat > /dev/null', `cat "${stubResponsePath}"`, 'exit 0'].join('\n'));
      stageEvents([
        { type: 'thread.started', thread_id: 't' },
        { type: 'turn.started' },
        { type: 'error', message: 'The model is not supported when using Codex with a ChatGPT account.' },
        { type: 'turn.failed', error: { message: 'The model is not supported when using Codex with a ChatGPT account.' } },
      ]);
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await expect(
        model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
      ).rejects.toThrow(/codex-cli turn failed: The model is not supported/);
    });
  });

  test('rejects a truncated stream that never reached turn.completed', async () => {
    await withStubEnv(async () => {
      installStub(recordingStub());
      stageEvents([
        { type: 'thread.started', thread_id: 't' },
        { type: 'turn.started' },
        { type: 'item.completed', item: { type: 'agent_message', text: 'half an ans' } },
      ]);
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await expect(
        model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
      ).rejects.toThrow(/without turn\.completed/);
    });
  });

  test('rejects on a non-zero exit (codex never reached the event stream)', async () => {
    await withStubEnv(async () => {
      installStub(['#!/bin/sh', 'cat > /dev/null', 'echo "error: unexpected argument" >&2', 'exit 2'].join('\n'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await expect(
        model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
      ).rejects.toThrow(/codex-cli exited 2/);
    });
  });

  test('SIGTERMs the child on AbortSignal', async () => {
    await withStubEnv(async () => {
      installStub(['#!/bin/sh', 'cat > /dev/null', 'sleep 30', 'echo "{}"'].join('\n'));
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      const ac = new AbortController();
      const promise = model.doGenerate({
        prompt: [userMessage('slow')],
        abortSignal: ac.signal,
      } as LanguageModelV2CallOptions);
      setTimeout(() => ac.abort(), 30);
      await expect(promise).rejects.toThrow(/aborted/);
    });
  });

  test('rejects cleanly when the codex binary is missing (no worker crash)', async () => {
    // Must surface via the spawn 'error' handler; the stdin EPIPE handler
    // swallows the pipe failure so it never becomes an unhandled rejection.
    await withEnv({ GBRAIN_CODEX_CLI_BIN: join(stubDir, 'nonexistent-codex') }, async () => {
      const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
      const model = new CodexCliLanguageModel('gpt-5.6-luna');
      await expect(
        model.doGenerate({ prompt: [userMessage('x')] } as LanguageModelV2CallOptions),
      ).rejects.toThrow(/codex-cli spawn failed/);
    });
  });

  test('doStream throws not-supported', async () => {
    const { CodexCliLanguageModel } = await import('../../src/core/ai/providers/codex-cli-language-model.ts');
    const model = new CodexCliLanguageModel('gpt-5.6-luna');
    await expect(model.doStream()).rejects.toThrow(/does not support streaming/);
  });
});
