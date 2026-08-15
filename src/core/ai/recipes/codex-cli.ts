import type { Recipe } from '../types.ts';

/**
 * GPT via the local `codex` CLI binary (`codex exec`), using its stored
 * ChatGPT-subscription session. No OPENAI_API_KEY needed — the CLI manages
 * its own auth state (`~/.codex/auth.json`, auth_mode "chatgpt") and the
 * gateway dispatches by subprocess.
 *
 * Same shape as `claude-cli`, one provider over: a ChatGPT Plus/Pro
 * subscriber who wants Minions subagent dispatch on their subscription
 * instead of per-token API charges picks per call —
 * `openai:gpt-5.2` (API key, per-token billing) vs
 * `codex-cli:gpt-5.6-luna` (subscription, no API key).
 *
 * Chat-only. Codex has no embedding model at all — not "we didn't wire it":
 * `codex exec` is a coding agent surface, there is no embedding subcommand.
 * Pair with openai/google/voyage for embeddings.
 *
 * Auth: `auth_env.required: []` because the CLI handles auth itself. The
 * `codex` binary on PATH (or `GBRAIN_CODEX_CLI_BIN`) IS the auth surface.
 * There is NO `CODEX_OAUTH_TOKEN`-style env equivalent of
 * `CLAUDE_CODE_OAUTH_TOKEN`; auth.json is the only handle, and it is
 * portable (verified: copying just that file into a clean $HOME works).
 *
 * Setup expectation: `codex` installed and `codex login` completed, or
 * `GBRAIN_CODEX_CLI_BIN` pointing at the binary.
 */
export const codexCli: Recipe = {
  id: 'codex-cli',
  name: 'Codex (via CLI)',
  tier: 'native',
  implementation: 'codex-cli',
  // The CLI owns auth; no env vars are required from the gateway side.
  auth_env: {
    required: [],
  },
  touchpoints: {
    // No embedding or expansion touchpoints — chat-only.
    chat: {
      // Slugs verified against the CLI's own model metadata
      // (~/.codex/models_cache.json on codex-cli 0.146.1): sol = frontier,
      // terra = balanced everyday, luna = fast/affordable.
      models: [
        'gpt-5.6-luna',
        'gpt-5.6-terra',
        'gpt-5.6-sol',
      ],
      // Verified, not assumed: gpt-5.6-luna emitted a well-formed
      // <use_tools> block on the first probe, including a two-call parallel
      // block. The adapter parses those into ai-sdk tool-call parts.
      supports_tools: true,
      // Each call is a fresh `codex exec` fed the whole rendered transcript,
      // so there is no session state to lose on crash/replay — the same
      // property that makes claude-cli loop-safe.
      supports_subagent_loop: true,
      // Codex benefits from OpenAI's automatic prefix caching and REPORTS the
      // hits (`cached_input_tokens`), but gbrain cannot DIRECT that cache:
      // there is no cache_control plane over `codex exec`. This flag gates
      // whether the gateway inserts cache breakpoints, so it stays false —
      // reporting is surfaced through usage.cachedInputTokens instead.
      supports_prompt_cache: false,
      // context_window from the CLI's model metadata; identical for all three.
      max_context_tokens: 272000,
      // Cost fields intentionally omitted. `src/core/model-pricing.ts` (the
      // one canonical chat-pricing table) has no gpt-5.6 entry, so any number
      // here would be invented and would seed exactly the cross-table drift
      // that invariant exists to prevent. The subscription bears the bill
      // anyway; token counts still reach the budget ledger via gateway.chat.
    },
  },
  // Friendly aliases mirror the claude-cli recipe's style (`sonnet`, `opus`):
  // the tier nickname alone resolves to the full slug, so config strings stay
  // short and a tier swap is a one-token edit.
  aliases: {
    'luna': 'gpt-5.6-luna',
    'terra': 'gpt-5.6-terra',
    'sol': 'gpt-5.6-sol',
  },
  setup_hint:
    'Install the Codex CLI (`codex`) and run `codex login` once to authenticate ' +
    'with your ChatGPT subscription. Set GBRAIN_CODEX_CLI_BIN if the binary is not on PATH.',
};
