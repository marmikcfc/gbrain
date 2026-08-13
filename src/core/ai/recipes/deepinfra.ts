import type { Recipe } from '../types.ts';

/**
 * DeepInfra — hosted open-weights inference behind an OpenAI-compatible
 * endpoint at https://api.deepinfra.com/v1/openai.
 *
 * Why a first-class recipe rather than routing through `litellm`: the proxy
 * recipe declares `models: []` + `default_dims: 0`, so the operator carries
 * the whole width contract by hand. That is exactly where the Qwen3 family
 * bites — `Qwen/Qwen3-Embedding-8B` emits 4096 dims natively, which exceeds
 * pgvector's 2000-dim ceiling for an HNSW index (4000 for `halfvec`). A brain
 * configured at any indexable width only works because DeepInfra honors the
 * OpenAI `dimensions` field for Matryoshka truncation server-side. Declaring
 * `model_dims` here lets `embeddingDimsForModel()` and the init-time
 * diagnostics see the native widths up front instead of discovering them as a
 * dim-mismatch on first embed.
 *
 * Embedding-only for now. DeepInfra also serves chat + rerank on the same
 * OpenAI-compatible surface; those touchpoints are deliberately omitted until
 * someone verifies the wire shapes against `gateway.chat()` / `gateway.rerank()`
 * rather than assuming they match.
 *
 * `model_dims` keys are the vendor's canonical, capitalized HuggingFace repo
 * ids — the strings DeepInfra's own docs and `/v1/openai/models` return. The
 * lookup in `embeddingDimsForModel()` is an exact match, so a differently-cased
 * id falls through to `default_dims`; `trust_custom_dims` keeps an explicit
 * `--embedding-dimensions` override authoritative in that case.
 */
export const deepinfra: Recipe = {
  id: 'deepinfra',
  name: 'DeepInfra',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'https://api.deepinfra.com/v1/openai',
  auth_env: {
    required: ['DEEPINFRA_API_KEY'],
    setup_url: 'https://deepinfra.com/dash/api_keys',
  },
  touchpoints: {
    embedding: {
      models: [
        'Qwen/Qwen3-Embedding-8B',
        'Qwen/Qwen3-Embedding-4B',
        'Qwen/Qwen3-Embedding-0.6B',
        'BAAI/bge-m3',
        'BAAI/bge-en-icl',
      ],
      // Native widths, verified against the live endpoint 2026-08-13. Only the
      // Qwen3 family accepts `dimensions` (Matryoshka); the BAAI models are
      // fixed-width and reject the parameter, which the dims.ts branch already
      // reflects by not emitting it for them.
      model_dims: {
        'Qwen/Qwen3-Embedding-8B': 4096,
        'Qwen/Qwen3-Embedding-4B': 2560,
        'Qwen/Qwen3-Embedding-0.6B': 1024,
        'BAAI/bge-m3': 1024,
      },
      // No safe recipe-wide default: the served models span 1024..4096 and
      // guessing one produces a silently wrong column width. Operators must
      // pass `--embedding-dimensions` (same posture as the litellm recipe).
      default_dims: 0,
      trust_custom_dims: true,
      // $0.01 / 1M input tokens — derived from the model metadata endpoint's
      // `pricing.cents_per_input_token: 0.000001` for Qwen3-Embedding-8B.
      cost_per_1m_tokens_usd: 0.01,
      price_last_verified: '2026-08-13',
      // DeepInfra publishes no hard batch cap for the OpenAI-compat path.
      // Declare a conservative token ceiling so the gateway pre-splits rather
      // than discovering an undocumented server-side limit as a 4xx mid-run.
      max_batch_tokens: 8192,
    },
  },
  setup_hint:
    'Create a key at https://deepinfra.com/dash/api_keys, then ' +
    '`export DEEPINFRA_API_KEY=...`. Inference requires a payment method on ' +
    'the account. Qwen3-Embedding-8B is 4096-dim natively — pass ' +
    '`--embedding-dimensions 1280` (or any value <=2000) so the column fits ' +
    "pgvector's HNSW ceiling; DeepInfra truncates server-side via Matryoshka.",
};
