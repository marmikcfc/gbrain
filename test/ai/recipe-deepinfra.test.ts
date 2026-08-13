/**
 * DeepInfra recipe smoke.
 *
 * DeepInfra serves open-weights models behind an OpenAI-compatible endpoint.
 * The load-bearing detail is width: `Qwen/Qwen3-Embedding-8B` is 4096-dim
 * natively, which no pgvector HNSW index can hold (2000 for `vector`, 4000 for
 * `halfvec`). The model is only usable because DeepInfra honors the OpenAI
 * `dimensions` field for Matryoshka truncation, so these tests pin that the
 * recipe declares native widths and that the dims path emits `dimensions` for
 * the Qwen3 family at a reduced width.
 *
 * Coverage:
 *  - Recipe registered with expected shape + auth env
 *  - Native model_dims declared for the served embedding models
 *  - default_dims is 0 (no safe guess across a 1024..4096 spread)
 *  - dims path emits `dimensions` for a truncated Qwen3 width
 *  - dims path omits `dimensions` at native width (fixed-dim backends 400)
 *  - fixed-width BAAI models never receive `dimensions`
 */

import { describe, expect, test } from 'bun:test';
import { getRecipe } from '../../src/core/ai/recipes/index.ts';
import { embeddingDimsForModel } from '../../src/core/ai/model-resolver.ts';
import { dimsProviderOptions } from '../../src/core/ai/dims.ts';

describe('recipe: deepinfra', () => {
  test('registered with the expected shape', () => {
    const r = getRecipe('deepinfra');
    expect(r).toBeDefined();
    expect(r!.tier).toBe('openai-compat');
    expect(r!.implementation).toBe('openai-compatible');
    expect(r!.base_url_default).toBe('https://api.deepinfra.com/v1/openai');
    expect(r!.auth_env?.required).toEqual(['DEEPINFRA_API_KEY']);
  });

  test('declares native widths for the served embedding models', () => {
    const r = getRecipe('deepinfra')!;
    expect(embeddingDimsForModel(r, 'Qwen/Qwen3-Embedding-8B')).toBe(4096);
    expect(embeddingDimsForModel(r, 'Qwen/Qwen3-Embedding-4B')).toBe(2560);
    expect(embeddingDimsForModel(r, 'Qwen/Qwen3-Embedding-0.6B')).toBe(1024);
    expect(embeddingDimsForModel(r, 'BAAI/bge-m3')).toBe(1024);
  });

  test('no recipe-wide default width — operator must declare one', () => {
    const r = getRecipe('deepinfra')!;
    expect(r.touchpoints.embedding!.default_dims).toBe(0);
    expect(r.touchpoints.embedding!.trust_custom_dims).toBe(true);
    // An unknown model falls through to default_dims rather than guessing.
    expect(embeddingDimsForModel(r, 'some/unlisted-model')).toBe(0);
  });

  test('Qwen3 at a truncated width threads `dimensions`', () => {
    // 1280 is the widest Matryoshka rung that still fits a pgvector HNSW index.
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-8B', 1280))
      .toEqual({ openaiCompatible: { dimensions: 1280 } });
  });

  test('Qwen3 at native width omits `dimensions`', () => {
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-8B', 4096))
      .toBeUndefined();
  });

  test('fixed-width BAAI models never receive `dimensions`', () => {
    expect(dimsProviderOptions('openai-compatible', 'BAAI/bge-m3', 1024))
      .toBeUndefined();
  });
});
