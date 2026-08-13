/**
 * Case-insensitive model-id matching on the openai-compatible dims path.
 *
 * Hosted providers serve HuggingFace repo ids with upstream capitalization.
 * DeepInfra's canonical id for the Qwen3 embedder is `Qwen/Qwen3-Embedding-8B`
 * — capital Q, E and B. `dimsProviderOptions` strips the org prefix but every
 * comparison after that tests a lower-case literal, so before this fix the
 * vendor's own id fell through the Matryoshka branch: no `dimensions` was
 * sent, DeepInfra returned the native 4096, and the embed hard-failed with
 * `Embedding dim mismatch: model returned 4096 but schema expects 1280`
 * against a pgvector column that cannot exceed 2000 dims under an HNSW index.
 *
 * Regression pin: the id a provider actually documents must resolve, whatever
 * its capitalization.
 */

import { describe, expect, test } from 'bun:test';
import { dimsProviderOptions } from '../../src/core/ai/dims.ts';

describe('dims: case-insensitive model ids (openai-compatible)', () => {
  test("DeepInfra's canonical `Qwen/Qwen3-Embedding-8B` threads dimensions", () => {
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-8B', 1280))
      .toEqual({ openaiCompatible: { dimensions: 1280 } });
  });

  test('all-lowercase HF-style id keeps working (no behavior change)', () => {
    expect(dimsProviderOptions('openai-compatible', 'qwen/qwen3-embedding-8b', 1280))
      .toEqual({ openaiCompatible: { dimensions: 1280 } });
  });

  test('mixed-case 0.6B and 4B variants resolve', () => {
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-0.6B', 512))
      .toEqual({ openaiCompatible: { dimensions: 512 } });
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-4B', 1280))
      .toEqual({ openaiCompatible: { dimensions: 1280 } });
  });

  test('native-width request still omits `dimensions` regardless of case', () => {
    // Fixed-dim openai-compatible backends (vLLM) reject the parameter even
    // when it equals the native width — pinned by dims-qwen3-native.test.ts.
    // Case normalization must not weaken that guard.
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-8B', 4096))
      .toBeUndefined();
    expect(dimsProviderOptions('openai-compatible', 'Qwen/Qwen3-Embedding-0.6B', 1024))
      .toBeUndefined();
  });

  test('capitalized OpenAI text-embedding-3 ids resolve too', () => {
    expect(dimsProviderOptions('openai-compatible', 'Azure/Text-Embedding-3-Large', 1536))
      .toEqual({ openaiCompatible: { dimensions: 1536 } });
  });

  test('unrelated models still return undefined (regression guard)', () => {
    expect(dimsProviderOptions('openai-compatible', 'BAAI/bge-m3', 1024))
      .toBeUndefined();
    expect(dimsProviderOptions('openai-compatible', 'Nomic-Embed-Text', 768))
      .toBeUndefined();
  });
});
