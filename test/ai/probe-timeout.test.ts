/**
 * Provider-aware probe budgets.
 *
 * `gbrain models doctor` fired every reachability probe with a hardcoded 5s
 * cap. That is a network-shaped budget, and it is wrong for subprocess-backed
 * providers: `claude-cli` spawns a whole CLI per call, and a cold
 * `claude --print` measures ~16s (a cold `codex exec` ~7s). Doctor therefore
 * reported `chat` and `expansion` as unreachable for routes that work fine in
 * production, where no such cap exists — and re-running never helped, because
 * the cost is the spawn itself, not a cold cache.
 *
 * This is the same divergence `resolveLiveRerankerTimeoutMs` was introduced to
 * fix, in the opposite direction: there the probe was more generous than
 * production and reported reachable while production timed out.
 *
 * Hermetic: env overrides go through `withEnv` (R1) so these cases stay safe
 * under the parallel runner — a bare `process.env` mutation here would leak
 * GBRAIN_PROBE_TIMEOUT_MS into whatever else the shard is executing.
 */

import { describe, expect, test } from 'bun:test';
import { probeTimeoutMsFor } from '../../src/commands/models.ts';
import { withEnv } from '../helpers/with-env.ts';

const ENV_KEY = 'GBRAIN_PROBE_TIMEOUT_MS';

describe('probeTimeoutMsFor', () => {
  test('subprocess-backed CLIs get the long budget', async () => {
    await withEnv({ [ENV_KEY]: undefined }, () => {
      // Must exceed the measured cold-start of each CLI, or doctor false-fails.
      expect(probeTimeoutMsFor('claude-cli:claude-haiku-4-5-20251001')).toBe(30_000);
      expect(probeTimeoutMsFor('claude-cli:claude-sonnet-4-6')).toBe(30_000);
      expect(probeTimeoutMsFor('codex-cli:gpt-5.6-luna')).toBe(30_000);
    });
  });

  test('HTTP providers keep the network budget', async () => {
    await withEnv({ [ENV_KEY]: undefined }, () => {
      expect(probeTimeoutMsFor('anthropic:claude-sonnet-4-6')).toBe(5_000);
      expect(probeTimeoutMsFor('litellm:deepseek-ai/DeepSeek-V4-Flash')).toBe(5_000);
      expect(probeTimeoutMsFor('openai:gpt-5.2')).toBe(5_000);
    });
  });

  test('unknown provider and bare model id fall back to the network budget', async () => {
    await withEnv({ [ENV_KEY]: undefined }, () => {
      expect(probeTimeoutMsFor('not-a-recipe:whatever')).toBe(5_000);
      expect(probeTimeoutMsFor('claude-sonnet-4-6')).toBe(5_000);
      expect(probeTimeoutMsFor(undefined)).toBe(5_000);
    });
  });

  test('env override wins for every provider', async () => {
    await withEnv({ [ENV_KEY]: '90000' }, () => {
      expect(probeTimeoutMsFor('claude-cli:claude-sonnet-4-6')).toBe(90_000);
      expect(probeTimeoutMsFor('anthropic:claude-sonnet-4-6')).toBe(90_000);
    });
  });

  test('non-numeric or non-positive override is ignored', async () => {
    for (const bad of ['abc', '0', '-1', '']) {
      await withEnv({ [ENV_KEY]: bad }, () => {
        expect(probeTimeoutMsFor('anthropic:claude-sonnet-4-6')).toBe(5_000);
        expect(probeTimeoutMsFor('claude-cli:claude-sonnet-4-6')).toBe(30_000);
      });
    }
  });
});
