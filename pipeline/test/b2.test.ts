import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MockLanguageModelV4 } from 'ai/test';
import { describe, expect, it } from 'vitest';
import { DEFAULT_OPTIONS, evaluateB2 } from '../src/evaluate/b2.ts';
import { Protocol } from '../src/protocol.ts';
import { demoCase, rubric } from './fixtures.ts';

type Prompt = { role: string; content: unknown }[];

/** Text of the whole conversation so far, and how many user turns it has. */
function read(prompt: Prompt): { text: string; userTurns: number } {
  const text = JSON.stringify(prompt);
  return { text, userTurns: prompt.filter((m) => m.role === 'user').length };
}

/** A scripted model: answers depend on which interaction is asked about and on the attempt. */
function scriptedModel() {
  return new MockLanguageModelV4({
    doGenerate: async ({ prompt }) => {
      const { text, userTurns } = read(prompt as Prompt);
      let body: unknown;
      if (text.includes('Взаимодействие call-1')) {
        body =
          userTurns === 1
            ? {
                greeting: { reasoning: 'r', evidence: [{ turn: 1, quote: 'здравствуйте меня зовут ольга', note: 'n' }] },
                empathy: { reasoning: 'r', evidence: [] },
                forbidden: { reasoning: 'r', evidence: [] },
              }
            : { greeting: { reasoning: 'r', evidence: [{ turn: 1, quote: 'Меня зовут Анна', note: 'n' }] } };
      } else if (text.includes('Взаимодействие chat-1')) {
        body = {
          greeting: { reasoning: 'r', evidence: [] },
          empathy: { reasoning: 'r', evidence: [{ turn: 2, quote: 'Приношу извинения', note: 'n' }] },
          forbidden: { reasoning: 'r', evidence: [] },
        };
      } else {
        body =
          userTurns === 1
            ? {
                greeting: { reasoning: 'r', answer: 'yes', evidence: ['E1'] },
                empathy: { reasoning: 'r', answer: 4, evidence: [] },
                forbidden: { reasoning: 'r', answer: 'no', evidence: [] },
              }
            : { empathy: { reasoning: 'r', answer: 4, evidence: ['E2'] } };
      }
      return {
        content: [{ type: 'text', text: JSON.stringify(body) }],
        finishReason: { unified: 'stop', raw: 'stop' },
        usage: {
          inputTokens: { total: 100, noCache: 100, cacheRead: 0, cacheWrite: 0 },
          outputTokens: { total: 20, text: 20, reasoning: 0 },
        },
        warnings: [],
      };
    },
  });
}

async function run(verify: boolean) {
  const dir = await mkdtemp(join(tmpdir(), 'b2-'));
  const protocol = new Protocol(join(dir, 'protocol.jsonl'));
  const result = await evaluateB2(
    demoCase,
    rubric,
    { model: scriptedModel(), concurrency: 2 },
    protocol,
    { runId: 'r', caseId: 'demo', scheme: verify ? 'b2' : 'b2-np', model: 'mock', engine: 'test' },
    { ...DEFAULT_OPTIONS, verify },
  );
  const calls = (await readFile(protocol.path, 'utf8'))
    .trim()
    .split('\n')
    .map((l) => JSON.parse(l) as { stage: string; interactionId: string | null; attempt: number; outcome: string });
  return { result, calls };
}

describe('scheme B2', () => {
  it('re-requests an unverified quote and an answer without evidence, then answers', async () => {
    const { result, calls } = await run(true);
    const byId = Object.fromEntries(result.answers.map((a) => [a.questionId, a]));
    expect(byId.greeting).toMatchObject({ status: 'answered', value: 'yes', evidence: ['E1'], attempts: 1 });
    expect(byId.empathy).toMatchObject({ status: 'answered', value: '4', evidence: ['E2'], attempts: 2 });
    expect(byId.forbidden).toMatchObject({ status: 'answered', value: 'no', evidence: [] });
    expect(byId.documented).toMatchObject({ status: 'not_applicable' }); // e-mail only, the case has none

    expect(result.rejectedQuotes).toBe(1);
    expect(result.evidence.map((e) => [e.id, e.questionId, e.verification.level])).toEqual([
      ['E1', 'greeting', 'exact'],
      ['E2', 'empathy', 'exact'],
    ]);

    const summary = calls.map((c) => `${c.stage}:${c.interactionId ?? '-'}:${c.attempt}:${c.outcome}`).sort();
    expect(summary).toEqual([
      'aggregate:-:1:rejected_no_evidence',
      'aggregate:-:2:accepted',
      'extract:call-1:1:rejected_quote',
      'extract:call-1:2:accepted',
      'extract:chat-1:1:accepted',
    ]);
  });

  it('B2-np keeps the fabricated quote and accepts the answer without evidence', async () => {
    const { result, calls } = await run(false);
    expect(result.rejectedQuotes).toBe(0);
    expect(result.evidence[0]).toMatchObject({ quote: 'здравствуйте меня зовут ольга', verification: { level: 'unverified' } });
    expect(result.answers.find((a) => a.questionId === 'empathy')).toMatchObject({ value: '4', evidence: [] });
    expect(calls).toHaveLength(3);
  });
});
