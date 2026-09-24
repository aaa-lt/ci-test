/**
 * Summarize runs against the author's gold answers into results/llm/summary.json.
 *
 *   pnpm tsx src/report.ts [run ...]      default: every run in results/runs/
 *
 * One case per model is far too little for accuracy claims; the numbers here describe
 * fitness (schema, citations, refusals, cost), and agreement with gold is reported as
 * counts, not as a rate with a confidence interval.
 */
import { existsSync } from 'node:fs';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { z } from 'zod';
import { readJson, ROOT } from './ingest/load.ts';

const RUNS = join(ROOT, 'results', 'runs');

const Gold = z.object({
  case: z.string(),
  answers: z.record(z.string(), z.object({ value: z.union([z.string(), z.number(), z.null()]) })),
});

const RunCase = z.object({
  key: z.object({ caseId: z.string(), model: z.string(), scheme: z.string(), engine: z.string() }),
  seconds: z.number(),
  score: z.object({ total: z.number().nullable(), autoFail: z.string().nullable(), unanswered: z.number() }),
  answers: z.array(
    z.object({ questionId: z.string(), status: z.string(), value: z.string().nullable(), attempts: z.number() }),
  ),
  evidence: z.array(z.object({ verification: z.object({ level: z.string() }) })),
  rejectedQuotes: z.number(),
});

const Call = z.object({
  stage: z.string(),
  attempt: z.number(),
  outcome: z.string(),
  quotes: z.object({ exact: z.number(), fuzzy: z.number(), failed: z.number(), unverified: z.number() }),
  inputTokens: z.number().nullable(),
  outputTokens: z.number().nullable(),
  latencyMs: z.number(),
  responseFormat: z.string().nullable(),
  error: z.string().nullable(),
});

const runs = process.argv.slice(2).length ? process.argv.slice(2) : await readdir(RUNS);
const summary: Record<string, unknown> = {};

for (const run of runs) {
  const dir = join(RUNS, run);
  const files = (await readdir(dir)).filter((f) => f.endsWith('.json'));
  const calls = existsSync(join(dir, 'protocol.jsonl'))
    ? (await readFile(join(dir, 'protocol.jsonl'), 'utf8')).trim().split('\n').map((l) => Call.parse(JSON.parse(l)))
    : [];
  const perCase = [];
  for (const f of files) {
    const r = await readJson(join(dir, f), RunCase);
    const goldPath = join(ROOT, 'data', 'gold', `${r.key.caseId}.json`);
    const gold = existsSync(goldPath) ? await readJson(goldPath, Gold) : null;
    const agree: string[] = [];
    const disagree: string[] = [];
    for (const a of r.answers) {
      const g = gold?.answers[a.questionId]?.value;
      if (g === undefined || a.status === 'not_applicable') continue;
      const expected = g === null ? null : String(g);
      (a.value === expected ? agree : disagree).push(`${a.questionId}: ${a.value ?? 'нет ответа'} / ${expected ?? 'отказ'}`);
    }
    perCase.push({
      case: r.key.caseId,
      engine: r.key.engine,
      score: r.score.total,
      autoFail: r.score.autoFail,
      unanswered: r.score.unanswered,
      agree: agree.length,
      disagree,
      seconds: r.seconds,
      evidence: countBy(r.evidence.map((e) => e.verification.level)),
      rejectedQuotes: r.rejectedQuotes,
    });
  }
  const quotes = calls.reduce(
    (acc, c) => ({
      exact: acc.exact + c.quotes.exact,
      fuzzy: acc.fuzzy + c.quotes.fuzzy,
      failed: acc.failed + c.quotes.failed,
      unverified: acc.unverified + c.quotes.unverified,
    }),
    { exact: 0, fuzzy: 0, failed: 0, unverified: 0 },
  );
  summary[run] = {
    calls: calls.length,
    outcomes: countBy(calls.map((c) => c.outcome)),
    retries: calls.filter((c) => c.attempt > 1).length,
    quotes,
    inputTokens: sum(calls.map((c) => c.inputTokens ?? 0)),
    outputTokens: sum(calls.map((c) => c.outputTokens ?? 0)),
    latencyMsMean: calls.length ? Math.round(sum(calls.map((c) => c.latencyMs)) / calls.length) : null,
    responseFormat: countBy(calls.map((c) => c.responseFormat ?? 'none')),
    errors: calls.flatMap((c) => (c.error ? [c.error.slice(0, 200)] : [])),
    cases: perCase,
  };
}

await mkdir(join(ROOT, 'results', 'llm'), { recursive: true });
await writeFile(join(ROOT, 'results', 'llm', 'summary.json'), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));

function countBy(items: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const i of items) out[i] = (out[i] ?? 0) + 1;
  return out;
}

function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}
