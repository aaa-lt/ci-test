/**
 * Evaluate cases with one command (FR-74).
 *
 *   pnpm eval --model hf-qwen3-8b --engine reference --cases demo,tariff [--scheme b2-np] [--run id]
 *
 * Writes results/runs/<run>/<case>.json (answers, evidence, score and the full key of the
 * run), results/runs/<run>/protocol.jsonl (every model call) and requests/*.json (one raw
 * request body per stage). Several invocations with the same --run append to one run.
 */
import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { DEFAULT_OPTIONS, evaluateB2 } from './evaluate/b2.ts';
import { PROMPT_VERSION } from './evaluate/prompts.ts';
import { loadCase, readJson, ROOT } from './ingest/load.ts';
import { isModelName, MODELS, type ModelSpec, toLlm } from './llm/models.ts';
import { Protocol } from './protocol.ts';
import { Rubric } from './rubric.ts';
import { score } from './score.ts';

const SCHEMES = { b2: { verify: true }, 'b2-np': { verify: false } } as const;

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    engine: { type: 'string', default: 'reference' },
    cases: { type: 'string' },
    scheme: { type: 'string', default: 'b2' },
    run: { type: 'string' },
    rubric: { type: 'string', default: 'v0' },
  },
});

const model = values.model ?? '';
if (!isModelName(model)) throw new Error(`--model must be one of ${Object.keys(MODELS).join(', ')}`);
const scheme = values.scheme;
if (scheme !== 'b2' && scheme !== 'b2-np') throw new Error('--scheme must be b2 or b2-np');
const cases = values.cases?.split(',') ?? (await readdir(join(ROOT, 'data', 'cases')));
const runId = values.run ?? `${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}-${model}-${scheme}`;
const runDir = join(ROOT, 'results', 'runs', runId);
const rubric = await readJson(join(ROOT, 'data', 'rubric', `${values.rubric}.json`), Rubric);
const spec: ModelSpec = MODELS[model];
const llm = toLlm(spec);
const options = { ...DEFAULT_OPTIONS, ...SCHEMES[scheme] };
const protocol = new Protocol(join(runDir, 'protocol.jsonl'));
await mkdir(runDir, { recursive: true });

for (const caseId of cases) {
  const started = performance.now();
  const c = await loadCase(caseId, values.engine);
  const result = await evaluateB2(c, rubric, llm, protocol, {
    runId,
    caseId,
    scheme,
    model,
    engine: values.engine,
  }, options);
  const answers = new Map(result.answers.map((a) => [a.questionId, a]));
  const caseScore = score(rubric, answers, result.rejectedQuotes);
  const out = {
    key: {
      runId,
      caseId,
      scheme,
      model,
      providerModel: spec.model,
      engine: values.engine,
      rubric: rubric.id,
      rubricRevision: rubric.revision,
      promptVersion: PROMPT_VERSION,
      decoding: { temperature: 0, structured: spec.structured, ...(spec.extraBody ?? {}) },
      options,
    },
    seconds: Math.round((performance.now() - started) / 100) / 10,
    score: caseScore,
    answers: result.answers,
    evidence: result.evidence,
    rejectedQuotes: result.rejectedQuotes,
    case: c,
  };
  await writeFile(join(runDir, `${caseId}.json`), JSON.stringify(out, null, 2));
  const summary = result.answers.map((a) => `${a.questionId}=${a.value ?? a.status}`).join(' ');
  console.log(`${caseId}: score ${caseScore.total ?? '—'}, ${out.seconds} s, rejected quotes ${result.rejectedQuotes}\n  ${summary}`);
}
