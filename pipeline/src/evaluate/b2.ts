/**
 * Scheme B2 (FR-39): isolated evidence extraction from every interaction, quote
 * verification with re-request (FR-46, FR-48), then one aggregation call over the verified
 * evidence of the whole case. Scheme B2-np (FR-40) is the same code with `verify: false`:
 * quotes are accepted as returned and nothing is re-requested.
 */
import type { ModelMessage } from 'ai';
import { z } from 'zod';
import type { Case, Interaction, Turn } from '../domain.ts';
import { callStructured } from '../llm/structured.ts';
import type { Llm } from '../llm/models.ts';
import type { Outcome, Protocol, ProtocolEntry } from '../protocol.ts';
import { answerValues, applies, allQuestions, type Question, type Rubric } from '../rubric.ts';
import { verifyQuote, type Verification } from '../verify.ts';
import {
  AGGREGATE_INSTRUCTIONS,
  aggregatePrompt,
  EXTRACT_INSTRUCTIONS,
  extractPrompt,
  type EvidenceView,
  PROMPT_VERSION,
} from './prompts.ts';

export type B2Options = {
  verify: boolean;
  /** Attempts per extraction or aggregation, the first one included (FR-48). */
  maxAttempts: number;
  /** Minimum similarity for the fuzzy verification level (FR-46). */
  fuzzyThreshold: number;
};

export const DEFAULT_OPTIONS: B2Options = { verify: true, maxAttempts: 3, fuzzyThreshold: 0.8 };

export type Evidence = {
  id: string;
  questionId: string;
  interactionId: string;
  turn: number;
  quote: string;
  note: string;
  verification: Verification | { level: 'unverified' };
};

export type Answer = {
  questionId: string;
  status: 'answered' | 'unanswered' | 'not_applicable';
  value: string | null;
  reasoning: string | null;
  evidence: string[];
  attempts: number;
};

export type B2Result = { answers: Answer[]; evidence: Evidence[]; rejectedQuotes: number };

type RunContext = {
  llm: Llm;
  protocol: Protocol;
  base: Pick<ProtocolEntry, 'runId' | 'caseId' | 'scheme' | 'model' | 'engine' | 'promptVersion' | 'rubricRevision'>;
  options: B2Options;
};

export async function evaluateB2(
  c: Case,
  rubric: Rubric,
  llm: Llm,
  protocol: Protocol,
  base: Omit<RunContext['base'], 'promptVersion' | 'rubricRevision'>,
  options: B2Options = DEFAULT_OPTIONS,
): Promise<B2Result> {
  const ctx: RunContext = {
    llm,
    protocol,
    base: { ...base, promptVersion: PROMPT_VERSION, rubricRevision: rubric.revision },
    options,
  };
  const questions = allQuestions(rubric);
  const applicable = questions.filter((q) => applies(q, c));

  const perInteraction = await mapLimit(c.interactions, llm.concurrency, (i) =>
    extractInteraction(
      ctx,
      i,
      applicable.filter((q) => q.appliesTo.includes(i.channel)),
    ),
  );
  const evidence = perInteraction.flatMap((r) => r.evidence).map((e, n) => ({ ...e, id: `E${n + 1}` }));
  const rejectedQuotes = perInteraction.reduce((acc, r) => acc + r.rejected, 0);

  const answered = await aggregate(ctx, c, applicable, evidence);
  const answers: Answer[] = questions.map(
    (q) =>
      answered.get(q.id) ?? {
        questionId: q.id,
        status: 'not_applicable',
        value: null,
        reasoning: null,
        evidence: [],
        attempts: 0,
      },
  );
  return { answers, evidence, rejectedQuotes };
}

/** Like Promise.all over `items`, with at most `limit` calls in flight; order is preserved. */
async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i] as T);
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
  return out;
}

// --- extraction ------------------------------------------------------------------------

function extractionSchema(questions: Question[], turns: number) {
  const item = z.object({
    turn: z.number().int().min(1).max(Math.max(1, turns)),
    quote: z.string(),
    note: z.string(),
  });
  const perQuestion = z.object({ reasoning: z.string(), evidence: z.array(item) });
  return z.object(Object.fromEntries(questions.map((q) => [q.id, perQuestion])));
}

type Extracted = { reasoning: string; evidence: { turn: number; quote: string; note: string }[] };

async function extractInteraction(
  ctx: RunContext,
  interaction: Interaction,
  questions: Question[],
): Promise<{ evidence: Omit<Evidence, 'id'>[]; rejected: number }> {
  const kept: Omit<Evidence, 'id'>[] = [];
  let rejected = 0;
  let pending = questions;
  const messages: ModelMessage[] = [{ role: 'user', content: extractPrompt(interaction, questions) }];

  for (let attempt = 1; attempt <= ctx.options.maxAttempts && pending.length > 0; attempt++) {
    const schema = extractionSchema(pending, interaction.turns.length);
    const result = await callStructured({
      llm: ctx.llm,
      schema,
      name: 'evidence',
      instructions: EXTRACT_INSTRUCTIONS,
      messages,
    });
    const entry = (
      outcome: Outcome,
      quotes: ProtocolEntry['quotes'],
      error: string | null,
      failedQuotes: ProtocolEntry['failedQuotes'] = [],
    ) =>
      ctx.protocol.record({
        ...ctx.base,
        stage: 'extract',
        failedQuotes,
        interactionId: interaction.id,
        questionIds: pending.map((q) => q.id),
        attempt,
        outcome,
        quotes,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        latencyMs: result.latencyMs,
        responseFormat: result.responseFormat,
        error,
      });

    if (!result.ok) {
      await entry(result.reason === 'schema' ? 'rejected_schema' : 'error', NO_QUOTES, result.error);
      if (result.reason === 'error') break;
      continue; // same messages again: the schema failure says nothing new to the model
    }
    await ctx.protocol.sample('extract', result.requestBody);

    const failed: { question: Question; quote: string; turn: number; best: number }[] = [];
    const counts = { ...NO_QUOTES };
    for (const q of pending) {
      const answer: Extracted | undefined = result.value[q.id];
      for (const e of answer?.evidence ?? []) {
        if (!ctx.options.verify) {
          counts.unverified++;
          kept.push({ questionId: q.id, interactionId: interaction.id, ...e, verification: { level: 'unverified' } });
          continue;
        }
        const v = verifyQuote(interaction, e.turn, e.quote, ctx.options.fuzzyThreshold);
        counts[v.level]++;
        if (v.level === 'failed') {
          failed.push({ question: q, quote: e.quote, turn: e.turn, best: v.bestSimilarity });
        } else if (!kept.some((k) => k.questionId === q.id && sameSpan(k.verification, v))) {
          kept.push({ questionId: q.id, interactionId: interaction.id, ...e, turn: v.turn, verification: v });
        }
      }
    }
    rejected += failed.length;
    await entry(
      failed.length ? 'rejected_quote' : 'accepted',
      counts,
      null,
      failed.map((f) => ({ questionId: f.question.id, turn: f.turn, quote: f.quote, bestSimilarity: f.best })),
    );

    // Re-request only the questions whose quotes did not verify; their verified quotes stay.
    pending = [...new Set(failed.map((f) => f.question))];
    if (pending.length) {
      messages.push(
        { role: 'assistant', content: result.rawText ?? JSON.stringify(result.value) },
        { role: 'user', content: retryFeedback(failed, pending) },
      );
    }
  }
  return { evidence: kept, rejected };
}

const NO_QUOTES = { exact: 0, fuzzy: 0, failed: 0, unverified: 0 };

function sameSpan(a: Evidence['verification'], b: Verification): boolean {
  return a.level !== 'unverified' && a.level !== 'failed' && b.level !== 'failed'
    && a.turn === b.turn && a.from === b.from && a.to === b.to;
}

function retryFeedback(failed: { quote: string; turn: number }[], pending: Question[]): string {
  const list = failed.map((f) => `- реплика ${f.turn}: «${f.quote}»`).join('\n');
  return `Эти цитаты не найдены в указанных репликах дословно:\n${list}\n\nПовтори ответ только для вопросов ${pending
    .map((q) => q.id)
    .join(', ')}. Копируй цитаты слово в слово из текста реплик или не приводи их.`;
}

// --- aggregation -----------------------------------------------------------------------

/**
 * Scale answers are integers, not digit strings: with "3" required as a string, Qwen3-235B
 * under Yandex's constrained decoding stalled on the value and emitted whitespace up to
 * the token limit (docs/llm-comparison.md). Everything else is an enum of value ids.
 */
function answerSchema(q: Question) {
  if (q.kind === 'scale') {
    return z.union([z.number().int().min(q.scale.min).max(q.scale.max), z.literal('no_answer')]);
  }
  return z.enum(['no_answer', ...answerValues(q)]);
}

function aggregationSchema(questions: Question[], evidenceIds: string[]) {
  const ids = evidenceIds.length ? z.array(z.enum(evidenceIds as [string, ...string[]])) : z.array(z.string()).max(0);
  return z.object(
    Object.fromEntries(
      questions.map((q) => [
        q.id,
        z.object({
          reasoning: z.string(),
          answer: answerSchema(q),
          evidence: ids,
        }),
      ]),
    ),
  );
}

async function aggregate(
  ctx: RunContext,
  c: Case,
  questions: Question[],
  evidence: Evidence[],
): Promise<Map<string, Answer>> {
  const views: EvidenceView[] = evidence.flatMap((e) => {
    const interaction = c.interactions.find((i) => i.id === e.interactionId);
    const turn: Turn | undefined = interaction?.turns.find((t) => t.index === e.turn);
    return interaction && turn ? [{ ...e, interaction, turn }] : [];
  });
  const ids = evidence.map((e) => e.id);
  const out = new Map<string, Answer>();
  let pending = questions;
  const messages: ModelMessage[] = [{ role: 'user', content: aggregatePrompt(c, questions, views) }];

  for (let attempt = 1; attempt <= ctx.options.maxAttempts && pending.length > 0; attempt++) {
    const result = await callStructured({
      llm: ctx.llm,
      schema: aggregationSchema(pending, ids),
      name: 'answers',
      instructions: AGGREGATE_INSTRUCTIONS,
      messages,
    });
    const entry = (outcome: Outcome, error: string | null) =>
      ctx.protocol.record({
        ...ctx.base,
        stage: 'aggregate',
        interactionId: null,
        failedQuotes: [],
        questionIds: pending.map((q) => q.id),
        attempt,
        outcome,
        quotes: NO_QUOTES,
        inputTokens: result.inputTokens,
        outputTokens: result.outputTokens,
        latencyMs: result.latencyMs,
        responseFormat: result.responseFormat,
        error,
      });
    if (!result.ok) {
      await entry(result.reason === 'schema' ? 'rejected_schema' : 'error', result.error);
      if (result.reason === 'error') break;
      continue;
    }
    await ctx.protocol.sample('aggregate', result.requestBody);

    const retry: Question[] = [];
    for (const q of pending) {
      const raw = result.value[q.id];
      const a = raw && { ...raw, answer: String(raw.answer) };
      if (!a) {
        retry.push(q);
        continue;
      }
      const needsEvidence = a.answer !== 'no_answer' && a.answer !== q.valueWithoutEvidence;
      if (ctx.options.verify && needsEvidence && a.evidence.length === 0) {
        retry.push(q);
        continue;
      }
      out.set(q.id, {
        questionId: q.id,
        status: a.answer === 'no_answer' ? 'unanswered' : 'answered',
        value: a.answer === 'no_answer' ? null : a.answer,
        reasoning: a.reasoning,
        evidence: a.evidence,
        attempts: attempt,
      });
    }
    await entry(retry.length ? 'rejected_no_evidence' : 'accepted', null);
    pending = retry;
    if (pending.length) {
      messages.push(
        { role: 'assistant', content: result.rawText ?? JSON.stringify(result.value) },
        {
          role: 'user',
          content: `Для вопросов ${pending.map((q) => q.id).join(', ')} ответ дан без доказательств. Повтори ответ только для них: укажи идентификаторы доказательств или ответь "no_answer".`,
        },
      );
    }
  }
  // FR-48: after the last attempt the question goes to a human.
  for (const q of pending) {
    out.set(q.id, {
      questionId: q.id,
      status: 'unanswered',
      value: null,
      reasoning: null,
      evidence: [],
      attempts: ctx.options.maxAttempts,
    });
  }
  return out;
}
