/**
 * Deterministic scoring (FR-49…FR-53, NFR-6): the model gives answers, this code gives
 * the score. Equal answers always produce an equal score.
 */
import { allQuestions, answerValues, type Question, type Rubric } from './rubric.ts';

export type AnswerStatus = 'answered' | 'unanswered' | 'not_applicable';
export type ScoredAnswer = { status: AnswerStatus; value: string | null };

/** Priority coefficients (FR-53); stored with every result so the ranking is reproducible. */
export type PriorityRule = { deficit: number; autoFail: number; unanswered: number; rejected: number };
export const DEFAULT_PRIORITY: PriorityRule = { deficit: 10, autoFail: 5, unanswered: 1, rejected: 0.2 };

export type Score = {
  sections: { id: string; score: number | null; answered: number; unanswered: number }[];
  /** Null when nothing in the case could be scored. */
  total: number | null;
  /** Some applicable question is unanswered (FR-50). */
  incomplete: boolean;
  autoFail: string | null;
  unanswered: number;
  priority: number;
  priorityRule: PriorityRule;
};

/** Position of the value on the worst→best order, mapped to [0, 1]. */
export function valueScore(q: Question, value: string): number {
  const order = answerValues(q);
  const i = order.indexOf(value);
  if (i < 0) throw new Error(`value ${value} is not allowed for ${q.id}`);
  return order.length === 1 ? 1 : i / (order.length - 1);
}

export function score(
  rubric: Rubric,
  answers: ReadonlyMap<string, ScoredAnswer>,
  rejected = 0,
  rule: PriorityRule = DEFAULT_PRIORITY,
): Score {
  let unanswered = 0;
  const sections = rubric.sections.map((section) => {
    let sum = 0;
    let weight = 0;
    let answered = 0;
    let missing = 0;
    for (const q of section.questions) {
      const a = answers.get(q.id);
      if (!a || a.status === 'not_applicable') continue; // FR-50: out of numerator and denominator
      if (a.status === 'unanswered' || a.value === null) {
        missing++;
        continue;
      }
      sum += q.weight * valueScore(q, a.value);
      weight += q.weight;
      answered++;
    }
    unanswered += missing;
    return { id: section.id, score: weight > 0 ? sum / weight : null, answered, unanswered: missing };
  });

  const scored = sections.flatMap((s, i) => {
    const w = rubric.sections[i]?.weight;
    return s.score === null || w === undefined ? [] : [{ score: s.score, weight: w }];
  });
  const weightSum = scored.reduce((acc, s) => acc + s.weight, 0);
  const autoFail =
    allQuestions(rubric).find((q) => {
      const a = answers.get(q.id);
      return a?.status === 'answered' && a.value !== null && (q.autoFail ?? []).includes(a.value);
    })?.id ?? null;
  const raw = weightSum > 0 ? scored.reduce((acc, s) => acc + s.score * s.weight, 0) / weightSum : null;
  const total = autoFail ? 0 : raw; // FR-51

  const deficit = total === null ? 0 : Math.max(0, rubric.passThreshold - total);
  const priority =
    rule.deficit * deficit + rule.autoFail * (autoFail ? 1 : 0) + rule.unanswered * unanswered + rule.rejected * rejected;

  return {
    sections: sections.map((s) => ({ ...s, score: s.score === null ? null : round(s.score) })),
    total: total === null ? null : round(total),
    incomplete: unanswered > 0,
    autoFail,
    unanswered,
    priority: round(priority),
    priorityRule: rule,
  };
}

const round = (x: number) => Math.round(x * 10000) / 10000;
