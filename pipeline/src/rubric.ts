/**
 * Rubric schema (FR-27…FR-35). Values of choice and yes/no questions are ordered from
 * worst to best (FR-30); a scale question has an integer range (FR-29).
 */
import { z } from 'zod';
import { Channel, type Case } from './domain.ts';

const Value = z.object({ id: z.string(), label: z.string() });

const Common = {
  id: z.string(),
  text: z.string(),
  guidance: z.string(),
  weight: z.number().positive(),
  appliesTo: z.array(Channel).min(1),
  multiSource: z.boolean(),
  source: z.string(),
  /** Values that end the whole case score at zero (FR-34). */
  autoFail: z.array(z.string()).optional(),
  /**
   * A value that may be given without a citation: the absence of evidence is the answer
   * ("no forbidden phrases were used"). Every other answer needs a verified quote.
   */
  valueWithoutEvidence: z.string().optional(),
};

export const Question = z.discriminatedUnion('kind', [
  z.object({ ...Common, kind: z.literal('yes_no'), values: z.array(Value).length(2), adverse: z.string() }),
  z.object({ ...Common, kind: z.literal('choice'), values: z.array(Value).min(2) }),
  z.object({
    ...Common,
    kind: z.literal('scale'),
    scale: z.object({ min: z.number().int(), max: z.number().int(), step: z.number().int().positive() }),
  }),
]);
export type Question = z.infer<typeof Question>;

export const Rubric = z.object({
  id: z.string(),
  revision: z.number().int().nonnegative(),
  title: z.string(),
  passThreshold: z.number().min(0).max(1),
  sections: z.array(
    z.object({ id: z.string(), title: z.string(), weight: z.number().positive(), questions: z.array(Question) }),
  ),
});
export type Rubric = z.infer<typeof Rubric>;

/** Allowed answer values as strings, worst first. Scale values become "1", "2", … */
export function answerValues(q: Question): string[] {
  if (q.kind !== 'scale') return q.values.map((v) => v.id);
  const out: string[] = [];
  for (let v = q.scale.min; v <= q.scale.max; v += q.scale.step) out.push(String(v));
  return out;
}

export function allQuestions(rubric: Rubric): Question[] {
  return rubric.sections.flatMap((s) => s.questions);
}

/** FR-32: a question applies when the case has at least one interaction of its channels. */
export function applies(q: Question, c: Case): boolean {
  return c.interactions.some((i) => q.appliesTo.includes(i.channel));
}
