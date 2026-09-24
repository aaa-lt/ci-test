import { describe, expect, it } from 'vitest';
import { score, type ScoredAnswer, valueScore } from '../src/score.ts';
import { allQuestions } from '../src/rubric.ts';
import { rubric } from './fixtures.ts';

const answers = (entries: Record<string, ScoredAnswer>) => new Map(Object.entries(entries));
const q = (id: string) => {
  const found = allQuestions(rubric).find((x) => x.id === id);
  if (!found) throw new Error(id);
  return found;
};

describe('valueScore', () => {
  it('maps the worst-to-best order onto [0, 1]', () => {
    expect(valueScore(q('greeting'), 'no')).toBe(0);
    expect(valueScore(q('greeting'), 'yes')).toBe(1);
    expect(valueScore(q('empathy'), '4')).toBe(0.75);
    expect(valueScore(q('forbidden'), 'no')).toBe(1);
  });

  it('refuses a value outside the question', () => {
    expect(() => valueScore(q('empathy'), '7')).toThrow();
  });
});

describe('score', () => {
  it('weights questions and sections and skips not applicable ones', () => {
    const s = score(
      rubric,
      answers({
        greeting: { status: 'answered', value: 'yes' },
        empathy: { status: 'answered', value: '3' }, // 0.5, weight 3
        forbidden: { status: 'answered', value: 'no' },
        documented: { status: 'not_applicable', value: null },
      }),
    );
    expect(s.sections.map((x) => x.score)).toEqual([0.625, 1]); // (1 + 3 * 0.5) / 4
    expect(s.total).toBe(0.9063); // (0.625 * 1 + 1 * 3) / 4
    expect(s.incomplete).toBe(false);
    expect(s.priority).toBe(0);
  });

  it('marks the score incomplete and raises priority for unanswered questions', () => {
    const s = score(
      rubric,
      answers({
        greeting: { status: 'answered', value: 'no' },
        empathy: { status: 'unanswered', value: null },
        forbidden: { status: 'answered', value: 'no' },
        documented: { status: 'not_applicable', value: null },
      }),
      2,
    );
    expect(s.sections[0]).toEqual({ id: 's1', score: 0, answered: 1, unanswered: 1 });
    expect(s.total).toBe(0.75);
    expect(s.incomplete).toBe(true);
    expect(s.priority).toBe(1.9); // 10 * 0.05 + 1 * 1 + 0.2 * 2
  });

  it('zeroes the total on an automatic fail', () => {
    const s = score(
      rubric,
      answers({
        greeting: { status: 'answered', value: 'yes' },
        empathy: { status: 'answered', value: '5' },
        forbidden: { status: 'answered', value: 'yes' },
        documented: { status: 'not_applicable', value: null },
      }),
    );
    expect(s.autoFail).toBe('forbidden');
    expect(s.total).toBe(0);
    expect(s.priority).toBe(13); // 10 * 0.8 + 5
  });

  it('has no total when nothing could be scored', () => {
    const s = score(rubric, answers({}));
    expect(s.total).toBeNull();
    expect(s.priority).toBe(0);
  });
});
