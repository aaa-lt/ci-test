import type { Case, Interaction, Turn } from '../src/domain.ts';
import type { Rubric } from '../src/rubric.ts';

export function turn(index: number, role: Turn['role'], text: string, words?: [string, number][]): Turn {
  return {
    index,
    role,
    start: words?.[0]?.[1] ?? null,
    end: null,
    at: null,
    text: words ? words.map(([w]) => w).join(' ') : text,
    words: words ? words.map(([w, s]) => ({ text: w, start: s, end: s + 0.3 })) : null,
    overlap: false,
  };
}

export const call: Interaction = {
  id: 'call-1',
  channel: 'call',
  operator: 'op-1',
  startedAt: null,
  turns: [
    turn(1, 'operator', '', [['компания', 0.8], ['кварта', 1.2], ['меня', 1.6], ['зовут', 1.9], ['анна', 2.2]]),
    turn(2, 'client', '', [['списывают', 3.0], ['за', 3.5], ['кино', 3.7], ['пакет', 4.0], ['плюс', 4.3]]),
    turn(3, 'operator', '', [['я', 5.0], ['вам', 5.2], ['перезвоню', 5.4], ['сегодня', 6.0]]),
  ],
};

export const chat: Interaction = {
  id: 'chat-1',
  channel: 'chat',
  operator: 'op-2',
  startedAt: '2026-09-11T09:12:40+03:00',
  turns: [
    turn(1, 'client', 'Мне обещали перезвонить и не перезвонили.'),
    turn(2, 'operator', 'Приношу извинения! Всё проверю.'),
  ],
};

export const demoCase: Case = { id: 'demo', title: 't', interactions: [call, chat] };

export const rubric: Rubric = {
  id: 'test',
  revision: 1,
  title: 't',
  passThreshold: 0.8,
  sections: [
    {
      id: 's1',
      title: 'Начало',
      weight: 1,
      questions: [
        {
          id: 'greeting',
          text: 'Представился?',
          guidance: 'g',
          kind: 'yes_no',
          values: [{ id: 'no', label: 'Нет' }, { id: 'yes', label: 'Да' }],
          adverse: 'no',
          weight: 1,
          appliesTo: ['call', 'chat'],
          multiSource: false,
          source: 'authored',
        },
        {
          id: 'empathy',
          text: 'Эмпатия',
          guidance: 'g',
          kind: 'scale',
          scale: { min: 1, max: 5, step: 1 },
          weight: 3,
          appliesTo: ['call', 'chat'],
          multiSource: false,
          source: 'authored',
        },
      ],
    },
    {
      id: 's2',
      title: 'Итог',
      weight: 3,
      questions: [
        {
          id: 'forbidden',
          text: 'Запрещённые фразы?',
          guidance: 'g',
          kind: 'yes_no',
          values: [{ id: 'yes', label: 'Да' }, { id: 'no', label: 'Нет' }],
          adverse: 'yes',
          autoFail: ['yes'],
          valueWithoutEvidence: 'no',
          weight: 1,
          appliesTo: ['call', 'chat', 'email'],
          multiSource: false,
          source: 'authored',
        },
        {
          id: 'documented',
          text: 'Задокументировано?',
          guidance: 'g',
          kind: 'yes_no',
          values: [{ id: 'no', label: 'Нет' }, { id: 'yes', label: 'Да' }],
          adverse: 'no',
          weight: 1,
          appliesTo: ['email'],
          multiSource: false,
          source: 'authored',
        },
      ],
    },
  ],
};
