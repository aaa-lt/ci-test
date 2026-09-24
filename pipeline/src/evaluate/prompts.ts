/**
 * Prompts of scheme B2. The structure is described in the prompt as well as passed as a
 * schema: a constrained server never shows the schema to the model, and a json_object
 * server ignores it (see docs/llm-comparison.md, check 4).
 */
import type { Case, Interaction, Turn } from '../domain.ts';
import type { Question } from '../rubric.ts';

/** Bumped on any wording change; stored with every call and result (FR-74, FR-75). */
export const PROMPT_VERSION = 'b2-v2';

const CHANNEL_NAME = { call: 'телефонный звонок', chat: 'переписка в чате', email: 'письмо' } as const;
const ROLE_NAME = { operator: 'Оператор', client: 'Клиент' } as const;

export const EXTRACT_INSTRUCTIONS = `Ты специалист по контролю качества обслуживания в контакт-центре.
Тебе дают одно взаимодействие клиента с компанией и вопросы оценочной рубрики. Отвечать на вопросы не нужно. Нужно найти в этом взаимодействии фрагменты, по которым на вопросы можно ответить: и подтверждающие, и опровергающие.

Правила цитирования:
- Цитата — дословный фрагмент одной реплики длиной от двух до тридцати слов. Копируй его без изменений, даже если в тексте ошибки распознавания речи или нет знаков препинания.
- Не пересказывай, не исправляй, не соединяй фрагменты разных реплик.
- В поле turn укажи номер реплики из квадратных скобок, из которой взята цитата.
- В поле note кратко напиши, что показывает фрагмент применительно к вопросу.
- Если во взаимодействии нет ничего, относящегося к вопросу, верни пустой список evidence.

Для каждого вопроса сначала напиши в reasoning одно–три предложения о том, что в этом взаимодействии относится к вопросу, затем перечисли фрагменты.

Ответ — один JSON-объект. Ключи — идентификаторы вопросов, значения — объекты вида
{"reasoning": "...", "evidence": [{"turn": 3, "quote": "...", "note": "..."}]}.`;

export const AGGREGATE_INSTRUCTIONS = `Ты специалист по контролю качества обслуживания в контакт-центре.
Тебе дают вопросы оценочной рубрики и доказательства: дословные фрагменты, извлечённые из взаимодействий одного обращения клиента. Взаимодействия перечислены в порядке времени. Одно обращение может включать звонок, чат и письма, их могли вести разные операторы.

Ответь на каждый вопрос, опираясь только на приведённые доказательства.
- Учитывай время: обещание, данное в раннем взаимодействии, проверяется по более поздним; итог обращения определяется последним по времени.
- В поле evidence перечисли идентификаторы доказательств (E1, E2, …), на которых основан ответ. Ответ без доказательств не принимается, кроме случаев, прямо указанных у вопроса.
- Если доказательств недостаточно, чтобы ответить уверенно, ответь "no_answer". Отказ лучше выдуманного ответа.
- Сначала напиши в reasoning одно–три предложения, затем выбери ответ.

Ответ — один JSON-объект. Ключи — идентификаторы вопросов, значения — объекты вида
{"reasoning": "...", "answer": "<допустимое значение или no_answer>", "evidence": ["E1"]}.`;

export function renderInteraction(i: Interaction): string {
  const head = `Взаимодействие ${i.id}: ${CHANNEL_NAME[i.channel]}${i.startedAt ? `, ${i.startedAt}` : ''}, оператор ${i.operator}.`;
  const note = i.channel === 'call' ? '\nТекст звонка получен распознаванием речи и может содержать ошибки.' : '';
  return `${head}${note}\n\n${i.turns.map(renderTurn).join('\n')}`;
}

function renderTurn(t: Turn): string {
  const time = t.start !== null ? ` (${clock(t.start)})` : t.at ? ` (${t.at})` : '';
  return `[${t.index}] ${ROLE_NAME[t.role]}${time}: ${t.text}`;
}

function clock(seconds: number): string {
  const s = Math.floor(seconds);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

export function renderQuestion(q: Question, withValues = true): string {
  const values =
    q.kind === 'scale'
      ? `целое число от ${q.scale.min} до ${q.scale.max} (числом, без кавычек)`
      : q.values.map((v) => `"${v.id}" (${v.label})`).join(', ');
  const lines = [`- ${q.id}: ${q.text}`, `  Как оценивать: ${q.guidance}`];
  if (withValues) lines.push(`  Допустимые ответы: ${values}.`);
  if (q.valueWithoutEvidence) lines.push(`  Ответ "${q.valueWithoutEvidence}" допустим без доказательств.`);
  if (q.multiSource) lines.push('  Для ответа может понадобиться несколько взаимодействий.');
  return lines.join('\n');
}

export function extractPrompt(i: Interaction, questions: Question[]): string {
  return `${renderInteraction(i)}\n\nВопросы:\n${questions.map((q) => renderQuestion(q)).join('\n')}`;
}

export type EvidenceView = {
  id: string;
  interaction: Interaction;
  turn: Turn;
  quote: string;
  note: string;
  questionId: string;
};

export function aggregatePrompt(c: Case, questions: Question[], evidence: EvidenceView[]): string {
  const timeline = c.interactions
    .map((i) => `- ${i.id}: ${CHANNEL_NAME[i.channel]}${i.startedAt ? `, ${i.startedAt}` : ''}, оператор ${i.operator}`)
    .join('\n');
  const blocks = questions.map((q) => {
    const own = evidence.filter((e) => e.questionId === q.id);
    const items = own.length
      ? own.map((e) => `  ${e.id} [${e.interaction.id}, реплика ${e.turn.index}, ${ROLE_NAME[e.turn.role].toLowerCase()}]: «${e.quote}» — ${e.note}`).join('\n')
      : '  доказательств не найдено';
    return `${renderQuestion(q)}\n  Доказательства:\n${items}`;
  });
  return `Обращение ${c.id}. Взаимодействия по порядку времени:\n${timeline}\n\nВопросы и доказательства:\n\n${blocks.join('\n\n')}`;
}

