/** Chat exports and cleaned e-mails as turns. */
import { z } from 'zod';
import { Role, type Turn } from '../domain.ts';

export const ChatExport = z.object({
  messages: z.array(z.object({ at: z.string(), role: Role, text: z.string() })),
});

export function chatTurns(chat: z.infer<typeof ChatExport>): Turn[] {
  return chat.messages.map((m, i) => ({
    index: i + 1,
    role: m.role,
    start: null,
    end: null,
    at: m.at,
    text: m.text,
    words: null,
    overlap: false,
  }));
}

/** Response of asr-service `/email/clean`. */
export const CleanEmail = z.object({
  sender: z.string(),
  sender_name: z.string(),
  to: z.string(),
  subject: z.string(),
  date: z.string().nullable(),
  body: z.string(),
  removed: z.record(z.string(), z.string()),
});
export type CleanEmail = z.infer<typeof CleanEmail>;

/** One e-mail is one turn: the new text of the message, without quotes and signature (FR-16). */
export function emailTurns(mail: CleanEmail, author: Role): Turn[] {
  return [
    { index: 1, role: author, start: null, end: null, at: mail.date, text: mail.body, words: null, overlap: false },
  ];
}
