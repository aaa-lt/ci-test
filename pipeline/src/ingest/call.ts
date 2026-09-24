/**
 * A recognized stereo call becomes one turn sequence (FR-15): segments of both channels
 * are ordered by start time; overlapping speech is kept as is and flagged on both turns;
 * at equal start times the operator goes first.
 */
import { z } from 'zod';
import { Role, type Turn } from '../domain.ts';

const Word = z.object({ text: z.string(), start: z.number(), end: z.number() });

/** Response of asr-service `/jobs/transcribe`, cached in results/transcripts/. */
export const Transcript = z.object({
  engine: z.string(),
  duration: z.number(),
  word_level: z.boolean(),
  channels: z.array(
    z.object({
      channel: z.number(),
      role: Role.nullable(),
      segments: z.array(
        z.object({ text: z.string(), start: z.number(), end: z.number(), words: z.array(Word).nullable() }),
      ),
    }),
  ),
  timings: z.record(z.string(), z.number()),
});
export type Transcript = z.infer<typeof Transcript>;

const ROLE_ORDER: Record<Role, number> = { operator: 0, client: 1 };

export function callTurns(transcript: Transcript): Turn[] {
  const pieces = transcript.channels.flatMap((ch) => {
    if (!ch.role) throw new Error(`channel ${ch.channel} has no role`);
    const role = ch.role;
    return ch.segments.map((s) => ({ role, ...s }));
  });
  pieces.sort((a, b) => a.start - b.start || ROLE_ORDER[a.role] - ROLE_ORDER[b.role]);

  return pieces.map((p, i) => ({
    index: i + 1,
    role: p.role,
    start: p.start,
    end: p.end,
    at: null,
    text: p.words ? p.words.map((w) => w.text).join(' ') : p.text,
    words: p.words,
    overlap: pieces.some((q) => q.role !== p.role && q.start < p.end && p.start < q.end),
  }));
}
