/**
 * The unified representation (FR-17): every channel becomes a sequence of turns with a
 * role, an operator and a time. The evaluation layer sees only these types and never
 * knows which channel a turn came from beyond the `channel` label (NFR-22).
 */
import { z } from 'zod';

export const Role = z.enum(['operator', 'client']);
export type Role = z.infer<typeof Role>;

export const Channel = z.enum(['call', 'chat', 'email']);
export type Channel = z.infer<typeof Channel>;

export type TimedWord = { text: string; start: number; end: number };

export type Turn = {
  /** 1-based position inside the interaction; this is the address the model cites. */
  index: number;
  role: Role;
  /** Seconds from the start of the recording, for calls. */
  start: number | null;
  end: number | null;
  /** Wall-clock time, for chat messages and e-mails. */
  at: string | null;
  text: string;
  /** Recognized words with timestamps, for calls; the turn text is these words joined. */
  words: TimedWord[] | null;
  /** Speech of the other party overlaps this turn (FR-15). */
  overlap: boolean;
};

export type Interaction = {
  id: string;
  channel: Channel;
  operator: string;
  startedAt: string | null;
  turns: Turn[];
};

export type Case = {
  id: string;
  title: string;
  interactions: Interaction[];
};

/** data/cases/<id>/case.json */
export const CaseManifest = z.object({
  id: z.string(),
  title: z.string(),
  interactions: z.array(
    z.discriminatedUnion('channel', [
      z.object({
        id: z.string(),
        channel: z.literal('call'),
        operator: z.string(),
        startedAt: z.string(),
        call: z.string(),
        stereo: z.object({ left: Role, right: Role }),
      }),
      z.object({ id: z.string(), channel: z.literal('chat'), operator: z.string(), file: z.string() }),
      z.object({
        id: z.string(),
        channel: z.literal('email'),
        operator: z.string(),
        file: z.string(),
        author: Role,
      }),
    ]),
  ),
});
export type CaseManifest = z.infer<typeof CaseManifest>;
