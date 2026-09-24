import { describe, expect, it } from 'vitest';
import { callTurns, type Transcript } from '../src/ingest/call.ts';

const seg = (text: string, start: number, end: number) => ({
  text,
  start,
  end,
  words: [{ text, start, end }],
});

const transcript: Transcript = {
  engine: 'test',
  duration: 10,
  word_level: true,
  timings: {},
  channels: [
    { channel: 0, role: 'operator', segments: [seg('здравствуйте', 0.5, 1.5), seg('отключаю', 4.0, 5.0), seg('так', 7.0, 7.5)] },
    { channel: 1, role: 'client', segments: [seg('добрый', 2.0, 3.0), seg('угу', 4.5, 4.8), seg('да', 7.0, 7.3)] },
  ],
};

describe('callTurns', () => {
  it('orders both channels by start time and numbers the turns', () => {
    const turns = callTurns(transcript);
    expect(turns.map((t) => [t.index, t.role, t.text])).toEqual([
      [1, 'operator', 'здравствуйте'],
      [2, 'client', 'добрый'],
      [3, 'operator', 'отключаю'],
      [4, 'client', 'угу'],
      [5, 'operator', 'так'],
      [6, 'client', 'да'],
    ]);
  });

  it('keeps overlapping speech whole and flags both sides', () => {
    const turns = callTurns(transcript);
    expect(turns.map((t) => t.overlap)).toEqual([false, false, true, true, true, true]);
    expect(turns[2]).toMatchObject({ start: 4.0, end: 5.0 });
  });

  it('refuses a channel without a role', () => {
    const broken = { ...transcript, channels: [{ ...transcript.channels[0]!, role: null }] };
    expect(() => callTurns(broken)).toThrow(/no role/);
  });
});
