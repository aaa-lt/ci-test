import { describe, expect, it } from 'vitest';
import { similarity, verifyQuote } from '../src/verify.ts';
import { call, chat } from './fixtures.ts';

describe('verifyQuote', () => {
  it('matches exactly regardless of case, punctuation and ё', () => {
    const v = verifyQuote(chat, 2, 'приношу извинения, всё проверю', 0.8);
    expect(v).toMatchObject({ level: 'exact', turn: 2, from: 0, to: 4, seek: null });
  });

  it('maps a call quote to the timestamp of its first word', () => {
    const v = verifyQuote(call, 1, 'Меня зовут Анна', 0.8);
    expect(v).toMatchObject({ level: 'exact', turn: 1, from: 2, to: 5, seek: 1.6 });
  });

  it('finds a quote cited with the wrong turn and readdresses it', () => {
    const v = verifyQuote(call, 1, 'я вам перезвоню сегодня', 0.8);
    expect(v).toMatchObject({ level: 'fuzzy', turn: 3, from: 0, to: 4, similarity: 1, seek: 5.0 });
  });

  it('accepts a word split by recognition on the fuzzy level', () => {
    const v = verifyQuote(call, 2, 'списывают за кинопакет плюс', 0.8);
    expect(v.level).toBe('fuzzy');
    if (v.level === 'fuzzy') expect(v).toMatchObject({ turn: 2, from: 0, to: 5, seek: 3.0 });
  });

  it('rejects a fabricated quote', () => {
    const v = verifyQuote(call, 3, 'деньги вернут в течение трёх дней', 0.8);
    expect(v.level).toBe('failed');
  });

  it('rejects an empty quote', () => {
    expect(verifyQuote(chat, 1, ' — ', 0.8).level).toBe('failed');
  });
});

describe('similarity', () => {
  it('is 1 for equal strings and falls with edits', () => {
    expect(similarity('abc', 'abc')).toBe(1);
    expect(similarity('abcd', 'abed')).toBe(0.75);
    expect(similarity('', '')).toBe(1);
  });
});
