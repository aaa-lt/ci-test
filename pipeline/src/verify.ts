/**
 * Quote verification against the stored transcript or text (FR-46).
 *
 * 1. Exact: the normalized quote is a contiguous token sequence of the cited turn.
 * 2. Fuzzy: otherwise, the most similar token window anywhere in the cited interaction,
 *    by normalized Levenshtein similarity, if it reaches the threshold. The address is
 *    then recomputed from the match, not taken from the model.
 *
 * Normalization is the shared tokenization (FR-45): case, punctuation and whitespace are
 * ignored, ё equals е. The ё rule goes beyond the wording of FR-46 and is a deliberate
 * addition: Russian texts use both spellings of the same word.
 */
import type { Interaction } from './domain.ts';
import { type Token, tokenize, tokenizeWords } from './tokenize.ts';

export type Verification =
  | {
      level: 'exact' | 'fuzzy';
      turn: number;
      /** Token range [from, to) within the turn. */
      from: number;
      to: number;
      similarity: number;
      /** Matched text as stored, normalized. */
      matched: string;
      /** Recording time of the first matched word, for calls with word timestamps (FR-63). */
      seek: number | null;
    }
  | { level: 'failed'; bestSimilarity: number };

export function verifyQuote(
  interaction: Interaction,
  turnIndex: number,
  quote: string,
  threshold: number,
): Verification {
  const q = tokenize(quote).map((t) => t.norm);
  if (q.length === 0) return { level: 'failed', bestSimilarity: 0 };

  const cited = interaction.turns.find((t) => t.index === turnIndex);
  if (cited) {
    const tokens = turnTokens(cited);
    const at = findSequence(tokens.map((t) => t.norm), q);
    if (at >= 0) return located('exact', cited, tokens, at, at + q.length, 1);
  }

  // Recognition splits and merges words ("кинопакет" / "кино пакет"), so window lengths
  // range around the quote length by a fifth of it, at least one token either way.
  const target = q.join(' ');
  const spread = Math.max(1, Math.round(q.length / 5));
  let best = { similarity: 0, turn: -1, from: 0, to: 0 };
  for (const turn of interaction.turns) {
    const norms = turnTokens(turn).map((t) => t.norm);
    for (let len = Math.max(1, q.length - spread); len <= Math.min(norms.length, q.length + spread); len++) {
      for (let from = 0; from + len <= norms.length; from++) {
        const sim = similarity(target, norms.slice(from, from + len).join(' '));
        if (sim > best.similarity) best = { similarity: sim, turn: turn.index, from, to: from + len };
      }
    }
  }
  const turn = interaction.turns.find((t) => t.index === best.turn);
  if (!turn || best.similarity < threshold) return { level: 'failed', bestSimilarity: round(best.similarity) };
  return located('fuzzy', turn, turnTokens(turn), best.from, best.to, best.similarity);
}

function turnTokens(turn: Interaction['turns'][number]): Token[] {
  return turn.words ? tokenizeWords(turn.words) : tokenize(turn.text);
}

function located(
  level: 'exact' | 'fuzzy',
  turn: Interaction['turns'][number],
  tokens: Token[],
  from: number,
  to: number,
  sim: number,
): Verification {
  const word = tokens[from]?.word;
  const seek = turn.words && word !== undefined ? (turn.words[word]?.start ?? null) : turn.start;
  return {
    level,
    turn: turn.index,
    from,
    to,
    similarity: round(sim),
    matched: tokens.slice(from, to).map((t) => t.norm).join(' '),
    seek,
  };
}

function findSequence(haystack: string[], needle: string[]): number {
  outer: for (let i = 0; i + needle.length <= haystack.length; i++) {
    for (let j = 0; j < needle.length; j++) if (haystack[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

/** 1 − Levenshtein distance / length of the longer string. */
export function similarity(a: string, b: string): number {
  if (a === b) return 1;
  const longer = Math.max(a.length, b.length);
  return longer === 0 ? 1 : 1 - levenshtein(a, b) / longer;
}

function levenshtein(a: string, b: string): number {
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur.push(Math.min((prev[j] ?? 0) + 1, (cur[j - 1] ?? 0) + 1, (prev[j - 1] ?? 0) + cost));
    }
    prev = cur;
  }
  return prev[b.length] ?? 0;
}

const round = (x: number) => Math.round(x * 1000) / 1000;
