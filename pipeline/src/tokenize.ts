/**
 * The single tokenization rule shared by storage, prompts and quote verification (FR-45).
 * A token is a maximal run of letters and digits; everything else separates tokens.
 */

export type Token = {
  /** Normalized form used for matching: lower case, ё→е. */
  norm: string;
  /** Index of the source word for call turns, so a matched token maps back to a timestamp. */
  word: number;
};

const TOKEN = /[\p{L}\p{N}]+/gu;

export function normalizeToken(raw: string): string {
  return raw.toLowerCase().replaceAll('ё', 'е');
}

export function tokenize(text: string, word = 0): Token[] {
  return Array.from(text.matchAll(TOKEN), (m) => ({ norm: normalizeToken(m[0]), word }));
}

/** Tokens of a recognized turn, each tagged with the index of the ASR word it came from. */
export function tokenizeWords(words: readonly { text: string }[]): Token[] {
  return words.flatMap((w, i) => tokenize(w.text, i));
}
