/** Call protocol (FR-75): one JSON line per model call, enough to recompute every metric. */
import { existsSync } from 'node:fs';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

export type Outcome = 'accepted' | 'rejected_schema' | 'rejected_quote' | 'rejected_no_evidence' | 'error';

export type ProtocolEntry = {
  runId: string;
  caseId: string;
  scheme: string;
  model: string;
  engine: string;
  promptVersion: string;
  rubricRevision: number;
  stage: 'extract' | 'aggregate';
  interactionId: string | null;
  questionIds: string[];
  attempt: number;
  outcome: Outcome;
  /** Verification levels of the quotes returned by this call. */
  quotes: { exact: number; fuzzy: number; failed: number; unverified: number };
  /** Quotes that did not verify, with the best similarity found: the raw material for error analysis. */
  failedQuotes: { questionId: string; turn: number; quote: string; bestSimilarity: number }[];
  inputTokens: number | null;
  outputTokens: number | null;
  latencyMs: number;
  responseFormat: string | null;
  error: string | null;
};

/** The Yandex folder id sits in model URIs; results are published, the id is not needed there. */
function redact(text: string): string {
  const folder = process.env.YANDEX_FOLDER_ID;
  return folder ? text.replaceAll(folder, '{folder}') : text;
}

export class Protocol {
  constructor(readonly path: string) {}

  async record(entry: ProtocolEntry): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true });
    await appendFile(this.path, `${JSON.stringify({ at: new Date().toISOString(), ...entry })}\n`);
  }

  /** Keep the first raw request body per stage: the evidence for what reaches the server. */
  async sample(stage: string, body: unknown): Promise<void> {
    const path = join(dirname(this.path), 'requests', `${stage}.json`);
    if (body === null || body === undefined || existsSync(path)) return;
    await mkdir(dirname(path), { recursive: true });
    const parsed: unknown = typeof body === 'string' ? JSON.parse(body) : body;
    await writeFile(path, redact(JSON.stringify(parsed, null, 2)));
  }
}
