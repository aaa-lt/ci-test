/**
 * One typed model call. Returns the parsed object or the reason it could not be had;
 * never throws for model misbehaviour, because a rejected answer is data (FR-75).
 */
import { APICallError, generateText, type ModelMessage, NoObjectGeneratedError, Output } from 'ai';
import type { z } from 'zod';
import type { Llm } from './models.ts';

export type CallResult<T> = {
  latencyMs: number;
  inputTokens: number | null;
  outputTokens: number | null;
  /** What was actually sent as response_format, read from the request body (check 4). */
  responseFormat: string | null;
  requestBody: unknown;
  rawText: string | null;
} & ({ ok: true; value: T } | { ok: false; reason: 'schema' | 'error'; error: string; status?: number });

/** Statuses worth waiting out: HF free credits answer 402 to bursts, providers 429 to rate limits. */
const BACKOFF_STATUS = new Set([402, 429]);
const BACKOFF_MS = [5_000, 15_000, 40_000];

export async function callStructured<T>(args: Parameters<typeof callOnce<T>>[0]): Promise<CallResult<T>> {
  for (const wait of [...BACKOFF_MS, null]) {
    const result = await callOnce(args);
    if (result.ok || result.reason !== 'error' || wait === null || !BACKOFF_STATUS.has(result.status ?? 0)) {
      return result;
    }
    await new Promise((resolve) => setTimeout(resolve, wait));
  }
  throw new Error('unreachable');
}

async function callOnce<T>(args: {
  llm: Llm;
  schema: z.ZodType<T>;
  name: string;
  instructions: string;
  messages: ModelMessage[];
}): Promise<CallResult<T>> {
  const started = performance.now();
  try {
    const result = await generateText({
      model: args.llm.model,
      instructions: args.instructions,
      messages: args.messages,
      output: Output.object({ schema: args.schema, name: args.name }),
      temperature: 0,
      // A runaway (whitespace loops under constrained decoding) stops here instead of at 6k.
      maxOutputTokens: 4000,
      maxRetries: 2,
      include: { requestBody: true, responseBody: true },
      ...(args.llm.providerOptions ? { providerOptions: args.llm.providerOptions } : {}),
    });
    const body = result.request.body;
    return {
      ok: true,
      value: result.output,
      latencyMs: Math.round(performance.now() - started),
      inputTokens: result.usage.inputTokens ?? null,
      outputTokens: result.usage.outputTokens ?? null,
      responseFormat: responseFormatOf(body),
      requestBody: body,
      rawText: result.text,
    };
  } catch (error) {
    const latencyMs = Math.round(performance.now() - started);
    if (NoObjectGeneratedError.isInstance(error)) {
      return {
        ok: false,
        reason: 'schema',
        error: String(error.cause ?? error.message),
        latencyMs,
        inputTokens: error.usage?.inputTokens ?? null,
        outputTokens: error.usage?.outputTokens ?? null,
        responseFormat: null,
        requestBody: null,
        rawText: error.text ?? null,
      };
    }
    return {
      ok: false,
      reason: 'error',
      error: error instanceof Error ? error.message : String(error),
      ...(APICallError.isInstance(error) && error.statusCode ? { status: error.statusCode } : {}),
      latencyMs,
      inputTokens: null,
      outputTokens: null,
      responseFormat: null,
      requestBody: null,
      rawText: null,
    };
  }
}

function responseFormatOf(body: unknown): string | null {
  const parsed: unknown = typeof body === 'string' ? JSON.parse(body) : body;
  if (typeof parsed !== 'object' || parsed === null || !('response_format' in parsed)) return null;
  const rf = parsed.response_format;
  if (typeof rf !== 'object' || rf === null || !('type' in rf)) return null;
  const strict = 'json_schema' in rf && typeof rf.json_schema === 'object' && rf.json_schema !== null
    && 'strict' in rf.json_schema ? `, strict=${String(rf.json_schema.strict)}` : '';
  return `${String(rf.type)}${strict}`;
}
