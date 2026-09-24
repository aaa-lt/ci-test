/**
 * Models behind one OpenAI-compatible interface (FR-71). A model is picked by name on the
 * command line; everything provider-specific lives in this table.
 */
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import type { JSONValue, LanguageModel } from 'ai';

export type ModelSpec = {
  provider: 'yandex' | 'hf' | 'local';
  /** Provider model id; `{folder}` is replaced with YANDEX_FOLDER_ID. */
  model: string;
  /**
   * `json_schema`: the schema goes to the server as response_format and, where the
   * server supports it, constrains decoding. `json_object`: only "answer in JSON" is sent
   * and the schema is enforced by parsing with retry (FR-44).
   */
  structured: 'json_schema' | 'json_object';
  /** Extra request body fields, e.g. switching off Qwen3 thinking. */
  extraBody?: { [key: string]: JSONValue };
  /** Parallel calls the provider tolerates; HF free credits answer 402 to concurrent requests. */
  concurrency: number;
  note: string;
};

export const MODELS = {
  'yandex-qwen3-235b': {
    provider: 'yandex',
    model: 'gpt://{folder}/qwen3-235b-a22b-fp8',
    structured: 'json_schema',
    extraBody: { chat_template_kwargs: { enable_thinking: false } },
    concurrency: 3,
    note: 'external upper bound candidate, same family as the local models',
  },
  'yandex-alice': {
    provider: 'yandex',
    model: 'gpt://{folder}/aliceai-llm',
    structured: 'json_schema',
    concurrency: 3,
    note: 'external upper bound candidate, Russian-specialized',
  },
  'hf-qwen3-8b': {
    provider: 'hf',
    model: 'Qwen/Qwen3-8B:nscale',
    structured: 'json_schema',
    extraBody: { chat_template_kwargs: { enable_thinking: false } },
    concurrency: 1,
    note: 'base of T-lite-it-2.1, instruction-tuned by Alibaba; the comparison pair',
  },
  'hf-t-lite': {
    provider: 'hf',
    model: 't-tech/T-lite-it-2.1:featherless-ai',
    structured: 'json_schema',
    concurrency: 1,
    note: 'main working model of the concept; featherless decodes it with a wrong vocabulary (docs/llm-comparison.md)',
  },
  // llama-server on a CI runner or on the host; one GGUF per server, the id is informational.
  'local-t-lite': {
    provider: 'local',
    model: 'T-lite-it-2.1-Q4_K_M',
    structured: 'json_schema',
    concurrency: 1,
    note: 'main working model; non-thinking only, per the model card',
  },
  'local-qwen3-8b': {
    provider: 'local',
    model: 'Qwen3-8B-Q4_K_M',
    structured: 'json_schema',
    extraBody: { chat_template_kwargs: { enable_thinking: false } },
    concurrency: 1,
    note: 'the comparison pair, same quantization as local-t-lite',
  },
} as const satisfies Record<string, ModelSpec>;

export type ModelName = keyof typeof MODELS;

export function isModelName(name: string): name is ModelName {
  return name in MODELS;
}

function env(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set; see .env.example`);
  return value;
}

/** What a call needs: the model and the provider-specific body fields. Tests pass a mock. */
export type Llm = {
  model: LanguageModel;
  concurrency: number;
  providerOptions?: { [provider: string]: { [key: string]: JSONValue } };
};

export function toLlm(spec: ModelSpec): Llm {
  const model = languageModel(spec);
  const base = { model, concurrency: spec.concurrency };
  return spec.extraBody ? { ...base, providerOptions: { [spec.provider]: spec.extraBody } } : base;
}

function languageModel(spec: ModelSpec): LanguageModel {
  const supportsStructuredOutputs = spec.structured === 'json_schema';
  if (spec.provider === 'yandex') {
    const folder = env('YANDEX_FOLDER_ID');
    // Yandex expects "Api-Key <key>" rather than a bearer token, and the folder as a header.
    const provider = createOpenAICompatible({
      name: 'yandex',
      baseURL: 'https://ai.api.cloud.yandex.net/v1',
      headers: { Authorization: `Api-Key ${env('YANDEX_API_KEY')}`, 'OpenAI-Project': folder },
      supportsStructuredOutputs,
      includeUsage: true,
    });
    return provider(spec.model.replace('{folder}', folder));
  }
  if (spec.provider === 'local') {
    const provider = createOpenAICompatible({
      name: 'local',
      baseURL: process.env.LOCAL_LLM_URL ?? 'http://127.0.0.1:8080/v1',
      supportsStructuredOutputs,
      includeUsage: true,
    });
    return provider(spec.model);
  }
  const provider = createOpenAICompatible({
    name: 'hf',
    baseURL: 'https://router.huggingface.co/v1',
    apiKey: env('HF_TOKEN'),
    supportsStructuredOutputs,
    includeUsage: true,
  });
  return provider(spec.model);
}
