/**
 * Loading a case from data/cases/<id>/ into the unified representation. Transcripts and
 * cleaned e-mails are produced once by asr-service and reused from results/ (FR-25).
 */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { z } from 'zod';
import { type Case, CaseManifest, type Interaction, Role } from '../domain.ts';
import { callTurns, Transcript } from './call.ts';
import { ChatExport, chatTurns, CleanEmail, emailTurns } from './text.ts';

export const ROOT = new URL('../../../', import.meta.url).pathname;
const CASES = join(ROOT, 'data', 'cases');
const TRANSCRIPTS = join(ROOT, 'results', 'transcripts');
const EMAILS = join(ROOT, 'results', 'emails');

export async function readJson<T extends z.ZodType>(path: string, schema: T): Promise<z.infer<T>> {
  return schema.parse(JSON.parse(await readFile(path, 'utf8')));
}

export async function loadCase(id: string, engine: string): Promise<Case> {
  const dir = join(CASES, id);
  const manifest = await readJson(join(dir, 'case.json'), CaseManifest);
  const interactions: Interaction[] = [];
  for (const item of manifest.interactions) {
    const base = { id: item.id, channel: item.channel, operator: item.operator };
    switch (item.channel) {
      case 'call': {
        const transcript =
          engine === REFERENCE ? await referenceTranscript(item.call) : await asrTranscript(item.call, engine);
        const roles = [item.stereo.left, item.stereo.right];
        transcript.channels.forEach((ch, i) => (ch.role = ch.role ?? roles[i] ?? null));
        interactions.push({ ...base, startedAt: item.startedAt, turns: callTurns(transcript) });
        break;
      }
      case 'chat': {
        const chat = await readJson(join(dir, item.file), ChatExport);
        interactions.push({ ...base, startedAt: chat.messages[0]?.at ?? null, turns: chatTurns(chat) });
        break;
      }
      case 'email': {
        const mail = await cleanEmail(id, item.id, join(dir, item.file));
        interactions.push({ ...base, startedAt: mail.date, turns: emailTurns(mail, item.author) });
        break;
      }
    }
  }
  return { id: manifest.id, title: manifest.title, interactions };
}

/**
 * Pseudo-engine: the script text itself, as if recognition were perfect. Running the
 * same case on "reference" and on a real engine separates recognition errors from model
 * errors (share of rejections caused by recognition, section 5 of the concept).
 */
export const REFERENCE = 'reference';

const Reference = z.object({
  channels: z.array(Role),
  utterances: z.array(z.object({ role: Role, text: z.string(), start: z.number(), end: z.number() })),
});

async function referenceTranscript(call: string): Promise<Transcript> {
  const ref = await readJson(join(ROOT, 'data', 'audio', `${call}.reference.json`), Reference);
  return {
    engine: REFERENCE,
    duration: Math.max(0, ...ref.utterances.map((u) => u.end)),
    word_level: false,
    channels: ref.channels.map((role, channel) => ({
      channel,
      role,
      segments: ref.utterances
        .filter((u) => u.role === role)
        .map((u) => ({ text: u.text, start: u.start, end: u.end, words: null })),
    })),
    timings: {},
  };
}

async function asrTranscript(call: string, engine: string): Promise<Transcript> {
  const path = join(TRANSCRIPTS, `${call}.${engine}.json`);
  if (!existsSync(path)) throw new Error(`no transcript ${path}; run tools/asr.py transcribe ${engine}`);
  return readJson(path, Transcript);
}

async function cleanEmail(caseId: string, interactionId: string, file: string): Promise<CleanEmail> {
  const cached = join(EMAILS, `${caseId}.${interactionId}.json`);
  if (existsSync(cached)) return readJson(cached, CleanEmail);
  const url = process.env.ASR_LOCAL_URL ?? 'http://127.0.0.1:8000';
  const resp = await fetch(`${url}/email/clean`, { method: 'POST', body: await readFile(file) });
  if (!resp.ok) throw new Error(`email/clean ${resp.status}: ${await resp.text()}`);
  const mail = CleanEmail.parse(await resp.json());
  await mkdir(EMAILS, { recursive: true });
  await writeFile(cached, JSON.stringify(mail, null, 2));
  return mail;
}
