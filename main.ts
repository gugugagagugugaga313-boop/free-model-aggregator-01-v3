// free-model-aggregator-01 — a routing gateway over ten simulated free-tier providers.
//
// The trial hands this program a brief (three JSON documents found by a schema member: the
// provider directory, the quality table, the workload manifest) and an env URL. It probes
// every provider's catalog route, registers one catalog, then answers the client's twenty
// requests, one at a time, each with the strongest free model that meets the request's
// requirements and still has budget on its provider. When a provider refuses, the request
// is answered by another model; a refusal is never the answer.
//
// Architecture: every request is first normalised into one canonical shape (system text,
// turns, tools, stops, caps, JSON mode, streaming), whatever wire the client used. That
// shape is then rendered into the chosen provider's wire (OpenAI, Anthropic, or Gemini),
// and the provider's answer is normalised into one canonical answer (text, token counts,
// stop family) and rendered back into the client's wire. Same-wire pairs skip the round
// trip and pass the provider's bytes through untouched. Nothing here reads the wall clock
// and nothing is random: the same requests and the same provider answers always produce
// the same bytes, so a replay reproduces the trial exactly.
//
// Routing: each provider carries a tally of attempts and transient refusals. A provider's
// budget for the trial is drawn uniformly from a small published window, it refuses a
// couple of early calls on a hidden schedule, and exactly one provider in ten goes dark
// partway through; the hazard estimate prices all three risks from what has been observed
// so far. Full-weight requests take the highest-rated model outright; lighter requests
// take the highest rating minus a hazard penalty, which spreads calls away from providers
// that are about to run dry. A transient refusal earns the same provider one immediate
// retry — the hidden failure schedule is sparse, so the second attempt usually lands —
// while a message that says the budget is spent or the provider is out retires that
// provider for the rest of the trial. Requests that need JSON mode never go to a wire
// that cannot carry it, because such a provider echoes JSON mode as absent and the
// request would fail its echo check.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/* ------------------------------------------------------------------ */
/* Small JSON helpers                                                  */
/* ------------------------------------------------------------------ */

/** Any JSON object: the one shape every parser in this file returns. */
type JsonObj = Record<string, unknown>;

/** True when the value is a JSON object (not null, not an array). */
function isObj(value: unknown): value is JsonObj {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** The value as a record, or an empty record when it is anything else. */
function asObj(value: unknown): JsonObj {
  return isObj(value) ? value : {};
}

/** A string field, or the fallback when the field is missing or not a string. */
function strField(host: JsonObj, key: string, fallback = ''): string {
  const found = host[key];
  return typeof found === 'string' ? found : fallback;
}

/** A finite number field, or null when the field is missing or not a number. */
function numField(host: JsonObj, key: string): number | null {
  const found = host[key];
  return typeof found === 'number' && Number.isFinite(found) ? found : null;
}

/** A boolean field, or null when the field is missing or not a boolean. */
function boolField(host: JsonObj, key: string): boolean | null {
  const found = host[key];
  return typeof found === 'boolean' ? found : null;
}

/** An array field, or an empty array when the field is missing or not an array. */
function arrField(host: JsonObj, key: string): unknown[] {
  const found = host[key];
  return Array.isArray(found) ? found : [];
}

/** The text carried by a string or by the text blocks of a block array, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const pieces: string[] = [];
  for (const block of content) {
    if (isObj(block) && block.type === 'text' && typeof block.text === 'string') pieces.push(block.text);
  }
  return pieces.join('\n');
}

/** A stop value as a list: a lone string becomes one entry, an array keeps its strings. */
function stopsOf(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((entry): entry is string => typeof entry === 'string');
}

/** Clamp a number into a closed interval. */
function clampTo(value: number, low: number, high: number): number {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

/* ------------------------------------------------------------------ */
/* The brief: three schema-tagged documents inside prose               */
/* ------------------------------------------------------------------ */

type WireKind = 'openai' | 'anthropic' | 'gemini';
type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** One provider as the directory lists it. `base` is a path (or an absolute URL). */
interface DirectoryProvider {
  id: string;
  name: string;
  wire: WireKind;
  auth: AuthKind;
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the quality table: the sealed truth about a model, where a row exists. */
interface QualityRow {
  model: string;
  provider: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
interface Needs {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One row of the workload manifest: the requests arrive in this order. */
interface ManifestRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Needs;
}

/** The brief, parsed. `problem` is set only when the brief could not be read at all. */
interface BriefData {
  providers: DirectoryProvider[];
  ratings: QualityRow[];
  manifest: ManifestRow[];
  problem: string | null;
}

const DIRECTORY_SCHEMA = 'free-model-aggregator-01/providers/v1';
const QUALITY_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/**
 * The index of the brace closing the object that opens at `open`, or -1. String
 * contents are skipped, including escaped quotes, so braces inside strings never
 * disturb the depth count.
 */
function closingBrace(text: string, open: number): number {
  let depth = 0;
  let quoted = false;
  for (let at = open; at < text.length; at += 1) {
    const ch = text[at];
    if (quoted) {
      if (ch === '\\') at += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return at;
    }
  }
  return -1;
}

/** Every top-level JSON object in the text that carries a string `schema`, in order. */
function schemaDocuments(text: string): JsonObj[] {
  const found: JsonObj[] = [];
  let scan = text.indexOf('{');
  while (scan !== -1) {
    const end = closingBrace(text, scan);
    if (end !== -1) {
      try {
        const parsed: unknown = JSON.parse(text.slice(scan, end + 1));
        if (isObj(parsed) && typeof parsed.schema === 'string') found.push(parsed);
      } catch {
        // Not a document on its own; the scan moves on to the next brace.
      }
    }
    scan = text.indexOf('{', scan + 1);
  }
  return found;
}

function wireKindOf(value: unknown): WireKind | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authKindOf(value: unknown): AuthKind {
  if (value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

/** One directory row, or null when the row names no usable provider. */
function directoryRowOf(value: unknown): DirectoryProvider | null {
  if (!isObj(value)) return null;
  const id = strField(value, 'id');
  const wire = wireKindOf(value.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path` first; an absolute base may hide under the other names.
  const base =
    strField(value, 'path') !== ''
      ? strField(value, 'path')
      : strField(value, 'base_path') !== ''
        ? strField(value, 'base_path')
        : strField(value, 'base_url') !== ''
          ? strField(value, 'base_url')
          : strField(value, 'base');
  const name = strField(value, 'name');
  return {
    id,
    name: name !== '' ? name : id,
    wire,
    auth: authKindOf(value.auth),
    base,
    keyEnv: strField(value, 'key_env'),
    key: strField(value, 'key'),
  };
}

/** One quality-table row, or null when the row names no model. */
function qualityRowOf(value: unknown): QualityRow | null {
  if (!isObj(value)) return null;
  const model = strField(value, 'model');
  const provider = strField(value, 'provider');
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: numField(value, 'rating_milli') ?? 0,
    context: numField(value, 'context_length'),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

/** One manifest row, or null when the row names no request. */
function manifestRowOf(value: unknown): ManifestRow | null {
  if (!isObj(value)) return null;
  const id = strField(value, 'id');
  if (id === '') return null;
  const requires = asObj(value.requires);
  const needs: Needs = {
    tools: requires.tools === true,
    vision: requires.vision === true,
    json: requires.json === true,
  };
  const minContext = numField(requires, 'min_context');
  if (minContext !== null) needs.minContext = minContext;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: needs,
  };
}

/** Read the brief file and pull its three documents out of the surrounding prose. */
function parseBrief(path: string): BriefData {
  const empty: BriefData = { providers: [], ratings: [], manifest: [], problem: null };
  if (path === '') return { ...empty, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directoryDoc = docs.find((doc) => doc.schema === DIRECTORY_SCHEMA && Array.isArray(doc.providers));
  if (directoryDoc === undefined) {
    return { ...empty, problem: `the brief at ${path} holds no ${DIRECTORY_SCHEMA} document` };
  }
  const providers: DirectoryProvider[] = [];
  for (const raw of arrField(directoryDoc, 'providers')) {
    const row = directoryRowOf(raw);
    if (row !== null) providers.push(row);
  }
  const ratings: QualityRow[] = [];
  const qualityDoc = docs.find((doc) => doc.schema === QUALITY_SCHEMA && Array.isArray(doc.ratings));
  if (qualityDoc !== undefined) {
    for (const raw of arrField(qualityDoc, 'ratings')) {
      const row = qualityRowOf(raw);
      if (row !== null) ratings.push(row);
    }
  }
  const manifest: ManifestRow[] = [];
  const workloadDoc = docs.find((doc) => doc.schema === WORKLOAD_SCHEMA && Array.isArray(doc.manifest));
  if (workloadDoc !== undefined) {
    for (const raw of arrField(workloadDoc, 'manifest')) {
      const row = manifestRowOf(raw);
      if (row !== null) manifest.push(row);
    }
  }
  return { providers, ratings, manifest, problem: null };
}

/* ------------------------------------------------------------------ */
/* The fleet: probing catalog routes and assembling the catalog        */
/* ------------------------------------------------------------------ */

/** One model as a provider's own catalog route listed it. Nulls mean "unsaid". */
interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What one provider's catalog probe learned. A failed probe carries no models. */
interface ProbeOutcome {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  problem: string | null;
}

/** One catalog entry, as `--list` prints it and the register body carries it. */
interface CatalogEntry {
  id: string;
  base_url: string;
  key_env: string;
  wire: WireKind;
  models: { id: string; free: boolean; context_length: number | null }[];
}

/** Join a directory base to the env URL; an absolute base stands on its own. */
function joinToEnv(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return root;
  return `${root}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, under its own auth scheme. */
function headersForKey(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/**
 * Whether a listing's pricing object prices the model at zero: true when every
 * price it names is zero, null when it names no prices at all, false otherwise.
 */
function freeFromPricing(entry: JsonObj): boolean | null {
  const pricing = asObj(entry.pricing);
  if (!isObj(entry.pricing)) return null;
  const amounts: number[] = [];
  for (const value of Object.values(pricing)) {
    if (typeof value === 'number') amounts.push(value);
    else if (typeof value === 'string' && value.trim() !== '') amounts.push(Number(value));
  }
  if (amounts.length === 0) return null;
  return amounts.every((amount) => Number.isFinite(amount) && amount === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling, or null. */
function listingModelOf(wire: WireKind, value: unknown): ListedModel | null {
  if (!isObj(value)) return null;
  const rawId = wire === 'gemini' ? strField(value, 'name').replace(/^models\//, '') : strField(value, 'id');
  if (rawId === '') return null;
  const display =
    strField(value, 'display_name') !== ''
      ? strField(value, 'display_name')
      : strField(value, 'displayName') !== ''
        ? strField(value, 'displayName')
        : rawId;
  const context = numField(value, 'context_length') ?? numField(value, 'context_window') ?? numField(value, 'inputTokenLimit');
  return {
    id: rawId,
    displayName: display,
    free: boolField(value, 'free') ?? freeFromPricing(value),
    context,
    tools: boolField(value, 'tools'),
    vision: boolField(value, 'vision'),
    json: boolField(value, 'json'),
  };
}

/** The models one catalog answer listed, in its wire's shape. */
function listingModelsOf(wire: WireKind, body: unknown): ListedModel[] {
  if (!isObj(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  const models: ListedModel[] = [];
  for (const raw of list) {
    const model = listingModelOf(wire, raw);
    if (model !== null) models.push(model);
  }
  return models;
}

const PROBE_BUDGET_MS = 5000;

/**
 * Probe one provider's catalog route in its own wire. Any failure — refused
 * connection, non-2xx, a body that is not JSON — leaves the provider out of the
 * catalog and off the routing table; the trial goes on without it.
 */
async function probeOne(envUrl: string, provider: DirectoryProvider): Promise<ProbeOutcome> {
  const root = joinToEnv(envUrl, provider.base);
  const target = provider.wire === 'gemini' ? `${root}/v1beta/models` : `${root}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...headersForKey(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const failed = (status: number, problem: string): ProbeOutcome => ({ provider, ok: false, status, models: [], problem });
  try {
    const res = await fetch(target, { headers, signal: AbortSignal.timeout(PROBE_BUDGET_MS) });
    if (!res.ok) return failed(res.status, `GET ${target} answered HTTP ${res.status}`);
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: listingModelsOf(provider.wire, body), problem: null };
  } catch (err) {
    return failed(0, `GET ${target} failed: ${(err as Error).message}`);
  }
}

/** The quality-table row for one provider's model, when the table has one. */
function qualityLookup(ratings: QualityRow[], providerId: string, modelId: string): QualityRow | null {
  return ratings.find((row) => row.provider === providerId && row.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where one exists (it is the
 * sealed truth), the listing's own word where it does not, paid when neither says.
 */
function freeStanding(model: ListedModel, row: QualityRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row first, the listing's fields second. */
interface ModelCaps {
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capsStanding(model: ListedModel, row: QualityRow | null): ModelCaps {
  return {
    context: row?.context ?? model.context ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. Unknown never meets. */
function capsMeet(needs: Needs, caps: ModelCaps): boolean {
  if (needs.tools === true && caps.tools !== true) return false;
  if (needs.vision === true && caps.vision !== true) return false;
  if (needs.json === true && caps.json !== true) return false;
  if (needs.minContext !== undefined && (caps.context === null || caps.context < needs.minContext)) return false;
  return true;
}

function byModelId(a: ListedModel, b: ListedModel): number {
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/**
 * The one catalog: every provider that answered, in the brief's order, each with
 * its models sorted by id and deduplicated. The base_url stays the directory's own
 * path, never joined to the env URL — the env's origin differs between a live pass
 * and its replay, and the catalog is part of the state a replay must reproduce.
 */
function assembleCatalog(probes: readonly ProbeOutcome[], ratings: QualityRow[]): { providers: CatalogEntry[] } {
  const entries: CatalogEntry[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogEntry['models'] = [];
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = qualityLookup(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: freeStanding(model, row),
        context_length: row?.context ?? model.context ?? null,
      });
    }
    entries.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers: entries };
}

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  displayName: string;
}

/** The free models across the providers that answered, in the brief's provider order. */
function servedList(probes: readonly ProbeOutcome[], ratings: QualityRow[]): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byModelId)) {
      const row = qualityLookup(ratings, probe.provider.id, model.id);
      if (!freeStanding(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, displayName: model.displayName });
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* The canonical request: one shape for both client wires              */
/*                                                                     */
/* The provider echoes exactly what it parsed — system text, user      */
/* text, tool names, max tokens, stops, the image, JSON mode — and     */
/* the client compares that echo with what it sent. The canonical      */
/* shape therefore keeps every echo-bearing field verbatim, in order,  */
/* and every renderer below is written against the provider parsers'   */
/* own rules: which message a parser reads last, which roles it        */
/* ignores, which content kinds carry text.                            */
/* ------------------------------------------------------------------ */

/** An image, either inline base64 data or a remote URL. */
interface ImageRef {
  mode: 'data' | 'url';
  media: string;
  data: string;
  url: string;
}

/** One assistant tool call: the id, the function name, the arguments as JSON text. */
interface CallStep {
  id: string;
  name: string;
  args: string;
}

/** One tool result: the call it answers and its content as text. */
interface ResultStep {
  id: string;
  text: string;
}

/** One segment of a user turn: prose, an image, or a tool result. */
type TurnSegment =
  | { kind: 'text'; text: string }
  | { kind: 'image'; image: ImageRef }
  | { kind: 'result'; result: ResultStep };

/** One conversation turn. User turns carry segments; assistant turns carry prose and calls. */
interface Turn {
  role: 'user' | 'assistant' | 'tool';
  segments: TurnSegment[];
  prose: string;
  calls: CallStep[];
  toolId: string;
  toolText: string;
}

/** One tool definition: the name the echo compares, plus its contract. */
interface ToolDef {
  name: string;
  description: string;
  schema: JsonObj;
}

type ChoiceKind = 'auto' | 'required' | { name: string } | null;

/** A client request, normalised. `jsonMode` is true only on the OpenAI wire. */
interface CanonRequest {
  system: string;
  turns: Turn[];
  tools: ToolDef[];
  choice: ChoiceKind;
  stops: string[];
  maxTokens: number | null;
  temperature: number | null;
  jsonMode: boolean;
  jsonType: string;
  stream: boolean;
  clientModel: string;
}

function blankTurn(role: Turn['role']): Turn {
  return { role, segments: [], prose: '', calls: [], toolId: '', toolText: '' };
}

/** The prose of a user turn: its text segments joined with newlines, in order. */
function turnProse(turn: Turn): string {
  const pieces: string[] = [];
  for (const segment of turn.segments) {
    if (segment.kind === 'text') pieces.push(segment.text);
  }
  return pieces.join('\n');
}

/** A data URL split into its media type and base64 payload, or null when it is not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** An OpenAI image_url part as an image reference. */
function imageFromUrl(url: string): ImageRef {
  const data = dataUrlOf(url);
  if (data !== null) return { mode: 'data', media: data.media, data: data.data, url: '' };
  return { mode: 'url', media: '', data: '', url };
}

/** A tool result's content as text: its text blocks, or its JSON when it holds something else. */
function resultTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions body as a canonical request. */
function openAIToCanon(body: JsonObj): CanonRequest {
  const systemPieces: string[] = [];
  const turns: Turn[] = [];
  for (const raw of arrField(body, 'messages')) {
    if (!isObj(raw)) continue;
    const role = strField(raw, 'role');
    if (role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemPieces.push(text);
      continue;
    }
    if (role === 'assistant') {
      const turn = blankTurn('assistant');
      turn.prose = blocksText(raw.content);
      for (const call of arrField(raw, 'tool_calls')) {
        if (!isObj(call) || !isObj(call.function)) continue;
        const args = call.function.arguments;
        turn.calls.push({
          id: strField(call, 'id'),
          name: strField(call.function, 'name'),
          args: typeof args === 'string' ? args : JSON.stringify(args ?? {}),
        });
      }
      turns.push(turn);
      continue;
    }
    if (role === 'tool') {
      const turn = blankTurn('tool');
      turn.toolId = strField(raw, 'tool_call_id');
      turn.toolText = typeof raw.content === 'string' ? raw.content : blocksText(raw.content);
      turns.push(turn);
      continue;
    }
    // Anything else travels as a user turn: the parsers only read user turns for prose.
    const turn = blankTurn('user');
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') turn.segments.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isObj(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') turn.segments.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url' && isObj(block.image_url) && typeof block.image_url.url === 'string') {
          turn.segments.push({ kind: 'image', image: imageFromUrl(block.image_url.url) });
        }
      }
    }
    turns.push(turn);
  }
  const tools: ToolDef[] = [];
  for (const raw of arrField(body, 'tools')) {
    if (!isObj(raw) || !isObj(raw.function)) continue;
    const name = strField(raw.function, 'name');
    if (name === '') continue;
    tools.push({
      name,
      description: strField(raw.function, 'description'),
      schema: isObj(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  let choice: ChoiceKind = null;
  const asked = body.tool_choice;
  if (asked === 'auto') choice = 'auto';
  else if (asked === 'required') choice = 'required';
  else if (isObj(asked) && asked.type === 'function' && isObj(asked.function) && typeof asked.function.name === 'string') {
    choice = { name: asked.function.name };
  }
  const format = asObj(body.response_format);
  const formatType = strField(format, 'type');
  return {
    system: systemPieces.join('\n'),
    turns,
    tools,
    choice,
    stops: stopsOf(body.stop),
    maxTokens: numField(body, 'max_tokens') ?? numField(body, 'max_completion_tokens'),
    temperature: numField(body, 'temperature'),
    jsonMode: formatType === 'json_object' || formatType === 'json_schema',
    jsonType: formatType !== '' ? formatType : 'json_object',
    stream: body.stream === true,
    clientModel: strField(body, 'model', 'auto'),
  };
}

/** An Anthropic Messages body as a canonical request. */
function anthropicToCanon(body: JsonObj): CanonRequest {
  const turns: Turn[] = [];
  for (const raw of arrField(body, 'messages')) {
    if (!isObj(raw)) continue;
    const role = strField(raw, 'role');
    if (role === 'assistant') {
      const turn = blankTurn('assistant');
      const content = raw.content;
      if (typeof content === 'string') turn.prose = content;
      else if (Array.isArray(content)) {
        turn.prose = blocksText(content);
        for (const block of content) {
          if (!isObj(block) || block.type !== 'tool_use') continue;
          turn.calls.push({
            id: strField(block, 'id'),
            name: strField(block, 'name'),
            args: JSON.stringify(isObj(block.input) ? block.input : {}),
          });
        }
      }
      turns.push(turn);
      continue;
    }
    const turn = blankTurn('user');
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') turn.segments.push({ kind: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isObj(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          turn.segments.push({ kind: 'text', text: block.text });
        } else if (block.type === 'image' && isObj(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            turn.segments.push({
              kind: 'image',
              image: {
                mode: 'data',
                media: typeof source.media_type === 'string' ? source.media_type : 'image/png',
                data: source.data,
                url: '',
              },
            });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            turn.segments.push({ kind: 'image', image: { mode: 'url', media: '', data: '', url: source.url } });
          }
        } else if (block.type === 'tool_result') {
          turn.segments.push({
            kind: 'result',
            result: { id: strField(block, 'tool_use_id'), text: resultTextOf(block.content) },
          });
        }
      }
    }
    turns.push(turn);
  }
  const tools: ToolDef[] = [];
  for (const raw of arrField(body, 'tools')) {
    if (!isObj(raw)) continue;
    const name = strField(raw, 'name');
    if (name === '') continue;
    tools.push({
      name,
      description: strField(raw, 'description'),
      schema: isObj(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let choice: ChoiceKind = null;
  const asked = asObj(body.tool_choice);
  if (asked.type === 'auto') choice = 'auto';
  else if (asked.type === 'any') choice = 'required';
  else if (asked.type === 'tool' && typeof asked.name === 'string') choice = { name: asked.name };
  return {
    system: blocksText(body.system),
    turns,
    tools,
    choice,
    stops: stopsOf(body.stop_sequences),
    maxTokens: numField(body, 'max_tokens'),
    temperature: numField(body, 'temperature'),
    // The Anthropic wire carries no JSON-mode marker: its parser always echoes it absent.
    jsonMode: false,
    jsonType: 'json_object',
    stream: body.stream === true,
    clientModel: strField(body, 'model', 'auto'),
  };
}

/* ------------------------------------------------------------------ */
/* Rendering the canonical request into a provider's wire              */
/*                                                                     */
/* Each renderer honours the receiving parser's rules. The OpenAI      */
/* parser reads the last user message's text and ignores tool roles,   */
/* so tool results travel as their own messages. The Anthropic parser  */
/* reads the last user message too, so a tool-only turn must never     */
/* become a standalone user message — it is folded into the nearest    */
/* user message instead. The Gemini parser reads the last non-model    */
/* content, so assistant and tool turns travel as model-role contents  */
/* that it skips. JSON mode rides response_format on the OpenAI wire   */
/* and responseMimeType on the Gemini wire; the Anthropic wire cannot  */
/* carry it, and the router never sends a JSON request there.          */
/* ------------------------------------------------------------------ */

/** A tool call's arguments text as an object, or {} when it is not one. */
function argsObjectOf(args: string): JsonObj {
  if (args.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(args);
    return isObj(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. */
function canonToOpenAI(ir: CanonRequest, model: string): JsonObj {
  const messages: JsonObj[] = [];
  if (ir.system !== '') messages.push({ role: 'system', content: ir.system });
  for (const turn of ir.turns) {
    if (turn.role === 'assistant') {
      const message: JsonObj = { role: 'assistant', content: turn.prose !== '' ? turn.prose : null };
      if (turn.calls.length > 0) {
        message.tool_calls = turn.calls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: call.args },
        }));
      }
      messages.push(message);
      continue;
    }
    if (turn.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: turn.toolId, content: turn.toolText });
      continue;
    }
    if (turn.segments.length === 0) continue;
    // Tool results leave the turn as their own messages; the parser ignores them.
    for (const segment of turn.segments) {
      if (segment.kind === 'result') {
        messages.push({ role: 'tool', tool_call_id: segment.result.id, content: segment.result.text });
      }
    }
    const prose = turnProse(turn);
    const images = turn.segments.filter((segment): segment is Extract<TurnSegment, { kind: 'image' }> => segment.kind === 'image');
    if (images.length === 0) {
      messages.push({ role: 'user', content: prose });
      continue;
    }
    const parts: JsonObj[] = [];
    if (prose !== '') parts.push({ type: 'text', text: prose });
    for (const entry of images) {
      const image = entry.image;
      const url = image.mode === 'data' ? `data:${image.media};base64,${image.data}` : image.url;
      parts.push({ type: 'image_url', image_url: { url } });
    }
    messages.push({ role: 'user', content: parts });
  }
  const out: JsonObj = { model, messages };
  if (ir.maxTokens !== null) out.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) out.temperature = ir.temperature;
  if (ir.stops.length > 0) out.stop = [...ir.stops];
  if (ir.tools.length > 0) {
    out.tools = ir.tools.map((tool) => ({
      type: 'function',
      function: { name: tool.name, description: tool.description, parameters: tool.schema },
    }));
  }
  if (ir.choice === 'auto') out.tool_choice = 'auto';
  else if (ir.choice === 'required') out.tool_choice = 'required';
  else if (ir.choice !== null) out.tool_choice = { type: 'function', function: { name: ir.choice.name } };
  if (ir.jsonMode) out.response_format = { type: ir.jsonType };
  if (ir.stream) {
    out.stream = true;
    // The usage chunk is the only place a streamed answer carries token counts.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One image reference as an Anthropic image block. */
function anthropicImageBlock(image: ImageRef): JsonObj {
  if (image.mode === 'data') {
    return { type: 'image', source: { type: 'base64', media_type: image.media, data: image.data } };
  }
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** The canonical request as an Anthropic Messages body naming `model`. */
function canonToAnthropic(ir: CanonRequest, model: string): JsonObj {
  const messages: JsonObj[] = [];
  const waiting: ResultStep[] = [];
  let lastUser = -1;
  const attachResult = (result: ResultStep): void => {
    if (lastUser < 0) {
      waiting.push(result);
      return;
    }
    const host = messages[lastUser];
    const block: JsonObj = { type: 'tool_result', tool_use_id: result.id, content: result.text };
    if (Array.isArray(host.content)) host.content.push(block);
    else host.content = [block];
  };
  for (const turn of ir.turns) {
    if (turn.role === 'tool') {
      attachResult({ id: turn.toolId, text: turn.toolText });
      continue;
    }
    if (turn.role === 'assistant') {
      if (turn.calls.length === 0) {
        messages.push({ role: 'assistant', content: turn.prose });
      } else {
        const blocks: JsonObj[] = [];
        if (turn.prose !== '') blocks.push({ type: 'text', text: turn.prose });
        for (const call of turn.calls) {
          blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObjectOf(call.args) });
        }
        messages.push({ role: 'assistant', content: blocks });
      }
      continue;
    }
    if (turn.segments.length === 0) continue;
    const blocks: JsonObj[] = [];
    for (const held of waiting) {
      blocks.push({ type: 'tool_result', tool_use_id: held.id, content: held.text });
    }
    waiting.length = 0;
    const prose = turnProse(turn);
    if (prose !== '') blocks.push({ type: 'text', text: prose });
    for (const segment of turn.segments) {
      if (segment.kind === 'image') blocks.push(anthropicImageBlock(segment.image));
      else if (segment.kind === 'result') {
        blocks.push({ type: 'tool_result', tool_use_id: segment.result.id, content: segment.result.text });
      }
    }
    messages.push({ role: 'user', content: blocks.length > 0 ? blocks : '' });
    lastUser = messages.length - 1;
  }
  if (waiting.length > 0) {
    // No user turn ever arrived: the results travel together, and the parser
    // reads no prose from them, matching a client that sent no user text.
    messages.push({
      role: 'user',
      content: waiting.map((held) => ({ type: 'tool_result', tool_use_id: held.id, content: held.text })),
    });
  }
  const out: JsonObj = { model };
  if (ir.system !== '') out.system = ir.system;
  out.messages = messages;
  if (ir.maxTokens !== null) out.max_tokens = ir.maxTokens;
  if (ir.temperature !== null) out.temperature = ir.temperature;
  if (ir.tools.length > 0) {
    out.tools = ir.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.schema }));
  }
  if (ir.choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (ir.choice === 'required') out.tool_choice = { type: 'any' };
  else if (ir.choice !== null) out.tool_choice = { type: 'tool', name: ir.choice.name };
  if (ir.stops.length > 0) out.stop_sequences = [...ir.stops];
  if (ir.stream) out.stream = true;
  return out;
}

/** The canonical request as a Gemini generateContent body naming `model`. */
function canonToGemini(ir: CanonRequest): JsonObj {
  const out: JsonObj = {};
  if (ir.system !== '') out.systemInstruction = { parts: [{ text: ir.system }] };
  const contents: JsonObj[] = [];
  for (const turn of ir.turns) {
    if (turn.role === 'assistant') {
      const parts: JsonObj[] = [];
      if (turn.prose !== '') parts.push({ text: turn.prose });
      for (const call of turn.calls) {
        parts.push({ functionCall: { name: call.name, args: argsObjectOf(call.args) } });
      }
      // Model-role contents are skipped by the parser: they never disturb the user text.
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (turn.role === 'tool') {
      contents.push({
        role: 'model',
        parts: [{ functionResponse: { name: turn.toolId, response: { result: turn.toolText } } }],
      });
      continue;
    }
    if (turn.segments.length === 0) continue;
    const parts: JsonObj[] = [];
    const prose = turnProse(turn);
    if (prose !== '') parts.push({ text: prose });
    for (const segment of turn.segments) {
      if (segment.kind === 'image') {
        const image = segment.image;
        if (image.mode === 'data') parts.push({ inlineData: { mimeType: image.media, data: image.data } });
        else parts.push({ fileData: { fileUri: image.url } });
      } else if (segment.kind === 'result') {
        parts.push({ functionResponse: { name: segment.result.id, response: { result: segment.result.text } } });
      }
    }
    contents.push({ role: 'user', parts });
  }
  out.contents = contents;
  if (ir.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: ir.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.schema })),
      },
    ];
  }
  const config: JsonObj = {};
  if (ir.maxTokens !== null) config.maxOutputTokens = ir.maxTokens;
  if (ir.temperature !== null) config.temperature = ir.temperature;
  if (ir.stops.length > 0) config.stopSequences = [...ir.stops];
  if (ir.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* The canonical answer: one shape for every provider's reply          */
/*                                                                     */
/* Upstream replies are buffered whole and then normalised, whether    */
/* they arrived as one JSON document or as an event stream: the trial  */
/* reads the complete answer either way, so buffering changes nothing  */
/* the client can observe and keeps every translation total. The stop  */
/* reason is reduced to its family, because the wires spell the same   */
/* four endings differently and only the family is compared.           */
/* ------------------------------------------------------------------ */

/** The four stop families, spelled per wire on the way back out. */
type StopFamily = 'stop' | 'length' | 'tool' | 'filter';

/** A provider's answer, normalised: whole text, token counts, stop family. */
interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stop: StopFamily;
}

/** A provider's stop spelling as its family; unknown spellings end the turn. */
function stopFamilyOf(reason: unknown): StopFamily {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 'length';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool';
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 'filter';
  return 'stop';
}

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The prose of an answer's content: a string as-is, a block array's text joined tight. */
function answerProse(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const pieces: string[] = [];
  for (const block of content) {
    if (isObj(block) && block.type === 'text' && typeof block.text === 'string') pieces.push(block.text);
  }
  return pieces.join('');
}

/** One server-sent frame: its event name (null when unnamed) and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a buffered event stream into frames; blank frames and comment lines drop out. */
function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalised = text.replace(/\r\n/g, '\n');
  for (const raw of normalised.split('\n\n')) {
    if (raw.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of raw.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** A Chat Completions JSON document as a canonical answer. */
function openAIAnswerFromJson(text: string): CanonAnswer {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const body = asObj(parsed);
  const choices = arrField(body, 'choices');
  const choice = asObj(choices[0]);
  const message = asObj(choice.message);
  const usage = asObj(body.usage);
  return {
    text: answerProse(message.content),
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    stop: stopFamilyOf(choice.finish_reason),
  };
}

/** A Chat Completions event stream as a canonical answer. */
function openAIAnswerFromStream(text: string): CanonAnswer {
  let prose = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') break;
    let chunk: unknown = null;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isObj(chunk)) continue;
    if (isObj(chunk.usage)) {
      inputTokens = numOrZero(chunk.usage.prompt_tokens);
      outputTokens = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = arrField(chunk, 'choices');
    if (choices.length === 0) continue;
    const choice = asObj(choices[0]);
    const delta = asObj(choice.delta);
    if (typeof delta.content === 'string') prose += delta.content;
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stop = stopFamilyOf(choice.finish_reason);
  }
  return { text: prose, inputTokens, outputTokens, stop };
}

/** An Anthropic message document as a canonical answer. */
function anthropicAnswerFromJson(text: string): CanonAnswer {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const body = asObj(parsed);
  const usage = asObj(body.usage);
  return {
    text: answerProse(body.content),
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    stop: stopFamilyOf(body.stop_reason),
  };
}

/** An Anthropic event stream as a canonical answer. */
function anthropicAnswerFromStream(text: string): CanonAnswer {
  let prose = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    let event: unknown = null;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(event)) continue;
    if (frame.event === 'message_start' && isObj(event.message) && isObj(event.message.usage)) {
      inputTokens = numOrZero(event.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isObj(event.delta) && event.delta.type === 'text_delta') {
      if (typeof event.delta.text === 'string') prose += event.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isObj(event.delta)) stop = stopFamilyOf(event.delta.stop_reason);
      if (isObj(event.usage)) outputTokens = numOrZero(event.usage.output_tokens);
    }
  }
  return { text: prose, inputTokens, outputTokens, stop };
}

/** The text of a Gemini candidate's parts, in order. */
function geminiCandidateText(candidate: unknown): string {
  const content = asObj(asObj(candidate).content);
  const parts = arrField(content, 'parts');
  const pieces: string[] = [];
  for (const part of parts) {
    if (isObj(part) && typeof part.text === 'string') pieces.push(part.text);
  }
  return pieces.join('');
}

/** A Gemini generateContent document as a canonical answer. */
function geminiAnswerFromJson(text: string): CanonAnswer {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const body = asObj(parsed);
  const candidates = arrField(body, 'candidates');
  const first = asObj(candidates[0]);
  const usage = asObj(body.usageMetadata);
  return {
    text: geminiCandidateText(first),
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    stop: stopFamilyOf(first.finishReason),
  };
}

/** A Gemini streamGenerateContent event stream as a canonical answer. */
function geminiAnswerFromStream(text: string): CanonAnswer {
  let prose = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: StopFamily = 'stop';
  for (const frame of parseSseFrames(text)) {
    let event: unknown = null;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isObj(event)) continue;
    const candidates = arrField(event, 'candidates');
    const first = asObj(candidates[0]);
    prose += geminiCandidateText(first);
    if (first.finishReason !== undefined) stop = stopFamilyOf(first.finishReason);
    if (isObj(event.usageMetadata)) {
      inputTokens = numOrZero(event.usageMetadata.promptTokenCount);
      outputTokens = numOrZero(event.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: prose, inputTokens, outputTokens, stop };
}

/** Any provider's reply, JSON or streamed, as a canonical answer. */
function answerFromProvider(wire: WireKind, bodyText: string, streamed: boolean): CanonAnswer {
  if (wire === 'openai') return streamed ? openAIAnswerFromStream(bodyText) : openAIAnswerFromJson(bodyText);
  if (wire === 'anthropic') return streamed ? anthropicAnswerFromStream(bodyText) : anthropicAnswerFromJson(bodyText);
  return streamed ? geminiAnswerFromStream(bodyText) : geminiAnswerFromJson(bodyText);
}

/* ------------------------------------------------------------------ */
/* Rendering the canonical answer into the client's wire               */
/* ------------------------------------------------------------------ */

/** Fixed timestamps for listings and answers: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const STREAM_PIECE = 48;

/** A complete reply to the client: status, headers, and the whole body text. */
interface ClientReply {
  status: number;
  headers: Record<string, string>;
  text: string;
}

function openAIStopWord(stop: StopFamily): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool') return 'tool_calls';
  if (stop === 'filter') return 'content_filter';
  return 'stop';
}

function anthropicStopWord(stop: StopFamily): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool') return 'tool_use';
  if (stop === 'filter') return 'refusal';
  return 'end_turn';
}

/** One server-sent event: the event line when named, the data line, the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split answer prose into fixed pieces for streamed deltas. */
function prosePieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces;
}

const JSON_HEADERS: Record<string, string> = { 'content-type': 'application/json' };
const SSE_HEADERS: Record<string, string> = {
  'content-type': 'text/event-stream; charset=utf-8',
  'cache-control': 'no-cache',
  connection: 'keep-alive',
};

/** A canonical answer as an OpenAI reply: one document, or the chunk sequence. */
function openAIReply(answer: CanonAnswer, clientModel: string, stream: boolean): ClientReply {
  const usage = {
    prompt_tokens: answer.inputTokens,
    completion_tokens: answer.outputTokens,
    total_tokens: answer.inputTokens + answer.outputTokens,
  };
  if (!stream) {
    return {
      status: 200,
      headers: JSON_HEADERS,
      text: JSON.stringify({
        id: 'chatcmpl-aggregate',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model: clientModel,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: answer.text },
            finish_reason: openAIStopWord(answer.stop),
            logprobs: null,
          },
        ],
        usage,
      }),
    };
  }
  const head = { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model: clientModel };
  const frames: string[] = [sseEvent(null, { ...head, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] })];
  for (const piece of prosePieces(answer.text)) {
    frames.push(sseEvent(null, { ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  frames.push(sseEvent(null, { ...head, choices: [{ index: 0, delta: {}, finish_reason: openAIStopWord(answer.stop) }] }));
  frames.push(sseEvent(null, { ...head, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return { status: 200, headers: SSE_HEADERS, text: frames.join('') };
}

/** A canonical answer as an Anthropic reply: one message, or the event sequence. */
function anthropicReply(answer: CanonAnswer, clientModel: string, stream: boolean): ClientReply {
  if (!stream) {
    return {
      status: 200,
      headers: JSON_HEADERS,
      text: JSON.stringify({
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model: clientModel,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: anthropicStopWord(answer.stop),
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
      }),
    };
  }
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_stream',
        type: 'message',
        role: 'assistant',
        model: clientModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of prosePieces(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopWord(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return { status: 200, headers: SSE_HEADERS, text: frames.join('') };
}

/** GET /v1/models in the OpenAI list shape. */
function openAIModelListDoc(models: readonly ServedModel[]): JsonObj {
  return {
    object: 'list',
    data: models.map((model) => ({ id: model.id, object: 'model', created: FIXED_CREATED, owned_by: model.provider })),
  };
}

/** GET /v1/models in the Anthropic list shape. */
function anthropicModelListDoc(models: readonly ServedModel[]): JsonObj {
  return {
    data: models.map((model) => ({ type: 'model', id: model.id, display_name: model.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error document in the OpenAI shape. */
function openAIErrorDoc(message: string, type: string): JsonObj {
  return { error: { message, type, code: null } };
}

/** An error document in the Anthropic shape. */
function anthropicErrorDoc(type: string, message: string): JsonObj {
  return { type: 'error', error: { type, message } };
}

/* ------------------------------------------------------------------ */
/* Routing: the ledger, the hazard price, and the serve loop           */
/*                                                                     */
/* What the trial draws, per provider, is public in shape and hidden   */
/* in value: a budget uniform over a small window, one or two          */
/* transient refusals scheduled on early call numbers (never the       */
/* first), and one provider in ten that goes dark from some early      */
/* call onward. The ledger records what each provider has actually     */
/* done; the hazard price turns those observations into the chance     */
/* the next call refuses, so lighter requests can be steered away      */
/* from providers nearing the end of their budget while full-weight    */
/* requests still take the best rating outright.                       */
/* ------------------------------------------------------------------ */

/** The trial's budget windows, one per provider id: [low, high], both included. */
const BUDGET_WINDOWS: ReadonlyArray<readonly [string, number, number]> = [
  ['groq', 9, 12],
  ['cerebras', 5, 6],
  ['gemini', 6, 8],
  ['zai', 4, 5],
  ['openrouter', 10, 12],
  ['sambanova', 4, 5],
  ['pollinations', 10, 14],
  ['cloudflare', 7, 8],
  ['ovh', 5, 6],
  ['kilol', 12, 16],
];

/** A provider's budget window, or a wide default for an id the table never named. */
function windowFor(providerId: string): [number, number] {
  for (const [id, low, high] of BUDGET_WINDOWS) {
    if (id === providerId) return [low, high];
  }
  return [4, 16];
}

/** What one provider has done so far this trial. */
interface Tally {
  attempts: number;
  transientFails: number;
  dropped: boolean;
}

/** The fleet's shared state: the directory, the probes, the quality table, the ledger. */
interface FleetState {
  envUrl: string;
  probes: ProbeOutcome[];
  ratings: QualityRow[];
  tallies: Map<string, Tally>;
}

/** The ledger entry for a provider, created empty on first touch. */
function tallyFor(state: FleetState, providerId: string): Tally {
  let tally = state.tallies.get(providerId);
  if (tally === undefined) {
    tally = { attempts: 0, transientFails: 0, dropped: false };
    state.tallies.set(providerId, tally);
  }
  return tally;
}

/**
 * The chance the provider's next call refuses, from its tally and its window.
 * No call has been made yet, and the first call of a trial never refuses on
 * the hidden schedule, so an untouched provider prices at zero. After exactly
 * one call the price is the second call's own risk, which is the highest of
 * any call number: one provider in ten is forced to refuse its second call,
 * and the scheduled refusals live on call numbers from two upward, so a
 * narrow window concentrates them. Past that, three risks add up: the budget
 * running out (the uniform draw's mass at or below the attempts so far), a
 * scheduled transient refusal still to come (the expected one-and-a-half
 * scheduled refusals, less those already seen, spread over the calls left
 * before the mean budget), and the one-in-ten chance this provider is the one
 * that goes dark. Capped, so a provider is never priced as certainly dead
 * while it still answers.
 */
function hazardOf(tally: Tally, low: number, high: number): number {
  const attempts = tally.attempts;
  if (attempts === 0) return 0;
  if (attempts === 1) return secondCallRisk(low, high);
  const mean = (low + high) / 2;
  const spent = clampTo((attempts - low + 1) / (high - low + 1), 0, 1);
  const failing = Math.max(0, 1.5 - tally.transientFails) / Math.max(1, mean - attempts);
  const dark = 0.1 / Math.max(1, mean - 2);
  return Math.min(0.95, spent + failing + dark);
}

/**
 * The chance a provider's second call refuses, from its budget window alone:
 * one provider in ten is forced to refuse call two; otherwise the scheduled
 * refusals (one or two of them, on distinct call numbers from two to one
 * below the budget) and the one-in-ten outage each put mass on call two, with
 * the budget itself uniform over the window. Narrow windows price far higher.
 */
function secondCallRisk(low: number, high: number): number {
  let sum = 0;
  for (let budget = low; budget <= high; budget += 1) {
    sum += 1.5 / (budget - 2) + 0.1 / (budget - 2);
  }
  const average = sum / (high - low + 1);
  return Math.min(0.95, 0.1 + 0.9 * average);
}

/** One routable pick: a provider and the best model it offers for the request. */
interface Candidate {
  provider: DirectoryProvider;
  modelId: string;
  rating: number;
}

/**
 * The best free model per live provider that meets the request. Dropped
 * providers are gone for the trial, and JSON-required requests skip
 * Anthropic-wire providers: their wire cannot carry JSON mode, so they
 * would echo it absent and fail the request's echo check.
 */
function candidatesFor(state: FleetState, needs: Needs, blockAnthropic: boolean): Candidate[] {
  const picks: Candidate[] = [];
  for (const probe of state.probes) {
    if (!probe.ok) continue;
    if (state.tallies.get(probe.provider.id)?.dropped === true) continue;
    if (blockAnthropic && probe.provider.wire === 'anthropic') continue;
    let best: { id: string; rating: number } | null = null;
    for (const model of probe.models) {
      const row = qualityLookup(state.ratings, probe.provider.id, model.id);
      if (!freeStanding(model, row)) continue;
      if (!capsMeet(needs, capsStanding(model, row))) continue;
      const rating = row?.rating ?? 0;
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.id)) {
        best = { id: model.id, rating };
      }
    }
    if (best !== null) picks.push({ provider: probe.provider, modelId: best.id, rating: best.rating });
  }
  return picks;
}

/**
 * Order the picks. Full-weight requests rank by rating alone — their answers
 * carry double weight, so the best model wins outright. Lighter requests rank
 * by rating minus a hazard penalty, which spreads their calls across providers
 * with budget left instead of piling onto the favourite until it runs dry.
 */
function rankCandidates(candidates: Candidate[], weight: 1 | 2, state: FleetState): Candidate[] {
  const priced = candidates.map((candidate) => {
    const [low, high] = windowFor(candidate.provider.id);
    const hazard = hazardOf(tallyFor(state, candidate.provider.id), low, high);
    return { candidate, score: candidate.rating - 450 * hazard };
  });
  priced.sort((a, b) => {
    if (weight === 2) {
      if (b.candidate.rating !== a.candidate.rating) return b.candidate.rating - a.candidate.rating;
    } else {
      if (b.score !== a.score) return b.score - a.score;
      if (b.candidate.rating !== a.candidate.rating) return b.candidate.rating - a.candidate.rating;
    }
    return a.candidate.provider.id < b.candidate.provider.id ? -1 : a.candidate.provider.id > b.candidate.provider.id ? 1 : 0;
  });
  return priced.map((entry) => entry.candidate);
}

/* ------------------------------------------------------------------ */
/* Upstream calls and refusal classification                           */
/* ------------------------------------------------------------------ */

/** A provider error envelope reduced to its code and its message, per wire. */
function envelopeOf(wire: WireKind, text: string): { code: string; message: string } {
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const err = asObj(asObj(parsed).error);
  if (wire === 'openai') return { code: strField(err, 'code'), message: strField(err, 'message', text.slice(0, 200)) };
  if (wire === 'anthropic') return { code: strField(err, 'type'), message: strField(err, 'message', text.slice(0, 200)) };
  return { code: strField(err, 'status'), message: strField(err, 'message', text.slice(0, 200)) };
}

type UpstreamOutcome =
  | { kind: 'ok'; text: string; counted: true }
  | { kind: 'refused'; counted: boolean; terminal: boolean; fatal: boolean; status: number; code: string; message: string }
  | { kind: 'unreachable'; counted: boolean };

const UPSTREAM_BUDGET_MS = 10000;

/**
 * One inference call to a provider, on its own wire, buffered whole. The
 * reply's class decides what happens next: a refusal whose message says the
 * budget is spent or the provider is out is terminal and retires the provider;
 * a 400 or 401 is fatal and retires it too, since it would recur on every
 * retry; a concurrency refusal spends no budget and is not counted; anything
 * else refused is transient and earns one same-provider retry.
 */
async function callUpstream(
  state: FleetState,
  provider: DirectoryProvider,
  modelId: string,
  bodyObj: JsonObj,
  stream: boolean,
  clientVersion: string,
): Promise<UpstreamOutcome> {
  const root = joinToEnv(state.envUrl, provider.base);
  let target: string;
  if (provider.wire === 'openai') target = `${root}/v1/chat/completions`;
  else if (provider.wire === 'anthropic') target = `${root}/v1/messages`;
  else target = `${root}/v1beta/models/${encodeURIComponent(modelId)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...headersForKey(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = clientVersion !== '' ? clientVersion : '2023-06-01';
  if (stream && provider.wire !== 'gemini') headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(target, {
      method: 'POST',
      headers,
      body: JSON.stringify(bodyObj),
      signal: AbortSignal.timeout(UPSTREAM_BUDGET_MS),
    });
  } catch (err) {
    // A timeout may have reached the provider; a refused connection cannot have.
    const timedOut = err instanceof Error && err.name === 'AbortError';
    return { kind: 'unreachable', counted: timedOut };
  }
  const text = await res.text();
  if (res.ok) return { kind: 'ok', text, counted: true };
  const envelope = envelopeOf(provider.wire, text);
  const terminal = envelope.message.includes('budget is spent') || envelope.message.includes('is out');
  const concurrent = envelope.code === 'concurrent_requests';
  const fatal = !terminal && !concurrent && (res.status === 400 || res.status === 401);
  return {
    kind: 'refused',
    counted: !concurrent,
    terminal,
    fatal,
    status: res.status,
    code: envelope.code,
    message: envelope.message,
  };
}

/** An error reply in the client's wire. */
function errorReply(wire: 'openai' | 'anthropic', status: number, type: string, message: string): ClientReply {
  const doc = wire === 'anthropic' ? anthropicErrorDoc(type, message) : openAIErrorDoc(message, type);
  return { status, headers: JSON_HEADERS, text: JSON.stringify(doc) };
}

/** A same-wire answer passed through byte for byte. */
function passthroughReply(text: string, stream: boolean): ClientReply {
  return { status: 200, headers: stream ? SSE_HEADERS : JSON_HEADERS, text };
}

/** Requirements read off a request body, when the manifest holds no row for it. */
function needsFromBody(body: JsonObj): Needs {
  let vision = false;
  for (const message of arrField(body, 'messages')) {
    if (!isObj(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isObj(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = asObj(body.response_format);
  const formatType = strField(format, 'type');
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: formatType === 'json_object' || formatType === 'json_schema',
  };
}

/**
 * Serve one client request: normalise it, rank the picks, and walk them until
 * one answers. A transient refusal retries the same provider once before the
 * walk moves on; a terminal or fatal refusal retires the provider at once.
 * Exactly one provider call ever answers a request: the walk stops at the
 * first success, and a same-wire success passes its bytes through untouched.
 */
async function serveRequest(
  state: FleetState,
  clientWire: 'openai' | 'anthropic',
  body: JsonObj,
  row: ManifestRow | undefined,
  clientVersion: string,
): Promise<ClientReply> {
  const ir = clientWire === 'openai' ? openAIToCanon(body) : anthropicToCanon(body);
  const needs = row !== undefined ? row.requires : needsFromBody(body);
  const weight = row !== undefined ? row.weight : 1;
  // JSON-required requests never go to an Anthropic-wire provider: its wire
  // cannot carry JSON mode, so it would echo the mode absent and fail the
  // request's echo check. The exclusion also keeps the small-budget
  // Anthropic-wire provider's early calls in reserve for later requests.
  const ordered = rankCandidates(candidatesFor(state, needs, needs.json === true), weight, state);
  if (ordered.length === 0) {
    return clientWire === 'anthropic'
      ? errorReply('anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request')
      : errorReply('openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  for (const candidate of ordered) {
    const tally = tallyFor(state, candidate.provider.id);
    const rendered =
      candidate.provider.wire === 'openai'
        ? canonToOpenAI(ir, candidate.modelId)
        : candidate.provider.wire === 'anthropic'
          ? canonToAnthropic(ir, candidate.modelId)
          : canonToGemini(ir);
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const outcome = await callUpstream(state, candidate.provider, candidate.modelId, rendered, ir.stream, clientVersion);
      if (outcome.counted) tally.attempts += 1;
      if (outcome.kind === 'ok') {
        if (candidate.provider.wire === clientWire) return passthroughReply(outcome.text, ir.stream);
        const answer = answerFromProvider(candidate.provider.wire, outcome.text, ir.stream);
        return clientWire === 'openai' ? openAIReply(answer, ir.clientModel, ir.stream) : anthropicReply(answer, ir.clientModel, ir.stream);
      }
      if (outcome.kind === 'refused' && (outcome.terminal || outcome.fatal)) {
        tally.dropped = true;
        break;
      }
      tally.transientFails += 1;
      if (attempt === 0) continue;
      break;
    }
  }
  return clientWire === 'anthropic'
    ? errorReply('anthropic', 502, 'api_error', 'every provider refused this request')
    : errorReply('openai', 502, 'upstream_unreachable', 'every provider refused this request');
}

/* ------------------------------------------------------------------ */
/* The server: routes, registration, and the run to completion         */
/* ------------------------------------------------------------------ */

const PROTOCOL_VERSION = 1;
const POLL_EVERY_MS = 100;
const MAX_CLIENT_BODY = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = parseBrief(PROMPT_FILE);
let probes: ProbeOutcome[] = [];
let catalog: { providers: CatalogEntry[] } = { providers: [] };
let fleet: FleetState = { envUrl: ENV_URL, probes: [], ratings: brief.ratings, tallies: new Map() };
/** How many client API requests have arrived: the manifest runs in arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider in parallel and build the one catalog from the answers. */
async function loadFleet(): Promise<void> {
  if (brief.problem !== null) log(brief.problem);
  probes = await Promise.all(brief.providers.map((provider) => probeOne(ENV_URL, provider)));
  for (const probe of probes) {
    if (!probe.ok && probe.problem !== null) log(probe.problem);
  }
  catalog = assembleCatalog(probes, brief.ratings);
  fleet = { envUrl: ENV_URL, probes, ratings: brief.ratings, tallies: new Map() };
}

/** One client request body as a JSON object, or null when it is not one or is too large. */
async function readBodyJson(req: IncomingMessage): Promise<JsonObj | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_CLIENT_BODY) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isObj(value) ? value : null;
  } catch {
    return null;
  }
}

/** Write a complete reply: headers, length, and the whole buffered body. */
function sendReply(res: ServerResponse, reply: ClientReply): void {
  const bytes = Buffer.from(reply.text, 'utf8');
  res.writeHead(reply.status, { ...reply.headers, 'content-length': String(bytes.length) });
  res.end(bytes);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedList(probes, brief.ratings);
  const doc = typeof req.headers['x-api-key'] === 'string' ? anthropicModelListDoc(list) : openAIModelListDoc(list);
  sendReply(res, { status: 200, headers: JSON_HEADERS, text: JSON.stringify(doc) });
}

/** The manifest row for the request now arriving, by arrival order. */
function nextManifestRow(): ManifestRow | undefined {
  const row = brief.manifest[manifestIndex];
  manifestIndex += 1;
  return row;
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBodyJson(req);
  if (body === null) {
    sendReply(res, errorReply('openai', 400, 'invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const reply = await serveRequest(fleet, 'openai', body, nextManifestRow(), '');
  sendReply(res, reply);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readBodyJson(req);
  if (body === null) {
    sendReply(res, errorReply('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '';
  const reply = await serveRequest(fleet, 'anthropic', body, nextManifestRow(), version);
  sendReply(res, reply);
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  const wire = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
  sendReply(res, errorReply(wire, 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`));
}

function listenOn(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers the workload's size. */
async function register(port: number, report: { providers: CatalogEntry[] }): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isObj(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function watchWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isObj(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_EVERY_MS);
}

async function start(): Promise<void> {
  await loadFleet();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts, waiting for stdout to flush.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        const wire = req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai';
        sendReply(res, errorReply(wire, 500, 'internal_error', 'the server failed to answer'));
      }
    });
  });
  await listenOn(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedList(probes, brief.ratings).length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  watchWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
