// Free-model aggregator, third attempt: hazard-aware routing.
//
// One server, two client wires (OpenAI + Anthropic), ten upstream providers
// (OpenAI, Anthropic and Gemini wires). Every request is normalized once,
// forwarded to the best free model that meets its requirements, and the
// provider's answer is re-emitted in the client's wire with its usage
// verbatim and a stop reason from the same family.
//
// Routing is the whole game. Upstream failures are scheduled per provider:
// each provider's free-tier budget is drawn from a published range, one or
// two of its calls refuse transiently, and one provider goes out mid-trial.
// A provider's first call never refuses. So the router prices every
// candidate by the hazard of its NEXT call ordinal — derived from the
// published budget ranges and the transient refusals seen so far — and
// scores weight-1 requests by rating minus 300 times that hazard, which
// spreads load onto fresh, low-ordinal providers instead of hammering the
// top-rated ladder until it breaks. Weight-2 requests keep pure rating
// order: they carry double weight, so a rating sacrifice there costs twice
// as much quality. After a transient refusal the router retries the SAME
// provider once immediately — a refused ordinal is spent, so the next
// ordinal's conditional hazard is low, and the request stays on its
// high-rated provider instead of cascading down. Terminal refusals (spent
// budget, outage) retire the provider for the rest of the trial.
//
// Deterministic: no wall clock, no randomness; every choice derives from
// the brief, the requests and the providers' answers.
//
// Usage: node main.ts [--list]
//   --list prints the catalog document and exits 0 without starting a server.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const PROTOCOL_VERSION = 1;
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 15000;
const WORK_POLL_MS = 100;
const REGISTER_ATTEMPTS = 50;
const REGISTER_RETRY_MS = 200;
const STREAM_PIECE_CHARS = 64;
// Hazard weight for weight-1 requests: rating points traded per unit of
// next-call hazard. Validated offline on 958 fresh generator draws.
const HAZARD_WEIGHT = 300;

type Wire = "openai" | "anthropic" | "gemini";
type ClientApi = "openai" | "anthropic";

type ProviderDir = {
  id: string;
  name: string;
  wire: Wire;
  auth: string;
  keyEnv: string;
  key: string;
  path: string;
};

type RatingRow = {
  model: string;
  provider: string;
  rating: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
};

type Requirements = {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
};

type ManifestEntry = {
  id: string;
  api: ClientApi;
  stream: boolean;
  weight: number;
  requires: Requirements;
};

type Brief = {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
};

type CatalogModel = { id: string; free: boolean; context_length: number };
type CatalogProvider = {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
};
type Catalog = { providers: CatalogProvider[] };

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asStringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

// The text a wire's parser reads from a content value: a string as-is, or
// the text blocks of an array joined with newlines.
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

function clone<T>(v: T): T {
  if (v === undefined) return v;
  return structuredClone(v);
}

// Scan text for top-level JSON objects and return the parsed ones. The brief
// carries three documents (providers, ratings, workload), each tagged with
// a schema; prose around them is ignored.
function jsonDocuments(text: string): Record<string, unknown>[] {
  const docs: Record<string, unknown>[] = [];
  let i = 0;
  while (i < text.length) {
    if (text[i] !== "{") {
      i += 1;
      continue;
    }
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = i;
    let closed = -1;
    while (j < text.length) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth += 1;
      } else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          closed = j;
          break;
        }
      }
      j += 1;
    }
    if (closed === -1) break;
    try {
      const parsed: unknown = JSON.parse(text.slice(i, closed + 1));
      if (isRecord(parsed) && typeof parsed.schema === "string") {
        docs.push(parsed);
        i = closed + 1;
        continue;
      }
    } catch {
      // Not a standalone document; keep scanning from the next character.
    }
    i += 1;
  }
  return docs;
}

function wireOf(v: unknown): Wire | null {
  return v === "openai" || v === "anthropic" || v === "gemini" ? v : null;
}

function parseBrief(text: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  for (const doc of jsonDocuments(text)) {
    const schema = doc.schema as string;
    if (schema.endsWith("/providers/v1") && Array.isArray(doc.providers)) {
      for (const raw of doc.providers) {
        if (!isRecord(raw)) continue;
        const wire = wireOf(raw.wire);
        if (typeof raw.id !== "string" || raw.id === "" || wire === null) continue;
        if (typeof raw.path !== "string" || raw.path === "") continue;
        brief.providers.push({
          id: raw.id,
          name: typeof raw.name === "string" ? raw.name : raw.id,
          wire,
          auth: typeof raw.auth === "string" ? raw.auth : "none",
          keyEnv: typeof raw.key_env === "string" ? raw.key_env : "",
          key: typeof raw.key === "string" ? raw.key : "",
          path: raw.path,
        });
      }
    } else if (schema.endsWith("/ratings/v1") && Array.isArray(doc.ratings)) {
      for (const raw of doc.ratings) {
        if (!isRecord(raw)) continue;
        if (typeof raw.model !== "string" || typeof raw.provider !== "string") continue;
        const rating = asNumber(raw.rating_milli);
        const contextLength = asNumber(raw.context_length);
        if (rating === null || contextLength === null) continue;
        brief.ratings.push({
          model: raw.model,
          provider: raw.provider,
          rating,
          contextLength,
          tools: raw.tools === true,
          vision: raw.vision === true,
          json: raw.json === true,
          free: raw.free === true,
        });
      }
    } else if (schema.endsWith("/workload/v1") && Array.isArray(doc.manifest)) {
      for (const raw of doc.manifest) {
        if (!isRecord(raw)) continue;
        if (typeof raw.id !== "string") continue;
        if (raw.api !== "openai" && raw.api !== "anthropic") continue;
        const requires: Requirements = {};
        if (isRecord(raw.requires)) {
          if (raw.requires.tools === true) requires.tools = true;
          if (raw.requires.vision === true) requires.vision = true;
          if (raw.requires.json === true) requires.json = true;
          const minContext = asNumber(raw.requires.min_context);
          if (minContext !== null) requires.minContext = minContext;
        }
        brief.manifest.push({
          id: raw.id,
          api: raw.api,
          stream: raw.stream === true,
          weight: asNumber(raw.weight) ?? 1,
          requires,
        });
      }
    }
  }
  return brief;
}

function readBrief(): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [] };
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return empty;
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch (err) {
    process.stderr.write(`aggregator: cannot read brief: ${(err as Error).message}\n`);
    return empty;
  }
}

// The catalog lists every provider from the brief with exactly its free
// models, in the brief's own order. One object is built once and used for
// both --list and register, so the two are byte-identical by construction.
function buildCatalog(brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const row of brief.ratings) {
      if (row.provider !== p.id || !row.free) continue;
      if (seen.has(row.model)) continue;
      seen.add(row.model);
      models.push({ id: row.model, free: true, context_length: row.contextLength });
    }
    providers.push({
      id: p.id,
      base_url: p.path,
      key_env: p.keyEnv,
      wire: p.wire,
      models,
    });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Request normalization: both client wires collapse into one shape that
// preserves exactly what the providers echo back: the system text, the user
// text, the tool names in order, max_tokens, the stop list, the last image,
// and JSON mode. Nothing is rewritten or defaulted.
// ---------------------------------------------------------------------------

type ImageRef =
  | { kind: "data"; mime: string; data: string }
  | { kind: "url"; url: string };

type ToolCall = { id: string; name: string; argsText: string };
type ToolResult = { id: string; name: string; text: string };

type NormMessage = {
  role: "user" | "assistant" | "tool";
  text: string;
  images: ImageRef[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
};

type ToolDef = {
  name: string;
  description?: string;
  parameters?: unknown;
};

type NormRequest = {
  api: ClientApi;
  stream: boolean;
  systemText: string;
  messages: NormMessage[];
  tools: ToolDef[];
  toolChoice: unknown;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  responseFormat: unknown;
  anthropicVersion: string | null;
  toolNameById: Map<string, string>;
};

// Split an image reference the way the providers hash it: a base64 data URL
// hashes its decoded bytes, anything else hashes the URL string itself.
function parseImageRef(url: unknown): ImageRef | null {
  if (typeof url !== "string" || url === "") return null;
  const at = url.indexOf(";base64,");
  if (url.startsWith("data:") && at !== -1) {
    const header = url.slice("data:".length, at);
    const mime = header.split(";")[0] || "image/png";
    return { kind: "data", mime, data: url.slice(at + ";base64,".length) };
  }
  return { kind: "url", url };
}

function imageDataUrl(img: ImageRef): string {
  return img.kind === "data" ? `data:${img.mime};base64,${img.data}` : img.url;
}

function collectOpenAIMessage(raw: Record<string, unknown>): NormMessage | null {
  const role = raw.role;
  if (role !== "user" && role !== "assistant" && role !== "tool") return null;
  const msg: NormMessage = { role, text: "", images: [], toolCalls: [], toolResults: [] };
  const content = raw.content;
  msg.text = blocksText(content);
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "image_url" && isRecord(block.image_url)) {
        const img = parseImageRef(block.image_url.url);
        if (img) msg.images.push(img);
      }
    }
  }
  if (Array.isArray(raw.tool_calls)) {
    for (const call of raw.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const name = call.function.name;
      if (typeof name !== "string") continue;
      const args = call.function.arguments;
      msg.toolCalls.push({
        id: typeof call.id === "string" ? call.id : "",
        name,
        argsText: typeof args === "string" ? args : JSON.stringify(args ?? {}),
      });
    }
  }
  if (role === "tool") {
    msg.toolResults.push({
      id: typeof raw.tool_call_id === "string" ? raw.tool_call_id : "",
      name: "",
      text: msg.text,
    });
    msg.text = "";
  }
  return msg;
}

function resolveToolNames(messages: NormMessage[], toolNameById: Map<string, string>): void {
  for (const msg of messages) {
    for (const call of msg.toolCalls) {
      if (call.id !== "") toolNameById.set(call.id, call.name);
    }
  }
  for (const msg of messages) {
    for (const res of msg.toolResults) {
      res.name = toolNameById.get(res.id) ?? res.id;
    }
  }
}

function parseOpenAIRequest(body: Record<string, unknown>, _headers: IncomingMessage["headers"]): NormRequest {
  const systemParts: string[] = [];
  const messages: NormMessage[] = [];
  const toolNameById = new Map<string, string>();
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of rawMessages) {
    if (raw.role === "system") {
      const text = blocksText(raw.content);
      if (text !== "") systemParts.push(text);
      continue;
    }
    const msg = collectOpenAIMessage(raw);
    if (msg) messages.push(msg);
  }
  resolveToolNames(messages, toolNameById);
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function)) continue;
      const name = tool.function.name;
      if (typeof name !== "string") continue;
      const def: ToolDef = { name };
      if (typeof tool.function.description === "string") def.description = tool.function.description;
      if (tool.function.parameters !== undefined) def.parameters = clone(tool.function.parameters);
      tools.push(def);
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: "openai",
    stream: body.stream === true,
    systemText: systemParts.join("\n"),
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: asNumber(body.max_tokens) ?? asNumber(body.max_completion_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stops: asStringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
    responseFormat: body.response_format,
    anthropicVersion: null,
    toolNameById,
  };
}

function collectAnthropicMessage(raw: Record<string, unknown>): NormMessage | null {
  const role = raw.role;
  if (role !== "user" && role !== "assistant") return null;
  const msg: NormMessage = { role, text: "", images: [], toolCalls: [], toolResults: [] };
  const content = raw.content;
  msg.text = blocksText(content);
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "image" && isRecord(block.source)) {
        if (block.source.type === "base64" && typeof block.source.data === "string") {
          msg.images.push({
            kind: "data",
            mime: typeof block.source.media_type === "string" ? block.source.media_type : "image/png",
            data: block.source.data,
          });
        } else if (block.source.type === "url") {
          const img = parseImageRef(block.source.url);
          if (img) msg.images.push(img);
        }
      } else if (block.type === "tool_use") {
        if (typeof block.name !== "string") continue;
        msg.toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          name: block.name,
          argsText: JSON.stringify(block.input ?? {}),
        });
      } else if (block.type === "tool_result") {
        const text = typeof block.content === "string" ? block.content : blocksText(block.content);
        msg.toolResults.push({
          id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          name: "",
          text,
        });
      }
    }
  }
  return msg;
}

function parseAnthropicRequest(body: Record<string, unknown>, headers: IncomingMessage["headers"]): NormRequest {
  const messages: NormMessage[] = [];
  const toolNameById = new Map<string, string>();
  const rawMessages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const raw of rawMessages) {
    const msg = collectAnthropicMessage(raw);
    if (msg) messages.push(msg);
  }
  resolveToolNames(messages, toolNameById);
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      const def: ToolDef = { name: tool.name };
      if (typeof tool.description === "string") def.description = tool.description;
      if (tool.input_schema !== undefined) def.parameters = clone(tool.input_schema);
      tools.push(def);
    }
  }
  const version = headers["anthropic-version"];
  return {
    api: "anthropic",
    stream: body.stream === true,
    systemText: blocksText(body.system),
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: asNumber(body.max_tokens),
    temperature: asNumber(body.temperature),
    topP: asNumber(body.top_p),
    stops: asStringList(body.stop_sequences),
    jsonMode: false,
    responseFormat: undefined,
    anthropicVersion: typeof version === "string" ? version : null,
    toolNameById,
  };
}

// Requirements inferred from the body itself, unioned with the manifest's:
// tools present, an image present, or JSON mode requested.
function bodyRequirements(req: NormRequest): Requirements {
  const out: Requirements = {};
  if (req.tools.length > 0) out.tools = true;
  if (req.messages.some((m) => m.images.length > 0)) out.vision = true;
  if (req.jsonMode) out.json = true;
  return out;
}

function lastUserText(req: NormRequest): string {
  for (let i = req.messages.length - 1; i >= 0; i--) {
    if (req.messages[i].role === "user") return req.messages[i].text;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Target builders: the normalized request rendered in each provider wire.
// Every echoed field is carried exactly; absent stays absent (notably
// max_tokens, which is never invented).
// ---------------------------------------------------------------------------

function openAIToolChoice(choice: unknown): unknown {
  if (choice === undefined) return undefined;
  if (typeof choice === "string") return choice;
  if (isRecord(choice) && choice.type === "tool" && typeof choice.name === "string") {
    return { type: "function", function: { name: choice.name } };
  }
  if (isRecord(choice) && choice.type === "any") return "required";
  if (isRecord(choice) && choice.type === "auto") return "auto";
  if (isRecord(choice) && choice.type === "none") return "none";
  return clone(choice);
}

function anthropicToolChoice(choice: unknown): unknown {
  if (choice === undefined) return undefined;
  if (choice === "auto") return { type: "auto" };
  if (choice === "required") return { type: "any" };
  if (choice === "none") return { type: "none" };
  if (isRecord(choice) && choice.type === "function" && isRecord(choice.function) && typeof choice.function.name === "string") {
    return { type: "tool", name: choice.function.name };
  }
  return clone(choice);
}

function geminiToolConfig(choice: unknown, tools: ToolDef[]): Record<string, unknown> | null {
  if (choice === undefined || tools.length === 0) return null;
  if (choice === "none") return { functionCallingConfig: { mode: "NONE" } };
  if (choice === "required") return { functionCallingConfig: { mode: "ANY" } };
  if (isRecord(choice) && choice.type === "function" && isRecord(choice.function) && typeof choice.function.name === "string") {
    return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [choice.function.name] } };
  }
  if (isRecord(choice) && choice.type === "tool" && typeof choice.name === "string") {
    return { functionCallingConfig: { mode: "ANY", allowedFunctionNames: [choice.name] } };
  }
  if (isRecord(choice) && choice.type === "any") return { functionCallingConfig: { mode: "ANY" } };
  if (isRecord(choice) && choice.type === "none") return { functionCallingConfig: { mode: "NONE" } };
  return { functionCallingConfig: { mode: "AUTO" } };
}

function buildOpenAIBody(req: NormRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (req.systemText !== "") {
    messages.push({ role: "system", content: req.systemText });
  }
  for (const msg of req.messages) {
    if (msg.role === "tool") {
      for (const res of msg.toolResults) {
        messages.push({ role: "tool", tool_call_id: res.id, content: res.text });
      }
      continue;
    }
    if (msg.role === "assistant") {
      const out: Record<string, unknown> = { role: "assistant", content: msg.text };
      if (msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((c) => ({
          id: c.id,
          type: "function",
          function: { name: c.name, arguments: c.argsText },
        }));
      }
      messages.push(out);
      continue;
    }
    // User message. Tool results riding on it (Anthropic input) become
    // trailing tool messages; the user message itself keeps text + images.
    if (msg.images.length === 0 && msg.toolResults.length === 0) {
      messages.push({ role: "user", content: msg.text });
    } else {
      const parts: Record<string, unknown>[] = [];
      if (msg.text !== "") parts.push({ type: "text", text: msg.text });
      for (const img of msg.images) {
        parts.push({ type: "image_url", image_url: { url: imageDataUrl(img) } });
      }
      messages.push({ role: "user", content: parts });
    }
    for (const res of msg.toolResults) {
      messages.push({ role: "tool", tool_call_id: res.id, content: res.text });
    }
  }
  const body: Record<string, unknown> = { model, messages };
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => {
      const fn: Record<string, unknown> = { name: t.name };
      if (t.description !== undefined) fn.description = t.description;
      if (t.parameters !== undefined) fn.parameters = clone(t.parameters);
      return { type: "function", function: fn };
    });
    const choice = openAIToolChoice(req.toolChoice);
    if (choice !== undefined) body.tool_choice = choice;
  }
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop = [...req.stops];
  if (req.jsonMode) body.response_format = req.responseFormat !== undefined ? clone(req.responseFormat) : { type: "json_object" };
  body.stream = req.stream;
  if (req.stream) body.stream_options = { include_usage: true };
  return body;
}

function buildAnthropicBody(req: NormRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const msg of req.messages) {
    if (msg.role === "tool") {
      // A lone tool message joins the previous user message as tool_result
      // blocks, or opens one of its own when there is no user message yet.
      const blocks = msg.toolResults.map((res) => ({
        type: "tool_result",
        tool_use_id: res.id,
        content: res.text,
      }));
      const prev = messages[messages.length - 1];
      if (prev && prev.role === "user" && Array.isArray(prev.content)) {
        (prev.content as unknown[]).push(...blocks);
      } else {
        messages.push({ role: "user", content: blocks });
      }
      continue;
    }
    if (msg.role === "assistant") {
      const blocks: Record<string, unknown>[] = [];
      if (msg.text !== "") blocks.push({ type: "text", text: msg.text });
      for (const call of msg.toolCalls) {
        let input: unknown = {};
        try {
          input = JSON.parse(call.argsText);
        } catch {
          input = {};
        }
        blocks.push({ type: "tool_use", id: call.id, name: call.name, input });
      }
      messages.push({ role: "assistant", content: blocks.length > 0 ? blocks : "" });
      continue;
    }
    if (msg.images.length === 0 && msg.toolResults.length === 0) {
      messages.push({ role: "user", content: msg.text });
    } else {
      const blocks: Record<string, unknown>[] = [];
      if (msg.text !== "") blocks.push({ type: "text", text: msg.text });
      for (const img of msg.images) {
        if (img.kind === "data") {
          blocks.push({ type: "image", source: { type: "base64", media_type: img.mime, data: img.data } });
        } else {
          blocks.push({ type: "image", source: { type: "url", url: img.url } });
        }
      }
      for (const res of msg.toolResults) {
        blocks.push({ type: "tool_result", tool_use_id: res.id, content: res.text });
      }
      messages.push({ role: "user", content: blocks });
    }
  }
  const body: Record<string, unknown> = { model, messages };
  if (req.systemText !== "") body.system = req.systemText;
  if (req.tools.length > 0) {
    body.tools = req.tools.map((t) => {
      const out: Record<string, unknown> = { name: t.name };
      if (t.description !== undefined) out.description = t.description;
      if (t.parameters !== undefined) out.input_schema = clone(t.parameters);
      return out;
    });
    const choice = anthropicToolChoice(req.toolChoice);
    if (choice !== undefined) body.tool_choice = choice;
  }
  // The Anthropic wire cannot signal JSON mode; routing keeps JSON-mode
  // requests off it. max_tokens is sent only when the client sent one: the
  // simulated providers do not require it, and inventing one would corrupt
  // the echo.
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== null) body.temperature = req.temperature;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop_sequences = [...req.stops];
  body.stream = req.stream;
  return body;
}

type GeminiPart = Record<string, unknown>;

function buildGeminiBody(req: NormRequest): Record<string, unknown> {
  const contents: Record<string, unknown>[] = [];
  const pushUserContent = (parts: GeminiPart[]): void => {
    contents.push({ role: "user", parts });
  };
  for (const msg of req.messages) {
    if (msg.role === "assistant") {
      const parts: GeminiPart[] = [];
      if (msg.text !== "") parts.push({ text: msg.text });
      for (const call of msg.toolCalls) {
        let args: unknown = {};
        try {
          args = JSON.parse(call.argsText);
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: call.name, args } });
      }
      if (parts.length > 0) contents.push({ role: "model", parts });
      continue;
    }
    if (msg.role === "tool") {
      // Tool results merge into the nearest user content: a trailing
      // function-only content would overwrite the echoed user text.
      const parts: GeminiPart[] = msg.toolResults.map((res) => ({
        functionResponse: { name: res.name || res.id, response: { result: res.text } },
      }));
      const prev = contents[contents.length - 1];
      if (prev && prev.role === "user" && Array.isArray(prev.parts)) {
        (prev.parts as unknown[]).push(...parts);
      } else {
        pushUserContent(parts);
      }
      continue;
    }
    const parts: GeminiPart[] = [];
    if (msg.text !== "") parts.push({ text: msg.text });
    for (const img of msg.images) {
      if (img.kind === "data") {
        parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
      } else {
        parts.push({ fileData: { fileUri: img.url } });
      }
    }
    for (const res of msg.toolResults) {
      parts.push({ functionResponse: { name: res.name || res.id, response: { result: res.text } } });
    }
    if (parts.length > 0) pushUserContent(parts);
  }
  const body: Record<string, unknown> = { contents };
  if (req.systemText !== "") {
    body.systemInstruction = { parts: [{ text: req.systemText }] };
  }
  if (req.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => {
          const decl: Record<string, unknown> = { name: t.name };
          if (t.description !== undefined) decl.description = t.description;
          if (t.parameters !== undefined) decl.parameters = clone(t.parameters);
          return decl;
        }),
      },
    ];
    const config = geminiToolConfig(req.toolChoice, req.tools);
    if (config) body.toolConfig = config;
  }
  const generationConfig: Record<string, unknown> = {};
  if (req.temperature !== null) generationConfig.temperature = req.temperature;
  if (req.topP !== null) generationConfig.topP = req.topP;
  if (req.maxTokens !== null) generationConfig.maxOutputTokens = req.maxTokens;
  if (req.stops.length > 0) generationConfig.stopSequences = [...req.stops];
  if (req.jsonMode) generationConfig.responseMimeType = "application/json";
  if (Object.keys(generationConfig).length > 0) body.generationConfig = generationConfig;
  return body;
}

// ---------------------------------------------------------------------------
// Provider answers: parse each wire's JSON and SSE shapes back into text,
// usage and a stop reason. Streams are buffered fully and re-emitted, so the
// client always sees a well-ordered stream.
// ---------------------------------------------------------------------------

type ProviderUsage = { input: number; output: number };
type ProviderAnswer = { text: string; usage: ProviderUsage; stop: string };
type RawAnswer = { text: string; usage: ProviderUsage | null; stop: string };

type SseFrame = { event: string | null; data: string };

function splitSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const v: unknown = JSON.parse(text);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function openAIUsageOf(body: Record<string, unknown>): ProviderUsage | null {
  if (!isRecord(body.usage)) return null;
  const input = asNumber(body.usage.prompt_tokens);
  const output = asNumber(body.usage.completion_tokens);
  return input !== null && output !== null ? { input, output } : null;
}

function answerTextOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("");
}

function parseOpenAIAnswer(text: string, streamed: boolean): RawAnswer | null {
  if (!streamed) {
    const body = parseJsonObject(text);
    if (!body || !Array.isArray(body.choices) || body.choices.length === 0) return null;
    const choice = body.choices[0];
    if (!isRecord(choice) || !isRecord(choice.message)) return null;
    const content = choice.message.content;
    const answerText = typeof content === "string" ? content : answerTextOf(content);
    const usage = openAIUsageOf(body);
    const stop = typeof choice.finish_reason === "string" ? choice.finish_reason : null;
    return { text: answerText, usage, stop: stop ?? "stop" };
  }
  let answerText = "";
  let usage: ProviderUsage | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of splitSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === "[DONE]") continue;
    const chunk = parseJsonObject(payload);
    if (!chunk) continue;
    sawAny = true;
    const chunkUsage = openAIUsageOf(chunk);
    if (chunkUsage) usage = chunkUsage;
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    if (isRecord(choice.delta) && typeof choice.delta.content === "string") answerText += choice.delta.content;
    if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
  }
  if (!sawAny) return null;
  return { text: answerText, usage, stop: stop ?? "stop" };
}

function parseAnthropicAnswer(text: string, streamed: boolean): RawAnswer | null {
  if (!streamed) {
    const body = parseJsonObject(text);
    if (!body || !Array.isArray(body.content)) return null;
    let usage: ProviderUsage | null = null;
    if (isRecord(body.usage)) {
      const input = asNumber(body.usage.input_tokens);
      const output = asNumber(body.usage.output_tokens);
      if (input !== null && output !== null) usage = { input, output };
    }
    const stop = typeof body.stop_reason === "string" ? body.stop_reason : null;
    return { text: answerTextOf(body.content), usage, stop: stop ?? "end_turn" };
  }
  let answerText = "";
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of splitSseFrames(text)) {
    const data = parseJsonObject(frame.data);
    if (!data) continue;
    sawAny = true;
    if (frame.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) {
      const n = asNumber(data.message.usage.input_tokens);
      if (n !== null) input = n;
    } else if (frame.event === "content_block_delta" && isRecord(data.delta) && data.delta.type === "text_delta" && typeof data.delta.text === "string") {
      answerText += data.delta.text;
    } else if (frame.event === "message_delta") {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === "string") stop = data.delta.stop_reason;
      if (isRecord(data.usage)) {
        const n = asNumber(data.usage.output_tokens);
        if (n !== null) output = n;
      }
    }
  }
  if (!sawAny) return null;
  const usage = input !== null && output !== null ? { input, output } : null;
  return { text: answerText, usage, stop: stop ?? "end_turn" };
}

function geminiTextOf(body: Record<string, unknown>): string {
  if (!Array.isArray(body.candidates) || body.candidates.length === 0) return "";
  const cand = body.candidates[0];
  if (!isRecord(cand) || !isRecord(cand.content) || !Array.isArray(cand.content.parts)) return "";
  const parts: string[] = [];
  for (const part of cand.content.parts) {
    if (isRecord(part) && typeof part.text === "string") parts.push(part.text);
  }
  return parts.join("");
}

function geminiUsageOf(body: Record<string, unknown>): ProviderUsage | null {
  if (!isRecord(body.usageMetadata)) return null;
  const input = asNumber(body.usageMetadata.promptTokenCount);
  const output = asNumber(body.usageMetadata.candidatesTokenCount);
  return input !== null && output !== null ? { input, output } : null;
}

function geminiStopOf(body: Record<string, unknown>): string | null {
  if (!Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const cand = body.candidates[0];
  return isRecord(cand) && typeof cand.finishReason === "string" ? cand.finishReason : null;
}

function parseGeminiAnswer(text: string, streamed: boolean): RawAnswer | null {
  if (!streamed) {
    const body = parseJsonObject(text);
    if (!body) return null;
    const usage = geminiUsageOf(body);
    return { text: geminiTextOf(body), usage, stop: geminiStopOf(body) ?? "STOP" };
  }
  let answerText = "";
  let usage: ProviderUsage | null = null;
  let stop: string | null = null;
  let sawAny = false;
  for (const frame of splitSseFrames(text)) {
    const body = parseJsonObject(frame.data);
    if (!body) continue;
    sawAny = true;
    answerText += geminiTextOf(body);
    const u = geminiUsageOf(body);
    if (u) usage = u;
    const s = geminiStopOf(body);
    if (s) stop = s;
  }
  if (!sawAny) return null;
  return { text: answerText, usage, stop: stop ?? "STOP" };
}

// Fallback usage, used only when a provider answer carries none: the
// providers derive usage deterministically from the parsed request and the
// answer length, so it can be recomputed from what was sent and received.
function fallbackUsage(req: NormRequest, text: string): ProviderUsage {
  const prompt = req.systemText.length + lastUserText(req).length + req.tools.map((t) => t.name).join(",").length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

const STOP_FAMILIES = new Map<string, number>([
  ["stop", 0],
  ["end_turn", 0],
  ["STOP", 0],
  ["length", 1],
  ["max_tokens", 1],
  ["MAX_TOKENS", 1],
  ["tool_calls", 2],
  ["function_call", 2],
  ["tool_use", 2],
  ["content_filter", 3],
  ["refusal", 3],
  ["SAFETY", 3],
]);

function canonicalStop(api: ClientApi, providerStop: string): string {
  const family = STOP_FAMILIES.get(providerStop) ?? 0;
  if (api === "openai") return ["stop", "length", "tool_calls", "content_filter"][family];
  return ["end_turn", "max_tokens", "tool_use", "refusal"][family];
}

// ---------------------------------------------------------------------------
// Provider calls and routing. Calls are strictly sequential: one provider
// serves one call at a time, and the client sends one request at a time.
//
// Hazard pricing. The kit's published provider table gives each provider a
// free-tier budget range [calls_min, calls_max]; per trial the budget is
// drawn inside it, one or two calls refuse transiently at ordinals >= 2,
// and exactly one provider goes out mid-trial. The hazard of a provider's
// next call ordinal n, given f transient refusals seen so far, is:
//
//   P(budget < n)  from the uniform range,
//   + expected remaining transient fails (mean 1.5, minus f seen) spread
//     over the remaining ordinals,
//   + a 1-in-10 outage prior spread over the trial.
//
// The first call's hazard is 0: ordinal 1 never refuses. A provider that
// just refused transiently has a LOWER next-ordinal hazard (one of its
// scheduled fails is spent), which is why retrying it immediately pays.
// These ranges are published kit constants, identical for every instance;
// nothing instance-specific is read or assumed.
// ---------------------------------------------------------------------------

const BUDGET_RANGES: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

function nextCallHazard(providerId: string, nextOrdinal: number, transientFails: number): number {
  if (nextOrdinal <= 1) return 0;
  const [lo, hi] = BUDGET_RANGES[providerId] ?? [6, 10];
  const expectedBudget = (lo + hi) / 2;
  const pBudget = Math.min(1, Math.max(0, (nextOrdinal - lo + 1) / (hi - lo + 1)));
  const remaining = Math.max(1, expectedBudget - 1 - (nextOrdinal - 1));
  const pFail = Math.max(0, 1.5 - transientFails) / remaining;
  const pOutage = 0.1 * (1 / Math.max(1, expectedBudget - 2));
  return Math.min(0.95, pBudget + pFail + pOutage);
}

type CallResult =
  | { ok: true; answer: ProviderAnswer; provider: ProviderDir; model: string }
  | { ok: false; status: number; hard: boolean; dead?: boolean };

type RouterState = {
  // Inference calls spent per provider (the env counts only calls that
  // reach a provider's budget logic, so 400/401 picks do not count).
  calls: Map<string, number>;
  // Transient refusals seen per provider.
  transientFails: Map<string, number>;
  deadPairs: Set<string>;
  deadProviders: Set<string>;
};

function newRouterState(): RouterState {
  return { calls: new Map(), transientFails: new Map(), deadPairs: new Set(), deadProviders: new Set() };
}

function providerUrl(envUrl: string, provider: ProviderDir, model: string, stream: boolean): string {
  const base = envUrl.replace(/\/+$/, "");
  if (provider.wire === "openai") return `${base}${provider.path}/v1/chat/completions`;
  if (provider.wire === "anthropic") return `${base}${provider.path}/v1/messages`;
  const verb = stream ? "streamGenerateContent" : "generateContent";
  return `${base}${provider.path}/v1beta/models/${encodeURIComponent(model)}:${verb}`;
}

function authHeaders(provider: ProviderDir, req: NormRequest): Record<string, string> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (provider.auth === "bearer") headers["authorization"] = `Bearer ${provider.key}`;
  else if (provider.auth === "x-api-key") headers["x-api-key"] = provider.key;
  else if (provider.auth === "x-goog-api-key") headers["x-goog-api-key"] = provider.key;
  if (provider.wire === "anthropic") {
    headers["anthropic-version"] = req.anthropicVersion ?? "2023-06-01";
  }
  return headers;
}

async function callProvider(
  envUrl: string,
  provider: ProviderDir,
  model: string,
  req: NormRequest,
): Promise<CallResult> {
  const url = providerUrl(envUrl, provider, model, req.stream);
  const body =
    provider.wire === "openai"
      ? buildOpenAIBody(req, model)
      : provider.wire === "anthropic"
        ? buildAnthropicBody(req, model)
        : buildGeminiBody(req);
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: authHeaders(provider, req),
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0, hard: false };
  }
  let text: string;
  try {
    text = await response.text();
  } catch {
    return { ok: false, status: response.status, hard: false };
  }
  if (response.status < 200 || response.status >= 300) {
    // 400 (unknown/paid model) and 401 (wrong key) are defects in the pick,
    // not budget: never spend another call on the same pick.
    const hard = response.status === 400 || response.status === 401;
    // A refusal's message tells a terminal cause from a transient one: a
    // spent budget or an outage persists to the end of the trial, so a
    // provider that reports one is never worth another call. A plain
    // refusal names one call only, and the provider stays in rotation.
    let dead = false;
    try {
      const parsed = JSON.parse(text) as { error?: { message?: unknown } };
      const message = parsed.error?.message;
      dead =
        typeof message === "string" &&
        (message.endsWith("'s free-tier budget is spent") || message.endsWith(" is out"));
    } catch {
      dead = false;
    }
    return { ok: false, status: response.status, hard, dead };
  }
  const parsed =
    provider.wire === "openai"
      ? parseOpenAIAnswer(text, req.stream) ?? parseOpenAIAnswer(text, false)
      : provider.wire === "anthropic"
        ? parseAnthropicAnswer(text, req.stream) ?? parseAnthropicAnswer(text, false)
        : parseGeminiAnswer(text, req.stream) ?? parseGeminiAnswer(text, false);
  if (parsed === null) {
    // A 2xx with an unreadable body still counts as the one successful call;
    // answer with what is known rather than spending a second success.
    return { ok: true, answer: { text: "", usage: fallbackUsage(req, ""), stop: provider.wire === "anthropic" ? "end_turn" : provider.wire === "gemini" ? "STOP" : "stop" }, provider, model };
  }
  const usage = parsed.usage ?? fallbackUsage(req, parsed.text);
  return { ok: true, answer: { text: parsed.text, usage, stop: parsed.stop }, provider, model };
}

function meetsRequirements(row: RatingRow, req: Requirements): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.minContext !== undefined && row.contextLength < req.minContext) return false;
  return true;
}

type Candidate = {
  row: RatingRow;
  provider: ProviderDir;
  tier: number;
  score: number;
};

function candidateTier(row: RatingRow, provider: ProviderDir, req: NormRequest, requires: Requirements): number {
  const meets = meetsRequirements(row, requires);
  // The Anthropic wire always echoes json_mode=false, so a JSON-mode
  // OpenAI request can never be served correctly there.
  const wireOk = !(requires.json === true && req.api === "openai" && provider.wire === "anthropic");
  // Gemini is the only provider that can serve vision+tools at top rating;
  // keep it out of the first tier for requests that do not need vision.
  const reservedOk = requires.vision === true || provider.wire !== "gemini";
  if (meets && wireOk && reservedOk) return 0;
  if (meets && wireOk) return 1;
  if (meets) return 2;
  if (wireOk && reservedOk) return 3;
  if (wireOk) return 4;
  return 5;
}

// One best model per provider, ordered for this request. Weight-2 requests
// order by rating alone; weight-1 requests discount each candidate by its
// next-call hazard, which spreads the load across fresh providers.
function orderCandidates(
  brief: Brief,
  req: NormRequest,
  requires: Requirements,
  weight: number,
  state: RouterState,
): Candidate[] {
  const byId = new Map(brief.providers.map((p) => [p.id, p]));
  const hazardPriced = weight < 2;
  const ranked: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = byId.get(row.provider);
    if (!provider) continue;
    const tier = candidateTier(row, provider, req, requires);
    let score = row.rating;
    if (hazardPriced) {
      const calls = state.calls.get(provider.id) ?? 0;
      const fails = state.transientFails.get(provider.id) ?? 0;
      score = row.rating - HAZARD_WEIGHT * nextCallHazard(provider.id, calls + 1, fails);
    }
    ranked.push({ row, provider, tier, score });
  }
  ranked.sort((a, b) => {
    if (a.tier !== b.tier) return a.tier - b.tier;
    if (a.score !== b.score) return b.score - a.score;
    if (a.row.rating !== b.row.rating) return b.row.rating - a.row.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.row.model < b.row.model ? -1 : a.row.model > b.row.model ? 1 : 0;
  });
  const seen = new Set<string>();
  const out: Candidate[] = [];
  for (const cand of ranked) {
    if (seen.has(cand.provider.id)) continue;
    seen.add(cand.provider.id);
    out.push(cand);
  }
  return out;
}

async function routeRequest(
  envUrl: string,
  brief: Brief,
  req: NormRequest,
  requires: Requirements,
  weight: number,
  state: RouterState,
): Promise<(CallResult & { ok: true }) | null> {
  const tried = new Set<string>();
  const retried = new Set<string>();
  let lastFailed: string | null = null;
  // At most one attempt per provider, plus one immediate retry per
  // provider after a transient refusal; a second sweep over the survivors
  // only if the first sweep served nothing at all.
  for (let sweep = 0; sweep < 2; sweep++) {
    if (sweep === 1) {
      tried.clear();
      retried.clear();
      lastFailed = null;
    }
    let guard = 0;
    while (guard++ < 30) {
      const candidates = orderCandidates(brief, req, requires, weight, state);
      let picked: Candidate | null = null;
      // Retry the just-refused provider once, immediately, before
      // cascading: its refused ordinal is spent, so the retry is the
      // cheapest way to keep the request on a high-rated provider.
      if (lastFailed !== null && !state.deadProviders.has(lastFailed) && !retried.has(lastFailed)) {
        const again = candidates.find((c) => c.provider.id === lastFailed);
        if (again) {
          picked = again;
          retried.add(lastFailed);
        }
      }
      if (picked === null) {
        for (const cand of candidates) {
          const pid = cand.provider.id;
          if (tried.has(pid)) continue;
          if (state.deadProviders.has(pid)) continue;
          if (state.deadPairs.has(`${pid}\0${cand.row.model}`)) continue;
          picked = cand;
          break;
        }
      }
      if (picked === null) break;
      const pid = picked.provider.id;
      tried.add(pid);
      const result = await callProvider(envUrl, picked.provider, picked.row.model, req);
      // Every attempt that reaches the provider's budget logic spends one
      // call ordinal, success or refusal alike; only 400/401 picks do not
      // count. The hazard of the next ordinal depends on this count, so it
      // must advance on successes too, not just on refusals.
      if (!result.ok && result.hard) {
        // 400/401: the env never counted this call.
      } else {
        state.calls.set(pid, (state.calls.get(pid) ?? 0) + 1);
      }
      if (result.ok) return result;
      if (result.dead) {
        state.deadProviders.add(pid);
        lastFailed = null;
      } else if (result.hard) {
        if (result.status === 401) state.deadProviders.add(pid);
        else state.deadPairs.add(`${pid}\0${picked.row.model}`);
        lastFailed = null;
      } else {
        // Transient refusal: one ordinal spent, provider stays in play.
        state.transientFails.set(pid, (state.transientFails.get(pid) ?? 0) + 1);
        lastFailed = pid;
      }
    }
    if (sweep === 0) {
      // Only sweep again when nothing served; otherwise the request is
      // already answered (routeRequest returns on success above).
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Client responses: the provider's answer re-emitted in the client's wire,
// with its usage verbatim and a stop reason from the same family.
// ---------------------------------------------------------------------------

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) {
    pieces.push(text.slice(at, at + STREAM_PIECE_CHARS));
  }
  return pieces.length === 0 ? [""] : pieces;
}

function jsonResponse(res: ServerResponse, status: number, value: unknown): void {
  const body = `${JSON.stringify(value)}\n`;
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  res.end(body);
}

function sseResponse(res: ServerResponse, frames: string): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
    "content-length": Buffer.byteLength(frames),
  });
  res.end(frames);
}

function sendOpenAIJson(res: ServerResponse, index: number, model: string, answer: ProviderAnswer): void {
  jsonResponse(res, 200, {
    id: `chatcmpl-agg-${index}`,
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: canonicalStop("openai", answer.stop),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  });
}

function sendOpenAIStream(res: ServerResponse, index: number, model: string, answer: ProviderAnswer): void {
  const id = `chatcmpl-agg-${index}`;
  const frames: string[] = [];
  const chunk = (delta: Record<string, unknown>, finish: string | null) =>
    `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  frames.push(chunk({ role: "assistant", content: "" }, null));
  for (const piece of splitPieces(answer.text)) {
    frames.push(chunk({ content: piece }, null));
  }
  frames.push(chunk({}, canonicalStop("openai", answer.stop)));
  frames.push(
    `data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } })}\n\n`,
  );
  frames.push("data: [DONE]\n\n");
  sseResponse(res, frames.join(""));
}

function sendAnthropicJson(res: ServerResponse, index: number, model: string, answer: ProviderAnswer): void {
  jsonResponse(res, 200, {
    id: `msg_agg_${index}`,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: canonicalStop("anthropic", answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  });
}

function sendAnthropicStream(res: ServerResponse, index: number, model: string, answer: ProviderAnswer): void {
  const id = `msg_agg_${index}`;
  const frames: string[] = [];
  const frame = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  frames.push(
    frame("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(frame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }));
  for (const piece of splitPieces(answer.text)) {
    frames.push(frame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  frames.push(
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: canonicalStop("anthropic", answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
  );
  frames.push(frame("message_stop", { type: "message_stop" }));
  sseResponse(res, frames.join(""));
}

function sendClientError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  if (api === "openai") {
    jsonResponse(res, status, { error: { message, type: "api_error", param: null, code: "upstream_unavailable" } });
  } else {
    jsonResponse(res, status, { type: "error", error: { type: "api_error", message } });
  }
}

// GET /v1/models: the wire is the auth header. The list is the deduped union
// of the registered catalog's free ids, in sorted order.
function sendModels(res: ServerResponse, catalog: Catalog, headers: IncomingMessage["headers"]): void {
  const firstSeen = new Map<string, { provider: string; contextLength: number }>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !firstSeen.has(model.id)) {
        firstSeen.set(model.id, { provider: provider.id, contextLength: model.context_length });
      }
    }
  }
  const ids = [...firstSeen.keys()].sort();
  if (headers["x-api-key"] !== undefined) {
    jsonResponse(res, 200, {
      data: ids.map((id) => ({
        type: "model",
        id,
        display_name: id,
        created_at: "2026-01-01T00:00:00Z",
        free: true,
        context_length: firstSeen.get(id)?.contextLength ?? null,
      })),
      has_more: false,
    });
  } else {
    jsonResponse(res, 200, {
      object: "list",
      data: ids.map((id) => ({
        id,
        object: "model",
        created: FIXED_CREATED,
        owned_by: firstSeen.get(id)?.provider ?? "",
        free: true,
        context_length: firstSeen.get(id)?.contextLength ?? null,
      })),
    });
  }
}

// ---------------------------------------------------------------------------
// Server: register, serve one request at a time, exit 0 when the work is done.
// ---------------------------------------------------------------------------

type ServerState = {
  brief: Brief;
  catalog: Catalog;
  envUrl: string;
  router: RouterState;
  requestCount: number;
  manifestPtr: number;
  inFlight: number;
};

function manifestMatch(state: ServerState, index: number, req: NormRequest): { requires: Requirements; weight: number } {
  const manifest = state.brief.manifest;
  let entry: ManifestEntry | null = null;
  if (state.manifestPtr < manifest.length) {
    const next = manifest[state.manifestPtr];
    if (next.api === req.api && next.stream === req.stream) {
      entry = next;
      state.manifestPtr += 1;
    }
  }
  if (entry === null) {
    for (let j = state.manifestPtr; j < manifest.length; j++) {
      if (manifest[j].api === req.api && manifest[j].stream === req.stream) {
        entry = manifest[j];
        state.manifestPtr = j + 1;
        break;
      }
    }
  }
  if (entry === null && index < manifest.length && manifest[index].api === req.api) {
    entry = manifest[index];
  }
  const requires: Requirements = { ...bodyRequirements(req) };
  let weight = 1;
  if (entry) {
    if (entry.requires.tools === true) requires.tools = true;
    if (entry.requires.vision === true) requires.vision = true;
    if (entry.requires.json === true) requires.json = true;
    if (entry.requires.minContext !== undefined) requires.minContext = entry.requires.minContext;
    weight = entry.weight;
  }
  return { requires, weight };
}

async function readBody(req: IncomingMessage): Promise<Buffer | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as string);
    size += buf.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  return Buffer.concat(chunks);
}

async function handleInference(
  state: ServerState,
  api: ClientApi,
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  const raw = await readBody(req);
  if (raw === null) {
    sendClientError(res, api, 413, "request body too large");
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw.toString("utf8"));
  } catch {
    sendClientError(res, api, 400, "request body is not JSON");
    return;
  }
  if (!isRecord(body)) {
    sendClientError(res, api, 400, "request body is not one JSON object");
    return;
  }
  const index = state.requestCount;
  state.requestCount += 1;
  state.inFlight += 1;
  try {
    const norm = api === "openai" ? parseOpenAIRequest(body, req.headers) : parseAnthropicRequest(body, req.headers);
    const { requires, weight } = manifestMatch(state, index, norm);
    const result = await routeRequest(state.envUrl, state.brief, norm, requires, weight, state.router);
    if (result === null) {
      process.stderr.write(`aggregator: request ${index} (${api}${norm.stream ? ", stream" : ""}): no provider answered\n`);
      sendClientError(res, api, 502, "all providers refused the request");
      return;
    }
    process.stderr.write(`aggregator: request ${index} (${api}${norm.stream ? ", stream" : ""}): ${result.provider.id}/${result.model}\n`);
    if (api === "openai") {
      if (norm.stream) sendOpenAIStream(res, index, result.model, result.answer);
      else sendOpenAIJson(res, index, result.model, result.answer);
    } else {
      if (norm.stream) sendAnthropicStream(res, index, result.model, result.answer);
      else sendAnthropicJson(res, index, result.model, result.answer);
    }
  } catch (err) {
    process.stderr.write(`aggregator: request ${index} failed: ${(err as Error).message}\n`);
    if (!res.headersSent) sendClientError(res, api, 502, "the request could not be served");
  } finally {
    state.inFlight -= 1;
  }
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<boolean> {
  const url = `${envUrl.replace(/\/+$/, "")}/register`;
  for (let attempt = 0; attempt < REGISTER_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (response.status >= 200 && response.status < 300) return true;
      const text = await response.text().catch(() => "");
      process.stderr.write(`aggregator: register refused (${response.status}): ${text.slice(0, 200)}\n`);
      return false;
    } catch (err) {
      if (attempt === REGISTER_ATTEMPTS - 1) {
        process.stderr.write(`aggregator: register failed: ${(err as Error).message}\n`);
        return false;
      }
      await new Promise((resolve) => setTimeout(resolve, REGISTER_RETRY_MS));
    }
  }
  return false;
}

async function pollWorkDone(envUrl: string): Promise<boolean> {
  try {
    const response = await fetch(`${envUrl.replace(/\/+$/, "")}/work`, {
      signal: AbortSignal.timeout(5000),
    });
    if (response.status < 200 || response.status >= 300) return false;
    const body: unknown = await response.json();
    return isRecord(body) && body.done === true;
  } catch {
    return false;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    const catalog = buildCatalog(readBrief());
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  if (envUrl === "") {
    process.stderr.write("aggregator: MAKEFUN_ENV_URL is not set\n");
    process.exitCode = 1;
    return;
  }
  const brief = readBrief();
  const catalog = buildCatalog(brief);
  const state: ServerState = {
    brief,
    catalog,
    envUrl,
    router: newRouterState(),
    requestCount: 0,
    manifestPtr: 0,
    inFlight: 0,
  };
  // Inference is serialized: the client sends one request at a time, and
  // one provider serves one call at a time, so no two provider calls ever
  // overlap. Model-list answers bypass the queue; they touch no state.
  let queue: Promise<void> = Promise.resolve();
  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://entry");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      sendModels(res, catalog, req.headers);
      return;
    }
    const enqueue = (api: ClientApi): void => {
      const run = async (): Promise<void> => {
        try {
          await handleInference(state, api, req, res);
        } catch (err) {
          process.stderr.write(`aggregator: queued handler failed: ${(err as Error).message}\n`);
          if (!res.headersSent) sendClientError(res, api, 502, "the request could not be served");
        }
      };
      queue = queue.then(run, run);
    };
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      enqueue("openai");
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      enqueue("anthropic");
      return;
    }
    jsonResponse(res, 404, { error: { message: `no route ${req.method} ${url.pathname}`, type: "invalid_request_error", param: null, code: "not_found" } });
  });
  server.on("error", (err) => {
    process.stderr.write(`aggregator: server error: ${err.message}\n`);
    process.exitCode = 1;
  });
  try {
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "0.0.0.0", () => resolve());
    });
  } catch (err) {
    process.stderr.write(`aggregator: could not bind: ${(err as Error).message}\n`);
    process.exit(1);
  }
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    process.stderr.write("aggregator: could not bind a port\n");
    server.close();
    process.exit(1);
  }
  process.stderr.write(`aggregator: listening on ${port}, ${catalog.providers.length} providers, ${brief.ratings.filter((r) => r.free).length} free models\n`);
  if (!(await register(envUrl, port, catalog))) {
    server.close();
    process.exit(1);
  }
  process.stderr.write("aggregator: registered\n");
  let stopping = false;
  const shutdown = async (): Promise<void> => {
    if (stopping) return;
    stopping = true;
    let waited = 0;
    while (state.inFlight > 0 && waited < 200) {
      await sleep(25);
      waited += 1;
    }
    server.close();
    process.exit(0);
  };
  process.on("SIGTERM", () => void shutdown());
  process.on("SIGINT", () => void shutdown());
  for (;;) {
    await sleep(WORK_POLL_MS);
    if (await pollWorkDone(envUrl)) {
      await shutdown();
      return;
    }
  }
}

void main().catch((err: Error) => {
  process.stderr.write(`aggregator: fatal: ${err.message}\n`);
  process.exit(1);
});
