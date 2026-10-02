import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// Extension under test. Imported fresh per test (query-string cache bust)
// because it keeps module-level guard state.
const EXT_URL = pathToFileURL(
	join(new URL(import.meta.url).pathname, "..", "..", "..", "extensions", "vision-context-guard.ts"),
).href;
let loadCount = 0;
async function loadExtension() {
	const mod = await import(`${EXT_URL}?fresh=${loadCount++}`);
	return mod.default;
}

function makePi() {
	const handlers = new Map();
	const commands = new Map();
	return {
		pi: {
			on: (event, handler) => {
				handlers.set(event, handler);
				return () => {};
			},
			registerCommand: (name, opts) => {
				commands.set(name, opts);
			},
			exec: async () => ({ code: 0, stdout: "", stderr: "" }),
		},
		handlers,
		commands,
	};
}

function imageBlock() {
	return { type: "image", data: "aGk=", mimeType: "image/png" };
}

function messagesWithImages(n) {
	return [{ role: "user", content: Array.from({ length: n }, imageBlock) }];
}

function countImages(messages) {
	let count = 0;
	for (const message of messages) {
		for (const block of message.content ?? []) {
			if (block?.type === "image") count++;
		}
	}
	return count;
}

function makeCtx(model) {
	const status = {};
	return {
		model,
		signal: undefined,
		modelRegistry: {
			findOfType: () => undefined,
			getModelsOfType: () => [],
			classify: async () => {
				throw new Error("classifier must not be consulted for bare images");
			},
		},
		getContextUsage: () => null,
		ui: {
			setStatus: (key, value) => {
				status[key] = value;
			},
			notify: () => {},
		},
		status,
	};
}

// Muse MSP registers baseUrl "http://localhost" as a placeholder; its real
// transport is stdio JSON-RPC to `muse serve` and the model itself is remote.
const MUSE_MSP = {
	provider: "muse-msp",
	id: "muse-spark-1.3",
	input: ["text", "image"],
	baseUrl: "http://localhost",
	contextWindow: 1_000_000,
};
const MTPLX_LOCAL = {
	provider: "mtplx",
	id: "hemingway-1-bf16-mtplx",
	input: ["text", "image"],
	baseUrl: "http://127.0.0.1:8001",
	contextWindow: 262144,
};
// Remote HTTP provider: large image histories 413 here (chat-4 on this
// exact model), so the guard must prune despite the non-localhost baseUrl.
const OPENROUTER_REMOTE = {
	provider: "openrouter",
	id: "z-ai/glm-5.3-flash",
	input: ["text", "image"],
	baseUrl: "https://openrouter.ai/api/v1",
	contextWindow: 1000000,
};
const CODEX_REMOTE = {
	provider: "openai-codex",
	id: "gpt-6-sol",
	input: ["text", "image"],
	baseUrl: "https://chatgpt.com/backend-api",
	contextWindow: 272000,
};

describe("vision-context-guard: provider scoping", () => {
	it("does not prune images or inject the guard prompt for muse-msp", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);

		const ctx = makeCtx(MUSE_MSP);
		const messages = messagesWithImages(3);
		const result = await handlers.get("context")({ messages }, ctx);
		assert.equal(result, undefined, "guard passes messages through untouched for muse-msp");
		assert.equal(countImages(messages), 3, "all images retained for remote muse-msp");

		const start = await handlers.get("before_agent_start")({ systemPrompt: "base" }, ctx);
		assert.equal(start, undefined, "no guard prompt appended for muse-msp");
	});

	it("still prunes images for genuinely local mtplx models", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);

		const ctx = makeCtx(MTPLX_LOCAL);
		const result = await handlers.get("context")({ messages: messagesWithImages(7) }, ctx);
		assert.equal(countImages(result.messages), 5, "only newest 5 images retained for local mtplx");

		const start = await handlers.get("before_agent_start")({ systemPrompt: "base" }, ctx);
		assert.match(start.systemPrompt, /Vision payload guard/, "guard prompt appended for mtplx");
	});

	it("prunes images for remote openrouter models (413 prevention)", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);

		const ctx = makeCtx(OPENROUTER_REMOTE);
		const result = await handlers.get("context")({ messages: messagesWithImages(7) }, ctx);
		assert.equal(countImages(result.messages), 5, "only newest 5 images retained for openrouter");

		const start = await handlers.get("before_agent_start")({ systemPrompt: "base" }, ctx);
		assert.match(start.systemPrompt, /Vision payload guard/, "guard prompt appended for openrouter");
	});

	it("prunes images for remote openai-codex models", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);

		const ctx = makeCtx(CODEX_REMOTE);
		const result = await handlers.get("context")({ messages: messagesWithImages(6) }, ctx);
		assert.equal(countImages(result.messages), 5, "only newest 5 images retained for codex");
	});
});
