import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

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
	return {
		pi: {
			on: (event, handler) => {
				handlers.set(event, handler);
				return () => {};
			},
			registerCommand: () => {},
			exec: async () => ({ code: 0, stdout: "", stderr: "" }),
		},
		handlers,
	};
}

function makeCtx(registry) {
	return {
		model: { provider: "mtplx", id: "qwen3.8-27b", input: ["text", "image"], contextWindow: 131072 },
		signal: undefined,
		modelRegistry: registry ?? {
			getModelsOfType: () => [{ id: "some-model" }],
			classify: async () => {
				throw new Error("classify must be stubbed per test");
			},
		},
		getContextUsage: () => null,
		ui: { setStatus: () => {}, notify: () => {} },
	};
}

function msg(role, text, data) {
	const content = [];
	if (text) content.push({ type: "text", text });
	content.push({ type: "image", data, mimeType: "image/png" });
	return { role, content };
}

function remainingImageData(messages) {
	const out = [];
	for (const m of messages) {
		for (const b of m.content ?? []) {
			if (b?.type === "image") out.push(b.data);
		}
	}
	return out;
}

// Deliberately generic model id: ranking must not depend on which classifier.
function registryWithScores(scores) {
	let calls = 0;
	const registry = {
		getModelsOfType: (type) => {
			assert.equal(type, "classifier");
			return [{ id: "some-model" }];
		},
		classify: async (_model, context) => {
			calls++;
			assert.equal(Object.keys(context.questions).length, scores.length, "one question per candidate");
			const answers = {};
			for (let i = 0; i < scores.length; i++) {
				answers[`keep_${i}`] = { type: "bool", probability: scores[i] };
			}
			return { stopReason: "stop", answers };
		},
	};
	return { registry, calls: () => calls };
}

function registryWithModels(models, classifyImpl) {
	return {
		getModelsOfType: (type) => {
			assert.equal(type, "classifier");
			return models;
		},
		classify: classifyImpl,
	};
}

function scoredAnswers(scores) {
	const answers = {};
	for (let i = 0; i < scores.length; i++) {
		answers[`keep_${i}`] = { type: "bool", probability: scores[i] };
	}
	return answers;
}

function sixMessages() {
	return [
		msg("user", "receipt for the acme purchase", "img0"),
		msg("user", "older screenshot", "img1"),
		msg("user", "random meme", "img2"),
		msg("user", "error dialog from the failing test", "img3"),
		msg("user", "config screen", "img4"),
		msg("user", "current screenshot", "img5"),
	];
}

function silentRegistry() {
	return {
		getModelsOfType: () => {
			throw new Error("registry must not be consulted");
		},
		classify: async () => {
			throw new Error("registry must not be consulted");
		},
	};
}

describe("vision-context-guard: classifier relevance ranking", () => {
	it("pins newest and fills remaining budget by keep-score, not recency", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const { registry, calls } = registryWithScores([0.9, 0.2, 0.1, 0.85, 0.3, 0.05]);
		const messages = [
			msg("user", "receipt for the acme purchase, keep for totals", "img0"),
			msg("assistant", "noted the receipt", "img1"),
			msg("user", "random meme", "img2"),
			msg("user", "error dialog from the failing test", "img3"),
			msg("user", "config screen", "img4"),
			msg("user", "another meme", "img5"),
			msg("user", "current screenshot, what changed?", "img6"),
		];
		const result = await handlers.get("context")({ messages }, makeCtx(registry));
		assert.equal(calls(), 1, "one batched classify call");
		assert.deepEqual(remainingImageData(result.messages), ["img0", "img1", "img3", "img4", "img6"]);
		assert.match(result.messages[2].content[1].text, /omitted/, "dropped image replaced with placeholder");
	});

	it("falls back to newest-first when classification throws", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const registry = {
			getModelsOfType: () => [{ id: "some-model" }],
			classify: async () => {
				throw new Error("backend down");
			},
		};
		const messages = [
			msg("user", "receipt for the acme purchase", "img0"),
			msg("user", "older screenshot", "img1"),
			msg("user", "screenshot", "img2"),
			msg("user", "screenshot", "img3"),
			msg("user", "screenshot", "img4"),
			msg("user", "current screenshot", "img5"),
		];
		const result = await handlers.get("context")({ messages }, makeCtx(registry));
		assert.deepEqual(remainingImageData(result.messages), ["img1", "img2", "img3", "img4", "img5"]);
	});

	it("falls back to newest-first on stopReason error", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const registry = {
			getModelsOfType: () => [{ id: "some-model" }],
			classify: async () => ({ stopReason: "error", errorMessage: "boom", answers: {} }),
		};
		const messages = [
			msg("user", "receipt for the acme purchase", "img0"),
			msg("user", "older screenshot", "img1"),
			msg("user", "screenshot", "img2"),
			msg("user", "screenshot", "img3"),
			msg("user", "screenshot", "img4"),
			msg("user", "current screenshot", "img5"),
		];
		const result = await handlers.get("context")({ messages }, makeCtx(registry));
		assert.deepEqual(remainingImageData(result.messages), ["img1", "img2", "img3", "img4", "img5"]);
	});

	it("falls back to newest-first when no classifier is registered", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const registry = {
			getModelsOfType: () => [],
			classify: async () => {
				throw new Error("must not be called without a model");
			},
		};
		const messages = [
			msg("user", "receipt for the acme purchase", "img0"),
			msg("user", "older screenshot", "img1"),
			msg("user", "screenshot", "img2"),
			msg("user", "screenshot", "img3"),
			msg("user", "screenshot", "img4"),
			msg("user", "current screenshot", "img5"),
		];
		const result = await handlers.get("context")({ messages }, makeCtx(registry));
		assert.deepEqual(remainingImageData(result.messages), ["img1", "img2", "img3", "img4", "img5"]);
	});

	it("skips the classifier entirely when images carry no ranking signal", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const messages = Array.from({ length: 6 }, (_, i) => ({
			role: "user",
			content: [{ type: "image", data: `img${i}`, mimeType: "image/png" }],
		}));
		const result = await handlers.get("context")({ messages }, makeCtx(silentRegistry()));
		assert.deepEqual(remainingImageData(result.messages), ["img1", "img2", "img3", "img4", "img5"]);
	});

	it("makes no classify call when images fit the budget", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const messages = [msg("user", "only image", "img0")];
		const result = await handlers.get("context")({ messages }, makeCtx(silentRegistry()));
		assert.deepEqual(remainingImageData(result.messages), ["img0"]);
	});

	it("tries the next classifier when the first one throws", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const seen = [];
		const registry = registryWithModels([{ id: "dead-cloud" }, { id: "live-local" }], async (model) => {
			seen.push(model.id);
			if (model.id === "dead-cloud") throw new Error("401 unauthorized");
			return { stopReason: "stop", answers: scoredAnswers([0.9, 0.2, 0.1, 0.85, 0.3]) };
		});
		const result = await handlers.get("context")({ messages: sixMessages() }, makeCtx(registry));
		assert.deepEqual(seen, ["dead-cloud", "live-local"], "failed first pick must not stop failover");
		assert.deepEqual(remainingImageData(result.messages), ["img0", "img1", "img3", "img4", "img5"]);
	});

	it("tries the next classifier when the first returns a non-stop result", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const seen = [];
		const registry = registryWithModels([{ id: "keyless" }, { id: "live-local" }], async (model) => {
			seen.push(model.id);
			if (model.id === "keyless") return { stopReason: "error", errorMessage: "no key", answers: {} };
			return { stopReason: "stop", answers: scoredAnswers([0.9, 0.2, 0.1, 0.85, 0.3]) };
		});
		const result = await handlers.get("context")({ messages: sixMessages() }, makeCtx(registry));
		assert.deepEqual(seen, ["keyless", "live-local"]);
		assert.deepEqual(remainingImageData(result.messages), ["img0", "img1", "img3", "img4", "img5"]);
	});

	it("falls back to newest-first when every classifier fails", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		const seen = [];
		const registry = registryWithModels([{ id: "dead-a" }, { id: "dead-b" }], async (model) => {
			seen.push(model.id);
			throw new Error("backend down");
		});
		const result = await handlers.get("context")({ messages: sixMessages() }, makeCtx(registry));
		assert.deepEqual(seen, ["dead-a", "dead-b"], "every candidate tried before giving up");
		assert.deepEqual(remainingImageData(result.messages), ["img1", "img2", "img3", "img4", "img5"]);
	});

	it("rethrows instead of failing over when the turn aborts", async () => {
		const { pi, handlers } = makePi();
		(await loadExtension())(pi);
		let calls = 0;
		const registry = registryWithModels([{ id: "first" }, { id: "second" }], async () => {
			calls++;
			throw new Error("backend down");
		});
		const controller = new AbortController();
		controller.abort();
		const ctx = makeCtx(registry);
		ctx.signal = controller.signal;
		await assert.rejects(handlers.get("context")({ messages: sixMessages() }, ctx));
		assert.ok(calls <= 1, `abort short-circuits failover (calls=${calls})`);
	});
});
