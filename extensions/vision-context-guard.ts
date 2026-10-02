import type { ClassifierQuestion, ImageContent } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const MAX_EDGE = 1600;
const DEFAULT_RETAINED_IMAGES = 5;
const LONG_CONTEXT_FRACTION = 0.35;
const MAX_SOURCE_BYTES = 64 * 1024 * 1024;
// Relevance ranking uses the first available registry classifier, if any.
// Recency fallback covers no-classifier setups.
const OMITTED_TEXT =
	"[Older image omitted from this request by the local vision context guard. Use existing textual observations or reread one specific image if it is essential.]";

let enabled = true;
let retainedImages = DEFAULT_RETAINED_IMAGES;

function protectsModel(ctx: ExtensionContext): boolean {
	const model = ctx.model;
	if (!model || !model.input.includes("image")) return false;
	// The failure mode is HTTP request-body size (provider 413s), which
	// applies to every HTTP(S) provider — remote ones like openrouter just
	// as much as localhost ones. Only exempt transports known not to be
	// HTTP-size-limited: muse-msp talks over a stdio JSON-RPC bridge (its
	// "http://localhost" baseUrl is a placeholder), so body limits do not
	// apply to it.
	if (model.provider === "muse-msp") return false;
	return true;
}

function extensionFor(mimeType: string): string {
	switch (mimeType.toLowerCase()) {
		case "image/jpeg":
		case "image/jpg":
			return ".jpg";
		case "image/gif":
			return ".gif";
		case "image/tiff":
			return ".tiff";
		case "image/bmp":
			return ".bmp";
		default:
			return ".png";
	}
}

async function resizeImage(image: ImageContent, pi: ExtensionAPI, signal?: AbortSignal): Promise<ImageContent> {
	let source: Buffer;
	try {
		source = Buffer.from(image.data, "base64");
	} catch {
		return image;
	}
	if (source.length === 0 || source.length > MAX_SOURCE_BYTES) return image;

	const dir = await mkdtemp(join(tmpdir(), "pi-vision-guard-"));
	const ext = extensionFor(image.mimeType);
	const input = join(dir, `input${ext}`);
	const output = join(dir, `output${ext}`);
	try {
		await writeFile(input, source);
		const result = await pi.exec("sips", ["-Z", String(MAX_EDGE), input, "--out", output], {
			signal,
			timeout: 30_000,
		});
		if (result.code !== 0) return image;
		const resized = await readFile(output);
		if (resized.length === 0) return image;
		return { type: "image", data: resized.toString("base64"), mimeType: image.mimeType };
	} catch {
		return image;
	} finally {
		await rm(dir, { recursive: true, force: true }).catch(() => undefined);
	}
}

async function resizeImages(
	content: ImageContent[],
	pi: ExtensionAPI,
	signal?: AbortSignal,
): Promise<ImageContent[]> {
	return Promise.all(content.map((image) => resizeImage(image, pi, signal)));
}

function imageCount(messages: any[]): number {
	let count = 0;
	for (const message of messages) {
		if (!Array.isArray(message?.content)) continue;
		for (const block of message.content) if (block?.type === "image") count++;
	}
	return count;
}

function pruneOldImages(messages: any[], keep: number): { messages: any[]; omitted: number; total: number } {
	const total = imageCount(messages);
	let remaining = keep;
	let omitted = 0;

	for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex--) {
		const message = messages[messageIndex];
		if (!Array.isArray(message?.content)) continue;
		for (let contentIndex = message.content.length - 1; contentIndex >= 0; contentIndex--) {
			const block = message.content[contentIndex];
			if (block?.type !== "image") continue;
			if (remaining > 0) {
				remaining--;
				continue;
			}
			message.content[contentIndex] = { type: "text", text: OMITTED_TEXT };
			omitted++;
		}
	}
	return { messages, omitted, total };
}

interface ImageSlot {
	messageIndex: number;
	contentIndex: number;
	role: string;
	/** Adjacent text in the same message — the ranking signal. Image bytes are never sent. */
	text: string;
	recency: number; // 0 = oldest
}

function collectImageSlots(messages: any[]): ImageSlot[] {
	const slots: ImageSlot[] = [];
	for (let mi = 0; mi < messages.length; mi++) {
		const message = messages[mi];
		if (!Array.isArray(message?.content)) continue;
		const text = message.content
			.filter((b: any) => b?.type === "text" && typeof b.text === "string")
			.map((b: any) => b.text as string)
			.join("\n")
			.slice(0, 300);
		for (let ci = 0; ci < message.content.length; ci++) {
			if (message.content[ci]?.type !== "image") continue;
			slots.push({ messageIndex: mi, contentIndex: ci, role: String(message.role ?? "?"), text, recency: slots.length });
		}
	}
	return slots;
}

function lastUserText(messages: any[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message?.role !== "user" || !Array.isArray(message.content)) continue;
		const text = message.content
			.filter((b: any) => b?.type === "text" && typeof b.text === "string")
			.map((b: any) => b.text as string)
			.join("\n")
			.trim();
		if (text) return text.slice(0, 500);
	}
	return "(unknown)";
}

/** One batched classifier call scoring every candidate. Returns scores aligned with slots, or undefined on any failure. */
async function scoreKeepers(
	slots: ImageSlot[],
	taskHint: string,
	ctx: ExtensionContext,
): Promise<number[] | undefined> {
	const clf = ctx.modelRegistry.getModelsOfType("classifier")[0];
	if (!clf) return undefined;
	const questions: Record<string, ClassifierQuestion> = {};
	for (let i = 0; i < slots.length; i++) {
		questions[`keep_${i}`] = {
			type: "bool",
			instructions: `Is image [${i}] still needed to complete the current task?`,
			criteria: {
				true: "still needed; dropping it would lose information",
				false: "no longer relevant; safe to drop",
			},
		};
	}
	try {
		const result = await ctx.modelRegistry.classify(
			clf,
			{
				state: {
					task: taskHint,
					images: slots.map((s, i) => ({
						index: i,
						role: s.role,
						context: s.text.slice(0, 300) || "(no surrounding text)",
					})),
				},
				questions,
			},
			{ signal: ctx.signal },
		);
		if (result.stopReason !== "stop") {
			if (result.stopReason === "aborted" || ctx.signal?.aborted) throw new Error("aborted");
			return undefined;
		}
		const scores: number[] = [];
		for (let i = 0; i < slots.length; i++) {
			const a = result.answers[`keep_${i}`];
			if (a?.type !== "bool") return undefined;
			scores.push(a.probability);
		}
		return scores;
	} catch (e) {
		if (ctx.signal?.aborted) throw e;
		return undefined;
	}
}

/**
 * Relevance-ranked prune within the same budget the recency guard used.
 * Newest image is always kept; remaining slots go to the highest classifier
 * keep-scores, falling back to recency when there is no ranking signal
 * (bare images) or the classifier is unavailable.
 */
async function pruneRanked(
	messages: any[],
	keep: number,
	ctx: ExtensionContext,
): Promise<{ messages: any[]; omitted: number; total: number; mode: "ranked" | "recent" }> {
	const slots = collectImageSlots(messages);
	const total = slots.length;
	if (total <= keep) return { messages, omitted: 0, total, mode: "recent" };
	const newest = slots[slots.length - 1];
	const candidates = slots.slice(0, -1);
	const candidateSlots = keep - 1;
	const keepSet = new Set<ImageSlot>([newest]);
	let mode: "ranked" | "recent" = "recent";
	if (candidates.length <= candidateSlots) {
		for (const c of candidates) keepSet.add(c);
	} else if (candidateSlots > 0) {
		const hasSignal = candidates.some((c) => c.text.length > 0);
		const scores = hasSignal ? await scoreKeepers(candidates, lastUserText(messages), ctx) : undefined;
		if (scores) {
			mode = "ranked";
			const ranked = candidates
				.map((c, i) => ({ c, s: scores[i] }))
				.sort((a, b) => b.s - a.s || b.c.recency - a.c.recency)
				.slice(0, candidateSlots);
			for (const r of ranked) keepSet.add(r.c);
		} else {
			const newestFirst = [...candidates]
				.sort((a, b) => b.recency - a.recency)
				.slice(0, candidateSlots);
			for (const c of newestFirst) keepSet.add(c);
		}
	}
	let omitted = 0;
	for (const s of slots) {
		if (keepSet.has(s)) continue;
		messages[s.messageIndex].content[s.contentIndex] = { type: "text", text: OMITTED_TEXT };
		omitted++;
	}
	return { messages, omitted, total, mode };
}

export default function (pi: ExtensionAPI) {
	pi.on("input", async (event, ctx) => {
		if (!enabled || !protectsModel(ctx) || !event.images?.length) return { action: "continue" };
		const images = await resizeImages(event.images, pi, ctx.signal);
		return { action: "transform", text: event.text, images };
	});

	pi.on("tool_result", async (event, ctx) => {
		if (!enabled || !protectsModel(ctx)) return;
		const imageIndexes = event.content
			.map((block, index) => (block.type === "image" ? index : -1))
			.filter((index) => index >= 0);
		if (imageIndexes.length === 0) return;

		const images = imageIndexes.map((index) => event.content[index] as ImageContent);
		const resized = await resizeImages(images, pi, ctx.signal);
		const content = [...event.content];
		for (let i = 0; i < imageIndexes.length; i++) content[imageIndexes[i]] = resized[i];
		return { content };
	});

	pi.on("context", async (event, ctx) => {
		if (!enabled || !protectsModel(ctx)) {
			ctx.ui.setStatus("vision-context-guard", undefined);
			return;
		}
		const usage = ctx.getContextUsage();
		const longContext = Boolean(
			usage && ctx.model && usage.tokens >= ctx.model.contextWindow * LONG_CONTEXT_FRACTION,
		);
		const keep = longContext ? 1 : retainedImages;
		const result = await pruneRanked(event.messages, keep, ctx);
		ctx.ui.setStatus(
			"vision-context-guard",
			result.omitted > 0
				? `vision ${Math.min(result.total, keep)}/${result.total} images${longContext ? " · long" : ""} · ${result.mode}`
				: `vision ≤${keep} image${keep === 1 ? "" : "s"}${longContext ? " · long" : ""}`,
		);
		return { messages: result.messages };
	});

	pi.on("before_agent_start", (event, ctx) => {
		if (!enabled || !protectsModel(ctx)) return;
		return {
			systemPrompt:
				event.systemPrompt +
				`\n\nVision payload guard: images are limited to ${MAX_EDGE}px. At most ${retainedImages} image payloads are retained per request (newest always kept, older ones relevance-ranked), dropping to one total beyond ${Math.round(LONG_CONTEXT_FRACTION * 100)}% context usage. Inspect the minimum necessary images, record conclusions as text, and do not repeatedly reread render history.`,
		};
	});

	pi.on("session_shutdown", (_event, ctx) => {
		ctx.ui.setStatus("vision-context-guard", undefined);
	});

	pi.registerCommand("vision-guard", {
		description: "Show, enable, disable, or set the retained local-vision image count",
		handler: async (args, ctx) => {
			const value = args.trim().toLowerCase();
			if (value === "off") enabled = false;
			else if (value === "on") enabled = true;
			else if (/^[1-6]$/.test(value)) {
				enabled = true;
				retainedImages = Number(value);
			} else if (value && value !== "status") {
				ctx.ui.notify("Usage: /vision-guard [status|on|off|1..6]", "warning");
				return;
			}
			ctx.ui.notify(
				`Vision guard ${enabled ? "on" : "off"}; max edge ${MAX_EDGE}px; retaining ${retainedImages} image(s) per request for image-input models (muse-msp exempt).`,
				"info",
			);
		},
	});
}
