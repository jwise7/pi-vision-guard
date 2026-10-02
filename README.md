# pi-vision-guard

A [Pi coding-agent](https://github.com/earendil-works/pi) extension that keeps image payloads from blowing up long sessions. Every screenshot, render, and pasted image gets re-sent on every request; past a handful of them, providers start rejecting requests (HTTP 413) or the context window fills with pixels instead of work. This guard caps how many images reach the model, shrinks the ones that do, and — when a decision-model classifier is available — keeps the *relevant* ones instead of just the newest.

## Install

```bash
pi install npm:pi-vision-guard
```

Requires Pi 1.0 or later (uses the native classifier registry). No configuration needed; the guard is on from the first session.

## Usage

```
/vision-guard [status|on|off|1..6]
```

- `status` (default) — show mode and retained-image budget.
- `on` / `off` — enable or disable the guard.
- `1..6` — set how many image payloads are retained per request (default 5).

The status bar shows what happened per request, e.g. `vision 5/9 images · ranked` or `vision ≤5 images`.

## How it works

Three layers, all automatic:

1. **Resize.** Attached and tool-result images are capped at 1600px on the long edge (macOS `sips`; other platforms skip this step and keep everything else).
2. **Budget.** At most 5 image payloads are sent per request, dropping to 1 total past 35% context usage. The newest image is always kept; older ones beyond budget become a short placeholder note.
3. **Relevance ranking.** When any Pi classifier is registered (a local Jev-style model, TypeSafe `jev-latest`, etc.), one batched keep/drop call ranks older images from their surrounding text and keeps the highest scorers. No classifier, no signal, or any failure → newest-first fallback. Image bytes are never sent to the classifier, only short text descriptors.

## Classifier ranking

Ranking lights up automatically with any registered Pi classifier — no settings. A local `clef-local/clef-flash` provider is preferred when present, otherwise the first available classifier is used. The questions are plain keep/drop judgments, so any Jev-family decision model works.

## ⚠️ Cache-churn caveat

This guard reprunes to exactly N images on *every* request: each new image flips one more old image block into placeholder text, which rewrites the transcript prefix. Providers that cache by exact prefix may therefore serve fewer requests from cache in image-heavy sessions than they would with no pruning at all. In practice the alternative is worse — unpruned image histories 413 or exhaust the window — but if you are tuning for maximum cache hits above all else, raise the budget (`/vision-guard 6`) or disable the guard and accept the size. Text-output pruners with batched schedules (e.g. pi-condense) have the same fundamental tradeoff; this guard simply chooses recency caps over batching because a single oversized vision request fails hard instead of degrading gracefully.

## Development

```bash
npm test   # node --test, no dependencies to install
```

Layout: `extensions/vision-context-guard.ts` is the whole extension; `tests/vision-guard/` holds the `node --test` suites (provider scoping + ranking/fallback, with a stubbed classifier registry).

## License

MIT
