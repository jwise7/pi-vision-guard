# Changelog

## 0.2.0

- Ranking uses the first available registered classifier with no built-in model preference (previously preferred a local clef provider); fully backend-agnostic, pairs with `pi-local-classifier` for offline use.
- Ranking tries each registered classifier in order until one answers: a keyless or failing first pick (e.g. a built-in cloud entry with no API key) no longer blocks a working backend later in the list. Recency fallback now only fires when no classifier answers at all.

## 0.1.0 — Initial release

- Cap image payloads per request (default 5, `/vision-guard 1..6`), dropping to 1 past 35% context usage.
- Resize attached and tool-result images to 1600px (macOS `sips`, graceful no-op elsewhere).
- Relevance-ranked pruning via any registered Pi classifier (newest always pinned, one batched keep/drop call, recency fallback).
- Exempt non-HTTP transports (`muse-msp` stdio bridge).
- Placeholder text plus a system-prompt note so the model banks image conclusions as text.
- `node --test` suites for provider scoping and ranking/fallback behavior.
