# Changelog

All notable changes to this project will be documented in this file.
Format: [Conventional Commits](https://www.conventionalcommits.org/)

## [0.1.0] — 2026-05-23

### Added
- PAP script parser: extracts 364 operator turns from 10 clinical cases
- ABCDE phase classifier: TF-IDF + LogisticRegression (train accuracy: 99%)
- Verbal act classifier: OneVsRest multi-label (8 act types)
- LLM few-shot classifier: Claude Haiku with 5 clinical examples
- Baseline fallback: automatic on LLM timeout or API error
- FastAPI service: `GET /health`, `POST /classify`
- Pydantic v2 contracts: strict input validation, typed output
- Docker deployment: non-root user, healthcheck, layer caching
- GitHub Actions: CI (lint+test), model eval, Docker build

### Architecture Decision
- LLM-first: baseline shows 37pp train/held-out accuracy gap
- Case-level split: prevents author-style vocabulary leakage
- Held-out: Mercedes + Luis (highest OOD challenge)

### Metrics (v0.1.0)
| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Train (8 cases) | 0.990 | 0.992 |
| Held-out (Mercedes+Luis) | 0.616 | 0.535 |
| Gap | 0.374 | 0.457 |

### Known Limitations
- Verbal act labels are weak (regex heuristics, not human annotations)
- No authentication on `/classify` endpoint
- Clinical risk detection not implemented (see writeup.md)
