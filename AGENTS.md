# Project Agent Rules

## 1. Mission

This project is a local text-to-image engine. It prioritizes:

- Reliable local image generation
- Measurable image quality
- Reproducibility
- Stability
- Low resource usage
- Minimal dependencies
- Controlled incremental development

## 2. Scope Discipline

**Agents MUST:**

- Inspect before modifying
- Understand the requested scope
- Make the smallest change that solves the task
- Preserve existing behavior unless the task explicitly requires changing it

**Agents MUST NOT:**

- Perform unrelated refactors
- Rename files unnecessarily
- Rewrite working code for style preference
- Introduce speculative abstractions
- Change architecture without explicit approval
- Expand task scope automatically

> When in doubt, stop and ask rather than expanding scope.

## 3. Change Safety

Before modifying code:

- Inspect relevant files
- Identify dependencies
- Identify regression risks

After modifying code:

- Run appropriate syntax/tests
- Inspect the diff
- Verify that only intended files changed

Never hide or overwrite unrelated user changes.

## 4. Dependencies

Do not add, remove, upgrade, or downgrade dependencies unless explicitly required by the task.

Avoid dependency expansion.

Every new dependency must have a documented reason.

Never create `.env` handling or add `python-dotenv` unless explicitly requested.

## 5. Secrets and Security

API keys and secrets MUST:

- Remain server-side
- Never be hardcoded
- Never appear in frontend code
- Never appear in logs
- Never appear in benchmark history
- Never appear in git commits
- Never be printed in reports

For Groq:

- Use `GROQ_API_KEY`
- Never expose its value
- Never ask the user to paste the key into chat

If a secret is accidentally exposed, stop and report it.

## 6. Image Generation Engine

Protect the generation pipeline.

Do not change:

- Model
- Scheduler
- Steps
- Guidance
- Resolution
- Dtype
- Memory/offload behavior
- Inference architecture

unless the task explicitly concerns that component.

Never forcibly terminate a running CUDA/PyTorch generation worker.

Preserve safe shutdown behavior.

Do not introduce concurrency changes casually.

The GTX 1060 6GB is a real deployment constraint.

Avoid changes that create unnecessary VRAM pressure or CUDA OOM risk.

## 7. Prompt Enhancement

Groq is an optional text-generation enhancement layer.

Important:

- Groq does NOT generate images.
- Stable Diffusion remains the image-generation engine.
- Prompt enhancement must never silently trigger image generation.
- Enhancement failures must not break normal local generation.
- API keys remain server-side.
- Enhanced prompts must be safely rendered as text.

Do not couple Groq availability to the basic image-generation path unless explicitly requested.

## 8. Benchmark System

The `benchmark/` directory is permanent project infrastructure.

`benchmark/history.json` is append-only.

Never:

- Delete historical benchmark records
- Rewrite previous measurements
- Fabricate scores
- Estimate missing measurements as if they were measured
- Silently change benchmark methodology

Benchmark prompts in `benchmark/prompts.json` are stable.

Do not modify benchmark prompts casually.

If the benchmark suite must change, document a new prompt-suite version.

The benchmark history is the source of truth for measured engine quality.

Engineering maturity and image quality are separate concepts.

Do not claim that an engine is "better" without benchmark evidence.

## 9. Benchmark Rules

When benchmarking:

Keep these stable whenever possible:

- Prompt
- Negative prompt
- Seed
- Resolution
- Model

Change only the parameter being investigated.

Record actual:

- Engine version
- Git commit
- Model
- Scheduler
- Steps
- Guidance
- Resolution
- Seed
- Generation time
- VRAM
- Scores
- Notes

Never fabricate missing values.

A benchmark result must be reproducible from its recorded configuration.

## 10. Evaluation

Use the project's documented 1–10 scoring system.

Evaluate:

- Prompt adherence
- Composition
- Detail
- Anatomy
- Artifacts
- Overall

Anatomy may be null when irrelevant.

Artifacts use the project's convention: higher score = fewer visible defects.

Overall is a human evaluation and must not be presented as an automatically calculated objective metric.

## 11. Testing

Before declaring a change complete:

Run the smallest appropriate verification first.

Depending on scope, this may include:

- Python syntax checks
- JavaScript syntax checks
- E2E tests
- Regression tests
- Benchmark comparison
- Security checks
- `git diff --check`

Do not claim a test passed unless it was actually run.

Do not claim a benchmark improvement unless it was actually measured.

## 12. Git Rules

Never commit or push unless explicitly requested.

Before a commit:

- Inspect `git status`
- Inspect `git diff`
- Run `git diff --check`
- Verify changed-file scope
- Verify no secrets
- Verify no generated/runtime files are staged

Do not commit:

- API keys
- `.env`
- Generated output images
- Runtime history
- Temporary files
- Local test artifacts
- Development-only E2E scripts unless explicitly requested

Never amend or rewrite historical commits unless explicitly requested.

## 13. Release Discipline

A release requires:

1. Implementation
2. Tests
3. Regression verification
4. Scope audit
5. Documentation verification
6. Release audit
7. Explicit commit
8. Explicit push

Do not combine unrelated changes into a release.

A release must have one clear purpose.

## 14. Severity

Use:

- **P0** — Correctness, data loss, security, broken core behavior, release-blocking regression
- **P1** — Serious functional or accessibility regression
- **P2** — Non-blocking defect, documentation gap, minor cleanup

Do not label cosmetic issues as P0/P1.

Any P0 or P1 must block release until resolved.

## 15. Files and Runtime Data

Respect the distinction between:

- Source code
- Tracked project configuration
- Benchmark history
- Runtime data
- Generated images
- Development-only tools

Do not commit runtime state unless explicitly required.

Do not delete user-generated images or history as part of cleanup.

## 16. Reporting

Every completed task should report:

- What changed
- What did not change
- Tests performed
- Benchmark results if applicable
- Remaining issues
- Git state when relevant

Never hide failures.

Never turn "not tested" into "passed."

Use precise language:

- PASS = actually verified
- FAIL = actually failed
- NOT RUN = not executed
- NOT MEASURED = no valid measurement exists

## 17. Final Safety Rule

The agent must never assume permission.

Explicit user scope overrides default agent behavior.

If a requested change conflicts with these rules:

- Explain the conflict
- Do not silently bypass the rule
- Ask for explicit clarification when necessary

> Inspect first. Change minimally. Measure honestly. Test explicitly. Never expand scope without permission.
