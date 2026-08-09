# ClaimAgent Investigator

**An AI-powered forensic claims investigation system that determines whether visual evidence actually supports an insurance damage claim.**

VisionClaim Investigator is a multi-agent pipeline built on top of a cloud Vision-Language Model (VLM). It takes a claimant's text description and submitted photo(s), runs them through a structured chain of specialized AI agents, and produces a structured, auditable verdict — complete with risk flags, coverage analysis, and a self-critiqued justification.

---

## What It Does

Most damage claim systems do a simple keyword match or a one-shot image classifier. This system is different. It breaks the problem into distinct reasoning steps — each handled by a focused agent — and combines them into a final forensic verdict that can be explained and audited.

The core question it answers: **"Does the visual evidence actually support what the claimant says happened?"**

---

## Live Dashboard

The project ships with a real-time investigation dashboard. You submit a claim and watch each agent complete in sequence, with live output at every step.

![VisionClaim Dashboard — Package Claim](app/static/screenshots/dashboard_package.png)

![VisionClaim Dashboard — Car Claim with Full Risk Analysis](app/static/screenshots/dashboard_car.png)

---

## How It Works — Investigation Workflow

A claim goes through the following agents in order. Each agent has one job and passes its output directly to the next.

```
Claimant submits: [ text description ] + [ photo(s) ]
          │
          ▼
  Agent 1 — Claim Extractor
  Reads the conversation and pulls out the damaged part and damage type.
  Example output: { "object_part": "front_wheel", "issue_type": "dent" }
          │
          ▼
  Agent 2 — Evidence Agent
  Maps that part/damage to a checklist of what visual evidence is required.
  Example: front_wheel dent → requires [ front_wheel, front_fender ] visible.
          │
          ▼
  Agent 3 — Image Analyzer
  Sends each photo to the VLM and asks what parts are actually visible,
  whether there is real damage, and how severe it looks.
          │
          ├── Authenticity Agent (Bonus)
          │   Reads EXIF metadata to check for missing timestamps, edited photos,
          │   or suspicious camera configurations.
          │
          └── Duplicate Agent (Bonus)
              Computes a perceptual hash of every image and cross-checks it
              against all previously processed claims from the same user.
          │
          ▼
  Agent 4 — Coverage Agent
  Cross-references: "What did the claimant say?" vs "What does the image show?"
  Calculates a coverage percentage and determines if the standard is met.
          │
          ▼
  Agent 5 — History Agent
  Checks whether this user has submitted multiple claims before.
  Flags repeat filers above a configurable threshold.
          │
          ▼
  Agent 6 — Contradiction Agent
  Looks for mismatches — wrong object in the photo, text inconsistent with
  image content, or prompt injection attempts in the claim text.
          │
          ▼
  Agent 7 — Verdict Agent
  Aggregates all agent outputs and writes the full 14-column structured verdict.
          │
          ▼
  Agent 10 — Self-Critique Agent (Bonus)
  Reviews the verdict, challenges any weak reasoning, and rewrites the
  justification to be grounded strictly in what the image shows.
          │
          ▼
  Final structured output (CSV row or JSON)
```

---

## Project Structure

```
ClaimAgents/
│
├── agents/
│   ├── base_agent.py              — Shared retry logic, API client, base class
│   ├── core/
│   │   ├── claim_extractor.py     — Extracts part and damage type from text
│   │   ├── evidence_agent.py      — Builds visual evidence requirements checklist
│   │   ├── image_analyzer.py      — VLM-based per-image visual analysis
│   │   ├── coverage_agent.py      — Cross-references visible vs required parts
│   │   ├── history_agent.py       — User claim frequency and risk profiling
│   │   ├── contradiction_agent.py — Detects mismatches and prompt injections
│   │   └── verdict_agent.py       — Aggregates all evidence into final verdict
│   └── bonus/
│       ├── authenticity_agent.py  — EXIF metadata integrity checks
│       ├── duplicate_agent.py     — Perceptual image hashing (cross-claim dedup)
│       └── critique_agent.py      — Self-critique and verdict revision
│
├── app/
│   ├── main.py                    — FastAPI server, WebSocket stream handler
│   └── static/
│       ├── index.html             — Dashboard UI
│       ├── style.css              — Dashboard styling
│       └── app.js                 — WebSocket client, live pipeline rendering
│
├── pipeline/
│   └── investigator.py            — Orchestrates agent execution order and streaming
│
├── models/
│   ├── model_manager.py           — OpenRouter API client, token tracking
│   └── schemas.py                 — Pydantic schemas for all agent I/O
│
├── evaluation/
│   ├── evaluator.py               — Accuracy metrics against ground truth
│   ├── metrics.py                 — Per-field scoring logic
│   └── compare_predictions.py     — Side-by-side mismatch viewer
│
├── utils/
│   ├── csv_handler.py             — Input CSV loading and user history mapping
│   ├── image_utils.py             — Image loading, base64 encoding, EXIF reading
│   └── prompt_builder.py          — Structured prompt templates for each agent
│
├── config.py                      — Central config: model IDs, thresholds, paths
├── main.py                        — Batch pipeline runner (CSV → CSV)
├── requirements.txt
└── README.md
```

---

## Output Schema

Every processed claim produces one structured row with 14 fields:

| Field | Description |
|---|---|
| `user_id` | Claimant identifier |
| `image_paths` | Paths to submitted photos |
| `user_claim` | Original claim text |
| `claim_object` | Category (`car` / `laptop` / `package`) |
| `evidence_standard_met` | `true` / `false` — did the images meet the evidence bar? |
| `evidence_standard_met_reason` | Explanation of what was covered and what was missing |
| `risk_flags` | Semicolon-joined flags (`none`, `claim_mismatch`, `possible_manipulation`, etc.) |
| `issue_type` | Damage type (`dent`, `crack`, `scratch`, `broken_part`, `water_damage`, …) |
| `object_part` | Specific part claimed (`front_wheel`, `screen`, `charging_case`, …) |
| `claim_status` | `supported` / `contradicted` / `not_enough_information` |
| `claim_status_justification` | The forensic reasoning, revised by the critique agent |
| `supporting_image_ids` | Which images actually backed the claim |
| `valid_image` | `true` / `false` — was the submitted image usable? |
| `severity` | `low` / `medium` / `high` / `critical` / `unknown` / `none` |

---

## Technology

| Component | Choice |
|---|---|
| Vision-Language Model | `qwen/qwen2.5-vl-72b-instruct` (primary), `qwen/qwen3-vl-8b-instruct` (fallback) |
| API Gateway | OpenRouter (OpenAI-compatible, no local GPU needed) |
| Web Framework | FastAPI + WebSockets |
| Frontend | Vanilla HTML/CSS/JS — no framework dependencies |
| Data Validation | Pydantic v2 |
| Image Processing | Pillow, piexif, imagehash |

The VLM choice matters here. Qwen2.5-VL-72B is one of the strongest open vision models available — it can reliably identify specific car parts, read damage types, and assess image quality in a single API call.

---

## Performance Notes

Running the full pipeline (all 10 agents including bonuses) on a single claim takes roughly:

| Agents enabled | Avg time per claim |
|---|---|
| Core only (7 agents) | 25 – 40 seconds |
| Core + all bonus agents | 45 – 70 seconds |
| Batch of 20 claims | ~15 – 20 minutes |

Time is dominated by API round-trips to OpenRouter. Each agent makes one structured VLM call. The system uses exponential backoff on rate-limit errors and isolates failures per claim so a single bad API response doesn't break the whole batch.

---

## Running the Web Dashboard

> The dashboard is the recommended way to use this project. It gives you a live view of each agent completing in real time.

**Step 1 — Set up your API key**

Copy `.env.example` to `.env` and fill in your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-your_key_here
```

Get a free key at [openrouter.ai](https://openrouter.ai).

**Step 2 — Install dependencies**

```
pip install -r requirements.txt
```

**Step 3 — Start the server**

```
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Step 4 — Open the dashboard**

Go to `http://localhost:8001` in your browser.

**Step 5 — Submit a claim**

- Enter a User ID
- Pick the claim object type (Car / Laptop / Package)
- Write or paste the claimant's description
- Upload a photo
- Hit **Run Investigation**

You'll see each of the 7 agents light up in sequence with their live output, the Evidence Board update with coverage analysis, and the Final Verdict panel fill in at the end.

---

## What Each Dashboard Panel Shows

**Investigation Pipeline** — the left-center panel. Each agent card shows its status (running / complete), token usage, and the raw structured output it produced. You can expand any card to inspect exactly what that agent returned.

**Evidence Board** — the right panel. Shows the submitted photo, the coverage checklist (required parts vs visible parts), and the risk analysis flags (duplicate check, authenticity check, user history, image validity).

**Final Investigation Verdict** — the bottom panel. The complete structured verdict: claim status, severity, evidence coverage percentage, refined part and issue type, supporting image IDs, and the full written justification.

**Diagnostic Console Log** — bottom right. Raw timestamped pipeline logs as they come in over the WebSocket connection.

---

## Risk Flags

The system can raise any of the following flags on a claim:

| Flag | Meaning |
|---|---|
| `none` | No issues detected |
| `claim_mismatch` | What the claimant described doesn't match what the image shows |
| `possible_manipulation` | Image authenticity score below threshold |
| `user_history_risk` | User has filed multiple claims (above configured threshold) |
| `manual_review_required` | Verdict confidence too low for automated decision |
| `blurry_image` | Image quality too poor for reliable analysis |
| `wrong_angle` | Photo doesn't show the claimed part |
| `damage_not_visible` | Image is valid but the described damage isn't visible |
| `cropped_or_obstructed` | Relevant area of the image is cut off or hidden |
| `text_instruction_present` | Prompt injection attempt detected in claim text |

---

## Evaluation

The `evaluation/` folder contains tools to measure how accurately the system performs against ground truth labels.

Run accuracy metrics:
```
python evaluation/evaluator.py -p outputs/sample_output.csv -g data/sample_claims.csv
```

See mismatches side by side:
```
python evaluation/compare_predictions.py -p outputs/sample_output.csv -g data/sample_claims.csv
```

---

## Key Engineering Decisions

**Why OpenRouter instead of local models?**
Running a 72B vision model locally needs 40+ GB of VRAM and a high-end GPU. OpenRouter makes the same model available via API with no hardware requirements and generous free-tier limits. The API is OpenAI-compatible so the client code is minimal.

**Why a multi-agent chain instead of one big prompt?**
A single "analyze this claim" prompt reliably hallucinates. Breaking it into focused steps — extract the part, build the evidence standard, analyze the image, check coverage — gives each agent a narrow, verifiable job. Errors are isolated and the reasoning chain is auditable at every step.

**Why WebSockets for the dashboard?**
The pipeline takes 30–70 seconds per claim. A regular HTTP request would just hang. WebSockets let each agent push its result to the browser the moment it finishes, making the investigation feel live and interactive rather than just a loading spinner.

**Why a self-critique agent?**
The verdict agent occasionally over-commits to a status when evidence is ambiguous. The critique agent acts as a second pass — it reads the verdict, challenges any assumptions, and rewrites the justification to be strictly grounded in what's visible in the image. This substantially reduces false positives.

---

## License

MIT
