# LLMQ — Presentation Assets Guide

> Everything you need for a professional demo, README, and pitch deck.

---

## 1. Architecture Diagram

**File:** [`docs/architecture.svg`](./architecture.svg)

### Layout Structure

The SVG uses a **3-layer horizontal architecture** (dark theme, 1200 × 820 px):

| Layer | Components | Color |
|-------|-----------|-------|
| **Entry Points** (top) | CLI · CI Pipeline · REST API · Dashboard | Indigo / Orange / Rose |
| **Core Engine** (middle) | Provider Factory · Eval Runner · Metrics Engine · Quality Gate | Cyan / Emerald / Amber |
| **Persistence** (bottom) | Storage (DuckDB) · Configuration · Test Dataset | Violet |

**End-to-End Flow (bottom bar):**
```
CLI / CI  →  Dataset  →  Provider  →  Metrics  →  Quality Gate  →  Storage  →  Dashboard
```

### Embedding in README

```markdown
## Architecture

![LLMQ Architecture](docs/architecture.svg)
```

### Embedding in HTML/Slides

```html
<img src="docs/architecture.svg" alt="LLMQ Architecture" width="100%">
```

---

## 2. Demo Script (60-Second Recording)

### Prerequisites
```bash
pip install -e .
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### Demo Sequence

Record the following 4 scenes in your terminal (use a tool like [asciinema](https://asciinema.org/) or screen capture):

#### Scene 1 — `llmq init` (~10 s)
```bash
# Start recording
mkdir demo-project && cd demo-project
llmq init
```
**What to show:** Config file creation, dataset generation, success message.

#### Scene 2 — `llmq eval` (~20 s)
```bash
llmq eval --provider groq
```
**What to show:** Test case execution progress, metrics table, quality gate verdict (✅ PASSED).

#### Scene 3 — CI Fail Example (~15 s)
```bash
# Simulate a strict CI run expected to fail
llmq eval --provider groq --fail-on-gate
echo "Exit code: $?"
```
**What to show:** Quality gate with ❌ FAILED result, non-zero exit code, the metrics that breached thresholds.

#### Scene 4 — Dashboard (~15 s)
```bash
llmq dashboard &
# Open browser to http://localhost:8000
```
**What to show:** Overview page with charts, historical runs table, provider comparison, drill-down to individual test case.

### Recording Tips

| Tool | Command | Output |
|------|---------|--------|
| **asciinema** | `asciinema rec demo.cast` | `.cast` file → upload to asciinema.org |
| **VHS** (charmbracelet) | `vhs demo.tape` | `.gif` / `.webm` for README |
| **OBS Studio** | Screen record terminal | `.mp4` for presentations |
| **terminalizer** | `terminalizer record demo` | `.gif` for README |

### VHS Tape File (Automated Terminal Recording)

Save as `docs/demo.tape`:
```tape
Output docs/demo.gif
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"

Type "# LLMQ — 60-Second Demo"
Enter
Sleep 1s

Type "llmq init"
Enter
Sleep 5s

Type "llmq eval --provider groq"
Enter
Sleep 15s

Type "llmq eval --provider groq --fail-on-gate"
Enter
Sleep 10s

Type "echo 'Exit code:' $?"
Enter
Sleep 2s

Type "llmq dashboard"
Enter
Sleep 3s

Type "# Open http://localhost:8000 in browser"
Enter
Sleep 5s
```

---

## 3. Screenshot Checklist

Capture these screenshots for README/docs/pitch:

### CLI Screenshots

| # | What to Capture | Command | Filename |
|---|----------------|---------|----------|
| 1 | **Init output** | `llmq init` | `docs/images/cli-init.png` |
| 2 | **Eval results** (passing) | `llmq eval --provider groq` | `docs/images/cli-eval-pass.png` |
| 3 | **Eval results** (failing) | `llmq eval --provider groq --fail-on-gate` | `docs/images/cli-eval-fail.png` |
| 4 | **Doctor output** | `llmq doctor` | `docs/images/cli-doctor.png` |
| 5 | **Provider list** | `llmq providers` | `docs/images/cli-providers.png` |
| 6 | **Run history** | `llmq runs --limit 5` | `docs/images/cli-runs.png` |
| 7 | **Compare** | `llmq compare` | `docs/images/cli-compare.png` |

### Dashboard Screenshots

| # | What to Capture | URL Path | Filename |
|---|----------------|----------|----------|
| 8 | **Overview / metrics page** | `http://localhost:8000` | `docs/images/dashboard-overview.png` |
| 9 | **Run detail drill-down** | `http://localhost:8000` (click a run) | `docs/images/dashboard-run-detail.png` |
| 10 | **Quality gate history** | Quality Gates section | `docs/images/dashboard-quality-gates.png` |
| 11 | **Provider comparison chart** | Comparison section | `docs/images/dashboard-comparison.png` |

### CI Screenshots

| # | What to Capture | Source | Filename |
|---|----------------|--------|----------|
| 12 | **GitHub Actions workflow** | Actions tab → LLM Quality Gate | `docs/images/ci-workflow.png` |
| 13 | **PR comment with results** | Pull Request → bot comment | `docs/images/ci-pr-comment.png` |
| 14 | **CI failure badge in action** | PR checks status | `docs/images/ci-failure-badge.png` |
| 15 | **Evaluation artifact** | Actions → Artifacts tab | `docs/images/ci-artifacts.png` |

### Screenshot Best Practices

- **Terminal theme:** Use a dark theme (e.g., Catppuccin, Dracula, One Dark)
- **Font:** JetBrains Mono or Fira Code, 14–16 px
- **Width:** 120+ columns for rich table output
- **Retina/HiDPI:** Capture at 2× for crisp images
- **Crop:** Remove window chrome; show only the terminal content
- **Format:** Use PNG for CLI, PNG or WebP for dashboard

---

## 4. README Badge Examples

### Recommended Badges

Copy-paste into the top of your README.md:

```markdown
[![Tests](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate)
```

### Extended Badge Set

```markdown
<!-- CI Status -->
[![CI](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml)

<!-- Python Version -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<!-- License -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Coverage -->
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate)

<!-- Code Style -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- PRs Welcome -->
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate/pulls)

<!-- PyPI (if published) -->
<!-- [![PyPI](https://img.shields.io/pypi/v/llmq.svg)](https://pypi.org/project/llmq/) -->

<!-- Quality Gate Custom Badge -->
[![Quality Gate](https://img.shields.io/badge/LLM%20Quality%20Gate-passing-brightgreen.svg)](https://github.com/Emart29/llm-quality-gate)

<!-- Providers -->
[![Providers](https://img.shields.io/badge/LLM%20Providers-8-informational.svg)](https://github.com/Emart29/llm-quality-gate#supported-providers)
```

### Badge Preview

| Badge | Purpose |
|-------|---------|
| ![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg) | CI pipeline status |
| ![Python](https://img.shields.io/badge/python-3.8+-blue.svg) | Python version requirement |
| ![MIT](https://img.shields.io/badge/License-MIT-yellow.svg) | License type |
| ![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg) | Test coverage |
| ![Black](https://img.shields.io/badge/code%20style-black-000000.svg) | Code formatter |
| ![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg) | Contribution welcome |
| ![Gate](https://img.shields.io/badge/LLM%20Quality%20Gate-passing-brightgreen.svg) | Custom quality gate badge |
| ![Providers](https://img.shields.io/badge/LLM%20Providers-8-informational.svg) | Number of supported providers |

### Dynamic CI Badge (Already in your README)

Your existing `tests.yml` badge:
```markdown
[![Tests](https://github.com/Emart29/llm-quality-gate/actions/workflows/tests.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/tests.yml)
```

Add the quality-gate workflow badge too:
```markdown
[![LLM Quality Gate](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml/badge.svg)](https://github.com/Emart29/llm-quality-gate/actions/workflows/llm-quality-gate.yml)
```

---

## 5. File Inventory

After following this guide, your `docs/` folder should look like:

```
docs/
├── architecture.svg            ← Architecture diagram (SVG)
├── PRESENTATION_ASSETS.md      ← This guide
├── ci-performance-optimizations.md
├── demo.tape                   ← VHS script for terminal recording
└── images/
    ├── evaluation-summary.svg  ← Dashboard mock SVG
    ├── historical-runs.svg     ← Dashboard mock SVG
    ├── model-comparison.svg    ← Dashboard mock SVG
    ├── cli-init.png            ← Screenshot: llmq init
    ├── cli-eval-pass.png       ← Screenshot: eval pass
    ├── cli-eval-fail.png       ← Screenshot: eval fail
    ├── cli-doctor.png          ← Screenshot: doctor
    ├── cli-providers.png       ← Screenshot: providers
    ├── cli-runs.png            ← Screenshot: runs
    ├── cli-compare.png         ← Screenshot: compare
    ├── dashboard-overview.png  ← Screenshot: dashboard
    ├── dashboard-run-detail.png
    ├── dashboard-quality-gates.png
    ├── dashboard-comparison.png
    ├── ci-workflow.png         ← Screenshot: GitHub Actions
    ├── ci-pr-comment.png       ← Screenshot: PR comment
    ├── ci-failure-badge.png    ← Screenshot: CI failure
    └── ci-artifacts.png        ← Screenshot: artifacts
```
