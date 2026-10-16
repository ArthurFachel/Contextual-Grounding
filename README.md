# The Role of Contextual Grounding in Geoscience Benchmarks

**Self-conditioned prompting for LLM evaluation on geoscientific tasks.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![BRACIS 2026](https://img.shields.io/badge/BRACIS-2026-green)](#citation)

This repository contains the code and experimental framework for the paper **"The Role of Contextual Grounding in Geoscience Benchmarks"**, accepted for publication at **BRACIS 2026** (proceedings forthcoming). We propose a self-conditioned prompting framework that augments raw geoscientific queries with structured semantic reasoning (domain classification, geological process identification, and difficulty indicators) and evaluate it across 4 open-source LLMs on the GeoBench benchmark.

**Paper authors:** Arthur Fachel, Otávio Parraga, André Luiz Durante Spigolon, George de Barros, Luiz Gonzaga Jr., Maurício Roberto Veronez, Rodrigo C. Barros, and Lucas Silveira Kupssinskü.

---

## Key Findings

- **Contextual grounding helps most models**: Mistral Large 3 gained up to **+22% accuracy** on True/False and **+18.44%** on Multiple Choice.
- **But it hurts models with small context windows**: Mixtral 8x7B degraded in True/False, Noun, and QA tasks due to context saturation.
- **Domain matters**: process-driven subdomains (Sedimentary Geology, Metamorphism, Historical Geology) benefit greatly, while spatially intensive ones (Structural Geology, Tectonic Stratigraphy) degrade.
- **Larger active parameter count** correlates with larger enhancement gains.

---

## What this repo does

This is the official implementation of the contextual grounding framework from the paper. It provides a CLI (`gas`) to:

1. **Generate** test cases from the GeoBench benchmark using an LLM
2. **Evaluate** model responses using an LLM-as-a-judge pipeline (DeepEval + Gemma 3 27B)

The framework supports two modes:
- **Baseline**: raw geoscientific queries with no enrichment
- **Enhanced**: queries augmented with structured semantic context (domain classification, geological process identification, difficulty indicators)

---

## Setup

### Prerequisites

- Python 3.11–3.12
- Poetry
- API keys for your chosen provider (AWS Bedrock, DeepInfra, or OpenAI-compatible)

### Installation

```bash
# Clone the repo
git clone https://github.com/ArthurFachel/Contextual-Grounding.git
cd Contextual-Grounding

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# Install Poetry and dependencies
pip install poetry
poetry install --with dev
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Core APIs
API_KEY=
OPENAI_API_KEY=
GEMINI_KEY=

# AWS / Bedrock (or compatible endpoints)
AWS_GENERATE_MODEL=
AWS_ENHANCE_MODEL=
AWS_ENDPOINT_URL=

# DeepInfra
DEEPINFRA_API_KEY=
DEEPINFRA_ENDPOINT_URL=

# Runtime / evaluation settings
TOKENIZERS_PARALLELISM=false
DEEPEVAL_RESULTS_FOLDER=./deepeval_results
CONFIDENT_METRIC_LOGGING_VERBOSE=0

# GPU (optional)
CUDA_VISIBLE_DEVICES=0
```

---

## Usage

Configure model parameters in `config.toml` (see `config_template.toml`), then run:

```bash
# Generate test cases
gas --log-level DEBUG --task TF --config ./config.toml --dotenv ./.env generate --limit 5

# Evaluate generated test cases
gas --log-level DEBUG --task TF --config ./config.toml --dotenv ./.env evaluate
```

### Tasks

| Flag | Task Type | Format |
|------|-----------|--------|
| `TF` | True/False | Closed-ended |
| `CHOICE` | Multiple Choice | Closed-ended |
| `NOUN` | Noun Meaning | Open-ended |
| `QA` | Question Answering | Open-ended |
| `COMPLETION` | Fill-in-the-blank | Open-ended |

### Pipeline

1. **Generate**: the model answers GeoBench questions (baseline or enhanced mode)
2. **Evaluate**: Gemma 3 27B (via Gemini API) judges responses on Alignment, Relevancy, and Correctness

---

## Models Evaluated

| Model | Active Params | Total Params | Context Length |
|---|---|---|---|
| LLaMA-4 Scout | 17B | 109B | 10M tokens |
| LLaMA-4 Maverick | 17B | 400B | 1M tokens |
| Mistral Large 3 | 41B | 675B | 256K tokens |
| Mixtral 8x7B | 12.9B | 46.7B | 32K tokens |

---

## Repository Structure

```
├── gas/
│   ├── cli.py              # CLI entry point (argparse)
│   ├── commons.py           # Shared utilities
│   ├── logger.py            # Logging configuration
│   ├── config_template.toml # Model and evaluation config
│   ├── data/
│   │   └── geobenchmark_provider.py  # GeoBench data loader
│   ├── metrics/
│   │   └── accuracy_score.py
│   ├── models/
│   │   ├── base.py          # Base model wrapper
│   │   ├── peft.py          # PEFT/LoRA model support
│   │   └── __init__.py      # AWSModel, DeepInfraModel
│   ├── pipelines/
│   │   ├── generation.py    # Test case generation pipeline
│   │   ├── evaluation.py   # Evaluation pipeline (LLM-as-judge)
│   │   └── pipeline.py      # Base pipeline
│   └── providers/
│       ├── data.py          # Data provider
│       ├── prompt.py        # Prompt templates
│       └── test_cases.py   # Test case definitions
├── results/                 # Evaluation outputs
├── config_template.toml     # Configuration template
├── pyproject.toml           # Project metadata & dependencies
└── README.md
```

---

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{fachel2026contextual,
  title={The Role of Contextual Grounding in Geoscience Benchmarks},
  author={Fachel, Arthur and Parraga, Otávio and Spigolon, André Luiz Durante and de Barros, George and Gonzaga Jr., Luiz and Veronez, Maurício Roberto and Barros, Rodrigo C. and Kupssinskü, Lucas Silveira},
  booktitle={Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS)},
  year={2026},
  note={Accepted for publication, proceedings forthcoming}
}
```

---

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This study was financed in part by CAPES (Finance Code 001), CNPq (Grant 443072/2024-8), FAPERGS (Grant 25/2551-0000891-3), and Petrobras/ANP (Grant 2023/00440-2). Supported by Kunumi Institute and PUCRS (MALTA Lab).
