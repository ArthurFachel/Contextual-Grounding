# 🌍 Geoscience AI Showdown 🌋

**A Python project for benchmarking and comparing LLMs for geoscience tasks.**

<div align="center">

[![Programming Language](https://img.shields.io/badge/Python-3.12-blue)]()
[![Code Style](https://img.shields.io/badge/code%20style-pep8-orange.svg)]()
[![Stability](https://img.shields.io/badge/stability-alpha-f4d03f.svg)]()

</div>

---

## 🚀 About GeoAI Showdown

Geoscience AI Showdown is a Python-based toolkit designed to benchmark and compare the performance of Large Language Models (LLMs) on various geoscience tasks. This project is a step forward in integrating AI and geoscience, enabling the evaluation of state-of-the-art models in tasks like geoscience reasoning, Q&A, and more.

---

## 🛠️ Setup

### ✅ Prerequisites

#### 1. Microsoft C++ Build Tools
Download the installer from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
During installation, select the following workloads:
  - Desktop development with C++
  - MSVC (Microsoft C++ compiler)
  - C++ CMake tools for Windows
  - Windows 11 SDK

#### 2. CUDA Toolkit
Verify CUDA Installation:
```bash
nvcc --version
```
If CUDA is not installed download it from [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

Check the current Nvidia driver version:
```bash
nvidia-smi
```
verify the driver version and [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#use-the-right-compat-package).
If the driver is incompatible or not installed, download the correct driver from [Nvidia Drivers](https://www.nvidia.com/en-us/drivers/).

### 🖥️ Environment Configuration

#### 1. Create a Virtual Environment
Inside the project directory, create a virtual environment:
```bash
python -m venv .venv
```
Then, activate the virtual environment:
```bash
.venv\Scripts\activate
```

#### 2. Install Dependencies
Install Poetry to manage dependencies:
```bash
pip install poetry
```

Then, install the project dependencies:
```bash
poetry install --with dev
```

#### 3. Set Environment Variables

Create a `.env` file in the root directory and configure the required variables:

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

Note: The `CUDA_VISIBLE_DEVICES` variable is set to `0` by default. If you have multiple GPUs, you can change this value to the desired GPU index or unset it to use all available GPUs.

## 🌀 Usage

In order to conduct a benchmark, first you need to configure the model parameters in the `config.toml` file. The configuration file allows you to specify the model you want to benchmark and their parameters. Please refer to config_template.toml for a template configuration file.


### Run the CLI
```bash
gas --help
```

The pipeline is separated into two main components: `generate` and `evaluate`.
- `generate`: This component is responsible for generating test cases using the specified LLM. It takes the model name and parameters from the configuration file and the LLM generates text based on the provided task.
- `evaluate`: This component evaluates the generated text against a set of geoscience-specific tasks. It uses the pre-defined evaluation metrics to assess the performance of the LLM.

### Example Command for Generating Test Cases
```bash
gas --log-level DEBUG --task TF --config ./config.toml --dotenv ./.env generate --limit 5
```
This command will generate test cases using the specified LLM and parameters from the configuration file. The `--limit` flag specifies the number of test cases samples to generate.

### Example Command for Evaluating Test Cases
```bash
gas --log-level DEBUG --task TF --config ./config.toml --dotenv ./.env evaluate
```

## 🧪 Features
- Benchmarking: Compare the performance of LLMs on geoscience-specific tasks (open-ended and closed-ended).
- Customizable Configurations: Easily modify model parameters in the `config.toml` file.
- Benchmark using GEOBENCH NPEE:
  - Open-ended: noun meaning, Q&A and completion (fill-in-the-blank)
  - Closed-ended: multiple choice, true/false
- Evaluation Metrics:
  - Open-ended: PromptAlignmentMetric, Answer Relevancy, Correctness, and Semantic Similarity with BERTScore
  - Closed-ended: Accuracy.
- GPU Support: Optimized for CUDA-enabled systems.

