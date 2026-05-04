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
