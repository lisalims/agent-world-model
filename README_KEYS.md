# AWM Keys and `.env` Setup

Use this file to configure API credentials without exporting variables manually in every shell session.

## 1) Create a local `.env`

```bash
cat > .env << 'EOF'
# Provider for synthesis LLM
AWM_SYN_LLM_PROVIDER=openai

# OpenAI-compatible synthesis credentials
OPENAI_API_KEY=your-openai-api-key
# OPENAI_BASE_URL=http://your-compatible-endpoint/v1

# Optional model override for synthesis
AWM_SYN_OVERRIDE_MODEL=gpt-5

# Required for scenario embedding model
EMBEDDING_OPENAI_API_KEY=your-embedding-api-key

# Azure OpenAI (use these instead of OPENAI_* when provider=azure)
# AZURE_ENDPOINT_URL=https://your-endpoint.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-azure-openai-api-key
EOF
```

## 2) Run commands with `.env`

Use `uv` to inject the env file when running AWM commands:

```bash
uv run --env-file .env awm --help
uv run --env-file .env awm gen scenario --input_path outputs/seed_scenario.jsonl --output_path outputs/gen_scenario.jsonl --target_count 1000
```

## 3) Security notes

- Keep `.env` local only.
- `.gitignore` is configured to ignore `.env` and `.env.*`.
- Use `.env.example` if you want to commit a template without real secrets.


uv run --env-file .env python test_emb.py
uv run --env-file .env python test_aoai.py 
