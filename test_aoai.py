import os
import sys
from pathlib import Path

from awm.gpt import GPTClient


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def require_env(keys: list[str]) -> None:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def main() -> int:
    load_dotenv()

    require_env([
        "AWM_SYN_LLM_PROVIDER",
        "AZURE_ENDPOINT_URL",
        "AZURE_OPENAI_API_KEY",
        "AWM_SYN_OVERRIDE_MODEL",
    ])

    provider = os.environ.get("AWM_SYN_LLM_PROVIDER")
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AWM_SYN_OVERRIDE_MODEL")

    print("[AOAI] provider:", provider)
    print("[AOAI] endpoint:", os.environ.get("AZURE_ENDPOINT_URL"))
    print("[AOAI] deployment/model:", model)

    client = GPTClient(timeout=60, max_retry_num=1)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Return exactly: AOAI_OK"},
        ],
        model=model,
        max_tokens=16,
        temperature=0,
    )

    if not response:
        raise RuntimeError("AOAI request returned empty content")

    print("[AOAI] response:", response.strip())
    print("[AOAI] success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[AOAI] failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
