import os
import sys
import json
from pathlib import Path
from urllib import request, error

from openai import OpenAI


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

    require_env(["EMBEDDING_OPENAI_API_KEY"])

    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_base_url = os.environ.get("EMBEDDING_OPENAI_BASE_URL", "https://api.openai.com/v1")

    azure_endpoint = os.environ.get("EMBEDDING_AZURE_ENDPOINT")
    azure_deployment = os.environ.get("EMBEDDING_AZURE_DEPLOYMENT") or embedding_model
    azure_api_version = os.environ.get("EMBEDDING_AZURE_API_VERSION", "2024-02-01")

    if azure_endpoint:
        endpoint = azure_endpoint.rstrip("/")
        url = f"{endpoint}/openai/deployments/{azure_deployment}/embeddings?api-version={azure_api_version}"
        payload = {"input": "Hello from my embeddings test!"}
        headers = {
            "Content-Type": "application/json",
            "api-key": os.environ.get("EMBEDDING_OPENAI_API_KEY", ""),
        }

        print("[EMB] mode: azure-rest")
        print("[EMB] endpoint:", azure_endpoint)
        print("[EMB] deployment:", azure_deployment)
        print("[EMB] api_version:", azure_api_version)

        req = request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

        parsed = json.loads(body)
        data = parsed.get("data", [])
        if not data or not data[0].get("embedding"):
            raise RuntimeError(f"Embedding response is empty: {parsed}")

        print("[EMB] embedding_dim:", len(data[0]["embedding"]))
        print("[EMB] success")
        return 0

    print("[EMB] mode: openai-compatible")
    print("[EMB] base_url:", embedding_base_url)
    print("[EMB] model:", embedding_model)

    client = OpenAI(
        api_key=os.environ.get("EMBEDDING_OPENAI_API_KEY"),
        base_url=embedding_base_url,
    )

    response = client.embeddings.create(
        input=["AWM embedding health check"],
        model=embedding_model,
    )

    if not response.data or not response.data[0].embedding:
        raise RuntimeError("Embedding response is empty")

    print("[EMB] embedding_dim:", len(response.data[0].embedding))
    print("[EMB] success")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[EMB] failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
