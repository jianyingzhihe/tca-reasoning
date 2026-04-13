from __future__ import annotations

from huggingface_hub import hf_hub_download


CHECKS = [
    ("google/gemma-3-4b-it", "config.json"),
    ("tianhux2/gemma3-4b-it-plt", "config.yaml"),
]


def main() -> int:
    ok = True
    for repo_id, filename in CHECKS:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[OK] {repo_id} -> {path}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[ERR] {repo_id}: {type(exc).__name__}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

