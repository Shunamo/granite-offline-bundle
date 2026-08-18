"""Qwen3-Embedding-0.6B 오프라인 로딩 검증.

Qwen3-Embedding은 query 인코딩 시 instruction 접두사를 쓰는 것이 권장 사용법이다.
config_sentence_transformers.json에 named prompt가 있으면 그걸 쓰고, 없으면
공식 모델카드의 기본 instruction을 폴백으로 쓴다 (하드코딩 추측이 아니라 파일에서
확인 후 분기).
"""

import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TEST_QUERIES = ["환자 등록 로직"]
TEST_PASSAGES = ["public void InsertPatient(...)", "SELECT * FROM OP_VISIT"]
DEFAULT_INSTRUCTION = "Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery:"


def _load_prompts(model_dir: str):
    cfg_path = os.path.join(model_dir, "config_sentence_transformers.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("prompts", {}), cfg.get("default_prompt_name")
    return {}, None


def check_default(model_dir: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir, device="cpu")
    prompts, default_name = _load_prompts(model_dir)
    print(f"prompts found in config: {list(prompts.keys())}, default={default_name}")

    query_prompt = prompts.get("query", DEFAULT_INSTRUCTION)
    q = model.encode([query_prompt + t for t in TEST_QUERIES], normalize_embeddings=True)
    p = model.encode(TEST_PASSAGES, normalize_embeddings=True)
    print(f"[default]  OK query_shape={q.shape} passage_shape={p.shape}")
    return q.shape[1]


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/test_qwen_offline.py <model_dir>", file=sys.stderr)
        sys.exit(2)
    model_dir = sys.argv[1]

    if not os.path.isdir(model_dir):
        print(f"FATAL: model_dir이 존재하지 않음: {model_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"model_dir={model_dir}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    dim = check_default(model_dir)
    print(f"\nALL OFFLINE CHECKS PASSED (embedding dim={dim})")


if __name__ == "__main__":
    main()
