"""multilingual-e5-small 오프라인 로딩 검증 (test_granite_offline.py와 동일 원칙).

e5 계열은 query:/passage: 접두사가 필요하므로 그 형태로 인코딩해서 확인한다.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TEST_QUERIES = ["query: 환자 등록 로직"]
TEST_PASSAGES = ["passage: public void InsertPatient(...)", "passage: SELECT * FROM OP_VISIT"]
EXPECTED_DIM = 384


def check_default(model_dir: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir, device="cpu")
    q = model.encode(TEST_QUERIES, normalize_embeddings=True)
    p = model.encode(TEST_PASSAGES, normalize_embeddings=True)
    if q.shape != (len(TEST_QUERIES), EXPECTED_DIM) or p.shape != (len(TEST_PASSAGES), EXPECTED_DIM):
        raise AssertionError(f"[default] unexpected shape: q={q.shape} p={p.shape}")
    print(f"[default]  OK query_shape={q.shape} passage_shape={p.shape}")


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/test_e5_offline.py <model_dir>", file=sys.stderr)
        sys.exit(2)
    model_dir = sys.argv[1]

    if not os.path.isdir(model_dir):
        print(f"FATAL: model_dir이 존재하지 않음: {model_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"model_dir={model_dir}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    check_default(model_dir)
    print("\nALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
