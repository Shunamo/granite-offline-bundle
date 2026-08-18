"""Granite Embedding 97M R2 오프라인 로딩 검증.

목적: 압축 해제된 로컬 모델 디렉터리가 huggingface.co에 전혀 접근하지 않고도
sentence-transformers로 로드/인코딩되는지 확인한다.

사용법:
    python tests/test_granite_offline.py <모델_디렉터리_경로>

CI(build-granite-offline.yml)에서는 huggingface.co를 /etc/hosts에서 블랙홀
처리한 뒤 이 스크립트를 호출해서 "진짜로 네트워크 없이도 되는지"까지 검증한다.
사내 PC에 반입한 뒤 로컬에서 재실행해서 그대로 재검증할 수도 있다.

app/embeddings.py는 건드리지 않는다 - 이건 순수 검증용 스크립트다.
"""

import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TEST_TEXTS = [
    "환자 등록 로직",
    "public void InsertPatient(...)",
    "SELECT * FROM OP_VISIT",
]
EXPECTED_DIM = 384


def check_default(model_dir: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir, device="cpu")
    vecs = model.encode(TEST_TEXTS, normalize_embeddings=True)
    if vecs.shape != (len(TEST_TEXTS), EXPECTED_DIM):
        raise AssertionError(f"[default] unexpected shape: {vecs.shape}")
    print(f"[default]  OK shape={vecs.shape}")


def check_backend(model_dir: str, backend: str) -> bool:
    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_dir, device="cpu", backend=backend)
    except Exception as e:
        print(f"[{backend}]  SKIP (이 아티팩트/환경에서 backend 사용 불가: {type(e).__name__}: {e})")
        return False

    vecs = model.encode(TEST_TEXTS, normalize_embeddings=True)
    if vecs.shape != (len(TEST_TEXTS), EXPECTED_DIM):
        raise AssertionError(f"[{backend}] unexpected shape: {vecs.shape}")
    print(f"[{backend}]  OK shape={vecs.shape}")
    return True


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/test_granite_offline.py <model_dir>", file=sys.stderr)
        sys.exit(2)
    model_dir = sys.argv[1]

    if not os.path.isdir(model_dir):
        print(f"FATAL: model_dir이 존재하지 않음: {model_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"model_dir={model_dir}")
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}  TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

    check_default(model_dir)
    onnx_ok = check_backend(model_dir, "onnx")
    openvino_ok = check_backend(model_dir, "openvino")

    print("\nALL OFFLINE CHECKS PASSED (default 필수, onnx/openvino는 있으면 검증 없으면 SKIP)")
    print(f"summary: default=OK onnx={'OK' if onnx_ok else 'SKIP'} openvino={'OK' if openvino_ok else 'SKIP'}")


if __name__ == "__main__":
    main()
