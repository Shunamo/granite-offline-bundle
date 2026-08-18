"""intfloat/multilingual-e5-small 스냅샷 다운로드 + 무결성 확인.

download_granite.py와 동일한 원칙: 파일명을 추측하지 않고, config.json +
*.safetensors(or pytorch_model.bin) 존재만 필수 검증하며 나머지는 glob으로 발견한다.
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "intfloat/multilingual-e5-small"
OUT_DIR = Path("multilingual-e5-small")
MIN_WEIGHT_MB = 1.0


def resolve_revision(pinned: str) -> str:
    if pinned:
        print(f"pinned revision requested: {pinned}")
        return pinned
    api = HfApi()
    info = api.model_info(REPO_ID, revision="main")
    print(f"resolved 'main' -> commit sha: {info.sha}")
    return info.sha


def main():
    pinned = os.environ.get("HF_REVISION_INPUT", "").strip()
    revision = resolve_revision(pinned)

    local_dir = snapshot_download(repo_id=REPO_ID, revision=revision, local_dir=str(OUT_DIR))
    root = Path(local_dir)
    print(f"downloaded to: {root.resolve()}")

    if not (root / "config.json").exists():
        print("FATAL: config.json missing at repo root", file=sys.stderr)
        sys.exit(1)

    weight_files = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model.bin"))
    if not weight_files:
        print("FATAL: no *.safetensors / pytorch_model.bin found at repo root", file=sys.stderr)
        for p in sorted(root.rglob("*")):
            print(f"  found: {p.relative_to(root)}")
        sys.exit(1)

    for f in weight_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"weight file: {f.name} ({size_mb:.1f} MB)")
        if size_mb < MIN_WEIGHT_MB:
            print(f"FATAL: {f.name} looks like an LFS pointer, not real binary ({size_mb:.3f} MB)", file=sys.stderr)
            sys.exit(1)

    onnx_files = sorted(p.relative_to(root).as_posix() for p in root.glob("onnx/**/*") if p.is_file())
    openvino_files = sorted(p.relative_to(root).as_posix() for p in root.glob("openvino/**/*") if p.is_file())
    all_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    print(f"\nonnx/ 하위 파일 {len(onnx_files)}개: {onnx_files}")
    print(f"openvino/ 하위 파일 {len(openvino_files)}개: {openvino_files}")

    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "weight_files": [f.name for f in weight_files],
        "onnx_files": onnx_files,
        "openvino_files": openvino_files,
        "all_files": all_files,
    }
    Path("manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("MODEL_REVISION.txt").write_text(f"repo_id={REPO_ID}\nrevision={revision}\n", encoding="utf-8")
    print("\nmanifest.json / MODEL_REVISION.txt 작성 완료")


if __name__ == "__main__":
    main()
