"""Granite Embedding 97M Multilingual R2 스냅샷 다운로드 + 무결성 확인.

GitHub Actions runner처럼 huggingface.co에 실제로 접근 가능한 환경에서만 실행한다.
사내망(HF 도메인 차단)에서는 절대 이 스크립트를 실행하지 않는다 — 애초에 실행이
안 될 것이다.

파일명을 추측하지 않는다: config.json / *.safetensors 존재만 필수로 검증하고,
onnx/openvino 파일은 glob으로 "있으면 있는 대로" 발견해서 manifest.json에 기록한다.
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "ibm-granite/granite-embedding-97m-multilingual-r2"
OUT_DIR = Path("granite-embedding-97m-multilingual-r2")
MIN_WEIGHT_MB = 1.0  # 이보다 작으면 LFS pointer 텍스트가 남아있는 것으로 간주


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

    local_dir = snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        local_dir=str(OUT_DIR),
    )
    root = Path(local_dir)
    print(f"downloaded to: {root.resolve()}")

    if not (root / "config.json").exists():
        print("FATAL: config.json missing at repo root", file=sys.stderr)
        sys.exit(1)

    weight_files = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model.bin"))
    if not weight_files:
        print(
            "FATAL: no *.safetensors / pytorch_model.bin found at repo root. "
            "레포 구조가 예상과 다를 수 있음 - 실제 파일 목록을 확인할 것.",
            file=sys.stderr,
        )
        for p in sorted(root.rglob("*")):
            print(f"  found: {p.relative_to(root)}")
        sys.exit(1)

    for f in weight_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"weight file: {f.name} ({size_mb:.1f} MB)")
        if size_mb < MIN_WEIGHT_MB:
            print(
                f"FATAL: {f.name} is only {size_mb:.3f} MB - looks like a Git LFS "
                "pointer file, not the real binary. Download did not resolve LFS content.",
                file=sys.stderr,
            )
            sys.exit(1)

    onnx_files = sorted(p.relative_to(root).as_posix() for p in root.glob("onnx/**/*") if p.is_file())
    openvino_files = sorted(p.relative_to(root).as_posix() for p in root.glob("openvino/**/*") if p.is_file())
    all_files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    print(f"\nonnx/ 하위 파일 {len(onnx_files)}개:")
    for p in onnx_files:
        print(f"  {p}")
    print(f"\nopenvino/ 하위 파일 {len(openvino_files)}개:")
    for p in openvino_files:
        print(f"  {p}")

    if not onnx_files:
        print("\nWARNING: onnx/ 폴더를 못 찾음 - 예상과 다르게 ONNX 아티팩트가 없을 수 있음 (계속 진행은 함)")
    if not openvino_files:
        print("WARNING: openvino/ 폴더를 못 찾음 - 예상과 다르게 OpenVINO 아티팩트가 없을 수 있음 (계속 진행은 함)")

    manifest = {
        "repo_id": REPO_ID,
        "revision": revision,
        "weight_files": [f.name for f in weight_files],
        "onnx_files": onnx_files,
        "openvino_files": openvino_files,
        "all_files": all_files,
    }
    Path("manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Path("MODEL_REVISION.txt").write_text(
        f"repo_id={REPO_ID}\nrevision={revision}\n", encoding="utf-8"
    )
    print("\nmanifest.json / MODEL_REVISION.txt 작성 완료")


if __name__ == "__main__":
    main()
