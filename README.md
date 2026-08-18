# granite-offline-bundle

사내망에서 huggingface.co가 차단되어 있어서, GitHub Actions(HF 접근 가능한 러너)에서
`ibm-granite/granite-embedding-97m-multilingual-r2`를 대신 받아 오프라인 zip으로
묶어주는 워크플로입니다. 이 저장소에는 회사 코드가 전혀 포함되지 않습니다 —
순수하게 공개된 HF 모델을 받아서 압축하는 작업만 합니다.

## 사용법

1. GitHub Actions 탭 → `Build Granite Embedding 97M R2 Offline Bundle` → `Run workflow`
2. 완료되면 해당 실행(run)의 Artifacts에서 다음을 받습니다:
   - `granite-embedding-97m-multilingual-r2-offline.zip`
   - `granite-embedding-97m-multilingual-r2-offline.sha256`
   - `MODEL_REVISION.txt` (다운로드한 정확한 HF commit SHA)
   - `manifest.json` (실제로 어떤 파일이 들어있는지 - onnx/openvino 포함 여부 등)
3. zip을 사내 PC로 가져온 뒤 체크섬 검증:
   ```bash
   sha256sum -c granite-embedding-97m-multilingual-r2-offline.sha256
   ```
4. 압축 해제 후 `tests/test_granite_offline.py <해제한_모델_경로>`로 로컬에서도
   오프라인 로딩이 되는지 재검증 가능합니다.

워크플로 자체가 CI 안에서 이미 "네트워크 차단 상태로 압축 해제 후 재로딩"까지
검증하고 나서 Artifact를 올리므로, 사내 반입 승인 요청 시 이 실행 로그를 근거로
쓸 수 있습니다.
