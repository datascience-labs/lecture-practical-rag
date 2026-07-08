#!/usr/bin/env python3

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image


PROMPT_HEADING_RE = re.compile(r"^###\s+(?P<slug>[^\s]+)\s*$")
FILENAME_RE = re.compile(r"^- 파일명:\s*(?P<filename>.+)\s*$")
PROMPT_RE = re.compile(r"^- 프롬프트:\s*(?P<prompt>.+)\s*$")


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        width_text, height_text = size_text.lower().split("x", 1)
        return int(width_text), int(height_text)
    except ValueError as exc:
        raise SystemExit(f"잘못된 크기 형식입니다: {size_text}") from exc


def load_prompt_entries(prompt_file: Path) -> list[dict[str, str]]:
    lines = prompt_file.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in lines:
        heading_match = PROMPT_HEADING_RE.match(line)
        if heading_match:
            if current:
                validate_entry(current, prompt_file)
                entries.append(current)
            current = {"slug": heading_match.group("slug")}
            continue

        if current is None:
            continue

        filename_match = FILENAME_RE.match(line)
        if filename_match:
            current["filename"] = filename_match.group("filename").strip()
            continue

        prompt_match = PROMPT_RE.match(line)
        if prompt_match:
            current["prompt"] = prompt_match.group("prompt").strip()

    if current:
        validate_entry(current, prompt_file)
        entries.append(current)

    if not entries:
        raise SystemExit(f"프롬프트 항목을 찾지 못했습니다: {prompt_file}")

    return entries


def validate_entry(entry: dict[str, str], prompt_file: Path) -> None:
    missing = [key for key in ("slug", "filename", "prompt") if key not in entry]
    if missing:
        raise SystemExit(
            f"프롬프트 파일 형식 오류: {prompt_file} / 누락 필드: {', '.join(missing)}"
        )


def load_env_file(env_path: Path) -> None:
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


def create_client() -> OpenAI:
    load_env_file(Path(".env"))

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY 또는 CHATGPT_API_KEY 환경 변수가 필요합니다.")

    client_kwargs = {"api_key": api_key}
    if os.getenv("OPENAI_BASE_URL"):
        client_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    if os.getenv("OPENAI_ORG_ID"):
        client_kwargs["organization"] = os.getenv("OPENAI_ORG_ID")
    if os.getenv("OPENAI_PROJECT"):
        client_kwargs["project"] = os.getenv("OPENAI_PROJECT")
    return OpenAI(**client_kwargs)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def cmd_prepare(args: argparse.Namespace) -> int:
    entries = load_prompt_entries(args.prompts)
    request_size = args.request_size or os.getenv("OPENAI_IMAGE_REQUEST_SIZE", "2048x1152")
    export_size = args.export_size or os.getenv("OPENAI_IMAGE_EXPORT_SIZE", "1920x1080")
    model = args.model or os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
    quality = args.quality or os.getenv("OPENAI_IMAGE_QUALITY", "medium")
    background = args.background or os.getenv("OPENAI_IMAGE_BACKGROUND", "auto")
    moderation = args.moderation or os.getenv("OPENAI_IMAGE_MODERATION", "auto")
    output_format = args.output_format or os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "png")

    requests = []
    manifest = []
    for entry in entries:
        custom_id = Path(entry["filename"]).stem
        request_body = {
            "model": model,
            "prompt": entry["prompt"],
            "size": request_size,
            "quality": quality,
            "background": background,
            "moderation": moderation,
            "output_format": output_format,
        }
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/images/generations",
                "body": request_body,
            }
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "slug": entry["slug"],
                "filename": entry["filename"],
                "prompt": entry["prompt"],
                "request_size": request_size,
                "export_size": export_size,
                "output_format": output_format,
            }
        )

    ensure_parent(args.jsonl)
    with args.jsonl.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")

    ensure_parent(args.manifest)
    args.manifest.write_text(
        json.dumps(
            {
                "prompt_file": str(args.prompts),
                "model": model,
                "request_size": request_size,
                "export_size": export_size,
                "quality": quality,
                "background": background,
                "moderation": moderation,
                "output_format": output_format,
                "entries": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"JSONL 생성 완료: {args.jsonl}")
    print(f"매니페스트 생성 완료: {args.manifest}")
    print(f"요청 수: {len(requests)}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    client = create_client()
    with args.jsonl.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/images/generations",
        completion_window="24h",
        metadata={
            "source": str(args.jsonl),
            "manifest": str(args.manifest),
        },
    )

    ensure_parent(args.batch_info)
    args.batch_info.write_text(json.dumps(batch.model_dump(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"업로드 파일 ID: {uploaded.id}")
    print(f"배치 ID: {batch.id}")
    print(f"상태: {batch.status}")
    print(f"배치 정보 저장: {args.batch_info}")
    return 0


def load_batch_id(args: argparse.Namespace) -> str:
    if args.batch_id:
        return args.batch_id
    if not args.batch_info.exists():
        raise SystemExit(f"배치 정보 파일이 없습니다: {args.batch_info}")
    batch_info = json.loads(args.batch_info.read_text(encoding="utf-8"))
    batch_id = batch_info.get("id")
    if not batch_id:
        raise SystemExit(f"배치 ID를 찾을 수 없습니다: {args.batch_info}")
    return batch_id


def cmd_status(args: argparse.Namespace) -> int:
    client = create_client()
    batch = client.batches.retrieve(load_batch_id(args))
    print(json.dumps(batch.model_dump(), ensure_ascii=False, indent=2))
    return 0


def extract_image_bytes(response_body: dict) -> bytes:
    data = response_body.get("data") or []
    if not data:
        raise ValueError("응답에 이미지 데이터가 없습니다.")

    image_item = data[0]
    if "b64_json" in image_item:
        return base64.b64decode(image_item["b64_json"])
    if "result" in image_item:
        return base64.b64decode(image_item["result"])
    raise ValueError("지원하지 않는 이미지 응답 형식입니다.")


def resize_image_bytes(image_bytes: bytes, export_size: str) -> bytes:
    export_width, export_height = parse_size(export_size)
    image = Image.open(BytesIO(image_bytes))
    if image.size == (export_width, export_height):
        return image_bytes

    resized = image.resize((export_width, export_height), Image.LANCZOS)
    buffer = BytesIO()
    image_format = image.format or "PNG"
    resized.save(buffer, format=image_format)
    return buffer.getvalue()


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {entry["custom_id"]: entry for entry in payload["entries"]}


def cmd_download(args: argparse.Namespace) -> int:
    client = create_client()
    batch = client.batches.retrieve(load_batch_id(args))
    if batch.status not in {"completed", "finalizing", "expired", "cancelled"}:
        raise SystemExit(f"배치가 아직 완료되지 않았습니다. 현재 상태: {batch.status}")
    if not batch.output_file_id:
        raise SystemExit("출력 파일이 아직 준비되지 않았습니다.")

    manifest = load_manifest(args.manifest)
    output_text = client.files.content(batch.output_file_id).text
    raw_output_path = args.output_dir / "batch_output.jsonl"
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.write_text(output_text, encoding="utf-8")

    saved = 0
    revised_prompts: dict[str, str] = {}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("error"):
            continue

        custom_id = item["custom_id"]
        manifest_entry = manifest.get(custom_id)
        if not manifest_entry:
            continue

        image_bytes = extract_image_bytes(item["response"]["body"])
        image_bytes = resize_image_bytes(image_bytes, manifest_entry["export_size"])
        destination = args.output_dir / manifest_entry["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        saved += 1

        data = item["response"]["body"].get("data") or []
        if data and data[0].get("revised_prompt"):
            revised_prompts[manifest_entry["filename"]] = data[0]["revised_prompt"]

    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        (args.output_dir / "batch_errors.jsonl").write_text(error_text, encoding="utf-8")

    if revised_prompts:
        (args.output_dir / "revised-prompts.json").write_text(
            json.dumps(revised_prompts, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"이미지 저장 완료: {saved}개 / {args.output_dir}")
    print(f"원본 배치 출력 저장: {raw_output_path}")
    if batch.error_file_id:
        print(f"에러 출력 저장: {args.output_dir / 'batch_errors.jsonl'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI Batch API로 슬라이드 이미지를 생성합니다.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="프롬프트 Markdown을 Batch JSONL로 변환")
    prepare.add_argument("--prompts", type=Path, default=Path("slides/image-prompts.md"))
    prepare.add_argument("--jsonl", type=Path, default=Path("out/openai/image-batch-input.jsonl"))
    prepare.add_argument("--manifest", type=Path, default=Path("out/openai/image-batch-manifest.json"))
    prepare.add_argument("--model")
    prepare.add_argument("--request-size")
    prepare.add_argument("--export-size")
    prepare.add_argument("--quality")
    prepare.add_argument("--background")
    prepare.add_argument("--moderation")
    prepare.add_argument("--output-format")
    prepare.set_defaults(func=cmd_prepare)

    submit = subparsers.add_parser("submit", help="생성된 JSONL을 업로드하고 배치 생성")
    submit.add_argument("--jsonl", type=Path, default=Path("out/openai/image-batch-input.jsonl"))
    submit.add_argument("--manifest", type=Path, default=Path("out/openai/image-batch-manifest.json"))
    submit.add_argument("--batch-info", type=Path, default=Path("out/openai/image-batch.json"))
    submit.set_defaults(func=cmd_submit)

    status = subparsers.add_parser("status", help="배치 상태 조회")
    status.add_argument("--batch-id")
    status.add_argument("--batch-info", type=Path, default=Path("out/openai/image-batch.json"))
    status.set_defaults(func=cmd_status)

    download = subparsers.add_parser("download", help="완료된 배치 결과를 PNG 파일로 저장")
    download.add_argument("--batch-id")
    download.add_argument("--batch-info", type=Path, default=Path("out/openai/image-batch.json"))
    download.add_argument("--manifest", type=Path, default=Path("out/openai/image-batch-manifest.json"))
    download.add_argument("--output-dir", type=Path, default=Path("slides/generated"))
    download.set_defaults(func=cmd_download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
