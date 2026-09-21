"""Unified Multimodal Audio Transcription & Translation using OpenRouter / Gemini.

A single-stack solution that processes audio/video (.mp3, .m4a, .wav, .mp4) across:
- German (financial, legal, technical)
- Chinese (business, market theory)
- English
- Hindi mixed with Sanskrit mantras, bīja syllables, and traditional discourses.

Preserves sacred Sanskrit mantras in original Devanagari and IAST transliteration,
while translating the surrounding conversational discourses into idiomatic English.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_OPENROUTER_MODEL = "google/gemini-3.7-flash"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an expert multilingual transcription and translation specialist with deep domain knowledge in:
1. Indian sacred traditions, Vedic/Tantric philosophy, Kaulantak Peeth discourses, and Sanskrit mantras.
2. German economic, statutory, and technical literature.
3. Chinese financial, philosophical, and market analysis.
4. English technical and business documentation.

TASK:
Transcribe and translate the provided audio file into structured English markdown.

CRITICAL INSTRUCTIONS FOR SANSKRIT & HINDI:
- Maintain sacred fidelity: DO NOT invent literal English translations for sacred Sanskrit mantras, stotras, stutis, or bīja syllables (e.g. ॐ, ह्रीं, क्लीं, स्त्रीं, हुं, फट्, स्वाहा, वषट्, नमः).
- For every mantra, formula, or ritual injunction (e.g. Pavitrīkaraṇa, Ācamana, Nyāsa), provide:
  1. The exact Sanskrit text in Devanagari script.
  2. The precise IAST transliteration (e.g., Oṃ Pavītrīkaraṇa...).
  3. A short contextual note explaining the ritual significance if mentioned in the discourse.
- For the surrounding conversational or explanatory Hindi, translate it into fluent, idiomatic English.

CRITICAL INSTRUCTIONS FOR GERMAN & CHINESE:
- Provide an accurate, high-fidelity English translation of the spoken content.
- Preserve domain-specific technical terminology (e.g., German statutory terms like Gesetzliche Rentenversicherung, or specific Chinese financial/philosophical terms) with parenthetical original terms where appropriate.

OUTPUT FORMAT:
Provide the output formatted cleanly in GitHub-flavored Markdown:
- # Title / Overview of the Audio
- ## Key Metadata & Language Detection
- ## Comprehensive Transcript & Translation (chronologically organized with clear section headings or speaker markers)
- ## Key Terms, Mantras & Glossaries (if applicable)
"""


def detect_mime_type(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    mapping = {
        ".mp3": "audio/mp3",
        ".m4a": "audio/m4a",
        ".wav": "audio/wav",
        ".aac": "audio/aac",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }
    return mapping.get(ext, "audio/mp3")


def load_env_file() -> None:
    """Load key-value pairs from .env if present in current or parent directories."""
    candidates = [Path(".env"), Path(__file__).resolve().parents[1] / ".env"]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    if k and k not in os.environ:
                        os.environ[k] = v


def transcribe_via_openrouter(
    audio_path: Path,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    custom_instruction: str | None = None,
) -> str:
    """Transcribe and translate audio using OpenRouter's multimodal Gemini endpoint."""
    print(f"Reading {audio_path} ({audio_path.stat().st_size / (1024*1024):.2f} MB)...")
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = detect_mime_type(audio_path)
    data_uri = f"data:{mime_type};base64,{audio_b64}"

    instruction = SYSTEM_PROMPT
    if custom_instruction:
        instruction += f"\n\nADDITIONAL USER INSTRUCTIONS:\n{custom_instruction}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/npu-rag",
        "X-Title": "NPU RAG Multimodal Transcriber",
        "Content-Type": "application/json",
    }

    # OpenRouter accepts multimodal data URIs in image_url / file or input_audio
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                        },
                    },
                ],
            }
        ],
    }

    print(f"Sending audio to OpenRouter model [{model}]...")
    with httpx.Client(timeout=300.0) as client:
        response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            # If image_url format fails for audio, attempt standard input_audio format
            if "format" in response.text.lower() or "unsupported" in response.text.lower():
                alt_payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_b64,
                                        "format": audio_path.suffix.lstrip(".").lower(),
                                    },
                                },
                            ],
                        }
                    ],
                }
                alt_resp = client.post(OPENROUTER_API_URL, headers=headers, json=alt_payload)
                if alt_resp.status_code == 200:
                    data = alt_resp.json()
                    return str(data["choices"][0]["message"]["content"])
            raise RuntimeError(f"OpenRouter API error (HTTP {response.status_code}): {response.text}")

        data = response.json()
        return str(data["choices"][0]["message"]["content"])


def transcribe_via_gemini_direct(
    audio_path: Path,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    custom_instruction: str | None = None,
) -> str:
    """Direct Google Gemini REST endpoint fallback."""
    print(f"Reading {audio_path} ({audio_path.stat().st_size / (1024*1024):.2f} MB)...")
    audio_bytes = audio_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = detect_mime_type(audio_path)

    instruction = SYSTEM_PROMPT
    if custom_instruction:
        instruction += f"\n\nADDITIONAL USER INSTRUCTIONS:\n{custom_instruction}"

    url = f"{GEMINI_API_URL.format(model=model)}?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": instruction},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": audio_b64,
                        }
                    },
                ]
            }
        ]
    }

    print(f"Sending audio directly to Google Gemini [{model}]...")
    with httpx.Client(timeout=300.0) as client:
        response = client.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Google Gemini API error (HTTP {response.status_code}): {response.text}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates returned: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)


def main() -> None:
    # `sys.stdout` is typed `TextIO`, which has no `reconfigure`; only the
    # concrete `TextIOWrapper` does. Narrowing rather than ignoring also makes
    # this correct when stdout is redirected to something that is not a
    # wrapper, where the call would otherwise raise at runtime.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Unified Multimodal Audio Transcriber & Translator (German, Chinese, Hindi/Sanskrit, English)"
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio or video file (.mp3, .m4a, .wav, .mp4)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output markdown file path (defaults to <audio_name>.transcript.md)"
    )
    parser.add_argument(
        "-m", "--model", type=str, default=None, help=f"Model identifier (default: {DEFAULT_OPENROUTER_MODEL})"
    )
    parser.add_argument(
        "--instruction", type=str, default=None, help="Optional extra prompt instruction for domain terms"
    )

    args = parser.parse_args()

    audio_file: Path = args.audio_file
    if not audio_file.exists():
        print(f"Error: File not found: {audio_file}", file=sys.stderr)
        sys.exit(1)

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")

    if not openrouter_key and not gemini_key:
        print(
            "Error: Neither OPENROUTER_API_KEY nor GEMINI_API_KEY is set.\n"
            "Please set OPENROUTER_API_KEY in your environment or in the .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = args.output or audio_file.with_suffix(".transcript.md")

    try:
        if openrouter_key:
            model = args.model or DEFAULT_OPENROUTER_MODEL
            result = transcribe_via_openrouter(audio_file, openrouter_key, model=model, custom_instruction=args.instruction)
        elif gemini_key:
            model = args.model or DEFAULT_GEMINI_MODEL
            result = transcribe_via_gemini_direct(audio_file, gemini_key, model=model, custom_instruction=args.instruction)
        else:
            # Unreachable: the guard above exits when neither key is set. Stated
            # as a raise rather than left implicit so the dispatch narrows both
            # keys to `str`, and so a future edit that removes the guard fails
            # here instead of passing `None` as an API key.
            raise RuntimeError("no transcription credential available")

        output_path.write_text(result, encoding="utf-8")
        print(f"\n[Success] Transcription and translation saved to:\n  {output_path.resolve()}")
    except Exception as e:
        print(f"\n[Failed] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
