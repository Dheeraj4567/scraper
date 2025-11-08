"""Download the recommended multimodal LLaVA model for Scarper.

This script fetches a Metal-friendly quantised build of LLaVA v1.6 Mistral 7B
(Q4_K_M) together with its multimodal projector. The model performs strongly on
M1 Pro machines while keeping memory usage practical (~8.5 GB at runtime).

Usage:
    python scripts/download_model.py --target ./models

Requires the `huggingface_hub` dependency and a Hugging Face access token if the
model repo requires authentication (currently public).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_REPO = "neoform-ai/LLaVA-v1.6-mistral-7b-GGUF"
MODEL_FILENAME = "llava-v1.6-mistral-7b.Q4_K_M.gguf"
MMPROJ_FILENAME = "llava-v1.6-mistral-7b-mmproj.gguf"


def download_file(filename: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(repo_id=MODEL_REPO, filename=filename, cache_dir=str(target_dir))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LLaVA weights")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("./models"),
        help="Directory where the GGUF files should be stored",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    print(f"Downloading model files into {target} ...")

    model_path = download_file(MODEL_FILENAME, target)
    projector_path = download_file(MMPROJ_FILENAME, target)

    print("Model file:", model_path)
    print("Projector file:", projector_path)
    print("Done. Update SCARPER_LLM_MODEL_PATH if you move the files elsewhere.")


if __name__ == "__main__":
    main()
