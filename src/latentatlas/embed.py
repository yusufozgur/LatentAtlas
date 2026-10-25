"""Embed the image, audio, video and text files in a directory with EmbeddingGemma 2.

Every file becomes one 768-d unit vector in a shared space, written to a parquet
file with columns `path`, `modality`, `sha256`, `embedder` and `embedding`.

Rerunning onto an existing output only embeds new or changed files: a file whose
content hash and modality match a row embedded with the same settings reuses
that row's embedding. Rows for files no longer in the directory are dropped.

    pixi run embed data/ embed/embeddings.parquet
"""

import argparse
import hashlib
from pathlib import Path

import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_ID = "google/embeddinggemma-2"

# The processor keeps only the first 30 s of audio by default. Raise both of its
# caps so a whole track fits, up to what the 8,192-token context allows
# (~327 s at 25 audio tokens per second). Longer audio is still truncated.
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 320
AUDIO_TOKENS_PER_SECOND = 25
AUDIO_PROCESSING_KWARGS = {"audio": {"max_length": MAX_AUDIO_SECONDS * SAMPLE_RATE, "truncation": True}}

# Everything that changes an embedding; cached rows with a different value are re-embedded
EMBEDDER = f"{MODEL_ID} max_audio={MAX_AUDIO_SECONDS}s"

EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"},
    "audio": {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
    "text": {".txt", ".md"},
}
EXT_TO_MODALITY = {ext: modality for modality, exts in EXTENSIONS.items() for ext in exts}


def find_files(directory: Path) -> list[tuple[Path, str]]:
    """Supported files under `directory`, recursively, with their modality."""
    return [
        (path, EXT_TO_MODALITY[path.suffix.lower()])
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in EXT_TO_MODALITY
    ]


def file_hash(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_cache(output: Path) -> dict[tuple[str, str], list[float]]:
    """Embeddings from a previous run with the current settings, keyed by (sha256, modality)."""
    if not output.exists():
        return {}
    previous = pl.read_parquet(output)
    if not {"sha256", "embedder"} <= set(previous.columns):
        return {}
    return {
        (sha256, modality): embedding
        for sha256, modality, embedding in previous.filter(pl.col("embedder") == EMBEDDER)
        .select("sha256", "modality", "embedding")
        .iter_rows()
    }


def write_output(output: Path, rows: list[dict]) -> None:
    """Write atomically, so an interrupted run never leaves a half-written file."""
    embeddings = pl.DataFrame(
        rows,
        schema={
            "path": pl.String,
            "modality": pl.String,
            "sha256": pl.String,
            "embedder": pl.String,
            "embedding": pl.Array(pl.Float32, len(rows[0]["embedding"])),
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".parquet.tmp")
    embeddings.write_parquet(partial)
    partial.replace(output)


def load_model() -> SentenceTransformer:
    return SentenceTransformer(
        MODEL_ID,
        # float32 is ~4x faster than the default bfloat16 on CPU
        model_kwargs={"dtype": torch.float32},
        processor_kwargs={"audio_seq_length": MAX_AUDIO_SECONDS * AUDIO_TOKENS_PER_SECOND},
    )


def embed_file(model: SentenceTransformer, path: Path, modality: str) -> list[float]:
    if modality == "text":
        embedding = model.encode(path.read_text(), prompt_name="Document", normalize_embeddings=True)
    else:
        # transformers decodes media from the path itself: images with Pillow, audio and
        # video with torchcodec (video sampled at 1 fps, at most 32 frames)
        embedding = model.encode(
            {modality: str(path)},
            processing_kwargs=AUDIO_PROCESSING_KWARGS if modality == "audio" else None,
            normalize_embeddings=True,
        )
    return embedding.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="directory to search for files, recursively")
    parser.add_argument("output", type=Path, help="parquet file to write")
    args = parser.parse_args()

    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    files = find_files(args.directory)
    if not files:
        parser.error(f"no supported files in {args.directory}")

    cache = load_cache(args.output)
    rows = []
    for path, modality in files:
        sha256 = file_hash(path)
        row = {"path": str(path), "modality": modality, "sha256": sha256, "embedder": EMBEDDER}
        rows.append(row | {"embedding": cache.get((sha256, modality))})
    todo = [row for row in rows if row["embedding"] is None]
    print(f"{len(rows) - len(todo)} of {len(rows)} files cached, {len(todo)} to embed")


    if todo:
        model = load_model()
        for row in tqdm(todo, desc="Embedding"):
            try:
                tqdm.write(f"Processing: {row['path']}")

                row["embedding"] = embed_file(
                    model,
                    Path(row["path"]),
                    row["modality"],
                )

            except Exception as exc:
                tqdm.write(
                    f"FAILED: {row['path']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            # Checkpoint after every successful embedding
            write_output(
                args.output,
                [r for r in rows if r["embedding"] is not None],
            )

    # Always synchronize output with currently existing files,
    # including when all new embeddings failed.
    successful_rows = [r for r in rows if r["embedding"] is not None]
    write_output(args.output, successful_rows)

    print(f"Wrote {len(successful_rows)} embeddings to {args.output}")

    

if __name__ == "__main__":
    main()
