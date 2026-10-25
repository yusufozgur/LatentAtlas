# LatentAtlas

LatentAtlas lets you embed and visualize your multimedia gallery inside the latent
space of a multimodal model.

Point it at a folder of images, music, videos and text files. Every file is embedded
with [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) into one shared
768-dimensional space, so a song, a video and a photo can sit next to each other when
the model considers them similar. An interactive notebook then projects that space down
to 2D, where you can click or lasso points to see which files they are, with thumbnails.

## Requirements

- Linux on x86-64
- [pixi](https://pixi.sh), which installs everything else, including Python, PyTorch
  and FFmpeg, into a project-local environment
- About 1.5 GB of disk for the model, downloaded from Hugging Face on first run

Embedding currently runs on the CPU only.

## Quick start

```bash
git clone <this repository>
cd LatentAtlas
pixi install

# embed every supported file under data/ (searched recursively)
pixi run embed data embed/embeddings.parquet

# explore the result
pixi run edit src/latentatlas/visualization.py
```

pixi tasks run from the project root, so relative paths are resolved from there.

## Embedding

`pixi run embed <directory> <output.parquet>` finds every supported file under
`<directory>` and writes one row per file:

| Column | Contents |
|---|---|
| `path` | the file's path |
| `modality` | `image`, `audio`, `video` or `text` |
| `sha256` | hash of the file's contents |
| `embedder` | the model and settings the embedding was made with |
| `embedding` | 768 float32 values, normalized to length 1 |

Supported files:

| Modality | Extensions | How it's embedded |
|---|---|---|
| Image | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tiff` | resized to the model's ~280-token budget (about 800 × 800 for a square image), keeping the aspect ratio; only the first frame of a GIF |
| Audio | `.mp3` `.wav` `.flac` `.ogg` `.m4a` `.opus` | resampled to 16 kHz mono; up to the first 320 seconds |
| Video | `.mp4` `.mov` `.mkv` `.webm` `.avi` | 1 frame per second, at most 32 frames spread across the clip |
| Text | `.txt` `.md` | the file's contents, up to 8,192 tokens |

Decoding is done by the model's own processor, via Pillow for images and
[torchcodec](https://github.com/pytorch/torchcodec) with FFmpeg for audio and video.

### Re-running is cheap

Running `embed` again onto the same output only embeds new or changed files. A file
whose contents (by SHA-256), modality and embedding settings match a row in the
existing output keeps that row's embedding, even if it was renamed or moved. Rows for
files that no longer exist are dropped, and changing the model or its settings
re-embeds everything.

The output is saved after every file, so an interrupted run picks up where it stopped.

### How long it takes

Timings on a 4-thread CPU, without a GPU:

| Input | Time |
|---|---|
| Image | 12–16 s |
| 3-minute song | about 1–1.5 min |
| 60-second video | about 3 min |

## Visualization

[`src/latentatlas/visualization.py`](src/latentatlas/visualization.py) is a
[marimo](https://marimo.io) notebook that reads `embed/embeddings.parquet` and shows:

- a 2D scatterplot of all files, using PCA or UMAP, colored and shaped by modality
- click, box or lasso selection, which lists the selected files' paths
- thumbnails for the selection: the image itself, a video's middle frame, or an audio
  file's embedded cover art

## License

LatentAtlas is released under the [MIT License](LICENSE). The EmbeddingGemma 2 model is
not part of this repository. It is downloaded from Hugging Face and is covered by its
own license.
