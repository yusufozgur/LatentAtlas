import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import functools
    import html
    import io
    import subprocess
    from pathlib import Path

    import marimo as mo
    import plotly.express as px
    import polars as pl
    from PIL import Image
    from sklearn.decomposition import PCA
    from torchcodec.decoders import VideoDecoder
    import umap

    return (
        Image,
        PCA,
        Path,
        VideoDecoder,
        functools,
        html,
        io,
        mo,
        pl,
        px,
        subprocess,
        umap,
    )


@app.cell
def _(mo):
    ROOT = mo.notebook_dir().parents[1]
    EMBEDDINGS_PATH = ROOT / "embed" / "embeddings.parquet"

    THUMBNAIL_SIZE = 192  # px, longest side
    MAX_THUMBNAILS = 60  # cap so a huge lasso selection stays responsive
    return EMBEDDINGS_PATH, MAX_THUMBNAILS, ROOT, THUMBNAIL_SIZE


@app.cell
def _(mo):
    reload_button = mo.ui.run_button(label="Reload embeddings")
    reload_button
    return (reload_button,)


@app.cell
def _(EMBEDDINGS_PATH, mo, pl, reload_button):
    mo.stop(not reload_button.value)

    embeddings = pl.read_parquet(EMBEDDINGS_PATH).with_row_index("row")
    embeddings
    return (embeddings,)


@app.cell
def _(mo):
    projection = mo.ui.radio(["PCA", "UMAP"], value="PCA", label="Projection", inline=True)
    projection
    return (projection,)


@app.cell
def _(PCA, embeddings, mo, pl, projection, px, umap):
    # fixed color + symbol per modality, so a modality keeps its look whichever files are present
    _modality_colors = {"image": "#2a78d6", "audio": "#eb6834", "video": "#1baf7a", "text": "#eda100"}
    _vectors = embeddings["embedding"].to_numpy()

    if projection.value == "UMAP":
        mo.stop(embeddings.height < 3, mo.md("_UMAP needs at least 3 files._"))
        # cosine suits unit-length embeddings; a fixed seed keeps the layout stable between runs
        _xy = umap.UMAP(
            n_neighbors=min(15, embeddings.height - 1), metric="cosine", random_state=42
        ).fit_transform(_vectors)
        _axis_labels = {"x": "UMAP 1", "y": "UMAP 2"}
    else:
        _pca = PCA(n_components=2)
        _xy = _pca.fit_transform(_vectors)
        _var = _pca.explained_variance_ratio_
        _axis_labels = {"x": f"PC1 ({_var[0]:.0%} of variance)", "y": f"PC2 ({_var[1]:.0%} of variance)"}

    _points = embeddings.select(
        "row",
        pl.col("path").str.split("/").list.last().alias("file"),
        "modality",
        pl.Series("x", _xy[:, 0]),
        pl.Series("y", _xy[:, 1]),
    )

    projection_fig = px.scatter(
        _points,
        x="x",
        y="y",
        color="modality",
        symbol="modality",
        hover_name="file",
        # px splits points into one trace per modality, so carry the row to map selections back
        custom_data=["row"],
        category_orders={"modality": list(_modality_colors)},
        color_discrete_map=_modality_colors,
        labels=_axis_labels,
        title=f"Embeddings projected to 2D with {projection.value}",
        template="plotly_white",
    )
    projection_fig.update_traces(
        marker={"size": 12, "line": {"width": 2, "color": "white"}},
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )
    projection_fig.update_layout(dragmode="lasso")

    projection_plot = mo.ui.plotly(projection_fig)
    projection_plot
    return projection_fig, projection_plot


@app.cell
def _(
    Image,
    Path,
    ROOT,
    THUMBNAIL_SIZE,
    VideoDecoder,
    functools,
    io,
    subprocess,
):
    def resolve_path(path: str) -> Path:
        """Paths in the parquet are relative to where embed ran, normally the project root."""
        return Path(path) if Path(path).is_absolute() else ROOT / path


    @functools.cache
    def thumbnail(path: Path, modality: str) -> bytes | None:
        """PNG thumbnail of an image, a video's middle frame, or an audio file's cover art."""
        try:
            match modality:
                case "image":
                    image = Image.open(path)
                case "video":
                    decoder = VideoDecoder(str(path), dimension_order="NHWC")
                    image = Image.fromarray(decoder[len(decoder) // 2].numpy())
                case "audio":
                    # torchcodec can't seek the attached-picture stream of an mp3, so ask ffmpeg
                    cover = subprocess.run(
                        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                         "-f", "image2pipe", "-c:v", "png", "-"],
                        capture_output=True,
                    )
                    if cover.returncode != 0 or not cover.stdout:
                        return None
                    image = Image.open(io.BytesIO(cover.stdout))
                case _:
                    return None
        except (OSError, RuntimeError, ValueError, IndexError):
            return None

        image = image.convert("RGB")
        image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return resolve_path, thumbnail


@app.cell
def _(
    MAX_THUMBNAILS,
    Path,
    THUMBNAIL_SIZE,
    html,
    mo,
    resolve_path,
    selected,
    thumbnail,
):
    _cards = []
    for _path, _modality in selected.head(MAX_THUMBNAILS).iter_rows():
        _png = thumbnail(resolve_path(_path), _modality)
        if _png is not None:
            _caption = mo.Html(
                f'<div style="max-width:{THUMBNAIL_SIZE}px;font-size:0.8em;overflow-wrap:anywhere">'
                f"{html.escape(Path(_path).name)}</div>"
            )
            _cards.append(mo.vstack([mo.image(_png, width=THUMBNAIL_SIZE, rounded=True), _caption], gap=0.25))

    _note = (
        mo.md(f"_Showing thumbnails for the first {MAX_THUMBNAILS} of {selected.height} selected._")
        if selected.height > MAX_THUMBNAILS
        else None
    )
    if _cards:
        _gallery = mo.vstack([_note, mo.hstack(_cards, wrap=True, justify="start", gap=1)])
    elif selected.height:
        _gallery = mo.md("_No thumbnails for this selection._")
    else:
        _gallery = None
    _gallery
    return


@app.cell
def _(embeddings, mo, pl, projection_fig, projection_plot):
    _rows = {
        int(projection_fig.data[p["curveNumber"]].customdata[p["pointIndex"]][0])
        for p in projection_plot.value
        if "curveNumber" in p and "pointIndex" in p
    }
    selected = embeddings.filter(pl.col("row").is_in(_rows)).select("path", "modality")

    mo.vstack(
        [
            mo.md(f"**{selected.height} selected**: click a point, or lasso / box-select several"),
            mo.ui.table(selected, selection=None) if selected.height else None,
        ]
    )
    return (selected,)


if __name__ == "__main__":
    app.run()
