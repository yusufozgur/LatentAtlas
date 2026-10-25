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
    from archetypes import AA
    import numpy as np

    return (
        AA,
        Image,
        PCA,
        Path,
        VideoDecoder,
        functools,
        html,
        io,
        mo,
        np,
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
    projection = mo.ui.radio(["PCA", "UMAP","AA"], value="PCA", label="Projection", inline=True)
    projection
    return (projection,)


@app.cell
def _(AA, PCA, embeddings, mo, np, pl, projection, px, umap):


    N_ARCHETYPES = 3
    _vectors = embeddings["embedding"].to_numpy()

    if projection.value == "UMAP":
        mo.stop(embeddings.height < 3, mo.md("_UMAP needs at least 3 files._"))
        _xy = umap.UMAP(
            n_neighbors=min(15, embeddings.height - 1),
            metric="cosine",
            random_state=42,
        ).fit_transform(_vectors)
        _axis_labels = {"x": "UMAP 1", "y": "UMAP 2"}

    elif projection.value == "AA":
        mo.stop(
            embeddings.height < N_ARCHETYPES or N_ARCHETYPES < 2,
            mo.md(f"_AA requires 2 ≤ N_ARCHETYPES ≤ {embeddings.height}._"),
        )

        # Reduce dimensionality before fitting AA
        _pca = PCA(n_components=min(50, *_vectors.shape))
        _reduced = _pca.fit_transform(_vectors)

        _aa = AA(
            n_archetypes=N_ARCHETYPES,
            random_state=42,
            n_init=3,
        )
        _weights = _aa.fit_transform(_reduced)

        if N_ARCHETYPES == 2:
            # Two archetypes: line segment
            _xy = np.column_stack([
                _weights[:, 1],
                np.zeros(embeddings.height),
            ])
            _vertices = np.array([[0, 0], [1, 0]])

        elif N_ARCHETYPES == 3:
            # Three archetypes: equilateral triangle
            _vertices = np.array([
                [0, 0],
                [1, 0],
                [0.5, np.sqrt(3) / 2],
            ])
            _xy = _weights @ _vertices

        else:
            # Four or more: project archetype weights into 2D
            _weight_pca = PCA(n_components=2)
            _xy = _weight_pca.fit_transform(_weights)

            # Each vertex represents a pure archetype
            _vertices = _weight_pca.transform(np.eye(N_ARCHETYPES))

        _axis_labels = {
            "x": "Archetype mixture X",
            "y": "Archetype mixture Y",
        }

    else:
        _pca = PCA(n_components=2)
        _xy = _pca.fit_transform(_vectors)
        _var = _pca.explained_variance_ratio_
        _axis_labels = {
            "x": f"PC1 ({_var[0]:.0%} of variance)",
            "y": f"PC2 ({_var[1]:.0%} of variance)",
        }

    points = embeddings.select(
        "row",
        pl.col("path").str.split("/").list.last().alias("file"),
        "modality",
        pl.Series("x", _xy[:, 0]),
        pl.Series("y", _xy[:, 1]),
        pl.col("path").str.split("_").list.first().str.replace("data/", "").alias("category"),
    )

    projection_fig = px.scatter(
        points,
        x="x",
        y="y",
        color="category",
        hover_name="file",
        custom_data=["row"],
        labels=_axis_labels,
        title=f"Embeddings projected to 2D with {projection.value}",
        template="plotly_white",
    )

    projection_fig.update_traces(
        marker={"size": 12, "line": {"width": 2, "color": "white"}},
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
    )

    if projection.value == "AA":
        if N_ARCHETYPES <= 3:
            # Draw line or triangle boundary
            _boundary = np.vstack([_vertices, _vertices[0]]) if N_ARCHETYPES == 3 else _vertices
            projection_fig.add_scatter(
                x=_boundary[:, 0],
                y=_boundary[:, 1],
                mode="lines",
                line={"color": "gray", "width": 1, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            )

        # Label archetype positions
        projection_fig.add_scatter(
            x=_vertices[:, 0],
            y=_vertices[:, 1],
            mode="markers+text",
            text=[f"A{i + 1}" for i in range(N_ARCHETYPES)],
            textposition="top center",
            marker={"size": 9, "symbol": "diamond", "color": "black"},
            name="Archetypes",
            hovertemplate="%{text}<extra></extra>",
        )

        projection_fig.update_yaxes(scaleanchor="x", scaleratio=1)

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
            _cards.append(
                mo.vstack(
                    [mo.image(_png, width=THUMBNAIL_SIZE, rounded=True), _caption],
                    gap=0.25,
                )
            )

    _note = (
        mo.md(f"_Showing thumbnails for the first {MAX_THUMBNAILS} of {selected.height} selected._")
        if selected.height > MAX_THUMBNAILS
        else None
    )

    gallery_w = 10
    if _cards:
        _rows = [
            mo.hstack(_cards[i:i+gallery_w], justify="start", gap=1)
            for i in range(0, len(_cards), gallery_w)
        ]
        _gallery = mo.vstack(
            ([_note] if _note is not None else []) + _rows,
            gap=1,
        )
    elif selected.height:
        _gallery = mo.md("_No thumbnails for this selection._")
    else:
        _gallery = None

    _gallery
    return


@app.cell
def _(embeddings, mo, pl, projection_fig, projection_plot):

    _rows = set()

    for p in projection_plot.value or []:
        _curve = p.get("curveNumber")
        _index = p.get("pointIndex")

        if _curve is None or _index is None:
            continue

        _trace = projection_fig.data[_curve]

        # Ignore archetype markers and boundary lines
        if _trace.customdata is None:
            continue

        _rows.add(int(_trace.customdata[_index][0]))

    selected = embeddings.filter(
        pl.col("row").is_in(_rows)
    ).select("path", "modality")

    mo.vstack(
        [
            mo.md(
                f"**{selected.height} selected**: "
                "click a point, or lasso / box-select several"
            ),
            mo.ui.table(selected, selection=None)
            if selected.height else None,
        ]
    )

    return (selected,)


if __name__ == "__main__":
    app.run()
