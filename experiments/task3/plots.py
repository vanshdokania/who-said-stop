"""Dependency-free SVG plots for Task 3 results."""

from pathlib import Path


COLORS = ("#245b8f", "#c55a11", "#548235", "#7030a0", "#008c95")


def _svg_text(x: float, y: float, text: str, **attributes: str) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attrs}>{text}</text>'


def _write_svg(path: Path, elements: list[str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="620" viewBox="0 0 1000 620">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(500, 28, title, text_anchor="middle", font_family="sans-serif", font_size="18"),
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def plot_training_likelihood(
    histories: list[dict[str, object]],
    path: Path,
) -> None:
    """Plot per-frame training log likelihood over EM iterations."""
    left, top, width, height = 95, 70, 730, 455
    valid = [item for item in histories if item["history"]]
    values = [
        float(value) / int(item["fitting_frames"])
        for item in valid
        for value in item["history"]
    ]
    if not values:
        raise ValueError("No EM likelihood histories are available to plot")
    minimum, maximum = min(values), max(values)
    span = max(maximum - minimum, 1e-12)
    max_iterations = max(len(item["history"]) for item in valid)
    x_denominator = max(max_iterations - 1, 1)
    elements = [
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222"/>',
        _svg_text(left + width / 2, 585, "EM iteration", text_anchor="middle", font_size="14"),
        _svg_text(25, top + height / 2, "Training average log-likelihood per frame", text_anchor="middle", font_size="14", transform=f"rotate(-90 25 {top + height / 2})"),
    ]
    for index, item in enumerate(valid):
        values_for_model = [
            float(value) / int(item["fitting_frames"])
            for value in item["history"]
        ]
        points = []
        for iteration, value in enumerate(values_for_model):
            x = left + iteration / x_denominator * width
            y = top + height - (value - minimum) / span * height
            points.append(f"{x:.2f},{y:.2f}")
        color = COLORS[index % len(COLORS)]
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        elements.append(
            _svg_text(
                left + width + 10,
                top + 18 + index * 22,
                f'{item["covariance_type"]} K={item["n_components"]}',
                fill=color,
                font_size="12",
            )
        )
    elements.extend(
        [
            _svg_text(left, top + height + 24, f"{minimum:.4f}", font_size="11"),
            _svg_text(left + width, top + height + 24, f"{maximum:.4f}", text_anchor="end", font_size="11"),
            _svg_text(left, top + height + 42, "1", font_size="11"),
            _svg_text(left + width, top + height + 42, str(max_iterations), text_anchor="end", font_size="11"),
        ]
    )
    _write_svg(path, elements, "UBM EM training log-likelihood")


def plot_heldout_likelihood(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    """Plot held-out average log likelihood against component count."""
    left, top, width, height = 95, 70, 730, 455
    valid = [row for row in rows if row["status"] == "success"]
    values = [float(row["heldout_average_log_likelihood"]) for row in valid]
    if not values:
        raise ValueError("No successful models are available to plot")
    minimum, maximum = min(values), max(values)
    span = max(maximum - minimum, max(abs(minimum), 1.0) * 1e-6)
    components = sorted({int(row["n_components"]) for row in valid})
    x_positions = {
        count: left + index / max(len(components) - 1, 1) * width
        for index, count in enumerate(components)
    }
    elements = [
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222"/>',
        _svg_text(left + width / 2, 585, "Mixture components", text_anchor="middle", font_size="14"),
        _svg_text(25, top + height / 2, "Held-out average log-likelihood per frame", text_anchor="middle", font_size="14", transform=f"rotate(-90 25 {top + height / 2})"),
    ]
    covariance_types = list(dict.fromkeys(str(row["covariance_type"]) for row in valid))
    for index, covariance_type in enumerate(covariance_types):
        series = sorted(
            (row for row in valid if row["covariance_type"] == covariance_type),
            key=lambda row: int(row["n_components"]),
        )
        points = []
        for row in series:
            x = x_positions[int(row["n_components"])]
            value = float(row["heldout_average_log_likelihood"])
            y = top + height - (value - minimum) / span * height
            points.append(f"{x:.2f},{y:.2f}")
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{COLORS[index % len(COLORS)]}"/>'
            )
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index % len(COLORS)]}" stroke-width="2"/>'
        )
        elements.append(
            _svg_text(
                left + width + 12,
                top + 18 + index * 22,
                covariance_type,
                fill=COLORS[index % len(COLORS)],
                font_size="12",
            )
        )
    for count, x in x_positions.items():
        elements.append(_svg_text(x, top + height + 22, str(count), text_anchor="middle", font_size="11"))
    _write_svg(path, elements, "Held-out likelihood by covariance and mixture count")


def plot_training_time(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    """Plot wall-clock fit time for each successful model configuration."""
    valid = [row for row in rows if row["status"] == "success"]
    if not valid:
        raise ValueError("No successful models are available to plot")
    left, top, width, height = 95, 70, 730, 455
    maximum = max(float(row["training_seconds"]) for row in valid)
    maximum = max(maximum, 1e-9)
    bar_height = min(16, height / len(valid) * 0.75)
    elements = [
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#222"/>',
        _svg_text(left + width / 2, 585, "Training time (seconds)", text_anchor="middle", font_size="14"),
    ]
    row_height = height / len(valid)
    for index, row in enumerate(valid):
        y = top + index * row_height + (row_height - bar_height) / 2
        bar_width = float(row["training_seconds"]) / maximum * width
        elements.append(
            f'<rect x="{left}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{COLORS[index % len(COLORS)]}"/>'
        )
        elements.append(
            _svg_text(
                left + width + 8,
                y + bar_height - 2,
                f'{row["covariance_type"]} K={row["n_components"]} ({float(row["training_seconds"]):.1f}s)',
                font_size="10",
            )
        )
    _write_svg(path, elements, "UBM model training time")
