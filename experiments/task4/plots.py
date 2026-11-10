"""Dependency-free SVG plots for Task 4 development comparisons."""

from pathlib import Path


def plot_relevance_metrics(rows: list[dict[str, object]], path: Path) -> None:
    """Plot EER across MAP relevance factors and the independent-GMM baseline."""
    width, height = 850, 500
    left, top, chart_width, chart_height = 90, 65, 660, 340
    minimum = 0.0
    maximum = max(float(row["eer"]) for row in rows)
    maximum = max(maximum * 1.15, 0.01)
    group_width = chart_width / len(rows)
    elements = [
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Development EER by adaptation method</text>',
        f'<text x="24" y="{top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + chart_height / 2})" font-family="sans-serif" font-size="14">EER</text>',
    ]
    for index, row in enumerate(rows):
        x = left + index * group_width + group_width * 0.22
        bar_width = group_width * 0.56
        eer = float(row["eer"])
        bar_height = (eer - minimum) / (maximum - minimum) * chart_height
        y = top + chart_height - bar_height
        color = "#4472c4" if row["method"] == "MAP" else "#c55a11"
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12">{eer:.4f}</text>'
        )
        label = f'r={row["relevance_factor"]}' if row["method"] == "MAP" else "Independent GMM"
        elements.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-size="11">{label}</text>'
        )
    elements.extend(
        [
            f'<text x="{left}" y="{top + chart_height + 55}" font-size="11">0.0000</text>',
            f'<text x="{left}" y="{top + 8}" font-size="11">{maximum:.4f}</text>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                *elements,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
