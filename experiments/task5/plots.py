"""Small dependency-free SVG plotting helpers for Task 5."""

from pathlib import Path


def plot_roc(
    rows: list[dict[str, float]],
    threshold: float,
    path: Path,
) -> None:
    """Write a ROC plot with the development EER operating point highlighted."""
    width, height = 720, 600
    left, top, chart_width, chart_height = 90, 55, 550, 440

    def x_position(value: float) -> float:
        return left + value * chart_width

    def y_position(value: float) -> float:
        return top + (1.0 - value) * chart_height

    selected = next(row for row in rows if row["threshold"] == threshold)
    points = " ".join(
        f"{x_position(row['far']):.2f},{y_position(row['tar']):.2f}"
        for row in rows
    )
    elements = [
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top}" stroke="#999" stroke-dasharray="5,5"/>',
        f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2"/>',
        f'<circle cx="{x_position(selected["far"]):.2f}" cy="{y_position(selected["tar"]):.2f}" r="5" fill="#d62728"/>',
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Development ROC</text>',
        f'<text x="{left + chart_width / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">False Acceptance Rate (FAR)</text>',
        f'<text x="24" y="{top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + chart_height / 2})" font-family="sans-serif" font-size="14">True Acceptance Rate (1 - FRR)</text>',
        f'<text x="{left - 34}" y="{top + chart_height + 4}" font-size="11">0</text>',
        f'<text x="{left - 34}" y="{top + 8}" font-size="11">1</text>',
        f'<text x="{left + chart_width - 8}" y="{top + chart_height + 18}" font-size="11">1</text>',
        f'<text x="{left + 12}" y="{top + 24}" font-size="12" fill="#d62728">EER threshold: {threshold:.6g}</text>',
    ]
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
