"""Dependency-free SVG plots for Task 6 ablations."""

from pathlib import Path


def _bar_chart(
    rows: list[dict[str, object]],
    labels: list[str],
    path: Path,
    title: str,
    x_label: str,
) -> None:
    width, height = 800, 500
    left, top, chart_width, chart_height = 95, 60, 650, 330
    maximum = max(float(row["eer"]) for row in rows)
    maximum = max(maximum * 1.15, 0.01)
    group_width = chart_width / len(rows)
    elements = [
        f'<line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#222"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
        f'<text x="25" y="{top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 25 {top + chart_height / 2})" font-family="sans-serif" font-size="14">Development EER</text>',
        f'<text x="{left - 40}" y="{top + chart_height + 4}" font-size="11">0</text>',
        f'<text x="{left - 40}" y="{top + 8}" font-size="11">{maximum:.4f}</text>',
    ]
    for index, (row, label) in enumerate(zip(rows, labels)):
        x = left + index * group_width + group_width * 0.2
        bar_width = group_width * 0.6
        eer = float(row["eer"])
        bar_height = eer / maximum * chart_height
        y = top + chart_height - bar_height
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#4472c4"/>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12">{eer:.4f}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-size="12">{label}</text>'
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


def plot_relevance_factor(rows: list[dict[str, object]], path: Path) -> None:
    """Plot Task 4 development EER against the MAP relevance factor."""
    ordered = sorted(rows, key=lambda row: int(row["relevance_factor"]))
    _bar_chart(
        ordered,
        [str(row["relevance_factor"]) for row in ordered],
        path,
        "Development EER by MAP relevance factor",
        "MAP relevance factor r",
    )


def plot_enrolment_size(rows: list[dict[str, object]], path: Path) -> None:
    """Plot development EER against selected enrolment utterance count."""
    ordered = sorted(rows, key=lambda row: int(row["enrolment_utterances"]))
    _bar_chart(
        ordered,
        [str(row["enrolment_utterances"]) for row in ordered],
        path,
        "Development EER by enrolment size",
        "Enrolment utterances per speaker",
    )


def plot_model_comparison(rows: list[dict[str, object]], path: Path) -> None:
    """Plot the reused Task 4 MAP and independent-GMM development EERs."""
    _bar_chart(
        rows,
        [str(row["setting"]) for row in rows],
        path,
        "Development EER: MAP vs independent GMM",
        "Speaker modelling method",
    )
