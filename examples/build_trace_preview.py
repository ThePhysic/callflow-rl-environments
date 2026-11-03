"""Render the recorded timeout trace as a compact, shareable figure."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    root = Path(__file__).resolve().parent
    rows = [
        json.loads(line) for line in (root / "timeout_after_commit.jsonl").read_text().splitlines()
    ]
    header = rows[0]
    steps = [row for row in rows if row.get("record") == "step"]
    lines = ["CallFlow RL  |  lost response after a committed booking", ""]
    for row in steps:
        action = row["action"]
        label = action["type"]
        if label == "CALL_TOOL":
            label = action["tool"] + "()"
            tool = row["observation"].get("last_tool") or {}
            label += " -> " + (tool.get("error") or "ok")
        elif label == "MESSAGE_USER":
            label = "report observed reservation status"
        elif label == "CONFIRM_INTENT":
            label = "request explicit customer approval"
        lines.append(f"{row['index'] + 1:02d}  {label:<49} {row['reward']:+.2f}")
    verdict = header["verdict"]
    lines += [
        "",
        f"Independent verification: success={verdict['task_success']}",
        f"Duplicate side effect: {verdict['duplicate_side_effect']}",
        "Trace: examples/timeout_after_commit.jsonl",
    ]
    figure, ax = plt.subplots(figsize=(11, 5.5), facecolor="#101921")
    ax.set_facecolor("#101921")
    ax.axis("off")
    ax.text(
        0.025,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=12,
        linespacing=1.65,
        color="#cce6da",
        transform=ax.transAxes,
    )
    figure.tight_layout(pad=1)
    figure.savefig(root / "trace_preview.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
