"""Table render engine: booktabs LaTeX + markdown twin writer (reference convention)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from studies.analysis.common.config import paper_root
from studies.analysis.common.io import atomic_write_text


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.caption}" if self.caption else "",
            "| " + " | ".join(self.headers) + " |",
            "| " + " | ".join("---" for _ in self.headers) + " |",
        ]
        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        for note in self.notes:
            lines.append(f"\n> {note}")
        return "\n".join(lines) + "\n"

    def to_latex(self) -> str:
        ncols = len(self.headers)

        def esc(text: str) -> str:
            return re.sub(r"([_%&$#])", r"\\\1", str(text))

        col_spec = "l" * ncols
        lines = [
            r"\begin{tabular}{" + col_spec + r"}",
            r"\toprule",
            " & ".join(esc(h) for h in self.headers) + r" \\",
            r"\midrule",
        ]
        for row in self.rows:
            lines.append(" & ".join(esc(c) for c in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        if self.caption:
            caption = esc(self.caption)
            notes = " ".join(esc(n) for n in self.notes)
            wrapped = [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{"
                + caption
                + (r" \label{tab:" + _label(self.caption) + r"}" if caption else "")
                + "}",
            ]
            if notes:
                wrapped.append(
                    r"\begin{minipage}{\linewidth}\small\vspace{2pt}" + notes + r"\end{minipage}"
                )
            wrapped += lines
            wrapped.append(r"\end{table}")
            lines = wrapped
        return "\n".join(lines) + "\n"


def _label(caption: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", caption.lower()).strip("-")[:40]


def save_table(table: Table, relative: str) -> list[Path]:
    """Write <paper_root>/<relative>.tex and .md (relative includes the asset id)."""
    base = paper_root() / relative
    tex = base.with_suffix(".tex")
    md = base.with_suffix(".md")
    atomic_write_text(tex, table.to_latex())
    atomic_write_text(md, table.to_markdown())
    return [tex, md]
