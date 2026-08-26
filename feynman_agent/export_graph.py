"""Render the agent's LangGraph as PNG / SVG / PDF.

    python -m feynman_agent.export_graph docs/graph.png docs/graph.svg

The DOT source is generated here and piped through the system ``dot``, so no
Python graphviz binding is needed — and the format is whatever ``dot`` supports.
Conditional edges are drawn dashed, matching what they are: branches the router
chooses between, not steps that always run.

Falls back to writing Mermaid source when ``dot`` is unavailable, so there is
always something to look at.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .graph import build

# Left to right: the pipeline is ten nodes long and two wide, so laying it along
# the page's long axis keeps the boxes big enough to read.
RANKDIR = "LR"

# Nodes that end a run, and the entry point — worth setting apart visually.
TERMINAL = {"report", "__end__"}
START = {"__start__"}

# The catalogue-lookup loop: these three nodes run once for the original
# integral and again for every diagram the reduction produces. Drawing it as a
# plain branch hides the one structural idea in the graph, so it gets its own
# colour and box.
LOOP_NODES = ["local_db", "loopedia", "advance"]
LOOP_EDGES = {
    ("loopedia", "advance"): "",
    ("advance", "local_db"): "next diagram",
    ("expand_masters", "local_db"): "each master",
}
LOOP_COLOR = "#c2410c"

# Nodes several stages can jump straight to, and the one source that reaches
# each of them along the main line. Every *other* way in is a shortcut: it has
# to be drawn, but letting it assign ranks pulls the stage it leaves from off
# the straight line, and the pipeline comes out zigzagging.
SHORTCUT_TARGETS = {"report": "ask_user", "read_papers": "arxiv"}


def to_dot(title: str = "Feynman integral agent") -> str:
    graph = build().get_graph()
    lines = [
        "digraph feynman_agent {",
        f"  rankdir={RANKDIR};",
        "  bgcolor=white;",
        f'  label="{title}";',
        '  labelloc=t; fontsize=16; fontname="Helvetica-Bold";',
        '  node [shape=box style="rounded,filled" fontname=Helvetica fontsize=11'
        ' color="#5b53a8" fillcolor="#f2f0ff" penwidth=1.2];',
        '  edge [fontname=Helvetica fontsize=9 color="#4a4a4a"];',
        "",
    ]

    for name in graph.nodes:
        if name in START:
            style = 'shape=circle label="start" fillcolor="#ffffff" color="#5b53a8" width=0.5'
        elif name == "__end__":
            style = 'shape=circle label="end" fillcolor="#bfb6fc" color="#5b53a8" width=0.5'
        elif name == "report":
            style = f'label="{name}" fillcolor="#bfb6fc"'
        elif name in LOOP_NODES:
            style = f'label="{name}" color="{LOOP_COLOR}" fillcolor="#fff3ea"'
        else:
            style = f'label="{name}"'
        lines.append(f'  "{name}" [{style}];')

    present = [n for n in LOOP_NODES if n in graph.nodes]
    if present:
        lines += [
            "",
            "  subgraph cluster_loop {",
            '    label="catalogue lookup — the original integral, then each master";',
            f'    fontsize=10; fontcolor="{LOOP_COLOR}"; style=rounded; color="{LOOP_COLOR}";',
            "    " + " ".join(f'"{n}";' for n in present),
            "  }",
        ]

    lines.append("")
    for edge in graph.edges:
        key = (edge.source, edge.target)
        if key in LOOP_EDGES:
            attrs = f'style=dashed color="{LOOP_COLOR}" fontcolor="{LOOP_COLOR}" penwidth=1.6'
            # Back edges: let them curve around rather than reorder the spine.
            # Without this the loop's target would be ranked after its source,
            # which is exactly backwards for an edge that goes back.
            if edge.target == "local_db":
                attrs += " constraint=false"
        elif edge.conditional:
            attrs = 'style=dashed color="#7a72c0"'
            if edge.source != SHORTCUT_TARGETS.get(edge.target, edge.source):
                attrs += " constraint=false"
        else:
            attrs = "style=solid penwidth=1.4"
        label = LOOP_EDGES.get(key) or edge.data or ""
        suffix = f' label="{label}"' if label else ""
        lines.append(f'  "{edge.source}" -> "{edge.target}" [{attrs}{suffix}];')

    lines += [
        "",
        "  subgraph cluster_legend {",
        '    label="edge kinds"; fontsize=10; style=dotted; color="#999999";',
        "    node [shape=point width=0.01 style=invis];",
        '    l0 -> l1 [style=solid penwidth=1.4 label="always"];',
        '    l2 -> l3 [style=dashed color="#7a72c0" label="branch"];',
        f'    l4 -> l5 [style=dashed color="{LOOP_COLOR}" fontcolor="{LOOP_COLOR}"'
        ' penwidth=1.6 label="loop"];',
        "  }",
        "}",
    ]
    return "\n".join(lines)


def render(path: Path, dot_source: str | None = None) -> Path:
    """Write one file; the suffix picks the format ``dot`` renders."""
    dot_source = dot_source or to_dot()
    fmt = path.suffix.lstrip(".").lower() or "png"

    if fmt == "dot":
        path.write_text(dot_source)
        return path
    if fmt in {"mmd", "mermaid"}:
        path.write_text(build().get_graph().draw_mermaid())
        return path

    exe = shutil.which("dot")
    if exe is None:
        fallback = path.with_suffix(".mmd")
        fallback.write_text(build().get_graph().draw_mermaid())
        raise RuntimeError(
            f"graphviz 'dot' not found, so {path.name} could not be rendered; "
            f"wrote Mermaid source to {fallback} instead "
            "(install graphviz, or paste it into mermaid.live)"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [exe, f"-T{fmt}", "-o", str(path)],
        input=dot_source,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed for {fmt}: {proc.stderr.strip()}")
    return path


def main(argv=None) -> int:
    targets = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not targets:
        targets = [Path("docs/graph.png"), Path("docs/graph.svg"), Path("docs/graph.pdf")]

    dot_source = to_dot()
    for target in targets:
        try:
            out = render(target, dot_source)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {out}  ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
