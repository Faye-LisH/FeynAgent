"""One run of the agent, from an input to a report — driven by whoever called.

The terminal and the browser want the same three things out of the graph: start
it, watch it work, and answer the one question it stops to ask. Only the
destination differs, so the loop lives here and the caller passes ``on`` for
what happened and ``answer`` for the interrupt. The CLI prints and reads a line
of stdin; the web interface puts the same events on a queue and waits for a
button.

Streaming is in two modes at once because the graph reports in two ways.
``updates`` fires as each node returns, which is where the finished stages come
from; ``custom`` carries what a slow node says while it is still inside itself
("NeatIBP: reducing ..., 8 propagators — minutes"), and without it a reduction
looks like a hang.
"""

from __future__ import annotations

from pathlib import Path

from langgraph.types import Command

from .graph import build
from .topology import slug

REPORTS = Path("reports")


def initial(raw_input: str, **options) -> dict:
    """The state a run starts from.

    ``trace`` and ``problems`` are accumulating channels, so they have to exist
    before the first node adds to one.
    """
    return {"raw_input": raw_input, "trace": [], "problems": [], **options}


def drive(payload, config, on, answer, graph=None) -> dict:
    """Run to the end, reporting through ``on`` and answering through ``answer``.

    ``on(kind, data)`` is called with ``("step", entry)`` for each finished
    stage and ``("note", text)`` for progress inside one. ``answer(question)``
    is called only if the graph interrupts, and returns the reply to resume
    with — the graph decides what to do with anything it does not recognise.
    """
    graph = build() if graph is None else graph
    pending, state = _to_stop(graph, payload, config, on)
    while pending is not None:
        resume = Command(resume=answer(pending.value))
        pending, state = _to_stop(graph, resume, config, on)
    return state


def _to_stop(graph, payload, config, on):
    """Run until the graph finishes or interrupts: ``(interrupt or None, state)``."""
    pending = None
    for mode, chunk in graph.stream(payload, config, stream_mode=["custom", "updates"]):
        if mode == "custom":
            on("note", chunk["progress"])
        elif "__interrupt__" in chunk:
            pending = chunk["__interrupt__"][0]
        else:
            for update in chunk.values():
                for entry in (update or {}).get("trace", []):
                    on("step", entry)
    return pending, graph.get_state(config).values


def write(state: dict, out: str | Path | None = None) -> Path:
    """Write the report and put the diagram beside it. Returns the report's path.

    The figure is named after the integral rather than after the report, so the
    link inside the Markdown resolves whatever the report itself was called.
    """
    topo = state.get("topo")
    stem = slug(topo) if topo is not None else "unidentified"
    path = Path(out) if out else REPORTS / f"{stem}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state["report"])
    if state.get("figure"):
        (path.parent / f"{stem}.svg").write_text(state["figure"])
    return path
