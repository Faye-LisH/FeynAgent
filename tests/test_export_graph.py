"""Tests for the graph exporter."""

import shutil

import pytest

from feynman_agent import export_graph


def test_dot_covers_every_node_and_edge():
    from feynman_agent import build

    graph = build().get_graph()
    dot = export_graph.to_dot()

    for name in graph.nodes:
        assert f'"{name}"' in dot
    for edge in graph.edges:
        assert f'"{edge.source}" -> "{edge.target}"' in dot
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")


def test_conditional_edges_are_dashed_and_direct_ones_solid():
    dot = export_graph.to_dot()
    # identify -> local_db is a router choice; report -> __end__ always runs.
    branch = next(l for l in dot.splitlines() if '"identify" -> "local_db"' in l)
    always = next(l for l in dot.splitlines() if '"report" -> "__end__"' in l)
    assert "dashed" in branch
    assert "solid" in always


def test_layout_is_horizontal():
    """Ten nodes long and two wide: top-down makes a tall ribbon of small boxes."""
    assert "rankdir=LR" in export_graph.to_dot()


def test_dot_source_can_be_written_without_graphviz(tmp_path):
    out = export_graph.render(tmp_path / "g.dot")
    assert out.exists() and out.read_text().startswith("digraph")


def test_mermaid_fallback_format(tmp_path):
    out = export_graph.render(tmp_path / "g.mmd")
    assert "graph TD" in out.read_text()


@pytest.mark.skipif(shutil.which("dot") is None, reason="graphviz not installed")
@pytest.mark.parametrize("fmt,magic", [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("svg", b"<")])
def test_renders_real_image_formats(tmp_path, fmt, magic):
    out = export_graph.render(tmp_path / f"g.{fmt}")
    assert out.read_bytes()[:4].startswith(magic)
