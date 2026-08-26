"""Draw the identified Feynman graph as a standalone SVG.

The first thing to check about an automatic identification is whether the agent
read the diagram the way it was meant, and a picture settles that faster than an
edge list does. Every line carries the momentum that was routed through it, so
the figure and the propagator list in the report are the same statement twice.

The layout is a circle. Internal vertices go on it in the cyclic order that
leaves the fewest lines cutting across the middle — a box then comes out as a
square and a double box as a hexagon with one chord. Parallel lines are bowed
apart, self-loops hang outwards, and external legs point away from the centre.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from .topology import Topology, leg_momenta

INK = "#1f2937"  # internal lines and vertices
LEG_INK = "#5b53a8"  # external legs, the accent the workflow diagram already uses
MASS_INK = "#c2410c"  # a line that carries a mass

R = 104  # radius of the vertex circle
LEG = 58  # how far an external leg reaches past its vertex
BOW = 30  # how far apart parallel lines are pushed
FAN = 0.62  # radians between two legs on the same vertex
GAP = 13  # from a line to its label
PAD = 12  # margin around everything drawn
FONT = 12.5


def svg(topo: Topology) -> str:
    """The whole diagram as one standalone SVG document."""
    pos = _positions(topo)
    shapes, labels, pts = [], [], list(pos)
    for drawn, said, corners in (_lines(topo, pos), _legs(topo, pos)):
        shapes += drawn
        labels += said
        pts += corners
    shapes += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="{INK}"/>' for x, y in pos]
    labels = _declutter(labels)
    for x, y, text, anchor, _ in labels:
        pts += _extent(x, y, text, anchor)

    x0 = min(p[0] for p in pts) - PAD
    y0 = min(p[1] for p in pts) - PAD
    w = max(p[0] for p in pts) + PAD - x0
    h = max(p[1] for p in pts) + PAD - y0
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}"'
            f' width="{w:.0f}" height="{h:.0f}" font-family="Helvetica,Arial,sans-serif">',
            f"<title>Feynman diagram {_esc(topo.nickel())}</title>",
            # Without this the figure is dark ink on whatever the reader's
            # Markdown viewer paints behind it, which in a dark theme is nothing.
            f'<rect x="{x0:.0f}" y="{y0:.0f}" width="{w:.0f}" height="{h:.0f}" fill="#ffffff"/>',
            *shapes,
            # Labels last and haloed, so no line is ever drawn through the text.
            '<g paint-order="stroke" stroke="#ffffff" stroke-width="3.5" stroke-linejoin="round">',
            *[_text(*label) for label in labels],
            "</g>",
            "</svg>",
            "",
        ]
    )


def caption(topo: Topology) -> str:
    """One line on how to read the figure, for printing under it."""
    text = "Momenta as routed below; every external leg is incoming."
    if topo.masses:
        text += " Thick coloured lines carry a mass."
    return text


# --------------------------------------------------------------------------
# Where things go
# --------------------------------------------------------------------------


def _positions(topo: Topology) -> list[tuple[float, float]]:
    n = topo.n_vertices
    if n == 1:
        return [(0.0, 0.0)]
    step = 2 * math.pi / n
    out = [(0.0, 0.0)] * n
    for i, v in enumerate(_ring_order(topo)):
        # Half a step past the top: a 4-cycle then lands square-on rather than
        # as a diamond, and a 2-vertex graph lies along the horizontal.
        angle = math.pi / 2 + step / 2 + i * step
        out[v] = (R * math.cos(angle), -R * math.sin(angle))
    return out


def _ring_order(topo: Topology) -> list[int]:
    """The order the vertices are met in going once round the circle.

    Scored by how many chords cross — the one thing that makes a small diagram
    unreadable — and then by how many lines leave the rim at all. These graphs
    are small, so every order is tried, exactly as the Nickel index tries every
    labelling; fixing vertex 0 and the direction of travel drops the rotations
    and reflections, which are the same picture.
    """
    n = topo.n_vertices
    pairs = {(min(a, b), max(a, b)) for a, b in topo.int_edges if a != b}
    # A ring of n vertices has n adjacent slots, so this many lines must cut
    # across it however they are ordered. Nothing beats no crossings and that,
    # and stopping there is what keeps the search off the factorial for the
    # graphs that have a good drawing at all.
    floor = (0, max(0, len(pairs) - n))
    best, best_cost = list(range(n)), None
    for tail in itertools.permutations(range(1, n)):
        if len(tail) > 1 and tail[0] > tail[-1]:
            continue
        cost = _ring_cost((0, *tail), pairs, n)
        if best_cost is None or cost < best_cost:
            best, best_cost = [0, *tail], cost
            if cost == floor:
                break
    return best


def _ring_cost(order, pairs, n: int) -> tuple[int, int]:
    pos = {v: i for i, v in enumerate(order)}
    chords = []
    for a, b in pairs:
        i, j = sorted((pos[a], pos[b]))
        if j - i > 1 and (i, j) != (0, n - 1):
            chords.append((i, j))
    crossings = sum(
        1
        for (i, j), (k, m) in itertools.combinations(chords, 2)
        if i < k < j < m or k < i < m < j
    )
    return crossings, len(chords)


def _outward(pos, v: int) -> tuple[float, float]:
    """Unit vector from the middle of the drawing towards a vertex."""
    x, y = pos[v]
    length = math.hypot(x, y)
    return (x / length, y / length) if length else (0.0, -1.0)


# --------------------------------------------------------------------------
# The parts, each returning (shapes, labels, points to fit inside the figure)
# --------------------------------------------------------------------------


def _lines(topo: Topology, pos):
    """The internal lines: the propagators, one path each."""
    parallel = defaultdict(list)
    for i, (a, b) in enumerate(topo.int_edges):
        parallel[(min(a, b), max(a, b))].append(i)

    shapes, labels, pts = [], [], []
    for (a, b), group in sorted(parallel.items()):
        for k, i in enumerate(group):
            if a == b:
                path, anchor = _self_loop(pos[a], _outward(pos, a), k)
            else:
                path, anchor = _arc(pos[a], pos[b], (k - (len(group) - 1) / 2) * BOW)
            mass = topo.masses.get(i)
            shapes.append(
                f'<path d="{path}" fill="none" stroke="{MASS_INK if mass else INK}"'
                f' stroke-width="{3.4 if mass else 1.8}"/>'
            )
            said = [topo.propagators[i]] if i < len(topo.propagators) else []
            said += [_pretty(mass)] if mass else []
            if said:
                labels.append((*anchor, " · ".join(said), "middle", INK))
            pts.append(anchor)
    return shapes, labels, pts


def _legs(topo: Topology, pos):
    """The external legs, drawn outwards with an arrow: all of them incoming."""
    at = defaultdict(int)
    for v in topo.legs:
        at[v] += 1
    momenta = leg_momenta(topo)

    shapes, labels, pts = [], [], []
    for v, count in sorted(at.items()):
        x, y = pos[v]
        ux0, uy0 = _outward(pos, v)
        for k in range(count):
            angle = math.atan2(uy0, ux0) + (k - (count - 1) / 2) * FAN
            ux, uy = math.cos(angle), math.sin(angle)
            ex, ey = x + LEG * ux, y + LEG * uy
            shapes.append(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{ex:.1f}" y2="{ey:.1f}"'
                f' stroke="{LEG_INK}" stroke-width="1.8"/>'
            )
            shapes.append(_arrow(x + 0.62 * LEG * ux, y + 0.62 * LEG * uy, -ux, -uy))
            pts.append((ex, ey))
        # One label per vertex, out along the bundle's own direction: momentum
        # conservation fixes what arrives there in total, and nothing in the
        # topology says how several legs on one vertex share it out.
        if momenta.get(v):
            labels.append(
                (
                    x + (LEG + GAP) * ux0,
                    y + (LEG + GAP) * uy0 + 4,
                    momenta[v],
                    _anchor(ux0),
                    LEG_INK,
                )
            )
    return shapes, labels, pts


def _arc(p, q, bow: float):
    """A line between two vertices, bowed out of the straight by ``bow``."""
    (x1, y1), (x2, y2) = p, q
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    length = math.hypot(x2 - x1, y2 - y1)
    nx, ny = (y1 - y2) / length, (x2 - x1) / length
    if bow == 0 and mx * nx + my * ny < 0:
        nx, ny = -nx, -ny  # a straight line keeps its label outside the ring
    gap = GAP if bow >= 0 else -GAP
    # Two lines that cross have their midpoints in the same place, so a label
    # landing near the middle of the drawing slides along its own line instead.
    # Everything else stays centred, which is where it looks right.
    t = 0.32 if not bow and math.hypot(mx, my) < 0.45 * R else 0.5
    lx, ly = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    # A quadratic curve only reaches half way to its control point, so aim twice
    # as far out as the line is meant to bend.
    return (
        f"M{x1:.1f},{y1:.1f} Q{mx + 2 * bow * nx:.1f},{my + 2 * bow * ny:.1f}"
        f" {x2:.1f},{y2:.1f}",
        (lx + (bow + gap) * nx, ly + (bow + gap) * ny),
    )


def _self_loop(p, out, k: int):
    """A line whose two ends meet: a tadpole, or what pinching can leave behind."""
    (x, y), (ux, uy) = p, out
    px, py = -uy, ux
    reach = 3 * BOW + k * BOW
    c1 = (x + reach * ux + 1.2 * BOW * px, y + reach * uy + 1.2 * BOW * py)
    c2 = (x + reach * ux - 1.2 * BOW * px, y + reach * uy - 1.2 * BOW * py)
    far = 0.75 * reach + GAP  # a cubic with both ends at one point turns at 3/4
    return (
        f"M{x:.1f},{y:.1f} C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} {x:.1f},{y:.1f}",
        (x + far * ux, y + far * uy),
    )


def _arrow(x: float, y: float, ux: float, uy: float) -> str:
    """A small filled triangle with its tip at (x, y), pointing along (ux, uy)."""
    px, py = -uy, ux
    return (
        f'<polygon points="{x:.1f},{y:.1f}'
        f' {x - 8 * ux + 4 * px:.1f},{y - 8 * uy + 4 * py:.1f}'
        f' {x - 8 * ux - 4 * px:.1f},{y - 8 * uy - 4 * py:.1f}" fill="{LEG_INK}"/>'
    )


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def _text(x: float, y: float, s: str, anchor: str, fill: str) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{FONT}"'
        f' fill="{fill}">{_esc(s)}</text>'
    )


def _extent(x: float, y: float, text: str, anchor: str):
    """Roughly where a label lands, so the figure is sized to hold it."""
    width = 0.58 * FONT * len(text)
    left = {"start": 0.0, "middle": -width / 2, "end": -width}[anchor]
    return [(x + left, y - FONT), (x + left + width, y + 0.4 * FONT)]


def _declutter(labels):
    """Move a label down until it stops overlapping the ones already placed.

    Two lines that cross want their labels in nearly the same place, and a
    momentum printed on top of another momentum is worse than one sitting a
    little off its own line. Planar graphs never reach the loop body.
    """
    placed, out = [], []
    for x, y, text, anchor, fill in labels:
        while any(_overlaps(_extent(x, y, text, anchor), box) for box in placed):
            y += FONT + 2
        placed.append(_extent(x, y, text, anchor))
        out.append((x, y, text, anchor, fill))
    return out


def _overlaps(a, b) -> bool:
    return a[0][0] < b[1][0] and b[0][0] < a[1][0] and a[0][1] < b[1][1] and b[0][1] < a[1][1]


def _anchor(ux: float) -> str:
    return "start" if ux > 0.3 else "end" if ux < -0.3 else "middle"


def _pretty(term: str) -> str:
    return term.replace("^2", "²")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
