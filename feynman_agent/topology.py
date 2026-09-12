"""Feynman graph topology: representation, Nickel index, and input parsing.

A topology is stored as internal edges between internal vertices (0..n-1) plus a
list of external legs, each recorded by the internal vertex it attaches to.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Nickel alphabet: 'e' (external leg) sorts before vertex 0, and '|' before all.
_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RANK = {"|": 0, "e": 1, **{c: 2 + i for i, c in enumerate(_SYMBOLS)}}

# Brute-force canonicalisation is exact but factorial; refuse beyond this.
MAX_EXACT_VERTICES = 9

# Highest vertex valence considered when rebuilding a graph from momenta.
MAX_VERTEX_DEGREE = 6


@dataclass
class Topology:
    """A Feynman graph plus whatever physics context we know about it."""

    int_edges: list[tuple[int, int]]
    legs: list[int]
    name: str = ""
    loop_momenta: list[str] = field(default_factory=list)
    # ``propagators[i]`` is the momentum on ``int_edges[i]``, flowing from the
    # first of its two vertices to the second. Both constructors keep to that
    # sense, which is what lets ``leg_momenta`` read the external momenta off
    # the graph instead of being told them.
    propagators: list[str] = field(default_factory=list)
    ext_momenta: list[str] = field(default_factory=list)
    kinematics: str = ""
    # Propagator index -> the mass-squared term subtracted from its momentum
    # squared, kept verbatim because ``m^2`` and ``msq`` are both in common use
    # and both are valid Mathematica. A missing key means the line is massless.
    masses: dict[int, str] = field(default_factory=dict)
    # The Loopedia configuration string it came from, if any. Kept because an
    # all-massless configuration is a statement and an absent one is not, and
    # ``masses`` alone cannot tell those apart.
    mass_config: str = ""
    # Irreducible scalar products the caller listed among the propagators, as
    # momenta. They are no lines of the graph, but they are the caller's choice
    # of basis, and a numerator is expanded in that basis rather than in one
    # invented here.
    isps: list[str] = field(default_factory=list)
    # A polynomial in dot products of the momenta, ``l1.k4`` or ``(l1.k4)^2 -
    # l2.l2``, sitting over the propagators. Empty means the scalar integral.
    numerator: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_vertices(self) -> int:
        return 1 + max(
            [v for e in self.int_edges for v in e] + list(self.legs) + [-1]
        )

    @property
    def n_loops(self) -> int:
        # L = P - V + 1 for a connected graph.
        return len(self.int_edges) - self.n_vertices + 1

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    def nickel(self) -> str:
        return nickel_index(self)

    def edge_list_string(self) -> str:
        """Loopedia edge-list syntax; external legs become extra degree-1 vertices."""
        nxt = self.n_vertices
        pairs = [(min(a, b) + 1, max(a, b) + 1) for a, b in self.int_edges]
        for v in self.legs:
            nxt += 1
            pairs.append((v + 1, nxt))
        return "[" + ",".join(f"({a},{b})" for a, b in sorted(pairs)) + "]"

    def summary(self) -> str:
        massive = f" ({len(self.masses)} massive)" if self.masses else ""
        return (
            f"{self.n_loops}-loop, {self.n_legs}-point, "
            f"{len(self.int_edges)} propagators{massive}, Nickel {self.nickel()}"
        )

    def mass_signature(self) -> tuple[int, tuple[int, ...]]:
        return mass_signature(self.masses.get(i, "") for i in range(len(self.int_edges)))

    @property
    def masses_given(self) -> bool:
        """Whether the caller stated the masses at all — massless counts."""
        return bool(self.masses or self.mass_config)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "nickel": self.nickel(),
            "loops": self.n_loops,
            "legs": self.n_legs,
            "propagators": len(self.int_edges),
            "int_edges": self.int_edges,
            "leg_vertices": self.legs,
            "momenta": self.propagators,
            "masses": self.masses,
            "notes": self.notes,
        }


def mass_signature(terms) -> tuple[int, tuple[int, ...]]:
    """(massless lines, sizes of the equal-mass groups) for one propagator set.

    Two integrals carry the same masses when this agrees, whatever order their
    propagators happen to be in and whatever the mass symbols are called. That
    independence is the point: the Nickel index comes from a minimal relabelling
    of the vertices, so nothing else about a mass assignment survives
    canonicalisation intact.
    """
    terms = list(terms)
    groups = Counter(t for t in terms if t)
    return sum(1 for t in terms if not t), tuple(sorted(groups.values()))


def propagator_terms(topo: Topology) -> list[str]:
    """The propagators as NeatIBP writes them: ``(q)^2``, or ``(q)^2-m^2``."""
    return [
        f"({q})^2-{topo.masses[i]}" if topo.masses.get(i) else f"({q})^2"
        for i, q in enumerate(topo.propagators)
    ]


def mass_symbols(topo: Topology) -> list[str]:
    """The distinct symbols occurring in the mass terms, in propagator order.

    These are the scalar variables NeatIBP demands a value for at the generic
    point, and it exits rather than guessing when one is missing.
    """
    seen: dict[str, None] = {}
    for i in sorted(topo.masses):
        for sym in re.findall(r"[A-Za-z]\w*", topo.masses[i]):
            seen.setdefault(sym)
    return list(seen)


def slug(topo: Topology) -> str:
    """A filename stem for one integral: the graph, and the masses when given.

    The Nickel index names a graph rather than an integral, so on its own it
    would have the massive and the massless version of one diagram overwrite
    each other's report and figure.
    """
    name = topo.nickel()
    if topo.masses_given:
        name += ":" + (topo.mass_config or "-".join(mass_symbols(topo)))
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


# --------------------------------------------------------------------------
# Nickel index
# --------------------------------------------------------------------------


def _nickel_for_labelling(edges, legs, n, perm) -> str:
    """Nickel notation for one specific vertex labelling."""
    higher = defaultdict(list)
    for a, b in edges:
        A, B = perm[a], perm[b]
        higher[min(A, B)].append(max(A, B))
    ext = Counter(perm[v] for v in legs)
    blocks = []
    for i in range(n):
        blocks.append("e" * ext[i] + "".join(_SYMBOLS[j] for j in sorted(higher[i])))
    return "|".join(blocks) + "|"


def _nickel_key(s: str) -> list[int]:
    return [_RANK[c] for c in s]


def nickel_index(topo: Topology) -> str:
    """Lexicographically minimal Nickel notation over all vertex labellings."""
    n = topo.n_vertices
    if n > MAX_EXACT_VERTICES:
        raise ValueError(f"graph has {n} vertices, above the exact-canonisation cap")
    best = None
    best_key = None
    for perm in itertools.permutations(range(n)):
        s = _nickel_for_labelling(topo.int_edges, topo.legs, n, perm)
        k = _nickel_key(s)
        if best_key is None or k < best_key:
            best, best_key = s, k
    return best


def _mass_term(ch: str) -> str:
    """One Loopedia mass symbol as an algebraic mass-squared term.

    ``0`` and ``z`` are massless, ``n`` is a generic nonzero mass, and a digit
    labels masses that are equal to each other — so ``n11|n|`` is the bubble
    whose two propagators carry the same mass.
    """
    return "" if ch in "0z" else ("m^2" if ch == "n" else f"m{ch}^2")


def _read_mass_config(nickel: str, config: str) -> tuple[dict[int, str], list[str]]:
    """Read an aligned Loopedia mass configuration into (masses, notes).

    The configuration mirrors the Nickel index symbol for symbol, so walking the
    two in step lands each mass on the edge ``from_nickel`` appended for that
    symbol. The caller checks that alignment first; a query wildcard such as
    ``e11|e|:*`` carries no per-slot information and never gets here.
    """
    masses: dict[int, str] = {}
    edge = 0
    off_shell = []
    for symbol, mass in zip(nickel, config):
        if symbol == "|":
            continue
        if symbol == "e":
            off_shell += [mass] if mass not in "0z" else []
        else:
            if _mass_term(mass):
                masses[edge] = _mass_term(mass)
            edge += 1
    notes = []
    if off_shell:
        # The external legs are the half of the configuration nothing downstream
        # can act on yet, so say so rather than dropping it silently.
        notes.append(
            f"mass configuration {config} puts {len(off_shell)} external leg(s) "
            "off shell; the generated kinematics still takes them on shell"
        )
    return masses, notes


def from_nickel(text: str) -> Topology:
    """Parse a Nickel index such as ``e12|e3|e3|e|`` into a Topology.

    A trailing ``:<mass configuration>`` is Loopedia's own notation for the
    masses — one symbol per Nickel symbol — so ``e11|e|:n11|n|`` is the bubble
    with two equal-mass propagators and an off-shell external leg.
    """
    text = text.strip().replace(" ", "")
    text, _, config = text.partition(":")
    blocks = [b for b in text.split("|")]
    if blocks and blocks[-1] == "":
        blocks.pop()
    edges: list[tuple[int, int]] = []
    legs: list[int] = []
    for i, block in enumerate(blocks):
        for ch in block:
            if ch == "e":
                legs.append(i)
            else:
                edges.append((i, _SYMBOLS.index(ch)))
    # Only a configuration with one symbol per Nickel symbol says anything about
    # this graph; ':*' and ':*:+1' are query forms Loopedia also puts in its URLs.
    config = config if len(config) == len(text) else ""
    masses, notes = _read_mass_config(text, config) if config else ({}, [])
    return Topology(
        int_edges=edges,
        legs=legs,
        name=f"nickel:{text}",
        masses=masses,
        mass_config=config,
        notes=notes,
    )


def from_edge_list(text: str) -> Topology:
    """Parse ``[(1,2),(2,3),(2,3),(3,4)]``; degree-1 vertices become external legs."""
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?", text)]
    deg = Counter(v for p in pairs for v in p)
    outer = {v for v, d in deg.items() if d == 1}
    inner = sorted(deg.keys() - outer)
    idx = {v: i for i, v in enumerate(inner)}
    edges, legs = [], []
    for a, b in pairs:
        if a in outer and b in outer:
            continue  # isolated line, not part of the graph
        if a in outer:
            legs.append(idx[b])
        elif b in outer:
            legs.append(idx[a])
        else:
            edges.append((idx[a], idx[b]))
    return Topology(int_edges=edges, legs=legs, name="edgelist")


# --------------------------------------------------------------------------
# Reconstruction from propagator momenta
# --------------------------------------------------------------------------

Vec = tuple[int, ...]


def parse_momentum(expr: str, basis: list[str]) -> Vec:
    """``l1+k1-k2`` -> coefficient vector over ``basis``."""
    expr = expr.strip().replace("-", "+-").replace(" ", "")
    coeffs = [0] * len(basis)
    for term in expr.split("+"):
        if not term:
            continue
        sign = -1 if term.startswith("-") else 1
        term = term.lstrip("-")
        m = re.match(r"^(\d*)\*?([A-Za-z]\w*)$", term)
        if not m:
            raise ValueError(f"cannot parse momentum term {term!r}")
        mult = int(m.group(1)) if m.group(1) else 1
        if m.group(2) not in basis:
            raise ValueError(f"unknown momentum symbol {m.group(2)!r}")
        coeffs[basis.index(m.group(2))] += sign * mult
    return tuple(coeffs)


def _neg(v: Vec) -> Vec:
    return tuple(-x for x in v)


def _add(a: Vec, b: Vec) -> Vec:
    return tuple(x + y for x, y in zip(a, b))


def _zero_sum_groups(endpoints, min_size, max_size):
    """All endpoint subsets summing to zero, no edge contributing both ends."""
    n = len(endpoints)
    out = []
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(range(n), size):
            owners = [endpoints[i][0] for i in combo if endpoints[i][0] is not None]
            if len(set(owners)) != len(owners):
                continue  # would create a self-loop at this vertex
            total = (0,) * len(endpoints[0][1])
            for i in combo:
                total = _add(total, endpoints[i][1])
            if not any(total):
                out.append(frozenset(combo))
    return out


def _exact_cover(groups, n_points, n_groups):
    """Pick exactly ``n_groups`` disjoint groups covering every endpoint."""
    by_point = defaultdict(list)
    for g in groups:
        for p in g:
            by_point[p].append(g)

    chosen: list[frozenset] = []

    def rec(covered: frozenset):
        if len(covered) == n_points:
            return len(chosen) == n_groups and list(chosen)
        if len(chosen) == n_groups:
            return None
        point = min(set(range(n_points)) - covered)
        for g in by_point[point]:
            if g & covered:
                continue
            chosen.append(g)
            got = rec(covered | g)
            if got:
                return got
            chosen.pop()
        return None

    return rec(frozenset())


def from_propagators(
    propagators: list[str],
    loop_momenta: list[str],
    ext_momenta: list[str],
    name: str = "",
    masses: dict[int, str] | None = None,
) -> Topology:
    """Rebuild the graph from propagator momenta by momentum conservation.

    Every internal line contributes endpoints +q and -q, every external leg one
    incoming momentum. A vertex is a set of endpoints summing to zero, so the
    graph is an exact cover of all endpoints by V = P - L + 1 such sets.
    """
    basis = list(loop_momenta) + list(ext_momenta)
    props = [parse_momentum(p, basis) for p in propagators]

    # External momenta are all incoming; if they do not sum to zero the caller
    # eliminated one leg via momentum conservation, so restore it.
    legs_mom = [
        tuple(1 if i == len(loop_momenta) + j else 0 for i in range(len(basis)))
        for j in range(len(ext_momenta))
    ]
    residual = (0,) * len(basis)
    for v in legs_mom:
        residual = _add(residual, v)
    notes = []
    if any(residual):
        legs_mom.append(_neg(residual))
        notes.append(
            f"restored implicit {len(ext_momenta) + 1}-th external leg "
            "from momentum conservation"
        )

    # Irreducible scalar products masquerade as propagators; drop up to two.
    for drop in range(0, 3):
        for keep in itertools.combinations(range(len(props)), len(props) - drop):
            sub = [props[i] for i in keep]
            topo = _cover_to_graph(sub, legs_mom, len(loop_momenta))
            if topo:
                edges, legs = topo
                if drop:
                    notes.append(
                        f"treated {drop} of {len(props)} entries as ISPs: "
                        + ", ".join(
                            propagators[i] for i in range(len(props)) if i not in keep
                        )
                    )
                return Topology(
                    int_edges=edges,
                    legs=legs,
                    name=name,
                    loop_momenta=list(loop_momenta),
                    ext_momenta=list(ext_momenta),
                    propagators=[propagators[i] for i in keep],
                    # ``keep`` is the surviving propagators in order, and the
                    # edges were built from exactly that sublist, so the masses
                    # re-key straight onto the new indices.
                    masses={
                        j: (masses or {})[i]
                        for j, i in enumerate(keep)
                        if (masses or {}).get(i)
                    },
                    isps=[propagators[i] for i in range(len(props)) if i not in keep],
                    notes=notes,
                )
    raise ValueError(
        "could not reconstruct a graph from these propagators "
        f"(tried dropping up to 2 ISPs; {len(props)} momenta, "
        f"{len(legs_mom)} external legs)"
    )


def _cover_to_graph(props: list[Vec], legs_mom: list[Vec], n_loop_dims: int):
    """Try to realise ``props`` as a graph; return (edges, legs) or None."""
    # Loop number = rank of the loop-momentum part of the propagator vectors.
    n_loops = _rank([p[:n_loop_dims] for p in props])
    n_v = len(props) - n_loops + 1
    if n_v < 1:
        return None

    endpoints: list[tuple[int | None, Vec]] = []
    for i, q in enumerate(props):
        endpoints.append((i, q))
        endpoints.append((i, _neg(q)))
    for m in legs_mom:
        endpoints.append((None, m))

    for min_size in (3, 2):
        groups = _zero_sum_groups(endpoints, min_size, MAX_VERTEX_DEGREE)
        cover = _exact_cover(groups, len(endpoints), n_v)
        if cover:
            return _groups_to_edges(cover, endpoints, len(props))
    return None


def _groups_to_edges(cover, endpoints, n_props):
    vertex_of = {}
    for vi, group in enumerate(cover):
        for p in group:
            vertex_of[p] = vi
    edges, legs = [], []
    seen = defaultdict(list)
    for i, (owner, _) in enumerate(endpoints):
        if owner is None:
            legs.append(vertex_of[i])
        else:
            seen[owner].append(vertex_of[i])
    for owner in range(n_props):
        # An endpoint's value is the momentum flowing *into* its vertex, so the
        # +q end is where the line arrives and the -q end is where it leaves.
        # Stored the other way round, every external momentum would come back
        # from ``leg_momenta`` with the wrong sign.
        arrives, leaves = seen[owner]
        edges.append((leaves, arrives))
    return edges, sorted(legs)


def sector_topology(topo: Topology, keep: set[int]) -> Topology:
    """The graph of a sub-sector: absent propagators are pinched, i.e. contracted.

    ``keep`` holds indices into ``topo.int_edges`` whose propagator power is
    positive. Every other line is shrunk away, merging the vertices it joined.
    """
    parent = list(range(topo.n_vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, (a, b) in enumerate(topo.int_edges):
        if i not in keep:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    roots = sorted({find(v) for v in range(topo.n_vertices)})
    idx = {r: i for i, r in enumerate(roots)}
    edges, notes, masses = [], [], {}
    for i, (a, b) in enumerate(topo.int_edges):
        if i in keep:
            u, v = idx[find(a)], idx[find(b)]
            if u == v:
                notes.append(f"propagator {i + 1} became a self-loop after pinching")
            if topo.masses.get(i):
                masses[len(edges)] = topo.masses[i]
            edges.append((u, v))
    legs = [idx[find(v)] for v in topo.legs]
    return Topology(
        int_edges=edges,
        legs=legs,
        name=f"{topo.name}[sector {''.join('1' if i in keep else '0' for i in range(len(topo.int_edges)))}]",
        masses=masses,
        notes=notes,
    )


def merge_legs(topo: Topology) -> Topology:
    """Collapse external legs that share a vertex into one.

    Pinching a propagator can leave several external lines on the same vertex.
    Their momenta simply add, so the integral is the one with a single leg
    carrying that sum: ``ee11|ee|`` is the ordinary bubble ``e11|e|``. Useful as
    a second lookup key when the literal sector graph is not catalogued.
    """
    return Topology(
        int_edges=list(topo.int_edges),
        legs=sorted(set(topo.legs)),
        name=f"{topo.name}[legs merged]",
        masses=dict(topo.masses),
        notes=list(topo.notes),
    )


def parse_g(expr: str) -> list[int]:
    """``G[1,1,0,1,-1]`` -> the list of propagator powers."""
    inside = expr[expr.index("[") + 1 : expr.rindex("]")]
    return [int(x.strip()) for x in inside.split(",")]


def g_string(powers: list[int]) -> str:
    """The inverse: propagator powers -> ``G[1,1,0,1,-1]``, as NeatIBP writes it."""
    return "G[" + ",".join(map(str, powers)) + "]"


def assign_momenta(topo: Topology) -> Topology:
    """Route momenta through a graph that was given only combinatorially.

    The spanning tree is chosen to keep the propagators simple: every root is
    tried and the routing with the fewest momentum terms wins. Routing is
    physics-neutral, but a convoluted one (``-l1-l2+k1+k3`` where ``l1+k1``
    would do) makes the downstream syzygy computation far slower.
    """
    best = None
    for root in range(topo.n_vertices):
        cand = _route_from(topo, root)
        cost = sum(sum(1 for c in v if c) for v in cand)
        if best is None or cost < best[0]:
            best = (cost, cand)
    flow = best[1]

    n_loops, n_legs = topo.n_loops, topo.n_legs
    basis = [f"l{i + 1}" for i in range(n_loops)] + [
        f"k{i + 1}" for i in range(max(n_legs - 1, 0))
    ]
    return Topology(
        int_edges=list(topo.int_edges),
        legs=list(topo.legs),
        name=topo.name,
        loop_momenta=[f"l{i + 1}" for i in range(n_loops)],
        ext_momenta=[f"k{i + 1}" for i in range(max(n_legs - 1, 0))],
        propagators=[render_momentum(q, basis) for q in flow],
        masses=dict(topo.masses),
        mass_config=topo.mass_config,
        notes=list(topo.notes),
    )


def _route_from(topo: Topology, root: int) -> list[Vec]:
    """One momentum routing, using the BFS spanning tree grown from ``root``."""
    edges = topo.int_edges
    n_loops, n_legs = topo.n_loops, topo.n_legs
    dim = n_loops + max(n_legs - 1, 0)

    def unit(i):
        return tuple(1 if j == i else 0 for j in range(dim))

    # Spanning tree by BFS over the internal graph.
    adj = defaultdict(list)
    for idx, (a, b) in enumerate(edges):
        adj[a].append((b, idx))
        adj[b].append((a, idx))
    parent, parent_edge, order, seen = {}, {}, [], {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        order.append(v)
        for w, idx in adj[v]:
            if w not in seen:
                seen.add(w)
                parent[w], parent_edge[w] = v, idx
                queue.append(w)
    tree = set(parent_edge.values())

    # Loop momenta on the non-tree edges, oriented as stored.
    flow: dict[int, Vec] = {}
    li = 0
    for idx in range(len(edges)):
        if idx not in tree:
            flow[idx] = unit(li)
            li += 1

    # External legs, all incoming; the last is minus the sum of the others.
    leg_mom: list[Vec] = []
    for j in range(n_legs):
        if j < n_legs - 1:
            leg_mom.append(unit(n_loops + j))
        else:
            tot = (0,) * dim
            for m in leg_mom:
                tot = _add(tot, m)
            leg_mom.append(_neg(tot))

    # Sweep leaves inward: each tree edge is fixed by conservation at its child.
    for v in reversed(order):
        if v not in parent:
            continue
        total = (0,) * dim
        for j, vert in enumerate(topo.legs):
            if vert == v:
                total = _add(total, leg_mom[j])
        for w, idx in adj[v]:
            if idx == parent_edge[v] or idx not in flow:
                continue
            a, b = edges[idx]
            # flow[idx] runs a -> b; it enters v when v is the head.
            total = _add(total, flow[idx] if b == v else _neg(flow[idx]))
        pe = parent_edge[v]
        a, b = edges[pe]
        # Orient the parent edge out of v, then store it in the edge's own sense.
        flow[pe] = total if a == v else _neg(total)

    return [flow[i] for i in range(len(edges))]


def leg_momenta(topo: Topology) -> dict[int, str]:
    """The external momentum arriving at each vertex that has a leg on it.

    Nothing records this: routing writes the momenta onto the internal lines and
    the legs are left implicit. But every internal momentum is known and each
    one flows from the first of its two vertices to the second, so whatever
    fails to cancel at a vertex is what came in from outside — all incoming, by
    the convention the routing uses. Several legs on one vertex cannot be told
    apart this way and get their sum, since the topology never said how they
    split it.
    """
    if not topo.propagators:
        return {}
    basis = topo.loop_momenta + topo.ext_momenta
    inflow = [(0,) * len(basis) for _ in range(topo.n_vertices)]
    for (a, b), q in zip(topo.int_edges, topo.propagators):
        flow = parse_momentum(q, basis)
        inflow[a] = _add(inflow[a], _neg(flow))
        inflow[b] = _add(inflow[b], flow)
    return {v: render_momentum(_neg(inflow[v]), basis) for v in set(topo.legs)}


def render_momentum(vec: Vec, basis: list[str]) -> str:
    parts = []
    for c, sym in zip(vec, basis):
        if c == 0:
            continue
        sign = "-" if c < 0 else ("+" if parts else "")
        mag = "" if abs(c) == 1 else str(abs(c))
        parts.append(f"{sign}{mag}{sym}")
    return "".join(parts) or "0"


def _rank(vectors: list[Vec]) -> int:
    """Rank over the rationals, restricted to the loop-momentum directions."""
    rows = [list(map(float, v)) for v in vectors]
    rank = 0
    ncols = len(rows[0]) if rows else 0
    used = [False] * len(rows)
    for c in range(ncols):
        piv = next((r for r in range(len(rows)) if not used[r] and abs(rows[r][c]) > 1e-9), None)
        if piv is None:
            continue
        used[piv] = True
        rank += 1
        for r in range(len(rows)):
            if r != piv and abs(rows[r][c]) > 1e-9:
                f = rows[r][c] / rows[piv][c]
                rows[r] = [x - f * y for x, y in zip(rows[r], rows[piv])]
    return rank
