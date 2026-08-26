"""Local database of known integrals: a JSON file plus a folder of notebooks.

Exact matching is by Nickel index. Anything that does not match exactly falls
back to a (loops, legs, propagators) signature so the agent can still report
near misses instead of silently giving up.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .topology import from_nickel, mass_signature

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_JSON = DATA / "local_db.json"
DEFAULT_NOTEBOOKS = DATA / "notebooks"


@dataclass
class Entry:
    nickel: str = ""
    name: str = ""
    result: str = ""
    reference: str = ""
    masters: list[str] = field(default_factory=list)
    # One mass-squared term per propagator, in this entry's own Nickel order,
    # empty for a massless line. Omitted entirely means every line is massless,
    # which is what every stored result was before masses could be expressed.
    masses: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = ""

    @property
    def has_result(self) -> bool:
        return bool(self.result)


@dataclass
class DBHit:
    exact: list[Entry] = field(default_factory=list)
    similar: list[Entry] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return any(e.has_result for e in self.exact)


def _load_json(path: Path) -> list[Entry]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return [Entry(source=str(path), **e) for e in raw.get("integrals", [])]


def _scan_notebooks(folder: Path) -> list[Entry]:
    """Pick up Nickel indices recorded inside .nb / .wl files.

    A notebook is registered by any ``Nickel index: <...>`` or ``(* nickel: ... *)``
    marker in its text, which keeps the convention obvious to whoever writes one.
    """
    if not folder.exists():
        return []
    out = []
    pattern = re.compile(r"[Nn]ickel(?:\s+index)?\s*[:=]\s*\"?([e0-9A-Z|]+\|)", re.I)
    for nb in sorted(list(folder.glob("*.nb")) + list(folder.glob("*.wl"))):
        text = nb.read_text(errors="replace")
        for nickel in dict.fromkeys(pattern.findall(text)):
            name = re.search(r"[Nn]ame\s*[:=]\s*\"([^\"]+)\"", text)
            ref = re.search(r"[Rr]eference\s*[:=]\s*\"([^\"]+)\"", text)
            out.append(
                Entry(
                    nickel=nickel,
                    name=name.group(1) if name else nb.stem,
                    reference=ref.group(1) if ref else "",
                    notes=f"Mathematica notebook {nb.name}",
                    result=f"see notebook {nb}",
                    source=str(nb),
                )
            )
    return out


def search(topo, json_path: Path | None = None, notebooks: Path | None = None) -> DBHit:
    json_path = json_path or DEFAULT_JSON
    notebooks = notebooks or DEFAULT_NOTEBOOKS
    entries = _load_json(json_path) + _scan_notebooks(notebooks)

    nickel = topo.nickel() if hasattr(topo, "nickel") else str(topo)
    sig = (topo.n_loops, topo.n_legs, len(topo.int_edges))
    # The Nickel index is the graph alone, so a stored massless result is not an
    # answer for the same graph with massive lines — it is a near miss.
    want = topo.mass_signature()
    massless = [""] * len(topo.int_edges)

    hit = DBHit(searched=[str(json_path), str(notebooks)])
    for e in entries:
        if e.nickel == nickel and mass_signature(e.masses or massless) == want:
            hit.exact.append(e)
        else:
            try:
                other = from_nickel(e.nickel)
            except (ValueError, KeyError):
                continue  # a malformed Nickel index in the DB is not fatal
            if (other.n_loops, other.n_legs, len(other.int_edges)) == sig:
                hit.similar.append(e)
    return hit
