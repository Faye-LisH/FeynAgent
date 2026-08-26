"""Loopedia client (https://loopedia.mpp.mpg.de, arXiv:1709.01266).

Loopedia is a bibliographic index: a record says *which paper* evaluated a given
graph, to which order in epsilon, not the closed-form expression. The site is a
plain POST form with two useful queries:

    graph=<edge list|Nickel>, q=:*   -> mass configurations available for a graph
    q=<nickel>:<massconfig>          -> the records for one configuration

Sending the edge list rather than our own Nickel index lets Loopedia do the
canonicalisation, so the lookup cannot be broken by a labelling disagreement.
"""

from __future__ import annotations

import codecs
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .topology import mass_signature

BASE = "https://loopedia.mpp.mpg.de/"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "loopedia"
TIMEOUT = 45


@dataclass
class Record:
    """One public Loopedia entry for a configuration."""

    reference: str = ""
    authors: str = ""
    orders: str = ""
    description: str = ""
    record_id: str = ""
    reducible: str = ""
    n_masters: str = ""
    equations: str = ""
    config: str = ""  # the ``<nickel>:<masses>`` query this record came back for

    def cite(self) -> str:
        bits = [b for b in (self.reference, self.authors) if b]
        return " — ".join(bits) if bits else "(unattributed record)"


@dataclass
class LookupResult:
    nickel: str = ""
    configurations: list[str] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)  # configurations with the wanted masses
    records: list[Record] = field(default_factory=list)
    found: bool = False
    error: str = ""

    @property
    def references(self) -> list[str]:
        seen, out = set(), []
        for r in self.records:
            if r.reference and r.reference not in seen:
                seen.add(r.reference)
                out.append(r.reference)
        return out


def _cp1252_fallback(exc):
    """Loopedia serves mostly UTF-8 with stray cp1252 bytes in submitted names."""
    return exc.object[exc.start : exc.end].decode("cp1252", "replace"), exc.end


codecs.register_error("loopedia", _cp1252_fallback)


def _post(fields: dict[str, str], offline: bool = False) -> str:
    """POST to Loopedia, caching the raw bytes so a decoding fix applies on reread."""
    body = urllib.parse.urlencode(fields).encode()
    key = hashlib.sha1(body).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{key}.html"
    if cached.exists():
        return cached.read_bytes().decode("utf-8", "loopedia")
    if offline:
        raise RuntimeError("offline mode and no cached Loopedia response")
    req = urllib.request.Request(
        BASE, data=body, headers={"User-Agent": "feynman-agent/0.1"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
        raw = fh.read()
    cached.write_bytes(raw)
    return raw.decode("utf-8", "loopedia")


def _text(chunk: str) -> str:
    chunk = re.sub(r"<script.*?</script>|<style.*?</style>", "", chunk, flags=re.S)
    chunk = re.sub(r"<[^>]+>", " ", chunk)
    return re.sub(r"\s{2,}", " ", html.unescape(chunk)).strip()


def _parse_configurations(page: str) -> tuple[str, list[str]]:
    nickel = ""
    m = re.search(r"Available Configurations for Nickel index\s*([^<]+?)\s*for", _text(page))
    if m:
        nickel = m.group(1).strip()
    configs = re.findall(r'name="q"\s+value="([^"]*:[^"]*)"\s+class="buttonM"', page)
    return nickel, configs


# Field labels Loopedia prints inside a record; any of them ends the previous one.
_STOPS = "|".join(
    [
        "Orders in .{0,3}", "Reference", "Authors", "Description", "Submitter",
        "Record", "Reducible", "Number of master integrals", "Integrand type",
        "Propagator powers", "Relevant equations in reference", "Additional material",
    ]
)


def _parse_records(page: str) -> list[Record]:
    """Records live between 'View public records' and the submission form."""
    body = page.split("View public records")[-1].split("Add new record")[0]
    out = []
    chunks = re.split(r"(?=Orders in .{0,3}:)", _text(body))
    for chunk in chunks:
        if "Reference:" not in chunk:
            continue

        def grab(label):
            m = re.search(rf"{label}:\s*(.*?)(?=\s+(?:{_STOPS}):|$)", chunk)
            return m.group(1).strip() if m else ""

        rec = Record(
            orders=grab("Orders in .{0,3}"),
            reference=grab("Reference"),
            authors=grab("Authors"),
            description=grab("Description"),
            reducible=grab("Reducible"),
            n_masters=grab("Number of master integrals"),
            equations=grab("Relevant equations in reference"),
        )
        m = re.search(r"Record\s+(\d{10}\.\w+)", chunk)
        rec.record_id = m.group(1) if m else ""
        if rec.reference or rec.description:
            out.append(rec)
    return out


def _config_signature(query: str):
    """The propagator-mass partition of one configuration.

    ``e11|e|:n11|n|`` holds the graph and its masses in one string, the second
    half mirroring the first symbol for symbol, so the propagator slots are the
    ones whose Nickel symbol is neither ``e`` nor ``|``. A wildcard query like
    ``e11|e|:*`` names no masses at all and cannot match anything.
    """
    nickel, _, config = query.partition(":")
    if len(nickel) != len(config):
        return None
    slots = [c for n, c in zip(nickel, config) if n not in "e|"]
    return mass_signature("" if c in "0z" else c for c in slots)


def _matches(query: str, topo) -> bool:
    """Does this configuration carry the masses that were asked for?

    Both sides are configuration strings in the canonical labelling when the
    caller supplied one, so they compare directly — which also settles the
    external legs, and those are most of what separates one configuration of a
    massless graph from another. Masses read off an integrand block come with no
    such string, and then only the propagator masses can be compared at all.
    """
    nickel, _, config = query.partition(":")
    if topo.mass_config and nickel == topo.nickel():
        return config == topo.mass_config
    return _config_signature(query) == topo.mass_signature()


def lookup(topo_or_query, offline: bool = False, max_configs: int = 6) -> LookupResult:
    """Look a topology up in Loopedia and collect every public record.

    ``topo_or_query`` may be a Topology, an edge-list string or a Nickel index.

    Loopedia indexes a graph once per mass configuration and only the first few
    are fetched, so a Topology whose masses were stated puts the configurations
    carrying those masses first — massless included, since asking for the
    massless case is as specific as asking for any other. Said nothing about
    masses and the catalogue's own order is kept, which is meaningful in itself.
    """
    graph = (
        topo_or_query.edge_list_string()
        if hasattr(topo_or_query, "edge_list_string")
        else str(topo_or_query)
    )
    want = topo_or_query if getattr(topo_or_query, "masses_given", False) else None
    try:
        page = _post(
            {
                "graph": graph,
                "q": ":*",
                "rloops": "0", "nloops": "999",
                "rlegs": "0", "nlegs": "999",
                "rscales": "0", "nscales": "999",
                "need": "", "omit": "",
            },
            offline=offline,
        )
    except Exception as exc:  # network down, site moved, timeout
        return LookupResult(error=f"Loopedia unreachable: {exc}")

    nickel, configs = _parse_configurations(page)
    matched = [c for c in configs if want is not None and _matches(c, want)]
    res = LookupResult(nickel=nickel, configurations=configs, matched=matched)
    if not configs:
        return res

    ordered = matched + [c for c in configs if c not in matched]
    for cfg in ordered[:max_configs]:
        try:
            detail = _post({"q": cfg}, offline=offline)
        except Exception:
            continue
        for rec in _parse_records(detail):
            rec.config = cfg
            res.records.append(rec)
    res.found = bool(res.records)
    return res


def save_json(res: LookupResult, path: Path) -> None:
    path.write_text(json.dumps(res.__dict__, default=lambda o: o.__dict__, indent=2))
