"""The LangGraph workflow.

    identify -> local_db -> loopedia -> neatibp -> masters -+
                    ^                                       |
                    +------- advance <-- loopedia <---------+
                                |
                                +-> assemble -> ask -> arxiv -> read_papers -> report

The lookup stages are a **loop**, not a straight line. A master integral is a
Feynman diagram like any other, so the diagrams NeatIBP reduces to are sent back
through the very same ``local_db`` and ``loopedia`` nodes the original integral
went through — reduction is the step that turns an unknown integral into
several possibly-known ones, and the catalogue is exactly where to check.

``queue`` is what makes one pair of nodes serve both passes: empty means the
node is working on the original integral, otherwise on the master at its head.

Each node appends to ``trace`` so the final report can say what was tried even
when no result is found. The graph short-circuits to the report as soon as an
answer turns up, and interrupts before arXiv so the user decides whether to
keep going. Past that point a paper title is not the answer either, so
``read_papers`` opens the best few and pulls the actual expressions out.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import arxiv_tool, draw, extract, localdb, loopedia, neatibp
from . import topology as T

MAX_ARXIV_QUERIES = 3 # maximum arxiv articles
#----------------------------------

def _merge_findings(old: dict, new: dict) -> dict:
    """Lookup records keyed by Nickel index, merged field by field.

    Each turn of the loop contributes one piece (local entries, then
    references), so they accumulate instead of overwriting each other.
    """
    out = dict(old)
    for nickel, fields in new.items():
        out[nickel] = {**out.get(nickel, {}), **fields}
    return out


class State(TypedDict, total=False):
    raw_input: str
    input_kind: str
    workdir: str
    neatibp_mode: str
    extract_mode: str
    max_papers: int  # how many papers read_papers may open
    offline: bool
    always_reduce: bool

    topo: Any
    trace: Annotated[list[dict], operator.add]
    problems: Annotated[list[str], operator.add]
    
    db_hit: Any
    loopedia_hit: Any
    ibp: Any
    queue: list[dict]  # diagrams still to look up; empty = working on the original
    findings: Annotated[dict, _merge_findings]  # nickel -> what the loop learned
    masters: list[dict]
    unresolved: list[str]
    arxiv: list
    search_arxiv: bool
    extracted: list
    answer: str
    answer_source: str  # which stage set it — the report formats them differently
    report: str
    figure: str  # the diagram, as SVG the report links to and the CLI writes out


def _step(name: str, status: str, detail: str = "", **extra) -> dict:
    return {"trace": [{"step": name, "status": status, "detail": detail, **extra}]}


def _progress(text: str) -> None:
    """Show progress for the whole workflow; indicating what the node is doing
    """
    from langgraph.config import get_stream_writer

    try:
        get_stream_writer()({"progress": text})
    except RuntimeError:
        pass


def _target(state: State):
    """What the lookup nodes are pointed at: ``(topology, queue_item_or_None)``.

    An empty queue means the original integral; otherwise the head of the queue.
    """
    queue = state.get("queue") or []
    return (queue[0]["topo"], queue[0]) if queue else (state["topo"], None)


def _found(record: dict) -> bool:
    """Resolved = a stored closed form locally, or at least one reference."""
    return bool(record.get("references") or any(e["result"] for e in record.get("local", [])))


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def identify(state: State) -> dict:
    """Turn whatever input into a canonical Topology."""
    raw = state["raw_input"].strip()
    kind = state.get("input_kind") or _guess_kind(raw)

    if kind == "figure": # reading figure requires llm
        from . import llm

        if not llm.available():
            reason = llm.missing_reason()
            return {
                **_step("identify", "failed", f"cannot read the figure: {reason}"),
                "problems": [
                    f"The input is a figure, but {reason}",
                    "Only figures need a model; the Nickel index or edge list runs the "
                    "same analysis with no key at all.",
                ],
            }
        try:
            _progress(f"reading {Path(raw).name} with {llm.describe()}")
            topo = llm.read_figure(raw)
        except Exception as exc:  # network, credit, quota, malformed reply
            return {
                **_step("identify", "failed", f"could not read the figure: {exc}"),
                "problems": [
                    f"Reading {raw} with {llm.describe()} failed: {exc}",
                    "Only figures need a model; the Nickel index or edge list runs the "
                    "same analysis with no key at all.",
                ],
            }
    elif kind == "nickel":
        topo = T.from_nickel(raw)
    elif kind == "edgelist":
        topo = T.from_edge_list(raw)
    elif kind == "integrand":
        topo = _parse_integrand(raw)
    else:
        return {
            **_step("identify", "failed", f"unrecognised input {raw[:60]!r}"),
            "problems": [f"Could not tell what kind of input {raw[:60]!r} is."],
        }

    if not topo.propagators:
        topo = T.assign_momenta(topo)
    topo.name = topo.name or "topology"
    return {
        "topo": topo,
        **_step("identify", "ok", topo.summary(), nickel=topo.nickel(), kind=kind),
    }


def local_db(state: State) -> dict:
    """Search the local JSON file and notebook folder first.

    Runs for the original integral, and again for every diagram the reduction
    produces — the loop re-enters here, so a master that happens to be in the
    local database is found by exactly the same code.
    """
    topo, item = _target(state)
    hit = localdb.search(topo)

    if item is not None:  # in the master loop: record, never set the headline
        entries = [{"name": e.name, "result": e.result, "source": e.source} for e in hit.exact]
        return {"findings": {item["origin"]: {"local": entries}}} if entries else {}

    if hit.found and not topo.numerator:
        entry = next(e for e in hit.exact if e.has_result)
        return {
            "db_hit": hit,
            "answer": entry.result,
            "answer_source": "local database",
            **_step("local database", "hit", f"{entry.name} ({entry.source})"),
        }
    # An entry for the same graph that was rejected on masses is worth naming:
    # "no entry" would be misleading when the massless case is right there.
    same_graph = [e for e in hit.similar if e.nickel == topo.nickel()]
    if hit.found:
        # Stored results are scalar integrals. With a numerator the graph is the
        # same and the integral is not; the reduction says what it is in terms
        # of the family's masters, and this entry may well be one of those.
        detail = f"{len(hit.exact)} entry/entries for this graph, but the numerator has to be reduced first"
    elif hit.exact:
        detail = f"{len(hit.exact)} entry/entries for this Nickel index but no stored result"
    elif same_graph:
        detail = f"{len(same_graph)} entry/entries for {topo.nickel()} but with different masses"
    else:
        detail = f"no entry for {topo.nickel()}"
    return {"db_hit": hit, **_step("local database", "miss", detail)}


def loopedia_node(state: State) -> dict:
    """Loopedia as the reference database — for the original and each master."""
    topo, item = _target(state)
    _progress(f"loopedia: {topo.nickel()}")
    res = loopedia.lookup(topo, offline=state.get("offline", False))

    if item is not None:
        return {"findings": {item["origin"]: {"references": res.references}}} if res.references else {}

    if res.error:
        return {
            "loopedia_hit": res,
            **_step("loopedia", "error", res.error),
            "problems": [f"Loopedia could not be reached: {res.error}"],
        }
    if res.found:
        return {
            "loopedia_hit": res,
            "answer": _format_loopedia(res),
            "answer_source": "Loopedia catalogue",
            **_step(
                "loopedia",
                "hit",
                f"{len(res.records)} record(s) across {len(res.configurations)} mass "
                f"configuration(s)"
                + (f", {len(res.matched)} matching the masses given" if res.matched else ""),
            ),
        }
    detail = (
        f"graph is known ({len(res.configurations)} configurations) but carries no public record"
        if res.configurations
        else "graph not in Loopedia"
    )
    return {"loopedia_hit": res, **_step("loopedia", "miss", detail)}


def neatibp_node(state: State) -> dict:
    """Reduce the topology to master integrals with NeatIBP."""
    topo = state["topo"]
    workdir = Path(state.get("workdir") or f"/tmp/feynman_agent/{topo.nickel().replace('|', '_')}")
    what = f" with numerator {topo.numerator}" if topo.numerator else ""
    _progress(
        f"NeatIBP: reducing {topo.nickel()}, {len(topo.int_edges)} propagators{what} "
        f"— minutes; logs under {workdir}"
    )
    res = neatibp.run(topo, workdir, mode=state.get("neatibp_mode", "auto"))
    if not res.ok:
        return {
            "ibp": res,
            **_step("NeatIBP", "failed", res.error or "no master integrals produced"),
            "problems": [f"IBP reduction did not produce master integrals: {res.error}"],
        }
    note = " (mocked)" if res.mocked else ""
    # The headline first: the CLI shows one line of this while the report shows
    # all of it, so anything the run had to do on the side goes underneath — and
    # carries its own dash, the report being one flattened line per stage.
    detail = f"{len(res.masters)} master integrals, {res.n_ibp} IBP relations{note}"
    targets = [f"— targets: {topo.numerator} = {res.expansion}"] if res.expansion else []
    return {
        "ibp": res,
        **_step(
            "NeatIBP",
            "ok",
            "\n".join([detail, *targets, *(f"— {n}" for n in res.notes)]),
            mocked=res.mocked,
        ),
    }


def expand_masters(state: State) -> dict:
    """Turn each master ``G[...]`` into its own diagram and queue it up.

    Pinching (contracting) the propagators whose power is zero gives the
    master's graph. Queueing them is what closes the loop back to ``local_db``:
    reduction is precisely the step that converts one unknown integral into
    several that the catalogues may well already know.
    """
    topo, res = state["topo"], state["ibp"]
    n_props = len(topo.int_edges)
    masters, queue, seen = [], [], set()
    for mi in res.masters:
        powers = T.parse_g(mi)
        keep = {i for i, a in enumerate(powers[:n_props]) if a > 0}
        sub = T.sector_topology(topo, keep)
        try:
            nickel = sub.nickel()
        except ValueError as exc:
            masters.append({"integral": mi, "status": "error", "detail": str(exc), "found": False})
            continue
        masters.append(
            {
                "integral": mi,
                "nickel": nickel,
                "loops": sub.n_loops,
                "legs": sub.n_legs,
                "props": len(sub.int_edges),
            }
        )
        if nickel not in seen:  # each distinct diagram is looked up once
            seen.add(nickel)
            queue.append({"topo": sub, "origin": nickel, "merged": False})

    return {
        "masters": masters,
        "queue": queue,
        **_step(
            "master sectors",
            "ok",
            f"{len(masters)} masters span {len(queue)} distinct diagram(s), queued for lookup",
        ),
    }

#----------------------------------------------------
# defermine what next step is
def advance(state: State) -> dict:
    """Close out the diagram at the head of the queue and move to the next.

    An unresolved one gets a single retry with its external legs merged, as
    another turn of the same loop: pinching often strands several on-shell legs
    on one vertex, and while that graph is rarely catalogued the merged-leg
    equivalent usually is (``ee11|ee|`` -> ``e11|e|``). Their momenta simply add,
    so the two are the same integral.
    """
    queue = list(state["queue"])
    item = queue[0]
    found = _found(state.get("findings", {}).get(item["origin"], {}))

    if not found and not item["merged"]:
        merged = T.merge_legs(item["topo"])
        if merged.nickel() != item["topo"].nickel():
            queue[0] = {**item, "topo": merged, "merged": True}
            return {"queue": queue}

    # Only the retry sets ``via``, so the Nickel index alone says what happened;
    # the report supplies the words.
    via = item["topo"].nickel() if found and item["merged"] else ""
    return {"queue": queue[1:], "findings": {item["origin"]: {"found": found, "via": via}}}


def closed_form(master: dict) -> str:
    """The stored closed form for a master, if the local database has one."""
    return next((e["result"] for e in master.get("local", []) if e.get("result")), "")


def assemble(state: State) -> dict:
    """Fold the loop's findings back onto the masters and judge the result.

    A reduction whose masters are all known is itself a result. Without this the
    agent would look every master up and then still report 'no result',
    throwing away the answer it just finished assembling.
    """
    findings = state.get("findings", {})
    masters = [{**m, **findings.get(m.get("nickel", ""), {})} for m in state.get("masters", [])]
    unresolved = sorted({m["nickel"] for m in masters if not m.get("found") and m.get("nickel")})
    if not masters:
        return {"unresolved": unresolved}

    resolved = sum(1 for m in masters if m.get("found"))
    with_form = [m for m in masters if closed_form(m)]
    detail = (
        f"{len(masters)} master integrals, {resolved} resolved by the catalogue loop"
        f", {len(with_form)} with a stored closed form"
    )
    if resolved < len(masters):
        return {
            "masters": masters,
            "unresolved": unresolved,
            **_step("assemble", "partial", detail),
        }

    ibp = state.get("ibp")
    where = ""
    if ibp is not None and ibp.workdir:
        where = (
            f"\nCombining them into the original integral needs the IBP coefficients: "
            f"NeatIBP wrote the system under {ibp.workdir}/outputs/*/results/ "
            f"(solve it with Kira or FiniteFlow for the target's coefficients)."
        )
    expansion = ""
    if ibp is not None and ibp.expansion:
        expansion = f"The numerator gives `{state['topo'].numerator}` = {ibp.expansion}, and "
    answer = (
        f"{expansion}IBP reduction resolves this integral completely: it reduces to "
        f"{len(masters)} master integral(s), every one of which is known "
        f"({len(with_form)} with a closed form stored locally, the rest by "
        f"literature reference). See **Master integrals** below for each one.{where}"
    )
    out = {"masters": masters, "unresolved": [], **_step("assemble", "ok", detail)}
    # A reference found for the parent earlier is the better headline; keep it.
    if not state.get("answer"):
        out["answer"] = answer
        out["answer_source"] = "IBP reduction"
    return out


def ask_user(state: State) -> dict:
    """Nothing conclusive yet — ask before going to arXiv."""
    answer = interrupt(
        {
            "question": "Search arXiv for the remaining diagrams (and read the results "
            "out of the best papers), or stop here?",
            "unresolved": state.get("unresolved") or ["the full topology"],
            "options": ["search", "stop"],
        }
    )
    reply = str(answer).strip().lower()
    want = reply in {"s", "y"} or reply.startswith(("search", "yes"))
    return {
        "search_arxiv": want,
        **_step("ask user", "ok", "search arXiv" if want else "stop requested"),
    }

#--------------------------------------------------
# search arxiv articles

def arxiv_node(state: State) -> dict:
    """Last resort: search for the diagrams the loop could not explain.

    After a reduction those are the unresolved *masters*, not the original
    integral — the parent is already accounted for by the reduction, so
    searching for it would answer a question that is no longer open. Only when
    there was no reduction at all does the original topology become the target.

    A query can only describe a diagram by its loop and leg count, since no
    search engine indexes Nickel indices; diagrams sharing those counts would
    produce identical results, so they share one query instead.
    """
    groups: dict[tuple[int, int], list[str]] = {}
    for m in state.get("masters", []):
        if m.get("found") or not m.get("nickel"):
            continue
        same = groups.setdefault((m["loops"], m["legs"]), [])
        if m["nickel"] not in same:
            same.append(m["nickel"])
    if not groups:
        topo = state["topo"]
        groups = {(topo.n_loops, topo.n_legs): [topo.nickel()]}

    results = []
    for (loops, legs), nickels in list(groups.items())[:MAX_ARXIV_QUERIES]:
        _progress(f"arXiv: searching for {loops}-loop {legs}-point ({', '.join(nickels)})")
        res = arxiv_tool.search(arxiv_tool.build_query_for(loops, legs), max_results=5)
        res.target = ", ".join(nickels)
        results.append(res)

    errors = [r.error for r in results if r.error]
    total = sum(len(r.papers) for r in results)
    skipped = len(groups) - len(results)
    detail = f"{total} candidate paper(s) for {len(results)} loop/leg signature(s)"
    if skipped:
        detail += f" ({skipped} more not searched, {MAX_ARXIV_QUERIES}-query cap)"
    return {
        "arxiv": results,
        **_step("arXiv", "error" if errors else ("ok" if total else "miss"), detail),
        "problems": errors,
    }


def _referenced(state: State) -> list[tuple]:
    """Papers the catalogues named, as ``(reference, which diagram)`` pairs.

    A Loopedia hit *is* a list of arXiv ids for this exact graph — better aimed
    than any search, and the reason this node is not reached only from arXiv.
    """
    out = []
    hit = state.get("loopedia_hit")
    if hit is not None and hit.records and state.get("topo") is not None:
        out += [(rec, state["topo"].nickel()) for rec in hit.records]
    for master in state.get("masters", []):
        out += [(ref, master["nickel"]) for ref in master.get("references", [])]
    return out


def extract_node(state: State) -> dict:
    """The reference is a pointer; go into the paper and get the result.

    Stopping at a citation would leave the last and most tedious step — finding
    which file or equation actually holds the expression — to the reader. The
    submitted source says it directly: modern papers attach the result as
    Mathematica input, older ones print the epsilon expansion in the LaTeX, and
    both usually contain a sentence saying which.

    Two ways to arrive here, and the catalogue route is the common one: papers
    Loopedia named for this graph, or, when nothing was catalogued at all, the
    papers the arXiv search turned up.
    """
    offline = state.get("offline", False)
    candidates = extract.pick(state.get("arxiv") or [])
    if not candidates:
        candidates = extract.from_references(_referenced(state), offline=offline)
    limit = state.get("max_papers") or extract.MAX_PAPERS
    picks = extract.select(candidates, limit)
    #-------------------------------------------------------------
    # An API call is network like any other, so --offline silences the model too.
    mode = state.get("extract_mode", "auto")
    summarise = mode in ("auto", "rewrite") and not offline
    # Rewriting the equations is a call each, where the summary is one call for
    # the paper, so it is the level nobody gets without asking.
    rewrite = mode == "rewrite" and not offline
    # It is also the slowest thing a run without a reduction does — a reasoning
    # model spends minutes per equation thinking — so the line says so, the way
    # the NeatIBP line does.
    slow = " — rewriting its equations, minutes" if rewrite else ""
    found = []
    for n, (paper, target) in enumerate(picks, 1):
        _progress(f"reading arXiv:{paper.arxiv_id} ({n}/{len(picks)}) {paper.title[:40]}{slow}")
        found.append(
            extract.extract(
                paper, target=target, offline=offline, summarise=summarise, rewrite=rewrite
            )
        )

    errors = [f"arXiv:{e.paper.arxiv_id}: {e.error}" for e in found if e.error]
    files = sum(len(e.files) for e in found)
    eqs = sum(len(e.equations) for e in found)
    detail = f"read {len(found)} paper(s): {files} data file(s), {eqs} result equation(s)"
    if rewrite:
        shown = sum(1 for e in found for q in e.equations if q.clean)
        detail += f", {shown} rewritten to read"
    # A reduction cites more papers than anyone wants downloaded, so say how
    # many were left — a silent cap reads as "that was everything".
    if len(candidates) > len(picks):
        detail += f" ({len(candidates) - len(picks)} more not opened, {limit}-paper cap)"
    return {
        "extracted": found,
        **_step("read papers", "ok" if files or eqs else "miss", detail),
        "problems": errors,
    }


# Status glyphs, shared with the CLI's progress lines so a stage looks the same
# while it runs and afterwards in the report.
MARKS = {"ok": "✓", "hit": "✓", "miss": "·", "partial": "~", "failed": "✗", "error": "✗"}


def _master_lines(master: dict) -> list[str]:
    """One master integral: the verdict, the diagram, and where it is known from."""
    if master.get("status") == "error":
        return [f"- ✗ `{master['integral']}` — no sector graph: {master['detail']}"]

    via = f", matched as `{master['via']}` with its external legs merged" if master.get("via") else ""
    out = [
        f"- {'✓' if master.get('found') else '?'} `{master['integral']}` — "
        f"`{master['nickel']}` ({master['loops']}L {master['legs']}pt){via}"
    ]
    names = ", ".join(e["name"] for e in master.get("local", []))
    where = [f"local: {names}"] if names else []
    where += [extract.cite(r) for r in master.get("references", [])[:3]]
    out.append(f"  - {' · '.join(where) or 'no reference found'}")
    form = closed_form(master)
    if form:
        out += ["", "  ```", *(f"  {ln}" for ln in form.splitlines()), "  ```"]
    return out

#-----------------------------------------------------------
def report(state: State) -> dict:
    """Either the result, or what was done and what is missing.

    Markdown, because the report is read after the run rather than during it:
    an epsilon expansion needs its line breaks, a reference needs to be
    clickable, and the answer belongs above the audit trail, not below it.

    The diagram goes first, before any of it. Everything below is only worth
    reading if the agent identified the right integral, and that is a question
    about a picture.
    """
    topo = state.get("topo")
    # Canonicalisation is brute force and the report asks for the name several
    # times over, so ask once.
    nickel = topo.nickel() if topo is not None else ""
    title = f"Feynman integral `{nickel}`" if topo is not None else "Feynman integral"
    lines = [f"# {title}", ""]
    figure = draw.svg(topo) if topo is not None else ""

    if topo is not None:
        lines += [
            f"![Feynman diagram of {nickel}]({T.slug(topo)}.svg)",
            "",
            f"*{draw.caption(topo)}*",
            "",
            "## Identified",
            "",
            topo.summary(),
            "",
            f"- **edge list** — `{topo.edge_list_string()}`",
            "- **propagators** — " + (", ".join(f"`{p}`" for p in topo.propagators) or "(none)"),
            *(
                [
                    "- **masses** — "
                    + ", ".join(
                        f"`{topo.propagators[i]}`: `{m}`" for i, m in sorted(topo.masses.items())
                    )
                ]
                if topo.masses
                else []
            ),
            *([f"- **numerator** — `{topo.numerator}`"] if topo.numerator else []),
            *(f"- **note** — {n}" for n in topo.notes),
            "",
        ]

    if state.get("answer"):
        source = state.get("answer_source", "")
        lines += ["## Result", ""]
        lines += [f"*From the {source}.*", ""] if source else []
        # A stored closed form is an expression, and Markdown would eat its
        # stars and underscores; the catalogue and reduction answers are prose.
        if source == "local database":
            lines += ["```", state["answer"], "```", ""]
        else:
            lines += [state["answer"], ""]
    else:
        lines += [
            "## No direct result",
            "",
            "Nothing conclusive — what the run did produce is below.",
            "",
        ]

    masters = state.get("masters")
    if masters:
        lines += [f"## Master integrals ({len(masters)})", ""]
        ibp = state.get("ibp")
        if ibp is not None and ibp.expansion:
            lines += [
                f"`{topo.numerator}` over these propagators is {ibp.expansion}, "
                "each term reduced to the masters below.",
                "",
            ]
        for m in masters:
            lines += _master_lines(m)
        lines.append("")

    extracted = [e for e in state.get("extracted") or [] if e.found or e.error]
    if extracted:
        lines += [
            "## Read from the papers themselves",
            "",
            "Candidates, not verified equalities: every paper normalises its integrals its",
            "own way, so the prefactor and the propagator ordering have to be matched first.",
            "",
        ]
        for e in extracted:
            lines += e.markdown()

    for res in state.get("arxiv") or []:
        if res.papers:
            lines += [f"## arXiv candidates for `{res.target}`", ""]
            lines += [f"- {p.markdown()}" for p in res.papers[:4]] + [""]

    lines += ["## What was done", ""]
    for entry in state.get("trace", []):
        detail = " ".join(entry["detail"].split())
        lines.append(f"- {MARKS.get(entry['status'], '·')} **{entry['step']}** — {detail}")
    lines.append("")

    #------------------------------------------------------------
    # failure informations
    failures = state.get("problems", [])
    if failures:
        lines += ["## problems", "", *(f"- {d}" for d in failures), ""]

    if not state.get("answer"):
        lines += ["## Likely useful next", "", *(f"- {s}" for s in _suggestions(state)), ""]

    return {"report": "\n".join(lines).rstrip() + "\n", "figure": figure}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _guess_kind(raw: str) -> str:
    if Path(raw).suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".gif", ".webp"}:
        return "figure"
    if raw.startswith("[") or raw.startswith("("):
        return "edgelist"
    if "|" in raw:
        return "nickel"
    if "Propagators" in raw or "^2" in raw or "LoopMomenta" in raw:
        return "integrand"
    return "unknown"


def _split_mass(entry: str) -> tuple[str, str]:
    """``(l1+k1)^2-msq`` -> ``("l1+k1", "msq")``; a bare momentum is massless.

    A massive line is written as its momentum squared minus a mass term, and
    that term is the whole of what distinguishes it from a massless one. It is
    kept verbatim rather than normalised, because ``m^2`` and ``msq`` are both
    in common use and NeatIBP accepts either.
    """
    mom, _, mass = entry.partition("^2")
    mom = mom.strip()
    if mom.startswith("(") and mom.endswith(")"):
        mom = mom[1:-1]
    return mom.strip(), mass.strip().lstrip("+-").strip()


def _parse_integrand(raw: str):
    """Read a NeatIBP-style kinematics block, or a bare propagator list.

    ``Propagators={l1^2-m^2,(l1+k1)^2}`` carries the masses inline, so each
    entry is split before the momenta are handed to the graph reconstruction —
    which only knows about momentum conservation, and would otherwise take a
    mass symbol for an external leg.
    """
    import re

    def grab(key):
        m = re.search(rf"{key}\s*=\s*\{{(.*?)\}}", raw, re.S)
        return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []

    loops = grab("LoopMomenta")
    exts = grab("ExternalMomenta")
    m = re.search(r"Propagators\s*=\s*(?:#\^2&/@)?\{(.*?)\}", raw, re.S)
    entries = [x.strip() for x in m.group(1).split(",")] if m else [
        x.strip() for x in raw.strip().strip("{}").split(",") if x.strip()
    ]
    split = [_split_mass(e) for e in entries]
    props = [mom for mom, _ in split]
    masses = {i: mass for i, (_, mass) in enumerate(split) if mass}
    if not loops:
        symbols = {s for p in props for s in re.findall(r"[A-Za-z]\w*", p)}
        loops = sorted(s for s in symbols if s.startswith("l"))
        exts = sorted(s for s in symbols if not s.startswith("l"))
    topo = T.from_propagators(props, loops, exts, name="integrand", masses=masses)
    # NeatIBP has no such line; it is this agent's, and neatibp.py turns it into
    # the index vectors NeatIBP does take.
    m = re.search(r"Numerator\s*=\s*([^;]+)", raw)
    topo.numerator = " ".join(m.group(1).split()) if m else ""
    return topo

#----------------------------------------------------------
def _format_loopedia(res: loopedia.LookupResult) -> str:
    if res.matched:
        lines = [
            f"Loopedia knows this graph (`{res.nickel}`). It indexes references rather than",
            "closed forms, and indexes a graph once per mass configuration — the ones",
            "carrying the masses asked for come first. The result lives in these papers:",
            "",
        ]
    else:
        lines = [
            f"Loopedia knows this graph (`{res.nickel}`). It indexes references, not closed",
            "forms, so the result itself lives in these papers:",
            "",
        ]
    for rec in res.records[:8]:
        cited = extract.cite(rec.reference) if rec.reference else "(unattributed record)"
        lines.append(f"- {cited}" + (f" — {rec.authors}" if rec.authors else ""))
        if rec.config:
            mark = " — **the masses asked for**" if rec.config in res.matched else ""
            lines.append(f"  - masses: `{rec.config.partition(':')[2]}`{mark}")
        if rec.orders:
            lines.append(f"  - orders in eps: `{rec.orders}`")
        if rec.n_masters:
            lines.append(f"  - masters: {rec.n_masters}")
        if rec.description:
            lines.append(f"  - {rec.description[:160]}")
    lines += ["", "*(cite [arXiv:1709.01266](https://arxiv.org/abs/1709.01266) for Loopedia)*"]
    return "\n".join(lines)


def _suggestions(state: State) -> list[str]:
    out = []
    ibp = state.get("ibp")
    if ibp is not None and ibp.mocked:
        out.append("Re-run with --neatibp real to compute the masters instead of mocking them.")
    if ibp is not None and ibp.workdir:
        out.append(f"NeatIBP inputs and logs are in {ibp.workdir}.")
    if state.get("unresolved"):
        out.append(
            "Unreferenced masters: "
            + ", ".join(state["unresolved"])
            + " — these are the sub-topologies to evaluate or look up by hand."
        )
    broken = [m for m in state.get("masters", []) if m.get("status") == "error"]
    if broken:
        out.append(
            f"{len(broken)} master(s) could not be turned into a graph at all: "
            + ", ".join(m["integral"] for m in broken)
        )
    # The biggest attachment, since a table of results dwarfs anything else in
    # the submission — and a link only helps for files arXiv serves singly.
    files = [f for e in state.get("extracted") or [] for f in e.files if f.direct]
    if files:
        best = max(files, key=lambda f: f.size)
        out.append(f"The most direct thing to try is the attached file {best.name}: {best.url}")
    topo = state.get("topo")
    if topo is not None and topo.n_legs > 4:
        out.append(
            "The kinematics template only covers up to 4 external legs; supply the "
            "invariants explicitly for this topology."
        )
    if not state.get("answer"):
        out.append(
            "Adding the result to data/local_db.json (or a notebook in data/notebooks/) "
            "makes the next lookup instant."
        )
    return out


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def _after_identify(state: State) -> str:
    return "report" if state.get("topo") is None else "local_db"


def _in_loop(state: State) -> bool:
    return bool(state.get("queue"))


def _after_local_db(state: State) -> str:
    if not _in_loop(state) and state.get("answer") and not state.get("always_reduce"):
        return "report"
    return "loopedia"


def _finish(state: State) -> str:
    """On the way out: read what the catalogues cited, if there is anything.

    A reference is not a result, so a run that ends with citations has not
    finished the job — the papers behind them are the job.
    """
    if state.get("extract_mode", "auto") == "off":
        return "report"
    picks = extract.from_references(_referenced(state), offline=state.get("offline", False))
    return "read_papers" if picks else "report"


def _after_loopedia(state: State) -> str:
    if _in_loop(state):
        return "advance"  # close this diagram out, then round again
    # A reference for the graph is an answer for its scalar integral only; a
    # numerator is a different integral of the same family and has to be
    # reduced to say what it is in terms of the masters that reference gives.
    if state.get("answer") and not (state.get("always_reduce") or state["topo"].numerator):
        return _finish(state)
    return "neatibp"


def _after_neatibp(state: State) -> str:
    return "expand_masters" if state["ibp"].ok else "ask_user"


def _next_diagram(state: State) -> str:
    """The loop condition: keep looking things up while the queue has entries."""
    return "local_db" if _in_loop(state) else "assemble"


def _after_assemble(state: State) -> str:
    return _finish(state) if not state.get("unresolved") else "ask_user"


def _after_ask(state: State) -> str:
    return "arxiv" if state.get("search_arxiv") else "report"


def _after_arxiv(state: State) -> str:
    papers = any(r.papers for r in state.get("arxiv") or [])
    return "read_papers" if papers and state.get("extract_mode", "auto") != "off" else "report"


# Each queued diagram costs up to six supersteps (two lookups and an advance,
# twice if the merged-leg retry fires), so the default limit of 25 would abort a
# reduction with more than about three masters.
RECURSION_LIMIT = 250

#--------------------------------------------------------
# the graph

def build(checkpointer=None, recursion_limit: int = RECURSION_LIMIT):
    g = StateGraph(State)
    g.add_node("identify", identify)
    g.add_node("local_db", local_db)
    g.add_node("loopedia", loopedia_node)
    g.add_node("neatibp", neatibp_node)
    g.add_node("expand_masters", expand_masters)
    g.add_node("advance", advance)
    g.add_node("assemble", assemble)
    g.add_node("ask_user", ask_user)
    g.add_node("arxiv", arxiv_node)
    g.add_node("read_papers", extract_node)
    g.add_node("report", report)

    g.add_edge(START, "identify")
    g.add_conditional_edges("identify", _after_identify, ["local_db", "report"])
    g.add_conditional_edges("local_db", _after_local_db, ["loopedia", "report"])
    g.add_conditional_edges(
        "loopedia", _after_loopedia, ["neatibp", "advance", "read_papers", "report"]
    )
    g.add_conditional_edges("neatibp", _after_neatibp, ["expand_masters", "ask_user"])
    # The loop: every master goes back through the same two lookup nodes.
    g.add_conditional_edges("expand_masters", _next_diagram, ["local_db", "assemble"])
    g.add_conditional_edges("advance", _next_diagram, ["local_db", "assemble"])
    g.add_conditional_edges("assemble", _after_assemble, ["ask_user", "read_papers", "report"])
    g.add_conditional_edges("ask_user", _after_ask, ["arxiv", "report"])
    g.add_conditional_edges("arxiv", _after_arxiv, ["read_papers", "report"])
    g.add_edge("read_papers", "report")
    g.add_edge("report", END)

    if checkpointer is None:
        checkpointer = _saver()
    return g.compile(checkpointer=checkpointer).with_config(recursion_limit=recursion_limit)


# The state carries our own dataclasses, so tell the checkpoint serializer which
# types it may rebuild. Naming them explicitly is quieter than the permissive
# default and keeps deserialization restricted to these.
_CHECKPOINT_TYPES = [
    ("feynman_agent.topology", "Topology"),
    ("feynman_agent.localdb", "DBHit"),
    ("feynman_agent.localdb", "Entry"),
    ("feynman_agent.loopedia", "LookupResult"),
    ("feynman_agent.loopedia", "Record"),
    ("feynman_agent.neatibp", "IBPResult"),
    ("feynman_agent.arxiv_tool", "ArxivResult"),
    ("feynman_agent.arxiv_tool", "Paper"),
    ("feynman_agent.extract", "Extraction"),
    ("feynman_agent.extract", "DataFile"),
    ("feynman_agent.extract", "Equation"),
]


def _saver():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES))
