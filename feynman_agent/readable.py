r"""Turn a paper's private LaTeX into LaTeX that renders — with the model.

A submission's display equations are written against macros the authors defined
in their own preamble, so what ``extract`` lifts out of the source reads like
this:

    \dps A_{7,1} &=& i \, \ESGamma^3 \lek -\lp q\rp^2-i \, \eta \rek^{-1-3\, \eps}\nnb\\

Every one of ``\dps``, ``\ESGamma``, ``\lek``, ``\lp``, ``\rp``, ``\rek``,
``\eps`` and ``\nnb`` is local to that one paper. Nothing outside it renders
them, and nobody reads them. The definitions travel in the same tarball though,
which makes the rewrite mechanical: expand the macros, straighten out the
eqnarray alignment, and what is left is ordinary amsmath that a Markdown viewer
typesets as mathematics.

Expanding TeX macros properly means implementing TeX, so this is a job for the
model — and equally a job where the model must not be inventive. The
instruction is to transcribe, never to evaluate or tidy the mathematics, and
the paper's own LaTeX stays in the report underneath, where any liberty taken
shows up against the source.

The same call also decides whether the equation is a result at all. ``extract``
scores equations by what they are written in, which is enough to find the
results but cannot tell a definition of a polylogarithm from a value of one; a
reader can, and so can the model.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from . import llm

KEEP = 3  # equations shown per paper; extract offers twice as many candidates
MAX_MACROS = 60  # definition lines sent along; more than any one equation uses

PROMPT = """This display equation comes verbatim from an arXiv paper's LaTeX source,
together with the macros that paper's preamble defines.

Say whether it is a RESULT for the diagram asked about — an evaluated integral,
an expansion in epsilon, a closed form — rather than a definition, a piece of
notation, or an intermediate step. If it is, rewrite it so that it renders
outside the paper:

- expand every macro the paper defines; leave standard LaTeX as it is
- transcribe the mathematics exactly: same numbers, same signs, same functions,
  same order. Never evaluate, simplify, correct or complete it
- amsmath only, and no $ or $$ delimiters: a multi-line equation becomes
  \\begin{{aligned}} ... \\end{{aligned}}, while \\label, \\nonumber, \\hspace and
  eqnarray's empty && columns go
- \\mathrm or \\text for upright names, never \\mbox — browser renderers do not
  have it
- an equation ending in "… (N chars in full)" was cut off before you saw it; end
  yours with \\dots

Answer in exactly this form and nothing else:

RESULT: yes or no
GIVES: at most twenty words of plain prose — which quantity, to what order in
epsilon, in which functions. No LaTeX in this line, and "unclear" rather than a
guess.
LATEX:
the rewritten equation, on as many lines as it needs

DIAGRAM ASKED ABOUT: {target}

MACROS THE PAPER DEFINES:
{macros}

EQUATION:
{equation}
"""

_RESULT = re.compile(r"^RESULT:\s*(\w+)", re.M)
_GIVES = re.compile(r"^GIVES:\s*(.*)", re.M)
_FENCE = re.compile(r"^```\w*\n|\n```$")


def available() -> bool:
    return llm.available()


def rewrite(equations, macros: dict[str, str], target: str = "") -> tuple[list, list[str]]:
    """The results among ``equations``, rewritten. Returns ``(chosen, notes)``.

    One call per equation, all of them at once. A reasoning model spends most
    of a rewrite thinking — 4449 of the 5028 tokens billed for the equation in
    the module docstring — so asking for three at once runs out of budget and
    returns nothing, where three calls together finish in the time one takes.
    An equation the model stumbles over then costs only itself.

    Nothing here is load-bearing. An empty list means the caller keeps the
    candidates it already had, printed in the LaTeX the authors wrote, which is
    the whole report as it stood before this step existed.
    """
    if not llm.available():
        return [], [f"equations left as the authors wrote them: {llm.missing_reason()}"]
    text = "\n".join(list(macros.values())[:MAX_MACROS]) or "none found in the source"
    # One proxy guard around the whole batch: it edits the environment, and the
    # per-call guards inside are inert while this one holds it. See llm.py.
    with llm.usable_proxies(), ThreadPoolExecutor(max_workers=len(equations)) as pool:
        answers = list(pool.map(lambda eq: _one(eq, text, target), equations))

    chosen = [eq for eq, (keep, _) in zip(equations, answers) if keep][:KEEP]
    failed = [why for _, why in answers if why]
    if chosen and not failed:
        return chosen, []
    if chosen:
        # Said out loud: a section where two equations are mathematics and the
        # third is source reads as a bug rather than as a model that gave up.
        return chosen, [
            f"{len(failed)} of {len(equations)} candidate equation(s) could not be read: "
            f"{failed[0]}"
        ]
    why = failed[0] if failed else f"{llm.describe()} found no results among the candidates"
    return [], [f"equations left as the authors wrote them: {why}"]


def _one(eq, macros: str, target: str) -> tuple[bool, str]:
    """One equation, judged and rewritten in place. Returns ``(keep it, why not)``.

    Plain text rather than JSON, because the payload is LaTeX: in JSON every
    backslash of it has to be doubled, which is a page of escaping for the model
    to get right and one more way for a reply full of good mathematics to arrive
    unreadable.
    """
    prompt = PROMPT.format(target=target or "not specified", macros=macros, equation=eq.block())
    try:
        reply = llm.ask(prompt).strip()
    except Exception as exc:  # network, quota, timeout
        return False, str(exc)
    if not reply:
        return False, (
            f"{llm.describe()} returned an empty reply — a reasoning model can spend the "
            f"whole {llm.max_tokens()}-token budget thinking; raise FEYNMAN_AGENT_MAX_TOKENS"
        )
    verdict = _RESULT.search(reply)
    if verdict is None:
        return False, f"unreadable reply from {llm.describe()}: {reply[:100]}"
    if verdict.group(1).lower() not in {"yes", "true"}:
        return False, ""  # a definition, not a result: dropped, and nothing to report

    gives = _GIVES.search(reply)
    eq.gives = gives.group(1).strip() if gives else ""
    # \mbox is plain TeX that no browser renderer implements, and it survives
    # the rewrite because the papers are full of it — Smirnov writes his
    # polylogarithms with it. The prompt asks for \text; this is the one liberty
    # taken with the reply, and it is a spelling rather than a quantity.
    eq.clean = _FENCE.sub("", reply.partition("LATEX:")[2].strip()).replace(r"\mbox", r"\text")
    return bool(eq.clean), ""
