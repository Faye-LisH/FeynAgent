"""Read the results out of the arXiv papers the search turned up.

A search hit is a pointer, not an answer. For a Feynman integral the answer sits
in one of three places in the submission, and this module fetches all three from
the source tarball arXiv serves at ``/e-print/<id>``:

  * **attached data files** — a modern paper ships its expressions as Mathematica
    input under ``anc/``, which is the only form you can actually compute with.
    Older papers put the same thing in top-level ``.txt`` files instead, so both
    are collected.
  * **the epsilon-expansion equations** in the LaTeX source — where a paper with
    no attachments puts its result. A result equation is recognised by what it
    is *written in* (polylogarithms, zeta values, powers of epsilon) rather than
    by any keyword in the prose around it.
  * **the sentence that says where the results are** ("in the ancillary files we
    provide analytic results for ...") plus the ancillary README, which between
    them describe the attachments better than any heuristic could.

The equations come out written in the paper's own macros, which no reader and
no renderer outside that paper can make sense of; ``readable`` hands them to the
model to be rewritten in standard LaTeX, and to be sorted into results and
not-results while it is there.

This is pattern matching over submitted source, so what comes out are
*candidates with provenance*, never a verified equality — every paper normalises
its integrals differently, and matching conventions to your own is a judgement
the extraction cannot make for you.
"""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import readable
from .arxiv_tool import Paper

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "arxiv"
SOURCE = "https://arxiv.org/e-print/{}"
ANC_URL = "https://arxiv.org/src/{}/anc/{}"
TIMEOUT = 60
UA = "feynman-agent/0.1 (Feynman integral lookup)"

MAX_PAPERS = 3  # each one is a tarball download; be a good arXiv citizen
MAX_EQUATIONS = 3
# Scoring finds equations that are *written like* results; only the model can
# tell a definition of a polylogarithm from a value of one, so it is given more
# to choose between than the report will show.
CANDIDATES = 2 * MAX_EQUATIONS
MAX_FILES = 8
# A weight-four result runs to pages of LaTeX, but the report is a rendered
# document rather than a terminal line, so most equations fit whole at this width.
EQ_CHARS = 1200
MIN_DATA_SIZE = 2000  # below this a stray .txt is a note, not a result

DATA_SUFFIXES = {".m", ".wl", ".nb", ".mx", ".txt", ".dat", ".json", ".csv", ".frm"}
# Attachments that are packaging, not physics — a build file is not a result.
SKIP_NAMES = {"cmakelists", "makefile", "license", "licence", "install", "manifest", "readme"}

# Papers whose abstract promises these are the ones worth downloading.
WANTED = (
    "analytic",
    "ancillary",
    "canonical",
    "closed form",
    "expansion",
    "explicit",
    "master integral",
    "polylogarithm",
)


@dataclass
class DataFile:
    name: str
    size: int
    url: str = ""
    direct: bool = True  # arXiv serves ``anc/`` files singly; the rest only in bulk

    def markdown(self) -> str:
        if self.direct:
            return f"[`{self.name}`]({self.url}) — {_size(self.size)}"
        return f"`{self.name}` — {_size(self.size)}, inside [the source tarball]({self.url})"


@dataclass
class Equation:
    latex: str
    file: str = ""
    section: str = ""
    label: str = ""
    markers: list[str] = field(default_factory=list)
    score: int = 0
    # The same equation in standard LaTeX, and one line on what it gives, both
    # from the model. Empty when no model ran: the report then prints the
    # source alone, as it always did.
    clean: str = ""
    gives: str = ""

    def where(self) -> str:
        bits = [b for b in (self.file, self.section) if b]
        if self.label:
            bits.append(f"eq. {self.label}")
        return " — ".join(bits) or "source"

    def block(self, width: int = EQ_CHARS) -> str:
        """The equation as the author wrote it, line breaks and all.

        Kept multi-line because it is printed inside a code fence: an ``align``
        environment is one row per order in epsilon, and flattening it to a
        single line is what made the old terminal report unreadable.
        """
        text = "\n".join(ln.strip() for ln in self.latex.strip().splitlines() if ln.strip())
        if len(text) <= width:
            return text
        return f"{text[:width]}\n… ({len(text)} chars in full)"


@dataclass
class Extraction:
    paper: Paper
    target: str = ""  # the diagram this paper was found for
    files: list[DataFile] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    pointers: list[str] = field(default_factory=list)
    readme: str = ""
    reading: str = ""  # what a model made of it, when one is configured
    notes: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def found(self) -> bool:
        return bool(self.files or self.equations)

    def markdown(self) -> list[str]:
        """This paper as one section of the report: the citation, then the goods."""
        out = [f"### [arXiv:{self.paper.arxiv_id}]({self.paper.url})", ""]
        byline = " — ".join(b for b in (self.paper.title, self.paper.authors) if b)
        if self.target:
            byline += f"{' — ' if byline else ''}result wanted for `{self.target}`"
        out += [byline, ""] if byline else []
        if self.error:
            out += [f"**{self.error}**", ""]

        if self.files:
            out += ["**Attached data files**", ""]
            out += [f"- {f.markdown()}" for f in self.files] + [""]
        readme = _readme_lines(self.readme)
        if readme:
            out += ["**From the ancillary README**", ""]
            out += [f"- {ln}" for ln in readme] + [""]
        if self.pointers:
            out += ["**What the paper says about its own results**", ""]
            out += [f'- "{p}"' for p in self.pointers[:2]] + [""]

        for e in self.equations:
            out += [f"**{e.where()}** — {', '.join(e.markers)}", ""]
            out += [f"*{e.gives}*", ""] if e.gives else []
            # The rewrite is what gets read; the source is what gets checked
            # against, so it stays underneath rather than being replaced.
            out += ["$$", e.clean, "$$", ""] if e.clean else []
            out += ["```latex", e.block(), "```", ""]
        if self.reading:
            out += ["**Read by the model**", ""]
            out += [f"> {ln}" if ln.strip() else ">" for ln in self.reading.splitlines()] + [""]
        out += [f"*{n}*" for n in self.notes] + ([""] if self.notes else [])
        return out


@dataclass
class Source:
    """The parts of a submission worth looking at."""

    tex: list[tuple[str, str]] = field(default_factory=list)  # (filename, text)
    files: list[DataFile] = field(default_factory=list)
    readme: str = ""


def _size(n: int) -> str:
    for unit in ("B", "kB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def _readme_lines(readme: str, limit: int = 3) -> list[str]:
    """The first few lines of an ancillary README — usually a file-by-file key."""
    out = []
    for line in readme.splitlines():
        line = line.strip()
        if len(line) > 8 and len(set(line)) > 3:  # skip banners of ***** and =====
            out.append(line[:150])
        if len(out) == limit:
            break
    return out


_ARXIV_ID = re.compile(
    r"(?:arXiv:)?((?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?)", re.I
)


def cached(arxiv_id: str) -> bool:
    return (CACHE / f"{arxiv_id.replace('/', '_')}.src").exists()


def cite(reference: str) -> str:
    """A catalogue reference as a Markdown link, when it names an arXiv id.

    Reports are read in a browser now, and a reference you have to copy into
    one is half a reference.
    """
    match = _ARXIV_ID.search(reference)
    return f"[{reference}](https://arxiv.org/abs/{match.group(1)})" if match else reference


def from_references(items, offline: bool = False) -> list[tuple[Paper, str]]:
    """Papers a catalogue named, as ``(reference, which diagram)`` pairs.

    This is better targeting than any search can manage: a Loopedia record is a
    statement that *this paper* evaluated *this graph*, where a query can only
    say "two-loop four-point". So there is no ranking here — the incoming order
    is kept, which is the catalogue's own, grouped by mass configuration and
    starting from the simplest. When masses were given, ``loopedia.lookup`` has
    already floated the matching configurations to the front, so the paper that
    evaluated *those* masses is the first one opened.

    A reference may be a bare string or a record carrying authors as well; the
    ones that name no arXiv id (journal-only citations) are skipped, since there
    is nothing to fetch for them.
    """
    out: dict[str, tuple[Paper, str]] = {}
    for ref, target in items:
        text = str(getattr(ref, "reference", ref))
        match = _ARXIV_ID.search(text)
        if not match:
            continue
        arxiv_id = match.group(1)
        if offline and not cached(arxiv_id):
            continue
        if arxiv_id in out:  # named for a second diagram: read once, say both
            paper, seen = out[arxiv_id]
            if target and target not in seen:
                out[arxiv_id] = (paper, f"{seen}, {target}")
            continue
        note = getattr(ref, "description", "")
        out[arxiv_id] = (
            Paper(
                title=note[:110],  # the id is printed anyway; a citation needs no title
                authors=getattr(ref, "authors", ""),
                published="",
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                summary=note,
            ),
            target,
        )
    return list(out.values())


def rank(paper: Paper) -> int:
    """How likely this paper is to contain an explicit result, from its abstract."""
    text = f"{paper.title} {paper.summary}".lower()
    return sum(w in text for w in WANTED)


def fetch(arxiv_id: str, offline: bool = False) -> bytes:
    """The submitted source, cached — arXiv asks not to be downloaded twice."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{arxiv_id.replace('/', '_')}.src"
    if cached.exists():
        return cached.read_bytes()
    if offline:
        raise RuntimeError("offline and no cached source")
    req = urllib.request.Request(SOURCE.format(arxiv_id), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
        raw = fh.read()
    cached.write_bytes(raw)
    return raw


def _is_readme(name: str) -> bool:
    return Path(name).name.lower().startswith("readme")


def _is_data(name: str, size: int) -> bool:
    """A file holding results rather than typesetting or explaining.

    ``anc/`` is the modern convention; before it existed papers simply dropped
    the tables next to the LaTeX, so a big top-level ``.txt`` counts too. Some
    submissions attach a whole software project, whose build files are not what
    anyone came for.
    """
    if Path(name).suffix.lower() not in DATA_SUFFIXES:
        return False
    if _is_readme(name) or Path(name).stem.lower() in SKIP_NAMES:
        return False
    return name.startswith("anc/") or (size >= MIN_DATA_SIZE and "/" not in name)


def unpack(raw: bytes, arxiv_id: str = "") -> Source:
    """Read the tarball in memory. Nothing is written to disk, so no extraction."""
    try:
        tar = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
    except tarfile.ReadError:  # a single gzipped .tex, which is the old style
        return Source(tex=[("source.tex", gzip.decompress(raw).decode("utf-8", "replace"))])

    src = Source()
    for m in tar.getmembers():
        if not m.isfile():
            continue
        name = m.name.removeprefix("./")
        if name.lower().endswith(".tex"):
            src.tex.append((name, tar.extractfile(m).read().decode("utf-8", "replace")))
        elif _is_data(name, m.size):
            direct = name.startswith("anc/")
            anc = name.removeprefix("anc/")
            url = ANC_URL.format(arxiv_id, anc) if direct else SOURCE.format(arxiv_id)
            src.files.append(DataFile(name=anc, size=m.size, url=url, direct=direct))
        if _is_readme(name) and not src.readme:
            src.readme = tar.extractfile(m).read().decode("utf-8", "replace")[:1200]
    src.files.sort(key=lambda f: -f.size)
    return src


ENVS = "equation|align|eqnarray|gather|multline|dmath"
_DISPLAY = re.compile(rf"\\begin\{{({ENVS})\*?\}}(.*?)\\end\{{\1\*?\}}", re.S)
# One level of nesting, so a title like "Results up to $\mathcal{O}(\eps^4)$" survives.
_SECTION = re.compile(r"\\(?:sub)*section\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_LABEL = re.compile(r"\\label\{([^}]*)\}")
_COMMENT = re.compile(r"(?<!\\)%.*")
# A period ends a sentence only when whitespace follows, so a filename like
# ``./LA`` or ``resultA.m`` does not cut the sentence in half.
_RUN = r"(?:[^.]|\.(?!\s|$))*"
_POINTER = re.compile(
    rf"{_RUN}\b(?:ancillary|attached to th|supplementary material|machine.readable)\b{_RUN}\.",
    re.I | re.S,
)
# Where a sentence really begins when there is no full stop in front of it.
_STRUCTURE = re.compile(r"\\(?:begin|end|(?:sub)*section\*?|paragraph|item)\b(?:\{[^}]*\})?")

# What an evaluated Feynman integral is written in. Two of these in one display
# equation is a far better result-detector than any keyword in the prose.
MARKERS = {
    "polylogs": r"\\(?:mathrm|text|operatorname)?\s*\{?\s*Li\}?_|\\Li_",
    "G/H functions": r"\bG\s*\(|\bH\s*\(|\\mathcal\{G\}",
    "zeta values": r"\\zeta",
    # \ep and \eps are the usual shorthands for the regulator in older sources.
    "eps expansion": r"\\(?:var)?(?:epsilon|eps|ep)\s*\^",
    "logs": r"\\ln|\\log",
    "powers of pi": r"\\pi\s*\^",
}


# \newcommand{\bea}{\begin{eqnarray}} and friends, including the \def spelling.
_SHORTHAND = re.compile(
    r"\\(?:new|renew)command\s*\*?\s*\{?\\(\w+)\}?\s*\{\s*(\\(?:begin|end)\s*\{\w+\*?\})\s*\}"
    r"|\\def\s*\\(\w+)\s*\{\s*(\\(?:begin|end)\s*\{\w+\*?\})\s*\}"
)


def _expand_shorthand(text: str) -> str:
    """Resolve aliases for the math environments before looking for equations.

    Older hep-ph sources almost universally write ``\\bea ... \\eea`` rather than
    ``\\begin{eqnarray} ... \\end{eqnarray}``. Left alone there is no
    ``\\begin{...}`` in the file to match, so a paper full of results looks like
    a paper with none — which is what happened to Smirnov's double-box papers,
    the canonical references for the graph being looked up.
    """
    pairs = {}
    for m in _SHORTHAND.finditer(text):
        name, body = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        pairs[name] = body
    if not pairs:
        return text
    # Drop the definitions first: rewriting \be inside its own \newcommand{\be}
    # would leave a stray unmatched \begin{equation}.
    text = _SHORTHAND.sub("", text)
    for name, body in pairs.items():
        # A function replacement, because a literal one would read the backslash
        # in \end{equation} as an escape sequence and raise.
        text = re.sub(rf"\\{name}(?![A-Za-z])", lambda _m, b=body: b, text)
    return text


def _strip(text: str) -> str:
    return _expand_shorthand(_COMMENT.sub("", text))


def _section_of(text: str, pos: int) -> str:
    heads = [m.group(1) for m in _SECTION.finditer(text, 0, pos)]
    if not heads:
        return ""
    title = re.sub(r"\\(?:mathcal|mathrm|text|boldsymbol|ensuremath)|[${}]", "", heads[-1])
    return re.sub(r"\s+", " ", title).strip()


def _diverse(eqs: list[Equation]) -> list[Equation]:
    """One equation per section before a second from any of them.

    A result block holds dozens of near-identical members (``g_30``, ``g_31``,
    ...); showing three of those says less about the paper than three drawn from
    different places.
    """
    seen, first, rest = set(), [], []
    for e in eqs:
        key = (e.file, e.section)
        (rest if key in seen else first).append(e)
        seen.add(key)
    return first + rest


def equations(src: Source) -> list[Equation]:
    """Display equations that look like an evaluated result, best first."""
    found = []
    for name, raw in src.tex:
        text = _strip(raw)
        for m in _DISPLAY.finditer(text):
            body = m.group(2).strip()
            if "=" not in body:
                continue
            hits = [k for k, p in MARKERS.items() if re.search(p, body)]
            # An equation still containing an integral sign is usually a
            # definition ("G(a;x) = int dt/(t-a) ..."), the opposite of an
            # evaluation, so it has to be that much more convincing otherwise.
            score = len(hits) - (r"\int" in body)
            if score < 2:
                continue
            label = _LABEL.search(body)
            found.append(
                Equation(
                    latex=body,
                    file=name,
                    section=_section_of(text, m.start()),
                    label=label.group(1) if label else "",
                    markers=hits,
                    score=score,
                )
            )
    found.sort(key=lambda e: -e.score)
    return _diverse(found)[:CANDIDATES]


_MACRO = re.compile(
    r"^[ \t]*\\(?:new|renew)command\s*\*?\s*\{?\\(\w+)\}?.*$" r"|^[ \t]*\\def\s*\\(\w+).*$",
    re.M,
)


def macros(src: Source, eqs: list[Equation]) -> dict[str, str]:
    r"""The paper's own definitions, for the macros these equations use.

    Whole lines rather than parsed bodies: a definition nests braces to any
    depth, and the model is a better TeX expander than a regex will ever be.

    Twice over, because a paper-private macro is normally written in terms of
    others — ``\dps`` is defined with ``\displaystyle``, ``\lek`` with
    ``\left[`` — and a definition that uses a macro nobody explains is no
    better than the macro it was meant to explain.
    """
    defined: dict[str, str] = {}
    for _, raw in src.tex:
        for m in _MACRO.finditer(_COMMENT.sub("", raw)):
            defined.setdefault(m.group(1) or m.group(2), m.group(0).strip()[:200])

    text = "\n".join(e.block() for e in eqs)
    used = {n for n in defined if re.search(rf"\\{n}(?![A-Za-z])", text)}
    for _ in range(2):
        bodies = "\n".join(defined[n] for n in used)
        used |= {n for n in defined if re.search(rf"\\{n}(?![A-Za-z])", bodies)}
    return {n: defined[n] for n in sorted(used)}


def pointers(src: Source) -> list[str]:
    """Sentences in which the authors say where they put the results."""
    out = []
    for _, raw in src.tex:
        for m in _POINTER.finditer(_strip(raw)):
            # A sentence has no full stop before it when an equation or a
            # heading precedes it, so the match starts inside that markup. Keep
            # only what follows the last structural marker.
            s = _STRUCTURE.split(m.group(0))[-1]
            s = re.sub(r"\\(?:cite|ref|label)\{[^}]*\}", "", s)
            # Drop command names but keep their argument: {\tt resultA.m} is a
            # filename the reader wants, "tt" is not.
            s = re.sub(r"\\[A-Za-z]+\s*", " ", s)
            s = re.sub(r"[{}$\\]|~", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            if 40 < len(s) < 400 and s not in out:
                out.append(s)
    return out


READING_PROMPT = """A Feynman-integral agent found this paper while looking for the result of a
diagram. Say what the paper actually gives, in at most six short plain-text lines:

GIVES: which integrals it evaluates, to what order in epsilon, in which functions
WHERE: the most direct way to get the expression — name the attached file, or the equation below
MATCH: whether this looks like the diagram asked about, and what must be checked to be sure

Quote nothing you were not shown here, and write "unclear" rather than guessing.

DIAGRAM ASKED ABOUT: {target}
TITLE: {title}
ABSTRACT: {abstract}
ATTACHED FILES: {files}
README: {readme}
WHAT THE PAPER SAYS ABOUT ITS RESULTS: {pointers}
EQUATIONS THAT LOOK LIKE RESULTS:
{equations}
"""


def _read_with_model(ex: Extraction) -> tuple[str, list[str]]:
    """Optional last step: have the configured model say what was found.

    The pattern matching finds the material; a model is what turns a page of
    LaTeX into "weight-four MPLs, in anc/NPL1_Solution_W4.m". Without a key the
    extraction still stands on its own, so this only ever adds.
    """
    from . import llm

    if not llm.available():
        return "", [f"no model reading: {llm.missing_reason()}"]
    prompt = READING_PROMPT.format(
        target=ex.target or "not specified",
        title=ex.paper.title,
        abstract=ex.paper.summary,
        files=", ".join(f.name for f in ex.files) or "none",
        readme=ex.readme[:800] or "none",
        pointers=" ".join(ex.pointers[:2]) or "nothing explicit",
        equations="\n\n".join(e.clean or e.block() for e in ex.equations) or "none found",
    )
    try:
        reply = llm.ask(prompt).strip()
    except Exception as exc:  # network, credit, quota — the extraction still stands
        return "", [f"model reading failed: {exc}"]
    if not reply:
        # Seen on a real run: a reasoning model bills its thinking against the
        # same budget and can return an empty message. Silence would look like
        # "this paper had nothing to say", which is a different claim entirely.
        return "", [
            f"{llm.describe()} returned an empty reply — a reasoning model can spend the "
            f"whole {llm.max_tokens()}-token budget thinking; raise FEYNMAN_AGENT_MAX_TOKENS"
        ]
    return reply, []


def extract(
    paper: Paper,
    target: str = "",
    offline: bool = False,
    summarise: bool = True,
    rewrite: bool = False,
):
    """Everything this paper has to offer about ``target``, with provenance.

    Two independent model steps, because they cost differently: ``summarise``
    is one call for the whole paper, ``rewrite`` is one per candidate equation
    and is what empties an API quota. Without the second the equations are
    printed as the authors wrote them, which is all the report ever showed
    before that step existed.
    """
    ex = Extraction(paper=paper, target=target)
    try:
        raw = fetch(paper.arxiv_id, offline=offline)
    except Exception as exc:  # withdrawn, network down, nothing cached
        ex.error = f"source not readable: {exc}"
        return ex
    if raw[:4] == b"%PDF":
        ex.error = "submitted as PDF only — there is no source to read"
        return ex

    src = unpack(raw, paper.arxiv_id)
    ex.files = src.files[:MAX_FILES]
    ex.readme = src.readme
    ex.equations = equations(src)
    ex.pointers = pointers(src)
    if rewrite and ex.found:
        clarify(ex, src)  # sorts the candidates, and keeps at most MAX_EQUATIONS
    else:
        ex.equations = ex.equations[:MAX_EQUATIONS]
    if summarise and ex.found:
        ex.reading, notes = _read_with_model(ex)
        ex.notes += notes
    return ex


def clarify(ex: Extraction, src: Source) -> None:
    """Sort the candidate equations into results, rewritten so they render.

    A call per equation, so it happens only when asked for. Everything about it
    is optional in the other sense too: without a key, or offline, or when
    the call fails, the candidates stay as they were and the report prints the
    authors' own LaTeX — which is what it printed before this step existed. The
    reason is said out loud rather than left as an unexplained difference
    between one paper's section and the next's.
    """
    if not ex.equations:
        return
    chosen, notes = readable.rewrite(ex.equations, macros(src, ex.equations), ex.target)
    ex.equations = chosen or ex.equations[:MAX_EQUATIONS]
    ex.notes += notes


def pick(results) -> list[tuple[Paper, str]]:
    """Every candidate a search turned up, most promising first.

    Ranking by abstract costs nothing — the abstract is already in hand — and it
    is what lets ``select`` spend a small download budget on the papers most
    likely to carry a result rather than on whichever came back first.
    """
    best: dict[str, tuple[int, Paper, str]] = {}
    for res in results:
        for p in res.papers:
            score = rank(p)
            old = best.get(p.arxiv_id)
            if old is None:
                best[p.arxiv_id] = (score, p, res.target)
            elif res.target and res.target not in old[2]:  # wanted for two diagrams
                best[p.arxiv_id] = (old[0], old[1], f"{old[2]}, {res.target}")
    ordered = sorted(best.values(), key=lambda t: -t[0])
    return [(p, target) for _, p, target in ordered]


def select(candidates, limit: int = MAX_PAPERS) -> list[tuple[Paper, str]]:
    """Which of the candidates to actually open, at most ``limit`` of them.

    A reduction can leave a dozen master integrals citing two dozen papers, and
    taking the first three in order spends the entire budget on whichever
    diagram happens to be listed first — the last masters are never read about
    at all. So every diagram gets a paper before any diagram gets a second one:
    one pass over the candidates taking each that covers a diagram nothing
    chosen yet covers, then the rest in the order given, which is by rank for a
    search and the catalogue's own for a lookup.

    A paper cited for two diagrams covers both, so four masters sharing two
    references leave budget over for something else.
    """
    picked: dict[str, tuple[Paper, str]] = {}
    covered: set[str] = set()
    for paper, target in candidates:
        if len(picked) >= limit:
            break
        diagrams = {d.strip() for d in target.split(",") if d.strip()}
        if diagrams - covered:
            picked[paper.arxiv_id] = (paper, target)
            covered |= diagrams
    for paper, target in candidates:
        if len(picked) >= limit:
            break
        picked.setdefault(paper.arxiv_id, (paper, target))
    return list(picked.values())
