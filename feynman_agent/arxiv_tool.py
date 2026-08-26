"""arXiv search, used only as the last resort and only with the user's consent."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    title: str
    authors: str
    published: str
    arxiv_id: str
    url: str
    summary: str = ""

    def markdown(self) -> str:
        """The citation as a link, skipping what a catalogue reference lacks."""
        bits = [f"[arXiv:{self.arxiv_id}]({self.url})"]
        if self.published:
            bits.append(f"({self.published})")
        if self.title:
            bits.append(self.title)
        return " ".join(bits) + (f" — {self.authors}" if self.authors else "")


@dataclass
class ArxivResult:
    query: str = ""
    papers: list[Paper] = field(default_factory=list)
    error: str = ""
    target: str = ""  # which diagram this search was about

    @property
    def found(self) -> bool:
        return bool(self.papers)


def build_query_for(n_loops: int, n_legs: int, extra: str = "") -> str:
    """Phrase a query from a diagram's vital statistics."""
    words = {1: "one-loop", 2: "two-loop", 3: "three-loop", 4: "four-loop"}
    loops = words.get(n_loops, f"{n_loops}-loop")
    points = {2: "two-point", 3: "three-point", 4: "four-point", 5: "five-point"}
    legs = points.get(n_legs, f"{n_legs}-point")
    base = f'abs:"{loops}" AND abs:"{legs}" AND abs:"master integrals"'
    return f"{base} AND abs:{extra}" if extra else base


def build_query(topo, extra: str = "") -> str:
    return build_query_for(topo.n_loops, topo.n_legs, extra)


def search(query: str, max_results: int = 8) -> ArxivResult:
    try:
        import arxiv
    except ImportError:
        return ArxivResult(query=query, error="the 'arxiv' package is not installed")
    try:
        client = arxiv.Client(page_size=max_results, delay_seconds=3, num_retries=2)
        search_obj = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = [
            Paper(
                title=r.title.strip().replace("\n", " "),
                authors=", ".join(a.name for a in r.authors[:4]),
                published=r.published.strftime("%Y-%m"),
                arxiv_id=r.get_short_id(),
                url=r.entry_id,
                summary=r.summary.strip().replace("\n", " ")[:400],
            )
            for r in client.results(search_obj)
        ]
        return ArxivResult(query=query, papers=papers)
    except Exception as exc:
        return ArxivResult(query=query, error=f"arXiv search failed: {exc}")
