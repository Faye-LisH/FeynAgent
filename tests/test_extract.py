"""Reading results out of an arXiv submission. Offline: the tarballs are built here.

The shapes tested are the ones real submissions actually take — an ``anc/``
directory (modern), results dropped next to the LaTeX (older), and a single
gzipped ``.tex`` with no tar at all (oldest).
"""

import gzip
import io
import tarfile

from feynman_agent import extract
from feynman_agent.arxiv_tool import ArxivResult, Paper


def _tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _paper(arxiv_id="2303.07395v2", summary="analytic results for master integrals"):
    return Paper(
        title="Two-loop master integrals",
        authors="A. Author",
        published="2023-03",
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        summary=summary,
    )


RESULT_TEX = rb"""
\section{Results}
\begin{align}
  g_{30} &= \frac{1}{4} + \epsilon^2 \left[ G(0,x) - \frac{1}{2}\log(-s) \right]
         + \zeta_3 \pi^2 \,.
\end{align}
\section{Definitions}
\begin{equation}
  G(a_1;x) = \int_0^x \frac{dt}{t-a_1} G(;t)\,, \qquad \log^n(x)\,.
\end{equation}
In ancillary files attached to this publication we provide analytic results
for all two-loop canonical master integrals in terms of MPLs.
% \begin{equation} this one is commented out \zeta_3 \pi^2 = \log \end{equation}
"""


def test_ancillary_files_are_found_with_a_direct_link():
    src = extract.unpack(
        _tarball(
            {
                "main.tex": RESULT_TEX,
                "anc/README.txt": b"Solution_W4.m: the analytic solution up to weight 4\n",
                "anc/Solution_W4.m": b"x" * 5000,
                "notes.txt": b"too small to be a result",
            }
        ),
        "2303.07395v2",
    )

    assert [f.name for f in src.files] == ["Solution_W4.m"]
    assert src.files[0].direct
    assert src.files[0].url == "https://arxiv.org/src/2303.07395v2/anc/Solution_W4.m"
    assert "weight 4" in src.readme


def test_results_beside_the_latex_count_too():
    """Papers predating anc/ simply dropped the tables next to the source."""
    src = extract.unpack(
        _tarball({"PInts.tex": RESULT_TEX, "TopoA_Mathematica.txt": b"y" * 90000}),
        "1306.6344v2",
    )

    assert [f.name for f in src.files] == ["TopoA_Mathematica.txt"]
    assert not src.files[0].direct  # arXiv only serves these inside the tarball
    assert src.files[0].url == "https://arxiv.org/e-print/1306.6344v2"


def test_single_gzipped_tex_submission():
    src = extract.unpack(gzip.compress(RESULT_TEX), "hep-ph/0101124v2")
    assert len(src.tex) == 1 and "g_{30}" in src.tex[0][1]


def test_an_evaluated_equation_is_kept_and_a_definition_is_not():
    """The filter is what the equation is written in, not the prose around it."""
    src = extract.unpack(_tarball({"main.tex": RESULT_TEX}), "x")
    eqs = extract.equations(src)

    assert len(eqs) == 1
    assert "g_{30}" in eqs[0].latex
    assert eqs[0].section == "Results"
    assert {"zeta values", "eps expansion"} <= set(eqs[0].markers)
    assert all(r"\int" not in e.latex for e in eqs), "a definition is not a result"


def test_commented_out_equations_are_ignored():
    src = extract.unpack(_tarball({"main.tex": RESULT_TEX}), "x")
    assert all("commented out" not in e.latex for e in extract.equations(src))


def test_the_sentence_saying_where_the_results_are():
    src = extract.unpack(_tarball({"main.tex": RESULT_TEX}), "x")
    (pointer,) = extract.pointers(src)
    assert pointer.startswith("In ancillary files attached to this publication")


def test_long_equations_are_truncated_but_say_so():
    eq = extract.Equation(latex="a = " + "b + " * 500)
    text = eq.block()
    assert len(text) < extract.EQ_CHARS + 40 and "chars in full" in text


def test_an_equation_keeps_the_line_breaks_the_author_wrote():
    """One line per order in epsilon is the whole readability of an align block."""
    eq = extract.Equation(latex="  f &= 1\n\n     + \\eps \\zeta_3\n")
    assert eq.block() == "f &= 1\n+ \\eps \\zeta_3"


def test_pdf_only_submission_is_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: b"%PDF-1.5 ...")
    ex = extract.extract(_paper())
    assert not ex.found and "PDF only" in ex.error


def test_offline_with_no_cache_is_a_note_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    ex = extract.extract(_paper(), offline=True)
    assert not ex.found and "offline" in ex.error


def test_a_cached_source_is_not_downloaded_again(monkeypatch, tmp_path):
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    (tmp_path / "hep-ph_0101124v2.src").write_bytes(gzip.compress(RESULT_TEX))
    # offline=True is the proof: any network access would raise instead.
    ex = extract.extract(_paper("hep-ph/0101124v2"), offline=True, summarise=False)
    assert ex.found and ex.equations


def test_extraction_reports_files_equations_and_provenance(monkeypatch):
    raw = _tarball(
        {
            "main.tex": RESULT_TEX,
            "anc/Solution_W4.m": b"x" * 5000,
            "anc/README.txt": b"Solution_W4.m: the analytic solution up to weight 4\n",
        }
    )
    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: raw)
    ex = extract.extract(_paper(), target="e11|e|", summarise=False)
    text = "\n".join(ex.markdown())

    assert ex.found
    assert "e11|e|" in text  # which diagram it was wanted for
    assert "Solution_W4.m" in text and "arxiv.org/src" in text
    assert "weight 4" in text  # the README key to the attachments
    assert "main.tex — Results" in text  # where in the paper the equation is
    assert "```latex" in text  # the equation is fenced, not run into the prose
    assert "g_{30}" in text


def test_model_reading_is_optional_and_its_absence_is_stated(monkeypatch):
    """No key must cost the extraction nothing but the summary line."""
    from feynman_agent import llm

    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: _tarball({"m.tex": RESULT_TEX}))
    monkeypatch.setattr(llm, "available", lambda: False)
    monkeypatch.setattr(llm, "missing_reason", lambda: "ANTHROPIC_API_KEY is not set")

    ex = extract.extract(_paper(), summarise=True)
    assert ex.found and ex.equations
    assert not ex.reading
    assert any("no model reading" in n for n in ex.notes)


def test_model_reading_is_shown_when_a_model_answers(monkeypatch):
    from feynman_agent import llm

    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: _tarball({"m.tex": RESULT_TEX}))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", lambda prompt: "GIVES: weight-four MPLs\nWHERE: eq. (3.1)")

    ex = extract.extract(_paper(), summarise=True)
    assert "weight-four MPLs" in ex.reading
    assert "> GIVES: weight-four MPLs" in "\n".join(ex.markdown())


def test_an_empty_model_reply_says_so_rather_than_reading_as_silence(monkeypatch):
    """Seen on a real run: a reasoning model can spend its whole budget thinking."""
    from feynman_agent import llm

    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: _tarball({"m.tex": RESULT_TEX}))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", lambda prompt: "   ")

    ex = extract.extract(_paper(), summarise=True)
    assert not ex.reading
    assert any("empty reply" in n and "FEYNMAN_AGENT_MAX_TOKENS" in n for n in ex.notes)


def test_a_failed_model_call_does_not_lose_the_extraction(monkeypatch):
    from feynman_agent import llm

    def boom(prompt):
        raise RuntimeError("credit balance is too low")

    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: _tarball({"m.tex": RESULT_TEX}))
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", boom)

    ex = extract.extract(_paper(), summarise=True)
    assert ex.equations, "the equations were found before the model was ever called"
    assert any("credit balance" in n for n in ex.notes)


def test_rewriting_the_equations_is_off_until_it_is_asked_for(monkeypatch):
    """The summary is one call for a paper; the rewrite is one per equation.

    That difference is the whole reason the two are separate switches, so the
    expensive one has to be asked for and the report otherwise prints the
    LaTeX the authors wrote.
    """
    from feynman_agent import llm

    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: _tarball({"m.tex": RESULT_TEX}))
    monkeypatch.setattr(llm, "available", lambda: True)
    seen = []

    def fake(prompt):
        seen.append(prompt)
        return "RESULT: yes\nGIVES: a bubble\nLATEX:\nI = 1"

    monkeypatch.setattr(llm, "ask", fake)

    ex = extract.extract(_paper(), summarise=True)  # what --extract auto asks for
    assert len(seen) == 1, "one call for the paper, none for its equations"
    assert ex.equations and not any(e.clean for e in ex.equations)
    assert "```latex" in "\n".join(ex.markdown())

    seen.clear()
    ex = extract.extract(_paper(), summarise=True, rewrite=True)
    assert len(seen) > 1 and all(e.clean for e in ex.equations)


ALIASED_TEX = rb"""
\newcommand{\bea}{\begin{eqnarray}}
\newcommand{\eea}{\end{eqnarray}}
\def\ep{\varepsilon}
\section{Master double boxes}
\bea
  K(x,\ep) &=& -\frac{4}{\ep^4} + \frac{5\ln x}{\ep^3} + 12 \zeta_3 + \pi^2 \,.
\eea
"""


def test_aliased_math_environments_are_expanded():
    """Older hep-ph sources write \\bea, never \\begin{eqnarray} — Smirnov's do.

    Unexpanded there is no \\begin{...} in the file at all, so the canonical
    double-box papers looked like papers with no results in them.
    """
    src = extract.unpack(_tarball({"p.tex": ALIASED_TEX}), "hep-ph/9905323")
    eqs = extract.equations(src)

    assert len(eqs) == 1
    assert r"K(x,\ep)" in eqs[0].latex
    assert eqs[0].section == "Master double boxes"
    assert "eps expansion" in eqs[0].markers, r"\ep^4 is an epsilon expansion"


def test_catalogue_references_become_papers_to_read():
    """A Loopedia record names an arXiv id — the extractor's native input."""
    from feynman_agent.loopedia import Record

    items = [
        (Record(reference="hep-ph/9905323", authors="V.A. Smirnov"), "e12|e3|34|5|e5|e|"),
        (Record(reference="arXiv:1306.6344", authors="T. Gehrmann"), "e12|e3|34|5|e5|e|"),
        ("Nucl.Phys. B580 (2000) 485", "e12|e3|34|5|e5|e|"),  # journal only: nothing to fetch
        ("arXiv:1306.6344", "e11|e|"),  # the same paper, cited for a second diagram
    ]
    picked = extract.from_references(items)

    assert [p.arxiv_id for p, _ in picked] == ["hep-ph/9905323", "1306.6344"]
    assert picked[0][0].authors == "V.A. Smirnov"
    assert picked[1][1] == "e12|e3|34|5|e5|e|, e11|e|"


def test_offline_reads_only_what_is_already_cached(monkeypatch, tmp_path):
    """Otherwise an offline run reports three download failures it cannot fix."""
    monkeypatch.setattr(extract, "CACHE", tmp_path)
    (tmp_path / "hep-ph_9905323.src").write_bytes(b"cached")
    items = [("hep-ph/9905323", "a"), ("arXiv:1306.6344", "b")]

    assert [p.arxiv_id for p, _ in extract.from_references(items, offline=True)] == [
        "hep-ph/9905323"
    ]


MACRO_TEX = rb"""
\newcommand{\dps}{\displaystyle}
\newcommand{\ESGamma}{\SGamma^3}
\def\SGamma{S_\Gamma}
\newcommand{\unused}{\mathcal{N}}
\begin{equation}
  \dps A = \ESGamma \, \zeta_3 \, \eps^2 \, \pi^2
\end{equation}
"""


def test_only_the_macros_an_equation_uses_travel_with_it():
    r"""The model can expand \ESGamma only if it is also told what \SGamma is.

    And only those: a preamble defines hundreds of macros, of which one
    equation uses a handful.
    """
    src = extract.unpack(_tarball({"m.tex": MACRO_TEX}), "x")
    (eq,) = extract.equations(src)

    found = extract.macros(src, [eq])
    assert set(found) == {"dps", "ESGamma", "SGamma"}
    assert found["SGamma"] == r"\def\SGamma{S_\Gamma}"


def _candidates():
    """One definition and one result, scored alike — telling them apart is the job."""
    return [
        extract.Equation(latex=r"G(a;x) = \int_0^x \frac{dt}{t-a}", file="m.tex"),
        extract.Equation(latex=r"\dps I &=& \lek \frac{1}{4\eps} \rek \nnb", file="m.tex"),
    ]


def test_the_model_picks_the_results_and_rewrites_them(monkeypatch):
    """What comes out of the source renders nowhere; what goes in the report must."""
    from feynman_agent import llm

    ex = extract.Extraction(paper=_paper(), target="e11|e|", equations=_candidates())
    src = extract.Source(tex=[("m.tex", "\\newcommand{\\lek}{\\left[}\n\\def\\eps{\\varepsilon}\n")])
    seen = []

    def fake(prompt):
        seen.append(prompt)
        if r"\int_0^x" in prompt:
            return "RESULT: no"
        return "RESULT: yes\nGIVES: a bubble\nLATEX:\n" + r"I = \left[ \frac{1}{4\varepsilon} \right]"

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", fake)
    extract.clarify(ex, src)

    (eq,) = ex.equations  # the definition was not a result
    assert eq.clean == r"I = \left[ \frac{1}{4\varepsilon} \right]" and eq.gives == "a bubble"
    assert len(seen) == 2, "one call per candidate, so a hard one costs only itself"
    assert r"\newcommand{\lek}" in seen[0], "the macros went with the equation"
    text = "\n".join(ex.markdown())
    assert "$$\nI = \\left[" in text and "```latex" in text, "rewrite above, source below"


def test_the_reply_is_unwrapped_and_made_renderable(monkeypatch):
    r"""Two things the model does that no browser renderer forgives.

    A fenced reply is still a reply — the fence is not part of the equation —
    and `\mbox` is plain TeX that KaTeX does not implement, which the papers
    are full of: Smirnov writes his polylogarithms with it.
    """
    from feynman_agent import llm

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(
        llm,
        "ask",
        lambda prompt: "RESULT: yes\nGIVES: a bubble\nLATEX:\n```latex\n"
        + r"K = 2{\mbox{Li}}_3(-y)" + "\n```",
    )
    ex = extract.Extraction(paper=_paper(), equations=_candidates()[:1])

    extract.clarify(ex, extract.Source())

    assert ex.equations[0].clean == r"K = 2{\text{Li}}_3(-y)"


def test_one_equation_failing_does_not_cost_the_others(monkeypatch):
    """The reason there is a call per equation rather than one for all of them."""
    from feynman_agent import llm

    def fake(prompt):
        if r"\int_0^x" in prompt:
            raise RuntimeError("connection reset")
        return "RESULT: yes\nGIVES: a bubble\nLATEX:\nI = 1"

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", fake)
    ex = extract.Extraction(paper=_paper(), equations=_candidates())

    extract.clarify(ex, extract.Source())

    assert [e.clean for e in ex.equations] == ["I = 1"]
    assert any("1 of 2 candidate equation(s) could not be read" in n for n in ex.notes)


def test_without_a_model_the_authors_own_latex_is_what_the_report_shows(monkeypatch):
    """The step is an improvement on the report, never a condition for having one."""
    from feynman_agent import llm

    monkeypatch.setattr(llm, "available", lambda: False)
    monkeypatch.setattr(llm, "missing_reason", lambda: "ANTHROPIC_API_KEY is not set")
    ex = extract.Extraction(paper=_paper(), equations=_candidates() * 4)

    extract.clarify(ex, extract.Source())

    assert len(ex.equations) == extract.MAX_EQUATIONS
    assert not any(e.clean for e in ex.equations)
    assert any("as the authors wrote them" in n for n in ex.notes)


def test_a_reply_in_the_wrong_shape_leaves_the_equations_alone(monkeypatch):
    from feynman_agent import llm

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "ask", lambda prompt: "I could not read that equation.")
    ex = extract.Extraction(paper=_paper(), equations=_candidates())

    extract.clarify(ex, extract.Source())

    assert len(ex.equations) == 2 and not any(e.clean for e in ex.equations)
    assert any("unreadable reply" in n for n in ex.notes)


def test_papers_are_ranked_before_anything_is_downloaded():
    """The download cap is spent on the abstracts that promise a result."""
    vague = _paper("1111.1111", summary="we discuss some aspects of loop calculations")
    good = _paper("2222.2222", summary="explicit analytic results in ancillary files")
    picked = extract.select(extract.pick([ArxivResult(papers=[vague, good], target="e11|e|")]), 1)

    assert [p.arxiv_id for p, _ in picked] == ["2222.2222"]


def test_a_paper_wanted_for_two_diagrams_is_read_once():
    p = _paper("3333.3333")
    picked = extract.pick(
        [ArxivResult(papers=[p], target="e11|e|"), ArxivResult(papers=[p], target="e12|e2|e|")]
    )

    assert len(picked) == 1
    assert picked[0][1] == "e11|e|, e12|e2|e|"  # both, so the report says so


def test_the_download_cap_is_respected():
    papers = [_paper(f"90{i:02d}.0001") for i in range(10)]
    candidates = extract.pick([ArxivResult(papers=papers, target="x")])
    assert len(candidates) == 10  # ranking does not throw anything away
    assert len(extract.select(candidates)) == extract.MAX_PAPERS


def test_every_diagram_is_read_about_before_any_is_read_about_twice():
    """Eight masters and a three-paper budget: order alone would spend it all
    on whichever diagram happens to be listed first."""
    candidates = [
        (_paper("1111.1111"), "e11|e|"),
        (_paper("2222.2222"), "e11|e|"),
        (_paper("3333.3333"), "e11|e|"),
        (_paper("4444.4444"), "e12|e2|e|"),
        (_paper("5555.5555"), "e12|e3|e3|e|"),
    ]
    chosen = extract.select(candidates, 3)

    assert [p.arxiv_id for p, _ in chosen] == ["1111.1111", "4444.4444", "5555.5555"]


def test_a_paper_cited_for_two_diagrams_covers_both():
    """Which is what leaves budget over when masters share their references."""
    candidates = [
        (_paper("1111.1111"), "e11|e|, e12|e2|e|"),
        (_paper("2222.2222"), "e12|e2|e|"),
        (_paper("3333.3333"), "e12|e3|e3|e|"),
    ]
    chosen = extract.select(candidates, 2)

    assert [p.arxiv_id for p, _ in chosen] == ["1111.1111", "3333.3333"]


def test_more_diagrams_than_papers_allowed_still_spends_them_in_order():
    candidates = [(_paper(f"{i}{i}{i}{i}.1111"), f"nickel-{i}") for i in range(1, 6)]
    chosen = extract.select(candidates, 2)

    assert [target for _, target in chosen] == ["nickel-1", "nickel-2"]
