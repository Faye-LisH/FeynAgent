"""Workflow tests. These run offline: NeatIBP is mocked and Loopedia is cached."""

import os

import pytest

from feynman_agent.graph import build


def run(text, **over):
    graph = build()
    config = {"configurable": {"thread_id": f"test-{text}"}}
    state = {
        "raw_input": text,
        "neatibp_mode": "mock",
        "offline": True,
        # These are routing tests; reading papers has its own file, and doing it
        # here would make every catalogue hit unpack three cached tarballs.
        "extract_mode": "off",
        "trace": [],
        "problems": [],
        **over,
    }
    return graph.invoke(state, config)


def steps(result):
    return {e["step"]: e["status"] for e in result["trace"]}


def test_local_db_short_circuits():
    """A stored result ends the run without touching the network."""
    out = run("e11|e|")
    assert steps(out)["local database"] == "hit"
    assert "Gamma(nu1+nu2-d/2)" in out["answer"]
    assert "loopedia" not in steps(out)


def test_report_always_explains_what_was_done():
    out = run("e11|e|")
    assert "## What was done" in out["report"]
    assert "## Identified" in out["report"]


def test_the_report_is_markdown_and_fences_a_stored_formula():
    """The result is an expression: unfenced, Markdown would eat its stars."""
    out = run("e11|e|")
    body = out["report"].split("## Result")[1]
    assert body.startswith("\n\n*From the local database.*\n\n```\nG(nu1,nu2) = ")


def test_unparseable_input_reports_the_difficulty():
    out = run("not a diagram at all")
    assert steps(out)["identify"] == "failed"
    assert out["problems"]
    assert "## No direct result" in out["report"]


def test_integrand_input_is_identified():
    """The NeatIBP kinematics block for the double box."""
    block = """
    LoopMomenta={l1,l2};
    ExternalMomenta={k1,k2,k4};
    Propagators=#^2&/@{l1,l1+k1,l1+k1+k2,l2-k1-k2,l2+k4,l2,l1+l2,l1+k4,l2+k1}
    """
    out = run(block)
    assert out["topo"].nickel() == "e12|e3|34|5|e5|e|"
    assert steps(out)["identify"] == "ok"


def test_mock_neatibp_replays_the_recorded_double_box(tmp_path):
    """The double box short-circuits at Loopedia, so exercise the mock directly."""
    from feynman_agent import neatibp
    from feynman_agent import topology as T

    if not neatibp.recorded().joinpath("summary.txt").exists():
        pytest.skip("no recorded NeatIBP run in this checkout")
    topo = T.assign_momenta(T.from_nickel("e12|e3|34|5|e5|e|"))
    res = neatibp.run(topo, tmp_path, mode="mock")
    assert res.ok and res.mocked
    assert len(res.masters) == 8


def test_mock_refuses_to_invent_masters(tmp_path):
    from feynman_agent import neatibp
    from feynman_agent import topology as T

    res = neatibp.run(T.from_nickel("e111|e|"), tmp_path, mode="mock")
    assert not res.ok and not res.masters
    assert "unknown" in res.error


def test_a_finished_output_directory_is_cleared_before_a_new_run(tmp_path):
    """Asking for the same integral twice used to stop the agent dead.

    NeatIBP finds the first run's output in the way and asks whether it may
    delete it — with ``InputString``, on a stdin nothing is listening to, and a
    stdout that is captured, so the question is invisible and the run hangs
    until the half-hour timeout.
    """
    from feynman_agent import neatibp

    out = tmp_path / "outputs" / "nickel_e11_e"
    (out / neatibp.FINISHED_TAG).parent.mkdir(parents=True)
    (out / neatibp.FINISHED_TAG).write_text("")

    assert neatibp.clear_output(tmp_path / "nothing here") == (True, "")
    ok, note = neatibp.clear_output(out)
    assert ok and not out.exists() and "cleared" in note


def test_an_unfinished_output_directory_is_left_alone(tmp_path):
    """No finished tag means it may be a mission still running — NeatIBP's own warning."""
    from feynman_agent import neatibp

    out = tmp_path / "outputs" / "nickel_e11_e"
    (out / "tmp").mkdir(parents=True)
    ok, note = neatibp.clear_output(out)
    assert not ok and out.exists() and "still going" in note


def test_a_timed_out_run_does_not_leave_kernels_behind(tmp_path, monkeypatch):
    """``run.sh`` starts a chain of Mathematica kernels; killing the shell alone
    leaves them running, which is how a kernel outlived its agent by half an
    hour on this machine."""
    import os
    import time

    from feynman_agent import neatibp
    from feynman_agent import topology as T

    fake = tmp_path / "neatibp"
    fake.mkdir()
    # A stand-in for run.sh: a child that outlives it unless the group is killed.
    (fake / "run.sh").write_text(f"sleep 60 & echo $! > {tmp_path}/child.pid\nwait\n")
    monkeypatch.setenv("FEYNMAN_AGENT_NEATIBP", str(fake))

    res = neatibp.run(T.from_nickel("e11|e|"), tmp_path / "wd", mode="real", timeout=1)
    assert res.mocked and "timeout" in res.error

    child = int((tmp_path / "child.pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def _apply(state: dict, update: dict) -> dict:
    """Fold a node's output into the state the way LangGraph's reducers do."""
    from feynman_agent.graph import _merge_findings

    for key, value in update.items():
        if key == "findings":
            state["findings"] = _merge_findings(state.get("findings", {}), value)
        elif key in {"trace", "problems"}:
            state[key] = state.get(key, []) + value
        else:
            state[key] = value
    return state


def _turn(state: dict) -> dict:
    """One turn of the lookup loop, in the order the graph runs the nodes."""
    from feynman_agent.graph import advance, local_db, loopedia_node

    for node in (local_db, loopedia_node, advance):
        state = _apply(state, node(state))
    return state


def test_pinched_bubble_resolves_via_merged_legs():
    """ee11|ee| is not catalogued, but it is the bubble once its legs are merged.

    Driven through the loop nodes the way the graph drives them: look up, fail
    to resolve, retry merged, resolve.
    """
    from feynman_agent import topology as T

    state = _turn(
        {
            "topo": T.from_nickel("e11|e|"),
            "offline": True,
            "queue": [{"topo": T.from_nickel("ee11|ee|"), "origin": "ee11|ee|", "merged": False}],
            "findings": {},
        }
    )
    # First turn: the pinched graph itself is not catalogued, so advance retries.
    assert state["queue"][0]["merged"] and state["queue"][0]["topo"].nickel() == "e11|e|"

    # Second turn: the merged graph is the massless bubble, which is known.
    state = _turn(state)
    assert state["queue"] == []
    assert state["findings"]["ee11|ee|"]["found"]
    assert state["findings"]["ee11|ee|"]["via"].startswith("e11|e|")


def test_figure_reply_is_parsed_and_canonicalised(monkeypatch, tmp_path):
    """The figure path with the model call stubbed out.

    Covers everything the agent owns — JSON extraction, Topology construction,
    canonicalisation — leaving only the model's own reading of the picture.
    """
    from feynman_agent import llm

    reply = """Looking at the diagram, I see two squares sharing a rung.
    {"internal_edges": [[0,1],[0,2],[1,3],[1,5],[2,3],[3,4],[4,5]],
     "external_legs": [0,2,4,5], "comment": "double box"}"""

    class FakeReply:
        content = reply

    class FakeChat:
        def invoke(self, messages):
            return FakeReply()

    monkeypatch.setattr(llm, "_chat", lambda prov: FakeChat())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("FEYNMAN_AGENT_PROVIDER", raising=False)
    png = tmp_path / "d.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")  # never decoded, only base64-encoded

    topo = llm.read_figure(png)
    assert topo.nickel() == "e12|e3|34|5|e5|e|"
    assert topo.n_loops == 2 and topo.n_legs == 4


def test_image_block_matches_each_provider_family():
    """Anthropic takes a base64 source block; OpenAI-compatible takes a data URL."""
    from feynman_agent import llm

    anthropic = llm._image_block(llm.PROVIDERS["anthropic"], "image/png", "QUJD")
    assert anthropic["type"] == "image"
    assert anthropic["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "QUJD",
    }

    kimi = llm._image_block(llm.PROVIDERS["kimi"], "image/png", "QUJD")
    assert kimi["type"] == "image_url"
    assert kimi["image_url"]["url"] == "data:image/png;base64,QUJD"


def test_kimi_provider_selection(monkeypatch):
    from feynman_agent import llm

    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "kimi")
    monkeypatch.delenv("FEYNMAN_AGENT_MODEL", raising=False)
    monkeypatch.delenv("FEYNMAN_AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")

    prov = llm.provider()
    assert prov.openai_style
    assert llm.base_url(prov) == "https://api.moonshot.ai/v1"
    assert llm.model_name(prov) == "kimi-k3"
    assert llm.api_key(prov) == "sk-test"
    assert "kimi" in llm.describe() and "moonshot" in llm.describe()


def test_deepseek_is_reached_as_an_openai_endpoint(monkeypatch):
    """Everything but the figure step is text, so a text-only provider is fine."""
    from feynman_agent import llm

    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "deepseek")
    monkeypatch.delenv("FEYNMAN_AGENT_MODEL", raising=False)
    monkeypatch.delenv("FEYNMAN_AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    prov = llm.provider()
    assert prov.openai_style and llm.available()
    assert llm.base_url(prov) == "https://api.deepseek.com/v1"
    assert llm.model_name(prov) == "deepseek-chat"


def test_a_provider_that_cannot_see_says_so_before_the_call(monkeypatch, tmp_path):
    """Otherwise the failure is an API error about a content block, at the far end."""
    from feynman_agent import llm
    from feynman_agent.graph import identify

    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    figure = tmp_path / "diagram.png"
    figure.write_bytes(b"\x89PNG")

    out = identify({"raw_input": str(figure)})

    assert out["trace"][0]["status"] == "failed"
    assert "reads images" in " ".join(out["problems"])
    assert "Nickel index" in " ".join(out["problems"]), "and what to do instead"


def test_token_budget_is_generous_and_overridable(monkeypatch):
    """A reasoning model bills its thinking against max_tokens; 2000 was too few."""
    from feynman_agent import llm

    monkeypatch.delenv("FEYNMAN_AGENT_MAX_TOKENS", raising=False)
    assert llm.max_tokens() >= 8000
    monkeypatch.setenv("FEYNMAN_AGENT_MAX_TOKENS", "1234")
    assert llm.max_tokens() == 1234


def test_truncated_reply_says_so(monkeypatch, tmp_path):
    """Running out of tokens must not be reported as 'no JSON in the reply'."""
    from langchain_core.messages import AIMessage

    from feynman_agent import llm

    figure = tmp_path / "d.png"
    figure.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "kimi")
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    monkeypatch.setattr(
        llm,
        "_chat",
        lambda prov: _Stub(AIMessage(content="", response_metadata={"finish_reason": "length"})),
    )

    with pytest.raises(ValueError, match="token limit"):
        llm.read_figure(figure)


class _Stub:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, _messages):
        return self.reply


def test_model_and_base_url_overrides(monkeypatch):
    from feynman_agent import llm

    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "openai")
    monkeypatch.setenv("FEYNMAN_AGENT_MODEL", "some-vision-model")
    monkeypatch.setenv("FEYNMAN_AGENT_BASE_URL", "https://example.invalid/v1")
    prov = llm.provider()
    assert llm.model_name(prov) == "some-vision-model"
    assert llm.base_url(prov) == "https://example.invalid/v1"


def test_unknown_provider_is_rejected(monkeypatch):
    from feynman_agent import llm

    monkeypatch.setenv("FEYNMAN_AGENT_PROVIDER", "nope")
    with pytest.raises(ValueError, match="unknown FEYNMAN_AGENT_PROVIDER"):
        llm.provider()


def test_unparsable_proxy_is_dropped_for_the_call(monkeypatch):
    """A bare socks:// ALL_PROXY makes httpx raise before any request goes out."""
    from feynman_agent import llm

    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:10808")
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10808")
    with llm.usable_proxies() as dropped:
        assert dropped == ["ALL_PROXY"]
        assert "ALL_PROXY" not in os.environ
        assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:10808"
    assert os.environ["ALL_PROXY"] == "socks://127.0.0.1:10808"  # restored


def test_ordinary_proxy_is_left_alone(monkeypatch):
    """Both spellings are checked, so a test must control both."""
    from feynman_agent import llm

    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("all_proxy", "http://127.0.0.1:10808")
    with llm.usable_proxies() as dropped:
        assert not dropped
        assert os.environ["ALL_PROXY"] == "http://127.0.0.1:10808"


def test_neatibp_output_name_is_filesystem_safe():
    """A Nickel index carries '|' and ':', which would break NeatIBP's output dir."""
    from feynman_agent import neatibp
    from feynman_agent import topology as T

    topo = T.from_nickel("e12|e3|e3|e|")  # name becomes "nickel:e12|e3|e3|e|"
    name = neatibp.output_name(topo)
    assert "|" not in name and ":" not in name and name


def test_the_lookup_loop_is_actually_a_cycle():
    """local_db -> loopedia -> advance -> local_db must close, or masters are
    never re-checked against the catalogues."""
    from feynman_agent import build

    edges = {(e.source, e.target) for e in build().get_graph().edges}
    assert ("local_db", "loopedia") in edges
    assert ("loopedia", "advance") in edges
    assert ("advance", "local_db") in edges  # the back edge that makes it a loop
    assert ("expand_masters", "local_db") in edges  # reduction feeds the loop
    assert ("advance", "assemble") in edges  # and it terminates


def test_recursion_limit_clears_a_realistic_reduction():
    """Eight masters at up to six supersteps each overruns LangGraph's default."""
    from feynman_agent.graph import RECURSION_LIMIT

    assert RECURSION_LIMIT > 8 * 6 + 10


def _recorded_double_box():
    from feynman_agent import neatibp
    from feynman_agent import topology as T

    if not neatibp.recorded().joinpath("summary.txt").exists():
        pytest.skip("no recorded NeatIBP run in this checkout")
    topo = T.assign_momenta(T.from_nickel("e12|e3|34|5|e5|e|"))
    return topo, neatibp.parse_summary(neatibp.recorded().joinpath("summary.txt").read_text())


def test_masters_map_back_to_subtopologies():
    """Each master G[...] pinches down to a graph the agent can look up."""
    from feynman_agent.graph import expand_masters

    topo, ibp = _recorded_double_box()
    out = expand_masters({"topo": topo, "ibp": ibp})

    assert len(out["masters"]) == len(ibp.masters)
    # The corner integral must reproduce the parent topology.
    corner = next(m for m in out["masters"] if m["integral"] == "G[1, 1, 1, 1, 1, 1, 1, 0, 0]")
    assert corner["nickel"] == "e12|e3|34|5|e5|e|"
    # Every master is a genuine sub-topology: no more propagators than the parent.
    assert all(m["props"] <= len(topo.int_edges) for m in out["masters"])


def test_every_master_is_queued_once_for_the_lookup_loop():
    """The queue is what sends the reduced diagrams back through local_db."""
    from feynman_agent.graph import expand_masters

    topo, ibp = _recorded_double_box()
    out = expand_masters({"topo": topo, "ibp": ibp})

    origins = [item["origin"] for item in out["queue"]]
    assert origins == sorted(set(origins), key=origins.index)  # deduplicated
    assert set(origins) == {m["nickel"] for m in out["masters"]}
    assert all(item["merged"] is False for item in out["queue"])


def test_loop_resolves_reduced_diagrams_against_the_catalogues():
    """The point of the loop: masters get the same lookup the original got.

    Every queued diagram must come out with a verdict, and the ones that are
    catalogued must come out resolved — that is what turns a reduction into
    progress rather than a longer list of unknowns.
    """
    from feynman_agent.graph import assemble, expand_masters

    topo, ibp = _recorded_double_box()
    state = _apply({"topo": topo, "ibp": ibp, "offline": True, "findings": {}}, {})
    state = _apply(state, expand_masters(state))
    queued = {item["origin"] for item in state["queue"]}

    while state["queue"]:
        state = _turn(state)

    assert set(state["findings"]) == queued, "the loop must judge every queued diagram"
    state = _apply(state, assemble(state))
    resolved = [m for m in state["masters"] if m.get("found")]
    assert resolved, "the loop resolved nothing"
    # The parent double box is one of its own masters, and Loopedia knows it.
    assert any(m["nickel"] == "e12|e3|34|5|e5|e|" for m in resolved)


def test_a_master_with_a_stored_closed_form_is_carried_through():
    """A master that local_db knows outright must surface its result."""
    from feynman_agent import topology as T
    from feynman_agent.graph import assemble, closed_form

    state = _turn(
        {
            "topo": T.from_nickel("e12|e3|e3|e|"),
            "offline": True,
            "masters": [
                {"integral": "G[0,1,1,0]", "nickel": "e11|e|", "loops": 1, "legs": 2},
            ],
            "queue": [{"topo": T.from_nickel("e11|e|"), "origin": "e11|e|", "merged": False}],
            "findings": {},
        }
    )
    state = _apply(state, assemble(state))
    bubble = state["masters"][0]
    assert bubble["found"] and closed_form(bubble), "the stored closed form was dropped"
    assert "resolves this integral completely" in state["answer"]


def test_arxiv_targets_the_unresolved_masters_not_the_parent(monkeypatch):
    """Step (f) is about the diagrams the reduction could not explain."""
    from feynman_agent import arxiv_tool
    from feynman_agent import topology as T
    from feynman_agent.graph import arxiv_node

    asked = []

    def fake_search(query, max_results=8):
        asked.append(query)
        return arxiv_tool.ArxivResult(query=query, papers=[])

    monkeypatch.setattr(arxiv_tool, "search", fake_search)
    out = arxiv_node(
        {
            "topo": T.from_nickel("e12|e3|34|5|e5|e|"),  # 2-loop 4-point parent
            "masters": [
                {"integral": "G[1,1,0]", "nickel": "e11|e|", "loops": 1, "legs": 2, "found": False},
                {"integral": "G[1,1,1]", "nickel": "e11|e|", "loops": 1, "legs": 2, "found": False},
                {"integral": "G[1,1,2]", "nickel": "ee1|ee|", "loops": 1, "legs": 2, "found": False},
                {"integral": "G[1,1,1,1]", "nickel": "x", "loops": 2, "legs": 4, "found": True},
            ],
        }
    )

    # One query: the three unresolved masters are all 1-loop 2-point, and the
    # query has no way to tell them apart, so repeating it would be waste.
    assert len(asked) == 1
    assert "one-loop" in asked[0] and "two-point" in asked[0]
    assert "two-loop" not in asked[0]  # not the parent, and not the resolved master
    assert out["arxiv"][0].target == "e11|e|, ee1|ee|"  # both diagrams it stands for


def test_a_run_narrates_itself_to_stderr_and_writes_the_report_to_a_file(capsys, tmp_path):
    """A run takes minutes, so stages are streamed as they finish, not collected.

    The split matters: the narration is for watching, the file is for reading,
    and stdout carries only the path so a shell can open it.
    """
    from feynman_agent.cli import main

    md = tmp_path / "db.md"
    main(["e11|e|", "--offline", "--reduce", "--neatibp", "mock", "--arxiv", "no", "-o", str(md)])
    out = capsys.readouterr()

    assert "✓ identify" in out.err
    assert "✓ local database" in out.err
    assert "… loopedia" in out.err, "a slow stage must speak before it finishes, not after"
    assert out.out.strip() == str(md)
    assert "## Identified" in md.read_text()


def test_the_report_file_is_named_after_the_integral(capsys, tmp_path, monkeypatch):
    """Runs accumulate rather than clobber, and re-running one refreshes it."""
    from feynman_agent.cli import main

    monkeypatch.chdir(tmp_path)
    main(["e11|e|", "--offline", "--quiet"])

    written = tmp_path / "reports" / "e11-e.md"
    assert written.exists()
    assert capsys.readouterr().out.strip() == "reports/e11-e.md"


def test_quiet_keeps_the_report_and_drops_the_narration(capsys):
    """``--out -`` is the escape hatch back to a pipe."""
    from feynman_agent.cli import main

    main(["e11|e|", "--offline", "--quiet", "--out", "-"])
    out = capsys.readouterr()

    assert out.err == ""
    assert "## Identified" in out.out and "## Result" in out.out


def test_reading_the_papers_is_wired_between_the_search_and_the_report():
    """A list of titles is not a result, so the search cannot be the last step."""
    from feynman_agent import build

    edges = {(e.source, e.target) for e in build().get_graph().edges}
    assert ("arxiv", "read_papers") in edges
    assert ("read_papers", "report") in edges
    assert ("arxiv", "report") in edges  # nothing to read is still a valid end


def test_a_catalogue_hit_goes_on_to_read_what_it_cited():
    """Loopedia answers with arXiv ids, and a reference is not a result.

    This is the common path, not the arXiv one: most graphs that are known at
    all are known *to Loopedia*, and stopping there hands back a bibliography.
    """
    from feynman_agent import loopedia
    from feynman_agent import topology as T
    from feynman_agent.graph import _after_assemble, _after_loopedia

    hit = loopedia.LookupResult(
        records=[loopedia.Record(reference="hep-ph/9905323", authors="V.A. Smirnov")],
        found=True,
    )
    state = {"topo": T.from_nickel("e12|e3|34|5|e5|e|"), "loopedia_hit": hit, "answer": "known"}

    assert _after_loopedia(state) == "read_papers"
    assert _after_loopedia({**state, "extract_mode": "off"}) == "report"
    # A record with no arXiv id is nothing to fetch, so the run just ends.
    journal = loopedia.LookupResult(records=[loopedia.Record(reference="Nucl.Phys. B580")])
    assert _after_loopedia({**state, "loopedia_hit": journal}) == "report"

    # Same for masters resolved by reference rather than by a stored closed form.
    resolved = {
        "topo": state["topo"],
        "masters": [{"nickel": "e11|e|", "references": ["arXiv:1306.6344"]}],
    }
    assert _after_assemble(resolved) == "read_papers"


def test_the_paper_budget_is_shared_out_and_says_what_it_left(monkeypatch):
    """Eight references across four masters and room for three papers.

    Read in the order given, all three would go to the first master and the
    other three diagrams would go unread — silently, since nothing said how many
    candidates there were.
    """
    import io
    import tarfile

    from feynman_agent import extract
    from feynman_agent.graph import extract_node

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        body = rb"\begin{align} f &= 1 - \eps^2 \zeta_3 \end{align}"
        info = tarfile.TarInfo("p.tex")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: buf.getvalue())

    masters = [
        {"nickel": f"nickel-{i}", "references": [f"arXiv:130{i}.1111", f"arXiv:140{i}.2222"]}
        for i in range(4)
    ]
    out = extract_node({"masters": masters, "extract_mode": "fetch"})

    assert [e.target for e in out["extracted"]] == ["nickel-0", "nickel-1", "nickel-2"]
    assert "5 more not opened, 3-paper cap" in out["trace"][0]["detail"]

    # --max-papers raises it, and then there is nothing left to report.
    more = extract_node({"masters": masters, "extract_mode": "fetch", "max_papers": 8})
    assert len(more["extracted"]) == 8
    assert "not opened" not in more["trace"][0]["detail"]


def test_max_papers_reaches_the_state_from_the_command_line(monkeypatch):
    from feynman_agent import cli

    seen = {}
    monkeypatch.setattr(
        cli.session,
        "drive",
        lambda payload, config, on, answer: seen.update(payload) or {"report": "", "topo": None},
    )
    monkeypatch.setattr(cli, "_write", lambda state, out: None)
    cli.main(["e11|e|", "--offline", "--max-papers", "9"])
    assert seen["max_papers"] == 9


def test_extraction_off_stops_at_the_titles():
    from feynman_agent import arxiv_tool
    from feynman_agent.graph import _after_arxiv

    with_papers = {"arxiv": [arxiv_tool.ArxivResult(papers=[_stub_paper()])]}
    assert _after_arxiv(with_papers) == "read_papers"
    assert _after_arxiv({**with_papers, "extract_mode": "off"}) == "report"
    assert _after_arxiv({"arxiv": [arxiv_tool.ArxivResult(papers=[])]}) == "report"


def _stub_paper(arxiv_id="1306.2799v2"):
    from feynman_agent import arxiv_tool

    return arxiv_tool.Paper(
        title="Analytic results for planar three-loop four-point integrals",
        authors="J. M. Henn et al.",
        published="2013-06",
        arxiv_id=arxiv_id,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        summary="We present analytic results in ancillary files.",
    )


def test_the_papers_the_search_returned_are_read_for_their_results(monkeypatch):
    """The whole point of step (f): come back with the expression, not a title."""
    import io
    import tarfile

    from feynman_agent import arxiv_tool, extract
    from feynman_agent.graph import extract_node, report

    buf = io.BytesIO()
    tex = rb"""\section{Results}
    \begin{align} f_1 &= 1 - \eps^2 \frac{\pi^2}{4} - 29 \eps^3 \zeta_3 \,. \end{align}
    Explicit results are in the ancillary file resultA.m."""
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("p.tex", tex), ("anc/resultA.m", b"m" * 9000)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(extract, "fetch", lambda aid, offline=False: buf.getvalue())

    state = {
        "arxiv": [arxiv_tool.ArxivResult(papers=[_stub_paper()], target="e12|e3|45|45|e6|6|e|")],
        "extract_mode": "fetch",  # no model, so this asserts on the parsing alone
    }
    state.update(extract_node(state))
    text = report({**state, "trace": [], "problems": []})["report"]

    assert "## Read from the papers themselves" in text
    assert "resultA.m" in text and "arxiv.org/src" in text  # the machine-readable result
    assert r"\zeta_3" in text  # and the expansion printed in the paper
    assert "e12|e3|45|45|e6|6|e|" in text  # which diagram it was wanted for
    assert "Candidates, not verified equalities" in text


def test_arxiv_falls_back_to_the_parent_when_there_was_no_reduction():
    """With NeatIBP failed there are no masters, so the original is the target."""
    from feynman_agent import arxiv_tool
    from feynman_agent import topology as T

    topo = T.from_nickel("e12|e3|34|5|e5|e|")
    assert "two-loop" in arxiv_tool.build_query(topo)
    assert "four-point" in arxiv_tool.build_query(topo)


# --------------------------------------------------------------------------
# Masses
# --------------------------------------------------------------------------


def test_a_massive_propagator_survives_the_integrand_parser():
    """``l1^2-m^2`` is a momentum and a mass, and used to be neither.

    The graph is rebuilt from momentum conservation alone, so a mass symbol left
    in the expression was read as an external leg — when it parsed at all.
    """
    block = """
    LoopMomenta={l1};
    ExternalMomenta={k1};
    Propagators={l1^2-m^2,(l1+k1)^2-msq};
    """
    out = run(block)
    topo = out["topo"]
    assert topo.nickel() == "e11|e|"
    assert topo.ext_momenta == ["k1"]  # 'm' is a mass, not a fourth momentum
    assert set(topo.masses.values()) == {"m^2", "msq"}


def test_the_isps_dropped_by_the_reconstruction_take_their_indices_with_them():
    """Masses are keyed by propagator, so dropping an ISP must re-key them."""
    from feynman_agent.graph import _parse_integrand

    topo = _parse_integrand(
        "LoopMomenta={l1,l2};ExternalMomenta={k1,k2,k4};"
        "Propagators={l1^2-msq,(l1+k1)^2,(l1+k1+k2)^2,(l2-k1-k2)^2,(l2+k4)^2,"
        "l2^2,(l1+l2)^2-msq,(l1+k4)^2,(l2+k1)^2};"
    )
    assert topo.nickel() == "e12|e3|34|5|e5|e|"
    assert len(topo.propagators) == 7  # the two ISPs went
    assert {topo.propagators[i] for i in topo.masses} == {"l1", "l1+l2"}


def test_a_massless_stored_result_is_not_an_answer_for_a_massive_graph():
    """The Nickel index is the graph alone; the local DB holds the massless bubble."""
    out = run("e11|e|:n11|n|")
    assert steps(out)["local database"] == "miss"
    assert "different masses" in dict((e["step"], e["detail"]) for e in out["trace"])[
        "local database"
    ]
    assert "Gamma(nu1+nu2-d/2)" not in (out.get("answer") or "")


def test_an_entry_declaring_the_same_masses_is_still_an_exact_hit(tmp_path):
    """The mass check must not lock the database out of ever answering."""
    import json

    from feynman_agent import localdb
    from feynman_agent import topology as T

    db = tmp_path / "db.json"
    db.write_text(
        json.dumps(
            {
                "integrals": [
                    {"nickel": "e11|e|", "name": "equal-mass bubble",
                     "result": "B(m^2) = ...", "masses": ["mm", "mm"]},
                    {"nickel": "e11|e|", "name": "massless bubble", "result": "G(nu1,nu2) = ..."},
                ]
            }
        )
    )
    hit = localdb.search(T.from_nickel("e11|e|:n11|n|"), json_path=db, notebooks=tmp_path)
    # The symbols differ — 'mm' here, 'm1^2' from the configuration — and equal
    # masses are equal masses whatever they are called.
    assert [e.name for e in hit.exact] == ["equal-mass bubble"]
    assert [e.name for e in hit.similar] == ["massless bubble"]


def test_neatibp_is_given_the_masses_and_a_value_for_each():
    """NeatIBP exits when GenericPoint misses a scalar variable, mass or not."""
    from feynman_agent import neatibp
    from feynman_agent import topology as T

    text, notes, _ = neatibp.kinematics_text(T.assign_momenta(T.from_nickel("e11|e|:n11|n|")))
    assert "Propagators={(-l1+k1)^2-m1^2,(l1)^2-m1^2};" in text
    assert "m1->1/97" in text
    assert any("masses carried through" in n for n in notes)


def test_loopedia_picks_out_the_configuration_that_was_asked_for():
    """A graph is indexed once per mass configuration and only a few are fetched.

    Without this the massive bubble gets the references for the massless one,
    which are for a different integral entirely.
    """
    from feynman_agent import loopedia
    from feynman_agent import topology as T

    configs = ["e11|e|:zzz|z|", "e11|e|:n00|n|", "e11|e|:n11|n|", "e11|e|:*"]
    want = T.from_nickel("e11|e|:n11|n|")
    assert [c for c in configs if loopedia._matches(c, want)] == ["e11|e|:n11|n|"]


def test_two_configurations_can_differ_only_in_their_external_legs():
    """Both of these have seven massless propagators; the legs are the difference.

    Comparing masses propagator by propagator calls them the same integral, so
    the configuration strings are compared whole whenever there is one.
    """
    from feynman_agent import loopedia
    from feynman_agent import topology as T

    on_shell = "e12|e3|34|5|e5|e|:000|00|00|0|00|0|"
    off_shell = "e12|e3|34|5|e5|e|:000|00|00|0|10|1|"
    assert loopedia._config_signature(on_shell) == loopedia._config_signature(off_shell)

    assert loopedia._matches(off_shell, T.from_nickel(off_shell))
    assert not loopedia._matches(on_shell, T.from_nickel(off_shell))


def test_saying_nothing_about_masses_leaves_the_catalogues_own_order_alone():
    """An empty ``masses`` is silence, not a request for the massless case."""
    from feynman_agent import topology as T

    assert not T.from_nickel("e11|e|").masses_given
    assert T.from_nickel("e11|e|:zzz|z|").masses_given  # massless, but stated


def test_the_massive_and_massless_reports_do_not_overwrite_each_other():
    """The report is named after the integral, and the graph is not the integral."""
    from feynman_agent import topology as T

    assert T.slug(T.from_nickel("e11|e|")) == "e11-e"
    assert T.slug(T.from_nickel("e11|e|:n11|n|")) == "e11-e-n11-n"


def test_the_report_names_the_masses_and_which_records_carry_them():
    out = run("e11|e|:n11|n|", extract_mode="off")
    text = out["report"]
    assert "- **masses** — `-l1+k1`: `m1^2`, `l1`: `m1^2`" in text
    assert "(2 massive)" in text
    assert "masses: `n11|n|` — **the masses asked for**" in text


# --------------------------------------------------------------------------
# 12. The figure
# --------------------------------------------------------------------------


def test_the_report_opens_with_the_diagram():
    """Before anything else: everything below is only worth reading if the
    agent identified the right integral, which is a question about a picture."""
    out = run("e11|e|")
    head = out["report"].splitlines()[:3]
    assert head[0] == "# Feynman integral `e11|e|`"
    assert head[2] == "![Feynman diagram of e11|e|](e11-e.svg)"
    assert out["figure"].startswith("<svg")


def test_the_figure_is_well_formed_and_holds_every_line_and_leg():
    """The double box: seven propagators, four legs, six vertices."""
    import xml.etree.ElementTree as ET

    out = run("e12|e3|34|5|e5|e|")
    svg = out["figure"]
    root = ET.fromstring(svg)  # malformed XML raises here
    tag = lambda name: [e for e in root.iter() if e.tag.endswith(name)]
    assert len(tag("path")) == 7  # internal lines
    assert len(tag("line")) == 4  # external legs
    assert len(tag("circle")) == 6  # vertices
    for momentum in out["topo"].propagators:
        assert f">{momentum}<" in svg


def test_a_massive_line_is_drawn_as_a_massive_line():
    """Colour and weight say which lines carry a mass; the label says which mass."""
    from feynman_agent import draw

    out = run("e11|e|:n11|n|")
    assert out["figure"].count(f'stroke="{draw.MASS_INK}"') == 2
    assert ">-l1+k1 · m1²<" in out["figure"]
    assert "carry a mass" in out["report"]


def test_the_figure_is_written_where_the_report_links_to_it(tmp_path):
    """Two files, and a relative link between them that has to resolve."""
    import re

    from feynman_agent.cli import _write

    out = run("e11|e|:n11|n|")
    report = tmp_path / "somewhere" / "named-by-hand.md"
    _write(out, str(report))
    link = re.search(r"!\[[^]]*]\(([^)]+)\)", report.read_text()).group(1)
    assert (report.parent / link).read_text().startswith("<svg")
    assert link == "e11-e-n11-n.svg"  # named for the integral, not for the report
