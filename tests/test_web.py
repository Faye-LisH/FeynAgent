"""The browser front end, without a browser.

A real server on a real port, driving a stub agent: the routes, the event
stream, and the one exchange that is not a straight line — the graph stopping
to ask whether to search arXiv, and a button answering it.
"""

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from feynman_agent import neatibp, web

REPORT = "# report\n\n![diagram](x.svg)\n\nanswered {}\n"


@pytest.fixture
def server(monkeypatch):
    """The real handler, with a stub in place of the minutes-long agent run."""

    def drive(payload, config, on, answer):
        # A trace entry has a field of its own called "kind" — the input format.
        on("step", {"step": "identify", "status": "ok", "detail": "1-loop", "kind": "nickel"})
        on("note", "loopedia: e11|e|")
        reply = answer({"question": "Search arXiv?", "unresolved": ["e11|e|"]})
        return {"report": REPORT.format(reply), "figure": "<svg/>", "topo": None}

    monkeypatch.setattr(web.session, "drive", drive)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def get(url):
    return urllib.request.urlopen(url, timeout=10)


def post(url, data):
    request = urllib.request.Request(
        url, json.dumps(data).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(request, timeout=10))


def events(url, on_ask=None):
    """Read the stream to the end, answering the question when it arrives.

    From this one thread, which is the point of a server that threads: the
    answer has to get in while the run is still blocked waiting for it.
    """
    out = []
    with get(url) as stream:
        for line in stream:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            out.append(event)
            if event["kind"] == "ask" and on_ask:
                on_ask(event)
            if event["kind"] == "done":
                return out
    return out


def _run(server, **options):
    job = post(f"{server}/api/run", {"input": "e11|e|", "options": options})["job"]
    seen = events(
        f"{server}/api/events?job={job}",
        lambda event: post(f"{server}/api/answer", {"job": job, "reply": "search"}),
    )
    return job, seen


def test_the_page_and_its_two_assets_are_served(server):
    assert b"<title>Feynman agent</title>" in get(f"{server}/").read()
    assert get(f"{server}/app.js").headers["Content-Type"] == "text/javascript"
    assert get(f"{server}/app.css").headers["Content-Type"] == "text/css"


def test_a_run_streams_what_it_does_and_stops_to_ask(server):
    """The page has to show the same stages the terminal does, as they happen."""
    _, seen = _run(server)

    assert [e["kind"] for e in seen] == ["step", "note", "ask", "report", "done"]
    assert seen[0]["step"] == {
        "step": "identify",
        "status": "ok",
        "detail": "1-loop",
        "kind": "nickel",
    }, "a stage travels whole, including the field it calls kind"
    assert seen[2]["question"]["unresolved"] == ["e11|e|"]
    assert "answered search" in seen[3]["markdown"], "the button's answer reached the graph"


def test_the_figure_is_served_and_the_report_is_stored_beside_it(server, tmp_path):
    job, _ = _run(server)

    assert get(f"{server}/api/figure?job={job}").read() == b"<svg/>"
    written = post(f"{server}/api/save", {"job": job, "path": str(tmp_path / "r.md")})["path"]
    assert Path(written).read_text().startswith("# report")
    assert (tmp_path / "unidentified.svg").read_text() == "<svg/>"


def test_a_dropped_figure_is_saved_where_the_agent_can_open_it(server, monkeypatch, tmp_path):
    """The browser sends bytes and a name; the agent reads figures from paths."""
    monkeypatch.setattr(web, "UPLOADS", tmp_path / "up")
    request = urllib.request.Request(
        f"{server}/api/upload?name=/tmp/elsewhere/diagram.png", b"\x89PNG", method="POST"
    )
    path = Path(json.load(urllib.request.urlopen(request, timeout=10))["path"])

    assert path.read_bytes() == b"\x89PNG"
    assert path.parent == tmp_path / "up", "the client does not get to choose the directory"


def test_the_key_goes_to_the_variable_the_chosen_provider_reads(monkeypatch):
    """One field in the sidebar, three possible homes for what is typed in it."""
    for var in ("MOONSHOT_API_KEY", "FEYNMAN_AGENT_PROVIDER", "FEYNMAN_AGENT_NEATIBP"):
        monkeypatch.setenv(var, "")  # touched, so pytest puts each one back afterwards

    web.apply_settings({"provider": "kimi", "api_key": "sk-test", "neatibp": "/opt/NeatIBP"})

    assert os.environ["MOONSHOT_API_KEY"] == "sk-test"
    assert neatibp.where("neatibp") == Path("/opt/NeatIBP"), "and the tools moved with it"


def test_a_blank_field_keeps_what_is_already_configured(monkeypatch):
    monkeypatch.setenv("FEYNMAN_AGENT_SINGULAR", "/opt/Singular")
    web.apply_settings({"singular": "", "spasm": "  "})
    assert neatibp.where("singular") == Path("/opt/Singular")


def test_switching_provider_drops_the_last_one_s_model(monkeypatch):
    """Otherwise deepseek's key goes to kimi's endpoint asking for kimi's model.

    Model and base URL are overrides of a *provider's* defaults, so the page
    empties them when the provider changes — and an emptied one has to mean "no
    override" rather than "keep what is there", which is what blank means for
    every other field.
    """
    from feynman_agent import llm

    for var in ("FEYNMAN_AGENT_PROVIDER", "FEYNMAN_AGENT_MODEL", "FEYNMAN_AGENT_BASE_URL"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("FEYNMAN_AGENT_MODEL", "kimi-k3")

    web.apply_settings({"provider": "deepseek", "model": "", "base_url": ""})

    prov = llm.provider()
    assert llm.model_name(prov) == "deepseek-chat"
    assert llm.base_url(prov) == "https://api.deepseek.com/v1"


def test_the_sidebar_says_which_tools_are_not_there(server, tmp_path):
    """The one failure the agent reports worst: a path that is simply wrong."""
    singular = tmp_path / "Singular"
    singular.write_text("#!/bin/sh\n")

    found = post(
        f"{server}/api/check",
        {"neatibp": str(tmp_path), "singular": str(singular), "spasm": str(tmp_path / "no.so")},
    )

    assert found["singular"] and not found["spasm"]
    assert not found["neatibp"], "a directory with no run.sh in it is not an installation"


def test_the_sidebar_speaks_the_state_s_language():
    state = web._options({"kind": "", "extract": "fetch", "max_papers": "5", "offline": True})

    assert state["input_kind"] is None, "detect means let the agent decide"
    assert state["extract_mode"] == "fetch" and state["max_papers"] == 5
    assert state["offline"] and state["neatibp_mode"] == "auto"
