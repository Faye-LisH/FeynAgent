"""A browser front end: text or a dragged-in figure at one end, the report at the other.

The command line already runs the agent. What it cannot do is take a picture of
a diagram someone dropped on it, or keep the paths to NeatIBP, Singular and
SpaSM somewhere they can be seen and changed — they are the three things most
likely to be wrong on a new machine, and the ones that turn into a silent hang
rather than an error. So they live in a sidebar here, next to the model key and
every flag the CLI takes.

Nothing but the standard library. An agent that shells out to Mathematica does
not need a web framework on top of it, and ``ThreadingHTTPServer`` already does
the one thing this needs: a thread per request, so a reduction that runs for
minutes does not block the page it is reporting to. The run itself sits in a
thread of its own and talks to the browser through two queues — events out,
the answer to the arXiv question back in.

Settings are applied to this process's environment, so they are shared by
whatever is running: one browser, one agent, one machine. Bind it to localhost
(the default) — the process holds an API key and starts subprocesses.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import tempfile
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import extract, llm, neatibp, session

STATIC = Path(__file__).resolve().parent / "static"
UPLOADS = Path(tempfile.gettempdir()) / "feynman_agent" / "uploads"
TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}

# Sidebar field -> the variable the agent reads it from. The API key is not
# here: which variable it belongs in depends on the provider chosen next to it.
ENV = {
    "neatibp": "FEYNMAN_AGENT_NEATIBP",
    "singular": "FEYNMAN_AGENT_SINGULAR",
    "spasm": "FEYNMAN_AGENT_SPASM",
    "provider": "FEYNMAN_AGENT_PROVIDER",
    "model": "FEYNMAN_AGENT_MODEL",
    "base_url": "FEYNMAN_AGENT_BASE_URL",
}
# These two are overrides of a *provider's* defaults, and the page empties them
# when the provider changes. Blank therefore has to mean "no override" rather
# than "keep what is there": otherwise switching from kimi to deepseek would
# send deepseek's key to kimi's endpoint asking for kimi's model.
CLEARED = ("model", "base_url")

RUNS: dict[str, "Run"] = {}


@dataclass
class Run:
    """One agent run, and the two queues that connect it to a browser tab."""

    events: queue.Queue = field(default_factory=queue.Queue)
    replies: queue.Queue = field(default_factory=queue.Queue)
    state: dict = field(default_factory=dict)

    def emit(self, kind: str, **data) -> None:
        self.events.put({"kind": kind, **data})

    def on(self, kind: str, data) -> None:
        """What ``session.drive`` reports, in the shape the page consumes.

        A finished stage travels under its own key rather than spread across
        the event: a trace entry already carries a field called ``kind``, which
        is the input format the run was given.
        """
        self.emit("note", text=data) if kind == "note" else self.emit("step", step=data)

    def ask(self, question: dict) -> str:
        """Put the graph's question on the page and block until a button answers."""
        self.emit("ask", question=question)
        return self.replies.get()


def apply_settings(settings: dict) -> None:
    """Put the sidebar's values into the environment the run will read.

    Blank means "leave what is already there", so a field the user never filled
    in keeps whatever ``.env`` or the shell provided rather than erasing it.
    """
    llm.load_env()
    for name, var in ENV.items():
        value = str(settings.get(name) or "").strip()
        if value:
            os.environ[var] = value
        elif name in CLEARED and name in settings:
            os.environ.pop(var, None)
    key = str(settings.get("api_key") or "").strip()
    if key:  # after the provider, so it lands in that provider's variable
        os.environ[llm.provider().key_vars[0]] = key


def current_settings() -> dict:
    """What the sidebar shows when the page opens. The key is never sent back."""
    llm.load_env()
    prov = llm.provider()
    return {
        "neatibp": str(neatibp.where("neatibp")),
        "singular": str(neatibp.where("singular")),
        "spasm": str(neatibp.where("spasm")),
        "provider": prov.name,
        "providers": sorted(llm.PROVIDERS),
        "model": llm.model_name(prov),
        "base_url": llm.base_url(prov) or "",
        # Per provider, both of them: the sidebar can be switched to one whose
        # key is set and one whose key is not, and it should say which is which.
        "key_vars": {name: p.key_vars[0] for name, p in llm.PROVIDERS.items()},
        "has_key": {name: bool(llm.api_key(p)) for name, p in llm.PROVIDERS.items()},
        # What each provider falls back to, so switching between them shows the
        # model and endpoint that will actually be used rather than the last
        # one's.
        "defaults": {
            name: {"model": p.default_model, "base_url": p.base_url or "", "images": p.reads_images}
            for name, p in llm.PROVIDERS.items()
        },
        "max_papers": extract.MAX_PAPERS,
    }


def check(settings: dict) -> dict:
    """Which of the three tool paths are really there, as typed into the sidebar.

    Worth a route of its own because this is the failure the agent is worst at
    reporting: a missing NeatIBP degrades to a mocked reduction, and a Singular
    that is not where it was said to be takes the run down with it minutes
    later. Better to say so while the field is still under the cursor.
    """
    out = {}
    for name in ("neatibp", "singular", "spasm"):
        typed = str(settings.get(name) or "").strip()
        path = Path(typed) if typed else neatibp.where(name)
        out[name] = (path / "run.sh").exists() if name == "neatibp" else path.exists()
    return out


def _options(opts: dict) -> dict:
    """The run options, under the names the graph's state uses for them."""
    return {
        "input_kind": opts.get("kind") or None,
        "neatibp_mode": opts.get("neatibp_mode") or "auto",
        "extract_mode": opts.get("extract") or "auto",
        "max_papers": int(opts.get("max_papers") or extract.MAX_PAPERS),
        "workdir": opts.get("workdir") or None,
        "offline": bool(opts.get("offline")),
        "always_reduce": bool(opts.get("reduce")),
    }


def _work(run: Run, payload: dict, config: dict, arxiv: str) -> None:
    """The whole run, in a thread of its own. Ends by saying so, always.

    A failure here is the page's business, not the terminal's — the browser is
    where the run was started and where its trace already is.
    """
    fixed = {"yes": "search", "no": "stop"}.get(arxiv)
    answer = (lambda question: fixed) if fixed else run.ask
    try:
        run.state = session.drive(payload, config, run.on, answer)
        run.emit("report", markdown=run.state.get("report", ""))
    except Exception as exc:
        run.emit("error", text=f"{type(exc).__name__}: {exc}")
    finally:
        run.emit("done")


def start(payload: dict) -> str:
    """Apply the settings, start the run, and hand back the job it goes under."""
    apply_settings(payload.get("settings") or {})
    options = payload.get("options") or {}
    job = uuid.uuid4().hex[:8]
    RUNS[job] = run = Run()
    state = session.initial(payload["input"], **_options(options))
    config = {"configurable": {"thread_id": job}}
    args = (run, state, config, options.get("arxiv", "ask"))
    threading.Thread(target=_work, args=args, daemon=True).start()
    return job


def upload(name: str, body: bytes) -> Path:
    """Keep a dropped figure where the agent can open it by path."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / Path(name).name
    path.write_bytes(body)
    return path


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # so the page and its two assets share a connection

    def do_GET(self):
        url = urlparse(self.path)
        job = parse_qs(url.query).get("job", [""])[0]
        if url.path in ("/", "/index.html"):
            return self._static("index.html")
        if url.path in ("/app.js", "/app.css"):
            return self._static(url.path.lstrip("/"))
        if url.path == "/api/settings":
            return self._json(current_settings())
        if url.path == "/api/figure":
            return self._send(200, "image/svg+xml", RUNS[job].state.get("figure", "").encode())
        if url.path == "/api/events":
            return self._events(RUNS[job])
        self.send_error(404)

    def do_POST(self):
        url = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if url.path == "/api/upload":
            name = parse_qs(url.query).get("name", ["figure.png"])[0]
            return self._json({"path": str(upload(name, body))})

        payload = json.loads(body or b"{}")
        if url.path == "/api/check":
            return self._json(check(payload))
        if url.path == "/api/run":
            return self._json({"job": start(payload)})
        if url.path == "/api/answer":
            RUNS[payload["job"]].replies.put(payload["reply"])
            return self._json({"ok": True})
        if url.path == "/api/save":
            path = session.write(RUNS[payload["job"]].state, payload.get("path") or None)
            return self._json({"path": str(path.resolve())})
        self.send_error(404)

    # ----------------------------------------------------------------- replies

    def _events(self, run: Run):
        """Server-sent events: one JSON object per line until the run says done.

        The queue is the run's only way out, so this blocks on it rather than
        polling — a reduction can go ten minutes between two events.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # the stream ends when the run does
        self.end_headers()
        while True:
            event = run.events.get()
            try:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return  # the tab was closed; the run carries on and keeps its state
            if event["kind"] == "done":
                return

    def _static(self, name: str):
        path = STATIC / name
        self._send(200, TYPES[path.suffix], path.read_bytes())

    def _json(self, data: dict):
        self._send(200, "application/json", json.dumps(data).encode())

    def _send(self, code: int, kind: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        """Quiet: the page narrates itself, and the terminal only started it."""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="feynman-agent-web", description="Browser front end for the Feynman agent."
    )
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1", help="localhost by default; it holds a key")
    p.add_argument("--open", action="store_true", help="open the page in a browser")
    args = p.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Feynman agent at {url}  (ctrl-c to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
