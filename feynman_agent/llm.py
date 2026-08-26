"""Optional model layer, needed by no path from an integral to a result.

``read_figure`` reads a diagram image into an edge list — every other input
format is parsed deterministically. The model only ever *proposes* a graph;
``topology`` then rebuilds and canonicalises it, so a malformed answer cannot
slip through as a result. ``ask`` is a plain text call, used by ``readable`` to
rewrite a paper's equations into LaTeX that renders and by ``extract`` to say
what its attached files amount to; without a key the extraction still reports
the files and equations it found, in the LaTeX the authors wrote.

The provider is pluggable — the task is ordinary reading, not anything
Claude-specific. Pick one with ``FEYNMAN_AGENT_PROVIDER``:

    anthropic  (default)  ANTHROPIC_API_KEY    claude-opus-5
    kimi                  MOONSHOT_API_KEY     kimi-k3          (OpenAI-compatible)
    deepseek              DEEPSEEK_API_KEY     deepseek-chat    (OpenAI-compatible, text only)
    openai                OPENAI_API_KEY       gpt-4o

Any other OpenAI-compatible endpoint works via ``openai`` plus
``FEYNMAN_AGENT_BASE_URL``. Override the model with ``FEYNMAN_AGENT_MODEL``.

Only ``read_figure`` needs a model that takes images, and ``reads_images`` says
which do: a text-only provider is a perfectly good choice for everything else,
and says so plainly rather than failing inside the API when a picture arrives.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class Provider:
    name: str
    key_vars: tuple[str, ...]
    default_model: str
    base_url: str | None = None
    reads_images: bool = True  # only the figure input needs this

    @property
    def openai_style(self) -> bool:
        return self.name != "anthropic"


PROVIDERS = {
    "anthropic": Provider("anthropic", ("ANTHROPIC_API_KEY",), "claude-opus-5"),
    "kimi": Provider(
        "kimi",
        ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        "kimi-k3",
        "https://api.moonshot.ai/v1",
    ),
    # deepseek-chat and deepseek-reasoner are text-only, which costs nothing
    # here except the figure input: the equations and the paper summaries are
    # text either way.
    "deepseek": Provider(
        "deepseek",
        ("DEEPSEEK_API_KEY",),
        "deepseek-chat",
        "https://api.deepseek.com/v1",
        reads_images=False,
    ),
    "openai": Provider("openai", ("OPENAI_API_KEY",), "gpt-4o"),
}

PROMPT = """You are reading a Feynman diagram. Return ONLY a JSON object:

{"internal_edges": [[i,j], ...], "external_legs": [i, ...], "comment": "..."}

Rules:
- Number the internal (interaction) vertices 0,1,2,... Only vertices where
  three or more lines meet, or where an external leg attaches, are vertices.
- "internal_edges" lists every propagator as a pair of vertex indices. Repeat a
  pair for a doubled line (e.g. a bubble is [[0,1],[0,1]]).
- "external_legs" lists, for each external line, the vertex it attaches to.
- Do not include the external lines in "internal_edges".
"""


def load_env() -> None:
    """Read KEY=value lines from a project-local .env, if present.

    A real environment variable always wins, so exporting still overrides it.
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def provider() -> Provider:
    load_env()
    name = os.environ.get("FEYNMAN_AGENT_PROVIDER", "anthropic").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"unknown FEYNMAN_AGENT_PROVIDER {name!r}; choose one of {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[name]


def api_key(prov: Provider) -> str | None:
    return next((os.environ[v] for v in prov.key_vars if os.environ.get(v)), None)


def model_name(prov: Provider) -> str:
    return os.environ.get("FEYNMAN_AGENT_MODEL") or prov.default_model


def base_url(prov: Provider) -> str | None:
    return os.environ.get("FEYNMAN_AGENT_BASE_URL") or prov.base_url


def max_tokens() -> int:
    """Generous by default: a cap is not a spend, and reasoning models need room.

    The answer itself is a few hundred tokens, but a thinking model bills its
    reasoning against the same budget — kimi-k2.5 spent 1999 of 2000 tokens
    thinking and returned an empty message. Rewriting one page-long equation
    measured 5028 output tokens on kimi-k3, of which 4449 were reasoning, so a
    budget of 8000 is one equation away from the same silence.
    """
    return int(os.environ.get("FEYNMAN_AGENT_MAX_TOKENS", 16000))


def timeout() -> int:
    """How long one call may take before it is given up on.

    Not a performance setting — a limit on how long a step that the agent can
    do without is allowed to hold up one that it cannot. A reasoning model
    rewriting three pages of LaTeX takes minutes; a request that has gone
    astray takes forever, and there is nothing on the other end to say which is
    which.
    """
    return int(os.environ.get("FEYNMAN_AGENT_TIMEOUT", 600))


def describe() -> str:
    prov = provider()
    where = f" at {base_url(prov)}" if base_url(prov) else ""
    return f"{model_name(prov)} via {prov.name}{where}"


def available() -> bool:
    prov = provider()
    if not api_key(prov):
        return False
    package = "langchain_openai" if prov.openai_style else "langchain_anthropic"
    return importlib.util.find_spec(package) is not None


def missing_reason() -> str:
    """Why the figure path is unavailable, in terms the user can act on."""
    prov = provider()
    if not api_key(prov):
        return (
            f"{' or '.join(prov.key_vars)} is not set for provider {prov.name!r}. "
            f"Put it in {ENV_FILE} or export it."
        )
    package = "langchain_openai" if prov.openai_style else "langchain_anthropic"
    if importlib.util.find_spec(package) is None:
        return f"provider {prov.name!r} needs `pip install {package.replace('_', '-')}`"
    return ""


@contextlib.contextmanager
def usable_proxies():
    """Drop an ALL_PROXY that httpx cannot parse, for the duration of the call.

    httpx understands ``http(s)://``, and ``socks5://`` only when socksio is
    installed — a bare ``socks://`` raises ValueError before any request goes
    out. HTTP(S)_PROXY still applies, so dropping ALL_PROXY keeps the proxy.

    Hold this around a *batch* of concurrent calls and the batch is safe: the
    variable is process-wide, so several independent holders would have the
    first one to finish restoring it while the others were still building their
    clients. Nested holders are inert — the inner ones find nothing left to drop
    and so restore nothing — which makes the outermost one the only one that
    puts it back, after the last call has returned.
    """
    ok = {"http", "https"}
    if importlib.util.find_spec("socksio") is not None:
        ok |= {"socks5", "socks5h"}
    saved = {}
    for var in ("ALL_PROXY", "all_proxy"):
        value = os.environ.get(var)
        if value and value.split("://", 1)[0].lower() not in ok:
            saved[var] = os.environ.pop(var)
    try:
        yield sorted(saved)
    finally:
        os.environ.update(saved)


def _image_block(prov: Provider, media: str, data: str) -> dict:
    """The two provider families spell an inline image differently."""
    if prov.openai_style:
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _chat(prov: Provider):
    if prov.openai_style:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name(prov),
            api_key=api_key(prov),
            base_url=base_url(prov),
            max_tokens=max_tokens(),
            timeout=timeout(),
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model_name(prov), api_key=api_key(prov), max_tokens=max_tokens(), timeout=timeout()
    )


def ask(prompt: str) -> str:
    """One plain-text question to the configured model."""
    from langchain_core.messages import HumanMessage

    with usable_proxies():
        reply = _chat(provider()).invoke([HumanMessage(content=prompt)])
    return reply.content if isinstance(reply.content, str) else str(reply.content)


def read_figure(path: str | Path):
    """Return a Topology read from a diagram image."""
    from langchain_core.messages import HumanMessage

    from .topology import Topology

    prov = provider()
    if not prov.reads_images:
        raise ValueError(
            f"provider {prov.name!r} has no model that reads images; set "
            f"FEYNMAN_AGENT_PROVIDER to one that does "
            f"({', '.join(n for n, p in PROVIDERS.items() if p.reads_images)}), "
            f"or give the diagram as a Nickel index or edge list"
        )
    path = Path(path)
    media = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode()

    message = HumanMessage(
        content=[_image_block(prov, media, data), {"type": "text", "text": PROMPT}]
    )
    with usable_proxies():
        reply = _chat(prov).invoke([message])

    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        why = f": {text[:200]}"
        if reply.response_metadata.get("finish_reason") == "length":
            why = (
                f" — it hit the {max_tokens()}-token limit first"
                f" (a reasoning model can spend the whole budget thinking;"
                f" raise FEYNMAN_AGENT_MAX_TOKENS){why.rstrip(': ')}"
            )
        raise ValueError(f"no JSON in the reply from {describe()}{why}")
    spec = json.loads(match.group(0))
    return Topology(
        int_edges=[tuple(e) for e in spec["internal_edges"]],
        legs=list(spec["external_legs"]),
        name=f"figure:{path.name}",
        notes=[f"read from {path.name} by {describe()}: {spec.get('comment', '')}".strip()],
    )
