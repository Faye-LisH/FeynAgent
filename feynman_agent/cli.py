"""Command line entry point: ``python -m feynman_agent <input>``."""

from __future__ import annotations

import argparse
import os
import sys

from . import llm, session
from .extract import MAX_PAPERS
from .graph import MARKS

# A run takes minutes, so it reports as it goes. Progress goes to stderr; the
# report is Markdown and goes to a file, since that is where pages of epsilon
# expansions are actually readable. Its path is the only thing on stdout, so
# ``open $(python -m feynman_agent ... --quiet)`` works.


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _done(entry: dict) -> str:
    """One finished stage, in one line — the same text the report will show."""
    return f"  {MARKS.get(entry['status'], '·')} {entry['step']}: {entry['detail'].splitlines()[0][:95]}"


def _write(state: dict, out: str | None) -> None:
    """Put the report where it was asked for; ``-`` means stdout as usual.

    Printing to stdout gets no figure: there is nowhere to put one, and the
    link is then as dangling as any other page taken out of its folder.
    """
    if out == "-":
        print(state["report"], end="")  # the report already ends in a newline
        return
    print(session.write(state, out))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="feynman-agent",
        description="Identify and evaluate a Feynman integral.",
    )
    p.add_argument(
        "input",
        help="Nickel index, edge list, integrand/kinematics block, or a figure path",
    )
    p.add_argument(
        "--kind",
        choices=["nickel", "edgelist", "integrand", "figure"],
        help="skip input sniffing and force a format",
    )
    p.add_argument(
        "--neatibp",
        choices=["auto", "real", "mock"],
        default="auto",
        help="auto runs NeatIBP when its dependencies are present (default)",
    )
    p.add_argument("--workdir", help="where to put NeatIBP inputs and outputs")
    p.add_argument("--offline", action="store_true", help="use only cached Loopedia responses")
    p.add_argument(
        "--reduce",
        action="store_true",
        help="run the IBP reduction even when a reference was already found",
    )
    p.add_argument(
        "--arxiv",
        choices=["ask", "yes", "no"],
        default="ask",
        help="whether to fall back to an arXiv search (default: ask)",
    )
    p.add_argument(
        "--extract",
        choices=["auto", "rewrite", "fetch", "off"],
        default="auto",
        help="how deeply to read the papers arXiv returns (default: auto, a "
        "one-call summary of each); 'rewrite' also has the model rewrite every "
        "candidate equation so it renders, which is a call each and the fastest "
        "way to spend a quota; 'fetch' takes the source with no model call and "
        "prints the authors' own LaTeX; 'off' stops at the titles",
    )
    p.add_argument(
        "--max-papers",
        type=int,
        default=MAX_PAPERS,
        metavar="N",
        help=f"how many papers to open and read (default: {MAX_PAPERS}); every "
        "diagram gets one before any diagram gets a second",
    )
    p.add_argument(
        "--provider",
        choices=sorted(llm.PROVIDERS),
        help="who to ask for the optional model calls (default: anthropic, or "
        "FEYNMAN_AGENT_PROVIDER); only figure input needs one that reads images",
    )
    p.add_argument("--model", help="override the model id")
    p.add_argument(
        "-o",
        "--out",
        help="where to write the Markdown report, the figure going beside it "
        "(default: reports/<nickel>.md, '-' for stdout)",
    )
    p.add_argument("--quiet", action="store_true", help="no progress lines, only the report")
    args = p.parse_args(argv)

    if args.provider:
        os.environ["FEYNMAN_AGENT_PROVIDER"] = args.provider
    if args.model:
        os.environ["FEYNMAN_AGENT_MODEL"] = args.model

    def on(kind: str, data) -> None:
        if not args.quiet:
            _say(f"  … {data}" if kind == "note" else _done(data))

    def answer(question: dict) -> str:
        """--arxiv decided it in advance, or the user decides it now."""
        if args.arxiv != "ask":
            return "search" if args.arxiv == "yes" else "stop"
        _say(f"\n{question['question']}")
        _say("  unresolved: " + ", ".join(question["unresolved"]))
        return input("  [search/stop] > ").strip() or "stop"

    state = session.initial(
        args.input,
        input_kind=args.kind,
        neatibp_mode=args.neatibp,
        extract_mode=args.extract,
        max_papers=args.max_papers,
        workdir=args.workdir,
        offline=args.offline,
        always_reduce=args.reduce,
    )
    result = session.drive(state, {"configurable": {"thread_id": "cli"}}, on, answer)

    _write(result, args.out)
    return 0 if result.get("answer") else 1


if __name__ == "__main__":
    sys.exit(main())
