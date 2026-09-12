"""NeatIBP driver: generate inputs, run the real package, parse the summary.

NeatIBP (arXiv:2305.08783, https://github.com/yzhphy/NeatIBP) does syzygy-based
IBP reduction and reports the master integrals of a topology. It needs a
complete basis of P + ISP = L(L+1)/2 + L*E scalar products, so the missing
irreducible scalar products are generated here before writing the input files.

``mode="mock"`` skips the kernel entirely and replays a recorded run, which
keeps the agent testable without burning minutes of Mathematica time.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import numerator
from .topology import (
    Topology,
    Vec,
    _rank,
    assign_momenta,
    g_string,
    mass_symbols,
    propagator_terms,
)

# Where the three external tools live on this machine, and the variable that
# moves each one somewhere else.
TOOLS = {
    "neatibp": ("FEYNMAN_AGENT_NEATIBP", "/home/faye/Packages/NeatIBP"),
    "singular": ("FEYNMAN_AGENT_SINGULAR", "/home/faye/Packages/Singular4/Singular4/bin/Singular"),
    "spasm": ("FEYNMAN_AGENT_SPASM", "/home/faye/Packages/SparseRREF/lib/libspasm.so"),
}
# NeatIBP writes this once a mission has completed; it is the only thing that
# distinguishes a finished output directory from one still being written to.
FINISHED_TAG = Path("results") / "NeatIBP_finished.tag"


def where(tool: str) -> Path:
    """The installed path of one external tool, the environment first.

    Read on each call rather than bound at import, because the web interface's
    sidebar edits these while the process is running — a module constant would
    hold whatever the environment said at startup and quietly ignore the change.
    """
    var, default = TOOLS[tool]
    return Path(os.environ.get(var) or default)


def recorded() -> Path:
    """The reference run NeatIBP ships, which mock mode replays."""
    return where("neatibp") / "examples" / "dbox" / "outputs" / "dbox" / "results"


@dataclass
class IBPResult:
    masters: list[str] = field(default_factory=list)
    n_ibp: int = 0
    n_integrals: int = 0
    sectors: dict[str, int] = field(default_factory=dict)
    summary: str = ""
    # A numerator, written as the combination of target integrals it was sent
    # as: ``1/2 G[1,1,1,1,1,1,1,-1,0] - 1/2 G[0,1,1,1,1,1,1,0,0]``. Empty for the
    # scalar integral. Filled whether or not a kernel ran, since the masters a
    # mock replays are the family's and this is what the numerator is in them.
    expansion: str = ""
    # What the run had to do besides reduce: ISPs invented, masses passed
    # through, a previous run's output cleared out of the way. The report says
    # so, because two of the three change what the numbers below mean and the
    # third deleted something.
    notes: list[str] = field(default_factory=list)
    mocked: bool = False
    ok: bool = False
    error: str = ""
    workdir: str = ""
    log_tail: str = ""


# --------------------------------------------------------------------------
# Input generation
# --------------------------------------------------------------------------


def _sp_vector(mom: Vec, n_loops: int, n_ext: int) -> list[int]:
    """Expand q^2 in the basis of independent scalar products."""
    a, b = mom[:n_loops], mom[n_loops:]
    out = []
    for i in range(n_loops):
        for j in range(i, n_loops):
            out.append(a[i] * a[i] if i == j else 2 * a[i] * a[j])
    for i in range(n_loops):
        for x in range(n_ext):
            out.append(2 * a[i] * b[x])
    return out


def _parse(expr: str, basis: list[str]) -> Vec:
    from .topology import parse_momentum

    return parse_momentum(expr, basis)


def complete_basis(topo: Topology) -> list[str]:
    """The ISP momenta that complete the propagators to a full SP basis.

    The caller's own come first and are kept whatever they are: a numerator
    is expanded in this basis, and the coefficients should be in the ISPs the
    caller chose, not in whichever ones a search happened to find first. More
    are invented only if those still leave the basis short.
    """
    L, E = topo.n_loops, len(topo.ext_momenta)
    basis = topo.loop_momenta + topo.ext_momenta
    need = L * (L + 1) // 2 + L * E
    vecs = [_sp_vector(_parse(p, basis), L, E) for p in topo.propagators + topo.isps]
    if _rank([tuple(v) for v in vecs]) >= need:
        return list(topo.isps)

    candidates: list[str] = []
    for i in range(L):
        for x in range(E):
            candidates.append(f"{topo.loop_momenta[i]}+{topo.ext_momenta[x]}")
    for i, j in itertools.combinations(range(L), 2):
        candidates.append(f"{topo.loop_momenta[i]}+{topo.loop_momenta[j]}")
    for i in range(L):
        candidates.append(topo.loop_momenta[i])
        for x, y in itertools.combinations(range(E), 2):
            candidates.append(
                f"{topo.loop_momenta[i]}+{topo.ext_momenta[x]}+{topo.ext_momenta[y]}"
            )

    chosen = list(topo.isps)
    for cand in candidates:
        if len(vecs) >= need:
            break
        v = _sp_vector(_parse(cand, basis), L, E)
        if _rank([tuple(x) for x in vecs + [v]]) > _rank([tuple(x) for x in vecs]):
            vecs.append(v)
            chosen.append(cand)
    return chosen


def invariants(ext: list[str]) -> list[tuple[str, str, str]]:
    """The external kinematics NeatIBP is told: ``(a, b, value)`` for a.b = value.

    Legs massless and on shell, with ``s`` and ``t`` up to four points. Five
    and more are left as the squares alone; the note in ``kinematics_text``
    says the rest is to be filled in by hand.
    """
    k = ext
    squares = [(x, x, "0") for x in k]
    if len(k) == 1:
        return [(k[0], k[0], "s")]
    if len(k) == 2:
        return squares + [(k[0], k[1], "s/2")]
    if len(k) == 3:
        return squares + [(k[0], k[1], "s/2"), (k[1], k[2], "t/2"), (k[0], k[2], "(-s-t)/2")]
    return squares


# Denominators for the mass values at the generic point. NeatIBP's own examples
# use small unrelated fractions; they only have to be distinct, and there are
# more here than a graph under the vertex cap can have propagators.
MASS_POINT = (97, 23, 41, 17, 13, 7, 61, 53, 29, 11, 79, 37, 43, 71, 19, 89)


def kinematics_text(topo: Topology) -> tuple[str, list[str], list[str]]:
    """Build kinematics.txt: masses as given, external legs massless and on shell.

    Returns the text, what building it involved, and the ISPs in the order
    they were appended — the basis that ``target_text`` writes vectors over.
    """
    n_ext = len(topo.ext_momenta)
    isps = complete_basis(topo)
    # ISPs are scalar products completing the basis, never real lines, so they
    # are massless whatever the propagators carry.
    props = propagator_terms(topo) + [f"({q})^2" for q in isps]
    notes = []

    k = topo.ext_momenta
    rules = invariants(k)
    text_rules = "{" + ",".join(f"{a}^2->{v}" if a == b else f"{a} {b}->{v}" for a, b, v in rules) + "}"
    values = [f"{x}->{at}" for x, at in (("s", "-1"), ("t", "-1/19")) if any(x in v for _, _, v in rules)]
    if n_ext > 3:
        notes.append(
            f"{n_ext + 1}-point kinematics is not templated; "
            "the invariant rules in kinematics.txt need to be filled in by hand."
        )

    # NeatIBP checks that GenericPoint covers every scalar variable in the input
    # and exits when one is missing, so the mass symbols have to be listed here
    # as well as appearing in the propagators.
    symbols = mass_symbols(topo)
    values += [f"{sym}->1/{MASS_POINT[i]}" for i, sym in enumerate(symbols)]

    text = "\n".join(
        [
            "LoopMomenta={" + ",".join(topo.loop_momenta) + "};",
            "ExternalMomenta={" + ",".join(k) + "};",
            "Propagators={" + ",".join(props) + "};",
            f"Kinematics={text_rules};",
            "GenericPoint={" + ",".join(values) + "};",
            "GenericD={d->1/137}",
            "",
        ]
    )
    invented = isps[len(topo.isps) :]
    if invented:
        notes.append("added ISPs to complete the basis: " + ", ".join(invented))
    if symbols:
        notes.append("masses carried through to NeatIBP: " + ", ".join(symbols))
    return text, notes, isps


def config_text(name: str) -> str:
    singular, spasm = where("singular"), where("spasm")
    return f"""(* generated by feynman_agent *)
kinematicsFile = workingPath<>"kinematics.txt";
targetIntegralsFile = workingPath<>"targetIntegrals.txt";

SingularApp = "{singular}";
SparseRREF`SpaSMLibrary = "{spasm}";

ReductionOutputName="{name}";
outputPath=Automatic;

OptionSimplification = 12;
FiniteFieldModulus=42013;
AzuritinoIntersectionDegreeBound=0
NeatIBPIntersectionDegreeBound=5

IntegralOrder = "MultiplePropagatorElimination"
NeedSymmetry=True;
CutIndices={{}};

MIFromAzuritino=True;
CriticalPointInAzuritino=True;

MemoryUsedLimit=Infinity;
ThreadUsedLimit=Infinity;
DeleteSingularTempFiles=True;
"""


def target_text(topo: Topology, isps: list[str]) -> tuple[str, str]:
    """targetIntegrals.txt, and the numerator's decomposition in words.

    The scalar integral is the corner of the top sector plus one dotted
    propagator. A numerator is instead the terms it expands to over the basis,
    negative entries standing for the ISP powers upstairs, and the second value
    says so: ``1/2 G[...] - 1/2 G[...]``. See numerator.py.
    """
    if topo.numerator:
        terms = numerator.expand(topo, isps, invariants(topo.ext_momenta))
        vectors = [v for _, v in terms]
        expansion = numerator.combination(terms)
    else:
        corner = [1] * len(topo.propagators) + [0] * len(isps)
        vectors = [corner, [2] + corner[1:]]
        expansion = ""
    return "{\n" + ",\n".join(g_string(v) for v in vectors) + "\n}\n", expansion


def output_name(topo: Topology) -> str:
    """NeatIBP uses this as a directory name, so keep it filesystem-safe.

    Topology names carry Nickel indices like ``nickel:e12|e3|e3|e|``; the pipes
    and colon silently break the output path.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", topo.name or "topo").strip("_")
    return safe or "topo"


def clear_output(out: Path) -> tuple[bool, str]:
    """Make room for a new run in ``out``, or refuse to. Returns (ok, note).

    Finding an output directory in the way, NeatIBP asks whether it may delete
    it — on stdin, which nothing here is listening to, so the run stops dead
    with the question sitting in a captured pipe. The answer has to be given in
    advance, and only one case has an unambiguous one: a run that finished,
    which is exactly what asking for the same integral twice leaves behind.

    Without the tag the directory may belong to a mission that is still going.
    That is the case NeatIBP's own prompt warns about at length, so it is left
    alone and the caller is told why.
    """
    if not out.is_dir():
        return True, ""
    if not (out / FINISHED_TAG).exists():
        return False, (
            f"an unfinished NeatIBP output directory is already in {out}; it may "
            "belong to a run that is still going, so it was left untouched"
        )
    shutil.rmtree(out)
    return True, f"cleared the finished NeatIBP run already in {out}"


def inputs(topo: Topology) -> tuple[dict[str, str], list[str], str]:
    """The three files NeatIBP reads, by name; what building them involved; the
    numerator's decomposition, if there was one. Pure, so that a run that never
    starts a kernel still has all three."""
    kin, notes, isps = kinematics_text(topo)
    targets, expansion = target_text(topo, isps)
    files = {
        "kinematics.txt": kin,
        "config.txt": config_text(output_name(topo)),
        "targetIntegrals.txt": targets,
    }
    return files, notes, expansion


def write_inputs(files: dict[str, str], workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (workdir / name).write_text(text)


# --------------------------------------------------------------------------
# Result parsing
# --------------------------------------------------------------------------


def parse_summary(text: str) -> IBPResult:
    res = IBPResult(summary=text)
    m = re.search(r"Total MI number:\s*(\d+)", text)
    n_mi = int(m.group(1)) if m else 0
    m = re.search(r"Total IBP number:\s*(\d+)", text)
    res.n_ibp = int(m.group(1)) if m else 0
    m = re.search(r"Total integral number:\s*(\d+)", text)
    res.n_integrals = int(m.group(1)) if m else 0
    block = re.search(r"-+MIs-+\s*\n(.*?)\n-+", text, re.S)
    if block:
        res.masters = re.findall(r"G\[[^\]]*\]", block.group(1))
    for sec, n in re.findall(r"sector (\{[^}]*\})\s*:\s*(\d+) MI", text):
        res.sectors[sec] = int(n)
    res.ok = bool(res.masters) or n_mi > 0
    return res


def singular_env() -> dict:
    """Singular ships its own shared libraries; put them on the loader path.

    Without this the binary dies on a missing libflint and NeatIBP's poll loop
    waits for a sector that will never finish, so the run hangs until timeout.
    """
    root = where("singular").parent.parent
    dirs = [d for d in (root / "lib", root.parent / "tmp" / "lib") if d.is_dir()]
    env = dict(os.environ)
    if dirs:
        env["LD_LIBRARY_PATH"] = ":".join(
            [str(d) for d in dirs] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
        )
    return env


def singular_works() -> bool:
    """Run a trivial Singular script — a broken Singular hangs NeatIBP, not fails it."""
    singular = where("singular")
    if not singular.exists():
        return False
    try:
        proc = subprocess.run(
            [str(singular), "-q", "-c", "int i=1+1; i; quit;"],
            capture_output=True,
            text=True,
            timeout=30,
            env=singular_env(),
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "2" in proc.stdout


def available() -> bool:
    return (
        shutil.which("math") is not None
        and where("neatibp").joinpath("run.sh").exists()
        and where("spasm").exists()
        and singular_works()
    )


def _mock(topo: Topology, reason: str) -> IBPResult:
    """Replay the recorded double-box run, or state plainly what is unknown."""
    summary = recorded() / "summary.txt"
    if topo.nickel() == "e12|e3|34|5|e5|e|" and summary.exists():
        res = parse_summary(summary.read_text())
        res.mocked = True
        res.error = f"{reason}; replayed recorded NeatIBP run from {summary}"
        return res
    return IBPResult(
        mocked=True,
        ok=False,
        error=(
            f"{reason}; no recorded run for Nickel {topo.nickel()}, so the master "
            "integrals are unknown. Re-run with mode='real' to compute them."
        ),
    )


def run(
    topo: Topology,
    workdir: Path,
    mode: str = "auto",
    timeout: int = 1800,
) -> IBPResult:
    """Run NeatIBP on ``topo``. mode is 'auto', 'real' or 'mock'."""
    if not topo.propagators:
        topo = assign_momenta(topo)
    files, notes, expansion = inputs(topo)
    if mode == "mock":
        res = _mock(topo, "mock mode")
    elif mode == "auto" and not available():
        why = "Singular is not runnable" if not singular_works() else "NeatIBP not available"
        res = _mock(topo, why)
    else:
        res = _real(topo, workdir, files, notes, timeout)
    res.expansion = expansion
    return res


def _real(topo: Topology, workdir: Path, files: dict, notes: list[str], timeout: int) -> IBPResult:
    """Write the inputs, start the kernel, wait; a mock stands in for any failure."""
    write_inputs(files, workdir)
    out = workdir / "outputs" / output_name(topo)
    ok, note = clear_output(out)
    if not ok:
        res = _mock(topo, note)
        res.workdir = str(workdir)
        return res
    notes += [note] if note else []
    proc = subprocess.Popen(
        ["bash", str(where("neatibp") / "run.sh"), str(workdir / "config.txt")],
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=singular_env(),
        # A session of its own, so giving up can take the whole tree with
        # it. ``run.sh`` starts a chain of Mathematica kernels, and killing
        # the shell alone leaves them running: one such kernel outlived its
        # agent by half an hour on this machine, still waiting on a prompt.
        start_new_session=True,
        # Every prompt NeatIBP has is an ``InputString`` on stdin, and an
        # inherited stdin is the user's terminal: the kernel blocks there
        # until the timeout — half an hour of apparent silence — for a
        # question nobody is shown, stdout being captured. At end of file it
        # gets ``EndOfFile`` instead, which Mathematica compares to neither
        # "y" nor anything else, so the surrounding ``If`` takes no branch
        # and the run simply carries on. That is why the output directory is
        # cleared above rather than left for NeatIBP to ask about: the
        # question is now unanswerable, so it must not be worth asking.
        stdin=subprocess.DEVNULL,
    )
    try:
        log, errors = proc.communicate(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as stopped:
        os.killpg(proc.pid, signal.SIGKILL)  # its own session, so pid == group
        proc.communicate()
        if isinstance(stopped, KeyboardInterrupt):
            raise
        res = _mock(topo, f"NeatIBP exceeded the {timeout}s timeout")
        res.workdir = str(workdir)
        return res

    summary = out / "results" / "summary.txt"
    if not summary.exists():
        res = _mock(topo, "NeatIBP produced no summary")
        res.workdir = str(workdir)
        res.log_tail = (log + errors)[-1500:]
        return res
    res = parse_summary(summary.read_text())
    res.workdir = str(workdir)
    res.log_tail = (log + errors)[-800:]
    res.notes = notes
    return res
