# Feynman Integral Agent

A LangGraph agent that identifies a Feynman integral, looks it up, and — when no
reference exists — reduces it to master integrals with NeatIBP and looks *those*
up instead. The agent always generates a structured Markdown/LaTeX report containing the results or, if the evaluation fails, it reports what it tried, where it
got stuck, and what would most likely help.

```bash
python -m feynman_agent 'e12|e3|34|5|e5|e|'    # a report in reports/
python -m feynman_agent.web --open             # the same thing in a browser
```

```
identify ─→ local DB ─→ Loopedia ─→ NeatIBP ─→ expand masters
               ↑  ↑         │                        │
               │  │         └─(in the loop)─→ advance│
               │  └──────── next diagram ────────┘   │
               └────────────  each master ───────────┘
                                              │
                                queue empty ──┴─→ assemble ─→ (ask)
                                                                │
                                     report ←─ read papers ←─ arXiv
                                                    ↑
                                    a catalogue hit cites papers too
```

**The lookup stages are a loop.** A master integral is a Feynman diagram like
any other, so the diagrams NeatIBP reduces to are sent back through the very
same `local_db` and `loopedia` nodes the original integral went through, one at
a time. That is the whole point of reducing: it turns one unknown integral into
several that the catalogues may well already know. A `queue` in the state is
what lets one pair of nodes serve both passes — empty means they are working on
the original integral, otherwise on the master at its head.

The graph short-circuits to the report the moment it has an answer, and
`interrupt()`s before the arXiv search so you decide whether to keep going. Past
that point a paper title is not an answer either, so the last stage opens the
best few submissions and pulls the expressions out of them.

![agent graph](docs/graph.png)

Regenerate that picture — laid out left to right, solid = always runs, dashed =
a router branch:

```bash
python -m feynman_agent.export_graph                       # docs/graph.{png,svg,pdf}
python -m feynman_agent.export_graph out.svg out.pdf       # or name your own
```

The format follows the file suffix — anything the system `dot` supports, plus
`.dot` and `.mmd` (Mermaid) which need no graphviz at all. DOT is generated from
the compiled graph and piped through `dot`, so there is no `pygraphviz` build to
fight; without graphviz installed it writes Mermaid source instead.

## Setup

```bash
source ~/.venv/bin/activate
pip install langgraph langchain-core langchain-anthropic arxiv
```

Only `langgraph` is needed for the core; `langchain-anthropic` is used solely
for the optional model calls, and `arxiv` solely for the last-resort literature
search. Reading papers needs nothing extra — it is `urllib` and `tarfile` — and
neither does the browser front end, which is `http.server`.

### API key (optional, two steps)

A model is used in three places, none of them on the path from a Nickel index to
a result: reading a **diagram image**, rewriting a paper's equations into LaTeX
that renders, and summarising what that paper gives. Without a key the figure
input is unavailable, and the paper reading still reports every file and
equation it found — in the authors' own macros, and without the summary. Put the
key in a project-local `.env` (gitignored), or export it (an exported variable
wins):

```bash
cp .env.example .env      # then edit it
```

### Choosing a provider

Nothing here is Anthropic-specific, so the provider is pluggable. Only the
**figure** step needs a model that reads images; the other two are text.

| `FEYNMAN_AGENT_PROVIDER` | key | default model | reads images | needs |
|---|---|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-opus-5` | yes | `langchain-anthropic` |
| `kimi` | `MOONSHOT_API_KEY` or `KIMI_API_KEY` | `kimi-k3` | yes | `langchain-openai` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` | no | `langchain-openai` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o` | yes | `langchain-openai` |

Kimi (Moonshot) is OpenAI-compatible, so it is reached through `langchain-openai`
pointed at `https://api.moonshot.ai/v1`:

```bash
pip install langchain-openai
printf 'FEYNMAN_AGENT_PROVIDER=kimi\nMOONSHOT_API_KEY=sk-...\n' >> .env
python -m feynman_agent diagram.png
```

#### DeepSeek

Same shape, a different endpoint (`https://api.deepseek.com/v1`):

```bash
pip install langchain-openai
printf 'FEYNMAN_AGENT_PROVIDER=deepseek\nDEEPSEEK_API_KEY=sk-...\n' >> .env
python -m feynman_agent 'e12|e3|34|5|e5|e|'
```

`deepseek-chat` (V3) is the default and `deepseek-reasoner` (R1) is the other
one — `--model deepseek-reasoner`, or `FEYNMAN_AGENT_MODEL`. R1 bills its
thinking against `max_tokens`, so leave the budget below alone if you pick it.

**Neither reads images**, which costs nothing for the two text steps and makes
the figure input impossible. That is said before the upload rather than after:

```
✗ identify: could not read the figure: provider 'deepseek' has no model that
  reads images; set FEYNMAN_AGENT_PROVIDER to one that does (anthropic, kimi,
  openai), or give the diagram as a Nickel index or edge list
```

Since the provider is one variable, a run that needs both is two runs — or one
`--provider` flag, which overrides `.env` for that run only.

#### Picking a Kimi model

Three ways, in increasing precedence: the provider default (`kimi-k3`),
`FEYNMAN_AGENT_MODEL` in `.env` or the environment, and `--model` per run.

```bash
python -m feynman_agent diagram.png --provider kimi --model kimi-k2.5   # this run only
FEYNMAN_AGENT_MODEL=kimi-k2.5 python -m feynman_agent diagram.png       # this shell
```

**Which ids exist depends on your account** — `kimi-latest` is not on every one,
and asking for a model you lack returns a confusing
`404 … Not found the model X or Permission denied`. List yours:

```bash
curl -s https://api.moonshot.ai/v1/models -H "Authorization: Bearer $MOONSHOT_API_KEY"
```

Measured on the account here, reading a plain one-loop box drawing:

| model | reads the box | note |
|---|---|---|
| `kimi-k3` (default) | ✅ `e12\|e3\|e3\|e\|` | best of the four |
| `kimi-k2.5` | ✅ `e12\|e3\|e3\|e\|` | reasoning model — needs the token headroom below |
| `moonshot-v1-128k-vision-preview` | ❌ read it as a bubble | older vision stack |
| `moonshot-v1-8k-vision-preview` | ❌ read it as a bubble | older vision stack |

Note the failure is *silent*: a wrong-but-valid graph canonicalises fine and the
run continues on the wrong integral. Prefer a `k` model.

**Token headroom.** A reasoning model bills its thinking against `max_tokens`,
and this is the setting most likely to cost you a feature silently. At the old
cap of 2000, `kimi-k2.5` spent 1999 tokens thinking and returned an empty
message. Rewriting one page-long equation on `kimi-k3` measured **8000 output
tokens, 7997 of them reasoning** — and the same equation with room to finish
took 5028, of which 4449 were reasoning, and came back correct. The cap is now
16000 (`FEYNMAN_AGENT_MAX_TOKENS`); hitting it is reported as such, naming the
variable, rather than as unparseable output. `FEYNMAN_AGENT_TIMEOUT` (default
600 s) bounds the wall clock of a single call.

Any other OpenAI-compatible endpoint works with `--provider openai` plus
`FEYNMAN_AGENT_BASE_URL`. `FEYNMAN_AGENT_MODEL` overrides the model for any
provider.

The two API families spell an inline image differently — Anthropic takes a
base64 `source` block, OpenAI-compatible ones take a `data:` URL — and the client
emits the right one per provider. That is the only wire difference, which is why
adding a provider is one entry in `PROVIDERS`.

**On Anthropic specifically:** `ANTHROPIC_API_KEY` is the only accepted form.
`langchain-anthropic` passes an empty key when it is unset, so
`ANTHROPIC_AUTH_TOKEN` and `ant auth login` profiles do **not** work.

**SOCKS proxies:** a bare `ALL_PROXY=socks://host:port` makes httpx raise
`Unknown scheme for proxy URL` before any request leaves the machine — it accepts
`socks5://` (with `socksio` installed) but not `socks://`. The client drops an
unparseable `ALL_PROXY` for the duration of the call and restores it afterwards;
`HTTP(S)_PROXY` still applies, so the proxy is still used.

## Use

```bash
python -m feynman_agent 'e11|e|'                       # Nickel index
python -m feynman_agent '[(1,2),(2,3),(2,3),(3,4)]'    # edge list
python -m feynman_agent "$(cat kinematics.txt)"        # NeatIBP kinematics block
python -m feynman_agent diagram.png                    # figure (needs ANTHROPIC_API_KEY)
```

Useful flags: `--neatibp {auto,real,mock}`, `--offline` (cached Loopedia and
papers only), `--arxiv {ask,yes,no}`, `--extract {auto,fetch,off}`,
`--max-papers N`, `--workdir DIR`, `-o/--out FILE`, `--quiet`, and `--reduce` to
run the IBP reduction even when a reference was already found.

There is also a browser front end — [below](#in-a-browser) — which is the same
agent with a drop target for diagrams and the tool paths in a sidebar.

### The report

The report is **Markdown, written to a file** — `reports/<nickel>.md` by
default, so runs accumulate and re-running one integral refreshes its report.
The path is the only thing on stdout, which makes this work:

```
open "$(python -m feynman_agent 'e12|e3|34|5|e5|e|' --quiet)"
```

`-o FILE` picks a different destination and `-o -` prints the report itself,
for piping. A file rather than scrollback because the interesting part is pages
of epsilon expansions: each one appears twice, first as a `$$` block of standard
LaTeX that GitHub, VS Code and the web interface all set as mathematics, and
under it the source the authors wrote, in fenced `latex` with their own line
breaks. Every reference comes out as a link.

It **opens with a drawing of the diagram**, written beside it as
`reports/<nickel>.svg`. Everything below the picture is only
worth reading if the agent identified the right integral, and that is not a
question you want to answer by parsing an edge list. Each line is labelled with
the momentum routed through it — the same momenta listed underneath — legs carry
arrows
because they are all incoming, and a massive line is drawn thick and coloured
with its mass beside it. The picture is generated from the `Topology`, so it
cannot drift from what the agent actually looked up.

Structure: **the figure** → `Identified` → `Result` (or `No direct result`) →
`Master integrals` → `Read from the papers themselves` → `arXiv candidates` →
`What was done` → `problems` → `Likely useful next`. The answer is at the top
and the audit trail below it; a run that found nothing still says what it tried.

### Watching a run

A run takes minutes, so it narrates itself: one line per stage as it finishes,
plus a line from the slow stages *while* they work.

```
  ✓ identify: 2-loop, 4-point, 7 propagators, Nickel e12|e3|34|5|e5|e|
  · local database: 1 entry/entries for this Nickel index but no stored result
  … loopedia: e12|e3|34|5|e5|e|
  ✓ loopedia: 11 record(s) across 11 mass configuration(s)
  … NeatIBP: reducing e12|e3|34|5|e5|e|, 7 propagators — minutes; logs under /tmp/…
  ✓ NeatIBP: 8 master integrals, 676 IBP relations
  ✓ master sectors: 8 masters span 7 distinct diagram(s), queued for lookup
  … loopedia: ee1122|e|e|
  … loopedia: e11|e22|e|          ← the merged-legs retry, live
  ~ assemble: 8 master integrals, 3 resolved by the catalogue loop
```

Progress goes to **stderr**; stdout carries only the path of the report file.
`--quiet` turns the narration off.

From Python:

```python
from feynman_agent import build

graph = build()
out = graph.invoke(
    {"raw_input": "e12|e3|34|5|e5|e|", "trace": [], "problems": []},
    {"configurable": {"thread_id": "1"}},
)
print(out["report"])
```

## In a browser

```bash
python -m feynman_agent.web --open          # http://127.0.0.1:8765
```

```
┌───────────────────┬────────────────────────────────────────────────┐
│ TOOLS             │  e12|e3|34|5|e5|e|                    [ Run ]  │
│  NeatIBP  …       │ ┌────────────────────────────────────────────┐ │
│  Singular …       │ │   Drop a diagram here, or click to choose  │ │
│  SparseRREF …     │ └────────────────────────────────────────────┘ │
│ MODEL             │  ✓ identify — 2-loop, 4-point, 7 propagators   │
│  provider, model, │  · local database — no entry for e12|e3|…      │
│  base URL, key    │  … loopedia: e12|e3|34|5|e5|e|                 │
│ RUN OPTIONS       │  ✓ loopedia — 11 record(s)                     │
│  every CLI flag   │ ─────────────────────────────────────────────  │
│                   │  reports/<nickel>.md          [ Store as .md ] │
│                   │  the report, rendered, equations set as maths  │
└───────────────────┴────────────────────────────────────────────────┘
```

Everything the command line does, plus the two things it cannot:

- **Drop a diagram on it.** The file is uploaded, the input becomes its path and
  the kind becomes `figure` — the same path the CLI takes, minus saving a crop
  somewhere and typing where you put it. A thumbnail of what you dropped stays
  on the page, above the drawing the agent made of what it read — which is the
  comparison you actually want, because a vision model that misreads a diagram
  produces a perfectly valid graph that is the wrong one.
- **The tool paths in a sidebar**, with the model provider, the model, and the
  API key. Switching provider fills in that provider's model and endpoint as
  the placeholders and forgets the last one's — kimi's model at DeepSeek's
  endpoint is a 404, not a run — and a provider that reads no images says so in
  red, next to the drop target it makes useless. Those three paths are the likeliest thing to be wrong on a new
  machine and they fail *quietly* — a Singular that cannot load `libflint` makes
  NeatIBP wait for a sector that never finishes. They are environment variables
  (`FEYNMAN_AGENT_NEATIBP`, `…_SINGULAR`, `…_SPASM`), read fresh on every use, so
  the sidebar changes them without a restart, and each is checked as you type it
  — a path that is not there is marked, rather than becoming a mocked reduction
  twenty minutes later. Blank means *keep what is configured*; the key is
  applied to whichever variable the chosen provider reads, and is never sent
  back to the page.

The run streams the same stage lines the terminal prints, stops on the same
arXiv question — with buttons instead of a prompt — and ends with the report
rendered in place: the diagram inline, references as links, and the equations
set as mathematics by KaTeX. **Store as .md** writes it through the same code the
CLI uses, so it lands in `reports/<nickel>.md` with its `.svg` beside it, or
wherever you name in the box.

No new dependency: it is `http.server` from the standard library, one page, and
three static files. It binds to localhost by default, which is where a process
holding an API key and starting Mathematica belongs.

## Inputs

| Form | Example | How it is read |
|---|---|---|
| Nickel index | `e12|e3|e3|e|` | parsed directly |
| Nickel + masses | `e11|e|:n11|n|` | Loopedia's mass configuration (see below) |
| Edge list | `[(1,2),(2,3),(2,3),(3,4)]` | degree-1 vertices become external legs |
| Integrand / kinematics | `Propagators={l1^2-m^2,(l1+k1)^2,...}` | graph rebuilt from momentum conservation, masses split off |
| Figure | `diagram.png` | Claude proposes an edge list, the code then verifies it |

Everything is canonicalised to a Nickel index, which is the key for every lookup.

**Reconstructing the graph from an integrand** is the interesting one. Each
internal line contributes endpoints `+q` and `−q`, each external leg one incoming
momentum, and a vertex is any set of endpoints summing to zero — so the graph is
an exact cover of all endpoints by `V = P − L + 1` such sets. That is solved
exactly rather than heuristically, and it identifies irreducible scalar products
for free: if no cover exists, entries are dropped until one does. Given the nine
`Propagators` of the shipped NeatIBP double-box example it recovers the 7-line
graph and correctly flags `l1+k4` and `l2+k1` as the two ISPs.

### Specifying masses

A Nickel index names a **graph**, and a graph is not an integral: `e11|e|` is the
massless bubble, the equal-mass bubble and the one-mass bubble, which have
nothing in common as functions. So masses ride alongside the index, in either of
two ways.

**Inline, in an integrand block** — the way NeatIBP writes them:

```
LoopMomenta={l1};
ExternalMomenta={k1,k2};
Propagators={l1^2-msq,(l1+k1)^2-msq,(l1+k1+k2)^2};
```

Each entry is a momentum squared minus a mass term. The term is kept verbatim,
so `m^2` and `msq` both work.

**As a Loopedia mass configuration** — a second Nickel-shaped string after a
colon, one symbol per Nickel symbol:

```
e11|e|:n11|n|                    two equal-mass propagators, off-shell legs
e12|34|34|e|e|:n00|00|00|1|1|    one off-shell leg, two of equal invariant mass
```

`0` and `z` are massless, `n` is a generic nonzero mass, and **a digit labels
masses that are equal to each other**. The slots facing an `e` are external legs,
the rest are propagators. You can paste these straight off Loopedia; its query
forms (`:*`, `:*:+1`) are recognised and ignored.

What changes when masses are given:

- **The local database stops answering with the wrong integral.** Ask for
  `e11|e|:n11|n|` and the stored *massless* bubble is no longer returned as the
  result — it is reported as "1 entry for `e11|e|` but with different masses".
- **Loopedia is filtered.** A graph is indexed once per mass configuration and
  only the first six are fetched, so the matching ones are floated to the front
  and marked in the report; the paper opened for its expressions is then the one
  that evaluated *your* masses. Every record shows its configuration either way.
- **NeatIBP gets the masses**, as an explicit propagator list plus a value for
  each mass symbol at the generic point — it exits rather than guess when a
  scalar variable has no value.

Saying nothing about masses is not the same as asking for the massless case:
silence leaves the catalogue's own ordering alone, while `:zzz|z|` is a request
like any other and gets the same filtering.

## Correctness checks

The Nickel index implementation is cross-validated against Loopedia itself:
posting an edge list makes Loopedia return its own canonical index, and it agrees
on every graph tried — including the double box (`e12|e3|34|5|e5|e|`). The test
suite additionally round-trips each topology through momentum routing and back
(`nickel → momenta → graph → nickel`).

## What each stage actually gives you

- **Local DB** (`data/local_db.json` + `data/notebooks/*.nb|*.wl`) — the only
  stage that can return a closed form. Drop a notebook containing a
  `nickel = "..."` marker into `data/notebooks/` and it becomes searchable.
- **Loopedia** — *a bibliographic index, not a table of results.* A record says
  which paper evaluated the graph, to which order in ε, and sometimes how many
  masters it has. Those records name **arXiv ids**, so the agent does not stop
  at the citation: it goes on to read the papers (below). Cite arXiv:1709.01266
  if you use them.
- **NeatIBP** — generates `kinematics.txt` / `config.txt` / `targetIntegrals.txt`,
  runs the real package, and parses `results/summary.txt`. It completes the
  propagator set to a full scalar-product basis first, adding the
  `L(L+1)/2 + L·E − P` missing ISPs.
- **Master lookup** — every master `G[...]` is mapped back to a graph by
  *pinching* (contracting) the propagators whose power is zero, and each distinct
  one is queued back through the local DB and Loopedia nodes. Pinching often
  strands several on-shell legs on one vertex; since their momenta simply add, an
  unresolved diagram gets one more turn of the loop with its legs merged
  (`ee11|ee|` → `e11|e|`), and the report says when a master was resolved that way.
  If every master comes back known, `assemble` says so as the result — with the
  caveat below.
- **arXiv** — only with your consent, and aimed at the *unresolved masters*
  rather than the original integral: after a reduction the parent is already
  accounted for, so the open question is the sub-diagrams. A query can only
  describe a diagram by loop and leg count (nothing indexes Nickel indices), so
  diagrams sharing those counts share one query instead of repeating an
  identical search — capped at three queries per run.
- **Reading the papers** — a citation is not a result, so up to three papers are
  opened and their results extracted (`--max-papers N` to change it). Every
  diagram gets one before any diagram gets a second, so a reduction with eight
  masters does not spend the whole budget on the first of them, and the report
  says how many candidates it did not open. Each equation is shown twice: once
  rewritten by the model into LaTeX that renders anywhere, and under it the
  source the authors wrote. See below.

### Getting the result out of a paper

A citation is a pointer; this stage hands back expressions. It downloads the
submitted source (`arxiv.org/e-print/<id>`, cached) and takes it apart:

| What | Why it is there |
|---|---|
| **Attached data files** | The result in the only form you can compute with. Modern papers put them under `anc/` and arXiv serves those singly, so the report gives a direct link; papers predating that convention dropped `.txt`/`.m` tables next to the LaTeX, which are found too. Build files and READMEs are excluded — a `CMakeLists.txt` is not a result. |
| **The ancillary README** | Usually a file-by-file key ("*the folder contains the full results up to weight 4*"), which beats any heuristic description. |
| **Result equations** | Display equations are ranked by *what they are written in* — polylogarithms, `G`/`H` functions, zeta values, powers of ε — which identifies a result far better than any keyword in the prose. Two such markers are required, an equation still containing `\int` is penalised as a definition rather than an evaluation, and the report says which file and section each came from. |
| **The sentence naming the files** | "*Explicit results for all integrals, and up to weight 6, can be found in the ancillary files resultA.m and resultE.m*" — the authors' own answer to the question. |
| **Equations you can read** | What comes out of a submission is written in that paper's own macros — `\dps A_{7,1} &=& i \, \ESGamma^3 \lek -\lp q\rp^2 \rek^{-1-3\eps}` — which renders nowhere and reads as nothing. The paper's macro definitions are in the same tarball, so both go to the model, which says whether the equation is a result at all and rewrites it in standard amsmath: `A_{7,1} = i S_\Gamma^3\left[-q^2-i\eta\right]^{-1-3\varepsilon}\left[\frac{1}{4\varepsilon^5} + \dots\right]`. Transcription, never evaluation — and the source stays underneath, where any liberty taken is visible against it. One call per equation, run together; without a key the report prints the authors' LaTeX and says why. |

**Which papers get opened** depends on how the run got here, and the catalogue
route is the common one:

| Arrived from | Papers read | Why those |
|---|---|---|
| a Loopedia hit, or masters resolved by reference | the ids in those records, in catalogue order | a record is a statement that *this paper* evaluated *this graph* — far better aimed than any query |
| the arXiv search (nothing was catalogued) | the top three by abstract | ranking happens before any download, so the budget goes to abstracts promising explicit results |

With a key configured, the model then says in three lines what the paper gives,
where to get it, and what must be checked for it to apply — the extraction
stands without it, and `--offline` silences it along with the downloads.

**One trap worth knowing about:** older hep-ph sources alias the math
environments (`\newcommand{\bea}{\begin{eqnarray}}`) and then write `\bea … \eea`
throughout. Unexpanded there is no `\begin{...}` in the file at all, so Smirnov's
double-box papers — the canonical references for that graph — read as papers
containing no equations. The aliases are resolved before scanning.

**These are candidates, not answers.** Every paper normalises its integrals its
own way, so the prefactor and propagator ordering still have to be matched by
hand; the report says so rather than presenting an extracted equation as *the*
result.

### What "resolved" does and does not mean

When all masters are known, the agent reports that the reduction resolves the
integral **and points at the IBP system rather than printing a closed form**.
Combining masters into the original integral needs the reduction coefficients,
which is the Kira/FiniteFlow stage NeatIBP feeds — so the report names the
`outputs/*/results/` directory instead of claiming an answer it has not
computed.

## Verified end to end

Real runs on this machine, not mocked:

| Topology | Result | Time |
|---|---|---|
| one-loop box `e12\|e3\|e3\|e\|` | 2 masters — `G[1,1,1,1]`, `G[0,1,1,0]` | ~90 s |
| double box `e12\|e3\|34\|5\|e5\|e\|` | 8 masters, 91 IBP relations | ~2 min |

The double box's master **count** matches the reference run shipped with NeatIBP
(also 8); the individual `G[...]` labels differ because the propagator basis and
momentum routing are generated rather than hand-written.

**Paper reading is verified end to end too**, on both routes into it:

- *From the catalogue.* `e12|e3|34|5|e5|e|` — the massless double box — is in
  Loopedia, so its records supply the ids. The run comes back with Smirnov's
  ε-expansion out of hep-ph/9905323 and hep-ph/9907385
  (`K_1(x,ε) = −4/ε⁴ + 5·ln x/ε³ − …`, labelled `eq. K1`, section "Master double
  boxes") together with Gehrmann–Tancredi–Weihs's Mathematica tables from
  1306.6344.
- *From the search.* The three-loop ladder `e12|e3|45|45|e6|6|e|` is in neither
  catalogue, so arXiv is searched; it comes back from Henn–Smirnov–Smirnov
  [1306.2799] with the attached `resultA.m` / `resultE.m`, the authors' sentence
  saying those hold all integrals to weight 6, and the ε-expansions from the
  paper — and correctly noted that the labelled equations give the
  normalisation, not the integrals.

The **figure path is verified end to end too**, against Kimi `kimi-k3`: a plain
matplotlib drawing of a double box was read straight to `e12|e3|34|5|e5|e|`, and
the run went on to return Loopedia's eight references for it (Smirnov
hep-ph/9905323 and the rest). A one-loop box drawing reads as `e12|e3|e3|e|`.
Not every model manages this — see the table above. Everything except the figure
path runs with no key at all.

## NeatIBP: real vs mock

`--neatibp auto` (the default) runs the real package when `math`, SpaSM, and a
**working** Singular are all present, and otherwise mocks. `--neatibp mock` never
starts a kernel: for the double box it replays the recorded run shipped with
NeatIBP, and for anything else it says plainly that the masters are unknown
rather than inventing them. Mock results are flagged as such in the report.

Real runs are slow — a one-loop box takes about 90 seconds; two loops take much
longer.

### Running the same integral twice

NeatIBP refuses to write into an output directory that already exists, and asks
on **stdin** whether it may delete it — a question the agent cannot answer and
the user never sees, since stdout is captured. A second real run of one integral
therefore used to sit silent until the half-hour timeout. The agent now settles
the directory before starting: a run carrying NeatIBP's `NeatIBP_finished.tag`
is cleared and the fact noted in the report, while one without the tag is left
alone and reported, because it may belong to a mission that is still going. The
kernel is also given no stdin at all, so no prompt anywhere in NeatIBP can ever
block a run again.

The same incident left a second trace worth fixing: when the timeout did fire it
killed `run.sh` and left the Wolfram kernel underneath it running, orphaned and
still waiting. NeatIBP is now started in its own session, and a timeout — or a
Ctrl-C — signals the whole group, so giving up takes the kernels with it.

### Singular must be loadable, and the failure is a hang

On this machine `Singular` could not start at all (`libflint.so.25` missing from
the loader path, though the library ships in `Singular4/tmp/lib`). That failure
mode is nasty: NeatIBP logs `Singular running returns $Failed / sector N failed`
but its polling loop keeps waiting for a sector that will never finish, so a
one-loop box **hung for over nine minutes** instead of erroring.

The runner therefore does two things: it puts Singular's bundled libraries on
`LD_LIBRARY_PATH` for the subprocess (`singular_env()`), and it preflights
Singular with a trivial script (`singular_works()`) so a broken install falls
back to the mock immediately, with the reason stated, instead of hanging.

## Limitations

- Kinematics templates cover up to 4 external legs, always on shell. Beyond that
  the agent writes a template and says the invariants need filling in.
- Nickel canonicalisation is exact but brute-force, capped at 9 internal vertices.
- Momentum routing for a bare Nickel index picks the spanning tree that minimises
  propagator complexity. This matters: the naive tree produced propagators like
  `-l1-l2+k1+k3` and made the syzygy computation dramatically slower, where the
  chosen one gives `-l1+k1`, matching the hand-written reference input.
- The figure lays the vertices out on a circle, in the order that leaves the
  fewest lines crossing. That is faithful and it recovers the conventional
  drawing of most families — a one-loop box is a square, the double box a
  hexagon with one chord, which is two squares sharing a rung — but a nonplanar
  graph must cross somewhere, and there a label can end up nudged off its own
  line. Only the integral itself is drawn, not each master.
- **Propagator** masses are modelled; external leg virtualities are not. An
  off-shell configuration is parsed and reported as a note, but the generated
  kinematics still puts every leg on shell, so `n11|n|` and `z11|z|` produce the
  same NeatIBP input.
- Matching a mass configuration against Loopedia compares the configuration
  strings whole when there is one, which settles the legs too. Masses read off an
  integrand block have no such string, and there only the propagator masses can
  be compared.
- Paper reading is pattern matching over LaTeX, so it finds *where* a result is
  and what it is written in — it does not match conventions, and cannot tell
  which of a paper's families is your diagram. Three papers are downloaded per
  run by default (`--max-papers`), three equations shown per paper.
- The rewritten equations are a **model transcribing**, not a verified
  translation. The instruction is to expand the paper's macros and change
  nothing else, and the source is printed underneath so the two can be compared
  — but nothing here checks that they agree. Without a key, offline, or with
  `--extract fetch`, the report shows the authors' LaTeX and says why.
- That step is also the slowest thing in a run that has no reduction in it. A
  reasoning model spends most of its budget *thinking* about a transcription:
  measured at ten to twenty minutes per paper on `kimi-k3`, against seconds on a
  model that does not reason. The calls for one paper run together, so it is the
  slowest equation that sets the time, not their number. `--extract fetch` skips
  the step entirely and `FEYNMAN_AGENT_TIMEOUT` bounds each call.
- The lookup loop spends supersteps: up to six per queued diagram (two lookups
  and an advance, twice if the merged-leg retry fires). LangGraph's default
  `recursion_limit` of 25 would abort a reduction with more than about three
  masters, so `build()` raises it to 250 — pass `build(recursion_limit=...)` to
  change it.

## Tests

```bash
python -m pytest tests/ -q     # 131 tests, offline (NeatIBP mocked, Loopedia cached,
                               # arXiv tarballs built in the test itself)
```

| File | What it covers |
|---|---|
| `test_topology.py` | Nickel canonicalisation, momentum routing, pinching, masses |
| `test_workflow.py` | Routing, the lookup loop, the report, the provider layer, the CLI |
| `test_extract.py` | Tarballs, equations, macros, and the model rewrite around them |
| `test_web.py` | The front end: a real server, its routes, and the arXiv question |
| `test_export_graph.py` | The graph picture |

Nothing calls a model or the network. Where a model would be, `llm.ask` is
replaced by a function that returns what the prompt asked for — which is also
how the reply-in-the-wrong-shape paths get tested.
