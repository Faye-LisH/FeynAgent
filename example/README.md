# Example: the double box with a numerator

The massless two-loop four-point double box, with two powers of `l1.k4` and a
`l2^2` in the numerator. The input is the shipped NeatIBP example's propagator
list — seven lines plus the two irreducible scalar products (ISPs) `(l1+k4)^2`
and `(l2+k1)^2` that complete the basis — with one line of this agent's own:

```
LoopMomenta={l1,l2};
ExternalMomenta={k1,k2,k4};
Propagators={l1^2,(l1+k1)^2,(l1+k1+k2)^2,(l2-k1-k2)^2,(l2+k4)^2,l2^2,(l1+l2)^2,(l1+k4)^2,(l2+k1)^2};
Numerator=(l1.k4)^2 - 3*l2^2;
```

## Run it

```bash
source ~/.venv/bin/activate     # or wherever the requirements are installed
example/run.sh                  # a few minutes; writes example/double_box_numerator.md
```

The script runs the agent from the repository root with `--arxiv no`, so it
never stops to ask a question, and puts the report in this folder with the
diagram beside it. Everything else is the default: NeatIBP runs for real when
its dependencies are found (`--neatibp mock` replays a recorded run instead,
and still computes the expansion below). The run shipped here took about five
minutes, most of it NeatIBP.

## What happens

1. **Identify.** The graph is rebuilt from momentum conservation: seven of the
   nine entries form a two-loop four-point graph, Nickel index
   `e12|e3|34|5|e5|e|`, and the two that fit nowhere are recognised as the
   ISPs — and kept, in the order given, as the basis for what follows.
2. **The catalogues do not stop the run.** The local database has an entry for
   this graph without a stored result, and Loopedia lists it under eleven mass
   configurations. For the scalar integral that would be the answer. With a
   numerator the run goes on to the reduction, because a reference for the
   graph answers a different integral of the same family.
3. **The numerator becomes target integrals.** NeatIBP takes an index vector
   per integral, a power in the numerator being a negative entry, so the
   numerator is written in the propagator basis first. With massless legs
   `l1.k4 = ((l1+k4)^2 - l1^2)/2 = (D8 - D1)/2` and `l2^2 = D6`, so

   ```
   1/4 G[-1,1,1,1,1,1,1,0,0] - 1/2 G[0,1,1,1,1,1,1,-1,0] - 3 G[1,1,1,1,1,0,1,0,0] + 1/4 G[1,1,1,1,1,1,1,-2,0]
   ```

   and that is `targetIntegrals.txt`, in place of the corner integral:

   ```
   {
   G[-1,1,1,1,1,1,1,0,0],
   G[0,1,1,1,1,1,1,-1,0],
   G[1,1,1,1,1,0,1,0,0],
   G[1,1,1,1,1,1,1,-2,0]
   }
   ```

4. **NeatIBP reduces them** to the family's eight master integrals: 2m39s and
   30 IBP relations in the run shipped here.
5. **Each master goes back through the catalogues.** Seven of the eight are
   resolved — two by a stored closed form (the massless sunset notebook in
   `data/notebooks/`), the rest by Loopedia references. The eighth,
   `G[1,0,1,1,0,1,0,0,0]` with graph `ee11|22|ee|`, has no reference, so the
   run would stop here to ask whether to search arXiv for it; `--arxiv no`
   answers no, which the report records as *stop requested*. Run with
   `--arxiv yes` to search, after which the papers found — and the ones cited
   for the other masters — are opened for their results, with a model
   summarising each when an API key is configured (`usage.md`, *API key*).

## The report

`double_box_numerator.md` is the output of one such run, `e12-e3-34-5-e5-e.svg`
the diagram it links to. Under *Identified* the numerator is named and the two
ISPs are noted; *Result* is Loopedia's list of papers for the graph, which is
where the masters are evaluated; *Master integrals* opens with the expansion
above and lists the eight masters, each with where it is known from and the
one unresolved marked `?`; *What was done* records the NeatIBP stage with the
same targets. The reduction coefficients themselves are under the run's
working directory (`/tmp/feynman_agent/e12_e3_34_5_e5_e_/outputs/`, or
`--workdir`), as the report says: combining the reduced targets back into one
expression is the step left to Kira or FiniteFlow.
