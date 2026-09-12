"""A numerator, as the NeatIBP target integrals it is a combination of.

NeatIBP takes no numerator. An integral is an index vector ``G[n1, ..., nN]``
over a complete basis of N = L(L+1)/2 + L*E propagators, and a propagator
raised to a power in the numerator is a negative entry — which is what every
example the package ships does by hand:

    G[1,1,1,1,1,1,1,-5,0]        ((l1+k4)^2)^5 over the seven lines of the double box

So a momentum in the numerator is first written in the basis. On the double box
``l1.k4 = ((l1+k4)^2 - l1^2 - k4^2)/2 = (D8 - D1)/2``, the legs being massless, and

    l1.k4 / (D1 ... D7)  =  1/2 G[1,1,1,1,1,1,1,-1,0] - 1/2 G[0,1,1,1,1,1,1,0,0].

The algebra is sympy's. Momenta are commuting symbols, so a squared momentum
expands to a polynomial whose degree-two monomials are the dot products — the
trick NeatIBP itself uses to read its propagators — the propagators are then a
linear system in those, and the numerator is substituted through and collected
by powers of the propagators.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from .topology import Topology, g_string, parse_momentum

_PARSE = standard_transformations + (convert_xor,)  # ^ is a power, as in Mathematica
_DOT = re.compile(r"([A-Za-z]\w*)\.([A-Za-z]\w*)")
_SQUARE = re.compile(r"\b([A-Za-z]\w*)\^2")


def expand(topo: Topology, isps: list[str], rules: list[tuple[str, str, str]]) -> list:
    """``topo.numerator`` as ``(coefficient, index vector)`` terms.

    The vector runs over ``topo.propagators + isps``; ``rules`` are the external
    kinematics as ``(a, b, value)`` triples meaning a.b = value.
    """
    moms = topo.loop_momenta + topo.ext_momenta
    sym = {m: sympy.Symbol(m) for m in moms}

    def dot(a: str, b: str) -> sympy.Symbol:
        if a not in moms or b not in moms:
            raise ValueError(f"unknown momentum in {a}.{b}; the momenta are {', '.join(moms)}")
        a, b = sorted((a, b), key=moms.index)
        return sympy.Symbol(f"{a}.{b}")

    def squared(q: str) -> sympy.Expr:
        """``(l1+k1)^2`` as a polynomial in dot products."""
        linear = sum(c * sym[m] for c, m in zip(parse_momentum(q, moms), moms))
        out = 0
        for exps, coeff in sympy.Poly(sympy.expand(linear**2), *sym.values()).terms():
            pair = [m for m, e in zip(moms, exps) for _ in range(e)]  # degree two: two names
            out += coeff * dot(*pair)
        return out

    def scalar(text: str) -> sympy.Expr:
        return parse_expr(text, transformations=_PARSE)

    external = {dot(a, b): scalar(v) for a, b, v in rules}
    forms = [squared(q) - (scalar(topo.masses[i]) if i in topo.masses else 0) for i, q in enumerate(topo.propagators)]
    forms += [squared(q) for q in isps]
    D = sympy.symbols(f"D1:{len(forms) + 1}")
    loops = topo.loop_momenta
    unknowns = [dot(a, b) for a in loops for b in moms[moms.index(a) :]]
    (solution,) = sympy.solve([sympy.Eq(d, f.subs(external)) for d, f in zip(D, forms)], unknowns, dict=True)

    # The numerator as typed, its dot products made atoms before sympy sees it.
    atoms: dict[str, sympy.Symbol] = {}

    def atom(a: str, b: str) -> str:
        atoms[f"sp_{a}_{b}"] = dot(a, b)
        return f"sp_{a}_{b}"

    text = _DOT.sub(lambda m: atom(m.group(1), m.group(2)), topo.numerator)
    text = _SQUARE.sub(lambda m: atom(m.group(1), m.group(1)) if m.group(1) in moms else m.group(0), text)
    num = parse_expr(text, local_dict=atoms, transformations=_PARSE)
    stray = [s.name for s in num.free_symbols if s.name in moms]
    if stray:
        raise ValueError(f"bare momentum {stray[0]!r} in the numerator: write dot products, l1.k4")

    poly = sympy.Poly(sympy.expand(num.subs(external).subs(solution)), *D)
    base = [1] * len(topo.propagators) + [0] * len(isps)
    return [
        (str(coeff).replace("**", "^"), [b - e for b, e in zip(base, exps)])
        for exps, coeff in poly.terms()
    ]


def combination(terms: list) -> str:
    """``1/2 G[...] - 1/2 G[...]``, the way a reduction is written down."""
    parts = []
    for coeff, vec in terms:
        coeff = f"({coeff})" if " " in coeff else coeff  # a sum needs its brackets
        parts.append(g_string(vec) if coeff == "1" else f"{coeff} {g_string(vec)}")
    return " + ".join(parts).replace("+ -", "- ")
