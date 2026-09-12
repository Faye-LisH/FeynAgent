"""A numerator becomes NeatIBP target integrals; every check is by hand-computed algebra."""

from feynman_agent import neatibp, numerator
from feynman_agent.graph import _parse_integrand

# The shipped NeatIBP double box: seven lines, then the two ISPs its authors chose.
DBOX = (
    "LoopMomenta={l1,l2};ExternalMomenta={k1,k2,k4};"
    "Propagators={l1^2,(l1+k1)^2,(l1+k1+k2)^2,(l2-k1-k2)^2,(l2+k4)^2,l2^2,(l1+l2)^2,"
    "(l1+k4)^2,(l2+k1)^2};"
)


def _targets(block: str) -> tuple[list[str], str]:
    topo = _parse_integrand(block)
    _, _, isps = neatibp.kinematics_text(topo)
    text, expansion = neatibp.target_text(topo, isps)
    return isps, expansion, text


def test_a_dot_product_becomes_a_negative_index():
    r"""l1.k4 = ((l1+k4)^2 - l1^2 - k4^2)/2 = (D8 - D1)/2, the legs being massless."""
    isps, expansion, text = _targets(DBOX + "Numerator=l1.k4;")
    assert isps == ["l1+k4", "l2+k1"], "the authors' ISPs, in their order, not invented ones"
    assert expansion == "-1/2 G[0,1,1,1,1,1,1,0,0] + 1/2 G[1,1,1,1,1,1,1,-1,0]"
    assert "G[1,1,1,1,1,1,1,-1,0]" in text and "G[1,1,1,1,1,1,1,0,0]" not in text


def test_a_mass_shifts_the_constant_term():
    """With D1 = l1^2 - msq the same identity picks up msq/2 times the corner."""
    _, expansion, _ = _targets(DBOX.replace("{l1^2,", "{l1^2-msq,") + "Numerator=l1.k4;")
    assert expansion.endswith("- msq/2 G[1,1,1,1,1,1,1,0,0]")


def test_powers_and_products_of_dot_products():
    _, squared, _ = _targets(DBOX + "Numerator=(l1.k4)^2;")
    assert squared == (
        "1/4 G[-1,1,1,1,1,1,1,0,0] - 1/2 G[0,1,1,1,1,1,1,-1,0] + 1/4 G[1,1,1,1,1,1,1,-2,0]"
    )
    # l2^2 is D6 itself, and an external product is a kinematic invariant.
    _, mixed, _ = _targets(DBOX + "Numerator=3*l2^2 + k1.k2;")
    assert mixed == "3 G[1,1,1,1,1,0,1,0,0] + s/2 G[1,1,1,1,1,1,1,0,0]"


def test_a_reducible_numerator_needs_no_isp():
    """One loop, one leg: the basis is the two propagators, and k1^2 = s."""
    isps, expansion, _ = _targets(
        "LoopMomenta={l1};ExternalMomenta={k1};Propagators={l1^2,(l1+k1)^2};Numerator=l1.k1;"
    )
    assert isps == []
    assert expansion == "-1/2 G[0,1] + 1/2 G[1,0] - s/2 G[1,1]"


def test_momenta_have_to_be_dotted():
    """A bare l1 would parse as a free symbol and end up in a coefficient, silently."""
    import pytest

    with pytest.raises(ValueError, match="dot products"):
        _targets(DBOX + "Numerator=l1*k4;")
    with pytest.raises(ValueError, match="unknown momentum"):
        _targets(DBOX + "Numerator=l1.p9;")


def test_the_combination_is_written_like_a_reduction():
    terms = [("1", [1, 1]), ("-1/2", [0, 1]), ("-msq*s/4 + 1/3", [1, 0])]
    assert numerator.combination(terms) == "G[1,1] - 1/2 G[0,1] + (-msq*s/4 + 1/3) G[1,0]"
