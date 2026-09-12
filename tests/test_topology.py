"""Topology tests.

The Nickel indices asserted here are the published ones, and they were also
cross-checked against Loopedia itself: posting the edge list of the box and the
double box makes Loopedia echo back exactly these strings.
"""

import pytest

from feynman_agent import topology as T

KNOWN = {
    "bubble": (T.Topology(int_edges=[(0, 1), (0, 1)], legs=[0, 1]), "e11|e|", 1, 2),
    "triangle": (T.Topology(int_edges=[(0, 1), (1, 2), (2, 0)], legs=[0, 1, 2]), "e12|e2|e|", 1, 3),
    "box": (
        T.Topology(int_edges=[(0, 1), (1, 2), (2, 3), (3, 0)], legs=[0, 1, 2, 3]),
        "e12|e3|e3|e|",
        1,
        4,
    ),
    "sunset": (T.Topology(int_edges=[(0, 1), (0, 1), (0, 1)], legs=[0, 1]), "e111|e|", 2, 2),
    "double box": (
        T.Topology(
            int_edges=[(0, 1), (0, 2), (1, 3), (1, 5), (2, 3), (3, 4), (4, 5)],
            legs=[0, 2, 4, 5],
        ),
        "e12|e3|34|5|e5|e|",
        2,
        4,
    ),
}


@pytest.mark.parametrize("name", list(KNOWN))
def test_nickel_index(name):
    topo, nickel, loops, legs = KNOWN[name]
    assert topo.nickel() == nickel
    assert topo.n_loops == loops
    assert topo.n_legs == legs


@pytest.mark.parametrize("name", list(KNOWN))
def test_nickel_roundtrip(name):
    _, nickel, _, _ = KNOWN[name]
    assert T.from_nickel(nickel).nickel() == nickel


@pytest.mark.parametrize("name", list(KNOWN))
def test_momentum_routing_roundtrip(name):
    """Routing momenta and rebuilding the graph must return the same topology."""
    _, nickel, _, _ = KNOWN[name]
    routed = T.assign_momenta(T.from_nickel(nickel))
    rebuilt = T.from_propagators(routed.propagators, routed.loop_momenta, routed.ext_momenta)
    assert rebuilt.nickel() == nickel


def test_edge_list_input():
    """Loopedia's own example edge list is the massless bubble."""
    assert T.from_edge_list("[(1,2),(2,3),(2,3),(3,4)]").nickel() == "e11|e|"


def test_double_box_from_neatibp_propagators():
    """The shipped NeatIBP example: 7 propagators plus 2 ISPs, 3 independent momenta."""
    topo = T.from_propagators(
        ["l1", "l1+k1", "l1+k1+k2", "l2-k1-k2", "l2+k4", "l2", "l1+l2", "l1+k4", "l2+k1"],
        ["l1", "l2"],
        ["k1", "k2", "k4"],
    )
    assert topo.nickel() == "e12|e3|34|5|e5|e|"
    assert len(topo.int_edges) == 7  # the two ISPs were identified and dropped
    assert topo.isps == ["l1+k4", "l2+k1"]  # but kept: they are the caller's basis
    assert topo.n_loops == 2 and topo.n_legs == 4


def test_sector_topology_pinches():
    """Setting a propagator power to zero contracts that edge."""
    dbox = T.from_nickel("e12|e3|34|5|e5|e|")
    top = T.sector_topology(dbox, set(range(7)))
    assert top.nickel() == dbox.nickel()

    powers = T.parse_g("G[1,1,0,1,1,0,1,0,0]")
    keep = {i for i, a in enumerate(powers[:7]) if a > 0}
    sub = T.sector_topology(dbox, keep)
    assert len(sub.int_edges) == 5
    assert sub.n_loops == 2  # pinching a tree line does not remove a loop


def test_merge_legs_recovers_the_bubble():
    """Two on-shell legs on one vertex act as one, so ee11|ee| is the bubble."""
    sector = T.from_nickel("ee11|ee|")
    assert sector.n_legs == 4
    assert T.merge_legs(sector).nickel() == "e11|e|"


def test_merge_legs_is_a_noop_when_legs_are_distinct():
    box = T.from_nickel("e12|e3|e3|e|")
    assert T.merge_legs(box).nickel() == box.nickel()


def test_parse_momentum():
    assert T.parse_momentum("l1+k1-k2", ["l1", "l2", "k1", "k2"]) == (1, 0, 1, -1)
    assert T.parse_momentum("-l2", ["l1", "l2"]) == (0, -1)


# --------------------------------------------------------------------------
# Masses
# --------------------------------------------------------------------------


def test_a_loopedia_mass_configuration_lands_on_the_right_propagators():
    """``n11|n|`` is the bubble with two equal-mass lines and off-shell legs.

    The configuration mirrors the Nickel index symbol for symbol, which is what
    puts the two ``1`` slots on the two propagators rather than on the legs.
    """
    topo = T.from_nickel("e11|e|:n11|n|")
    assert topo.nickel() == "e11|e|"  # masses are not part of the graph
    assert topo.masses == {0: "m1^2", 1: "m1^2"}
    assert any("off shell" in n for n in topo.notes)


def test_only_the_slots_facing_a_propagator_become_masses():
    """A configuration on a bigger graph must not slide out of alignment."""
    topo = T.from_nickel("e12|34|34|e|e|:000|11|11|0|n|")
    assert topo.masses == {i: "m1^2" for i in (2, 3, 4, 5)}


def test_a_wildcard_configuration_names_no_masses():
    """``:*`` and ``:*:+1`` are Loopedia query forms, not mass assignments."""
    for text in ("e11|e|:*", "e11|e|:*:+1", "e11|e|:zzz|z|"):
        assert T.from_nickel(text).masses == {}


def test_mass_signature_ignores_naming_and_ordering():
    """Two equal masses are two equal masses, whatever they are called.

    This is what lets a mass assignment survive canonicalisation: the Nickel
    index comes from a minimal relabelling, so nothing that depends on the
    propagator order can be compared across two sources.
    """
    assert T.mass_signature(["m^2", "", "m^2"]) == T.mass_signature(["msq", "msq", ""])
    assert T.mass_signature(["m1^2", "m2^2"]) != T.mass_signature(["m1^2", "m1^2"])
    assert T.from_nickel("e11|e|:n11|n|").mass_signature() == (0, (2,))


def test_pinching_carries_the_masses_of_the_surviving_lines():
    """A sub-sector keeps the masses of the lines it still has, re-indexed."""
    topo = T.from_nickel("e12|e3|34|5|e5|e|:000|00|00|0|0|0|")
    topo.masses = {0: "m^2", 3: "msq", 6: "m^2"}
    sub = T.sector_topology(topo, {0, 3, 5, 6})
    assert sub.masses == {0: "m^2", 1: "msq", 3: "m^2"}


def test_propagators_are_written_the_way_neatibp_reads_them():
    topo = T.assign_momenta(T.from_nickel("e11|e|:n11|n|"))
    assert T.propagator_terms(topo) == ["(-l1+k1)^2-m1^2", "(l1)^2-m1^2"]
    assert T.mass_symbols(topo) == ["m1"]


# --------------------------------------------------------------------------
# Reading the external momenta back off a routed graph
# --------------------------------------------------------------------------


def test_the_external_momenta_are_recoverable_from_the_internal_ones():
    """Routing never stores the legs, but conservation still fixes them."""
    topo = T.assign_momenta(T.from_nickel("e12|e3|34|5|e5|e|"))
    arriving = T.leg_momenta(topo)
    assert sorted(arriving.values()) == sorted(["k1", "k2", "k3", "-k1-k2-k3"])
    assert set(arriving) == set(topo.legs)  # nothing arrives where there is no leg


def test_both_constructors_run_a_line_from_its_first_vertex_to_its_second():
    """The one convention shared by routing and reconstruction.

    ``from_propagators`` learns the graph from a set of endpoint momenta and can
    write each edge either way round. Written the wrong way round it still gives
    the right Nickel index, and every external momentum comes out negated — so
    nothing here would catch it except this.
    """
    rebuilt = T.from_propagators(
        ["l1", "l1+k1", "l1+k1+k2", "l2-k1-k2", "l2+k4", "l2", "l1+l2"],
        ["l1", "l2"],
        ["k1", "k2", "k4"],
    )
    assert sorted(T.leg_momenta(rebuilt).values()) == sorted(["k1", "k2", "k4", "-k1-k2-k4"])


def test_legs_sharing_a_vertex_are_reported_as_their_sum():
    """Two legs on one vertex, and nothing recording how they split the total."""
    merged = T.assign_momenta(T.from_nickel("ee11|e|"))
    assert T.leg_momenta(merged) == {0: "k1+k2", 1: "-k1-k2"}
