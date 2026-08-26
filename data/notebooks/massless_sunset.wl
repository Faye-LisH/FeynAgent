(* Example of how a local Mathematica notebook is registered with the agent.

   The scanner in feynman_agent/localdb.py picks up any file in this folder
   carrying a Nickel index marker, so dropping a .nb or .wl here is enough to
   make it searchable.  Keep the three markers below. *)

name = "massless two-loop sunset";
nickel = "e111|e|";
reference = "arXiv:hep-ph/9711266";

(* Two-loop propagator-type integral with three massless lines. *)
sunset[nu1_, nu2_, nu3_, d_, psq_] :=
  (-psq)^(d - nu1 - nu2 - nu3) *
   Gamma[nu1 + nu2 + nu3 - d] Gamma[d/2 - nu1 - nu2] Gamma[d/2 - nu3] /
    (Gamma[nu1] Gamma[nu2] Gamma[nu3] Gamma[3 d/2 - nu1 - nu2 - nu3]) *
   Gamma[d/2 - nu1] Gamma[d/2 - nu2] / Gamma[d - nu1 - nu2];
