#!/usr/bin/env bash
# The massless double box with a numerator, start to finish.
#
# About five minutes: a real NeatIBP reduction (2m39s on this machine) and the
# catalogue lookups for its eight master integrals. The report lands beside
# this script. One master has no reference in Loopedia, so the run would stop
# to ask whether to search arXiv for it; --arxiv no answers no. Say yes, or
# drop the flag to be asked, and the papers found are read as well.
cd "$(dirname "$0")/.."
python -m feynman_agent "$(cat example/double_box_numerator.txt)" --arxiv no -o example/double_box_numerator.md
