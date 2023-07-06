#!/usr/local/bin/env python

# =============================================================================================
# MODULE DOCSTRING
# =============================================================================================

"""
Print YANK citations.

"""

# =============================================================================================
# MODULE IMPORTS
# =============================================================================================

# =============================================================================================
# CITATIONS
# =============================================================================================

citations = [
    "Kai Wang K, John D. Chodera, Yanzhi Yang, and Michael R. Shirts.\n"
    "Identifying ligand binding sites and poses using GPU-accelerated Hamiltonian replica exchange molecular dynamics."
    "\n"
    "J. Comput. Aid. Mol. Des. 27:989, 2013. http://dx.doi.org/10.1007/s10822-013-9689-8",
    ]

# =============================================================================================
# COMMAND DISPATCH
# =============================================================================================


def dispatch(args):
    print("Thank you for using YANK!")
    print("")
    print("Please cite the following:")
    print("")
    for citation in citations:
        print(citation)
        print("")
    # TODO: Also print citations for MultiState, pymbar, and other dependencies.

    return True

