from collections import deque

from rayuela.fsa.fsa import FSA
from rayuela.fsa.scc import SCC
from rayuela.base.state import State


class CC:
    def __init__(self, fsa: FSA) -> None:
        self.undirected_fsa = fsa.undirected()

        # TODO (Anej): Hacky: preprocess the φ-only FSA to have initial states
        initial_states = set(fsa.Q)
        for q in fsa:
            for a, t, w in fsa.arcs(q):
                # Remove the target state from the states with no incoming edges
                initial_states.discard(t)

        for q in initial_states:
            self.undirected_fsa.add_I(q, fsa.R.one)

        self.fsa = fsa.copy()
        self.R = self.fsa.R
        self.scc = SCC(self.undirected_fsa, single_I=False)

    def cc(self, reverse=False):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        ccs = reversed(list(self.scc.scc())) if not reverse else list(self.scc.scc())
        for cc in ccs:
            yield cc

    def component_to_fsa(self, cc: frozenset[State]) -> FSA:
        new_fsa = FSA(self.fsa.R)
        for q in cc:
            new_fsa.add_state(q)
            for a, t, w in self.fsa.arcs(q):
                new_fsa.add_arc(q, a, t, w)

        initial_states = set(new_fsa.Q)
        for q in new_fsa:
            for a, t, w in new_fsa.arcs(q):
                # Remove the target state from the states with no incoming edges
                initial_states.discard(t)

        for q in initial_states:
            new_fsa.add_I(q, new_fsa.R.one)

        return new_fsa
