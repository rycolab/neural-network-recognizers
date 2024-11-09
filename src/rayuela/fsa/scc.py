from collections import deque

from rayuela.base.semiring import Boolean, Real
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.fsa import FSA


class SCC:
    def __init__(self, fsa, single_I=True):
        self.fsa = fsa.single_I() if single_I else fsa.copy()
        self.R = self.fsa.R

    def scc(self):
        """
        Computes the SCCs of the FSA.
        Currently uses Kosaraju's algorithm.

        Guarantees SCCs come back in topological order.
        """
        for scc in self._kosaraju():
            yield scc

    def _kosaraju(self) -> "list[frozenset]":
        """
        Kosaraju's algorithm [https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm]
        Runs in O(E + V) time.
        Returns in the SCCs in topologically sorted order.
        """

        rev_fsa = self.fsa.reverse()
        components = []
        visited = set([])

        for q1 in self.fsa.finish():

            if q1 in visited:
                continue

            component = set([q1])
            queue = deque([q1])

            while queue:
                q2 = queue.pop()
                component.add(q2)
                visited.add(q2)

                for a, j, w in rev_fsa.arcs(q2):
                    if j not in visited:
                        queue.append(j)

            components.append(frozenset(component))

        return components

    def _tarjan(self):
        raise NotImplemented

    def _lehmann(self):
        """
        This is an SCC algorithm based on Lehmann's algorithm.
        It runs in O(V³).
        """

        pathsum = Pathsum(self.fsa)
        W = pathsum.lehmann()
        processed = set([])
        components = []

        for state1 in self.fsa.states:
            if state1 in processed:
                continue

            component = set([state1])

            for state2 in self.fsa.states:
                if (
                    W(state1, state2) == Boolean.one
                    and W(state2, state1) == Boolean.one
                ):
                    component.add(state2)
                    processed.add(state2)

            components.append(frozenset(component))

        return components

    def to_fsa(self, scc, αs):

        F = FSA(R=self.R)

        for i in self.fsa.Q:
            for a, j, w in self.fsa.arcs(i):
                if i in scc and j in scc:
                    F.add_arc(i, a, j, w)
                elif j in scc:
                    F.add_I(j, αs[i] * w)

        for i, w in self.fsa.I:
            if i in scc:
                F.add_I(i, w)

        return F
