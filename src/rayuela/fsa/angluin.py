"""This implements a weighted version of Angluin's algorithm for
learning finite-state automata from examples.
"""

from rayuela.base.symbol import Sym, ε
from rayuela.base.string import String
from rayuela.fsa.fsa import FSA
from rayuela.fsa.hankel import Hankel
from copy import deepcopy
from random import shuffle

from rayuela.fsa.transformer import Transformer

class Oracle:

    def __init__(self, fsa: FSA) -> None:
        self._fsa = fsa
        self.alphabet = self._fsa.Sigma
        self.prev_ineq = []

    def membership_query(self, string: str):
        return self._fsa.accept(string)

    def equivalence_query(self, guess: FSA, words):
        # Goal will be to check equivalence and if not,
        # then output a string that is accepted by one but not the other
        def equals(R, fsa1, fsa2):
            return Transformer.equivalent_nfa(fsa1, fsa2)
        guess.Sigma = self._fsa.Sigma

        if not equals(self._fsa.R, self._fsa, guess):
            shuffled_alph = deepcopy(list(self.alphabet))
            shuffle(shuffled_alph)
            if ε in shuffled_alph:
                shuffled_alph.remove(ε)
            for string in shuffled_alph:
                string = str(string)
                if guess.accept(string) != self.membership_query(string):
                    return False, string
            ctr = 1
            strings = deepcopy(list(self.alphabet))
            while True:
                curr_strings = []
                ctr += 1
                shuffle(shuffled_alph)
                shuffle(strings)
                if ε in strings:
                    strings.remove(ε)
                if ε in shuffled_alph:
                    shuffled_alph.remove(ε)
                for a in shuffled_alph:
                    new_strings = [str(symbol) + str(a) for symbol in strings]
                    curr_strings.extend(new_strings)

                    for string in new_strings:
                        if String(string) in words:
                            continue

                        # solution 1
                        if guess.accept(string) != self.membership_query(string) and string not in self.prev_ineq:
                            self.prev_ineq.append(string)
                            return False, string
                strings = curr_strings
        else:
            return True, None



def l_star(R, oracle):
    hankel = Hankel(oracle.alphabet, R)
    hankel.fill_table(oracle)

    while True:     
        closed, closed_proof = hankel.closed()
        if not closed:
            print("Not closed. Adding " + str(closed_proof) + " to prefixes")
            hankel.add_prefix(closed_proof)
            hankel.fill_table(oracle)
            continue

        consistent, consistent_proof = hankel.consistent1()
        if not consistent:
            print("Not consistent. Adding " + str(consistent_proof) + " to suffixes")
            string = String([Sym("")])
            for symbol in str(consistent_proof):
                string = String(str(string) + str(symbol))
                hankel.add_suffix(consistent_proof)
            hankel.fill_table(oracle)
            print(hankel.observation_table)
            continue

        # Closed and consistent means we guess a FSA
        guess = hankel.guess()
        equal, proof = oracle.equivalence_query(guess, hankel.S_Sigma)
        if not equal:
            print("Not equal. Adding " + str(proof) + " to table")
            # Adding to prefixes
            string = String([Sym("")])
            for symbol in proof:
                string = String(str(string) + str(symbol))
                hankel.add_prefix(string)
            string = String([Sym("")])
            hankel.fill_table(oracle)
        else:
            return guess
