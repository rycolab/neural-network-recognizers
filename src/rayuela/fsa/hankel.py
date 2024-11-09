"""This implements the Hankel matrices and the relevant functions."""
from rayuela.base.symbol import ε
from rayuela.base.string import String
from rayuela.fsa.fsa import FSA, State
from copy import deepcopy
from random import shuffle


class Hankel:

    def __init__(self, alph, R) -> None:
        self.alphabet = alph
        self.S = set([String([ε])])
        self.E = set([String([ε])])
        self.S_Sigma = set(String([s, a]) if s != String([ε]) else String([a])
                           for s in self.S
                           for a in self.alphabet if a != ε) | self.S
        self.observation_table = dict()
        self.normalized_table = dict()
        self.row_sums = dict()
        self.R = R

    def fill_table(self, oracle) -> None:
        for prefix in self.S_Sigma:
            for suffix in self.E:
                total_string = self.addition_eps_handler(prefix, suffix)
                if prefix not in self.observation_table:
                    self.observation_table[prefix] = {suffix : oracle.membership_query(
                        str(String(str(total_string))))}
                else:
                    if suffix not in self.observation_table[prefix]:
                        self.observation_table[prefix][suffix] = oracle.membership_query(
                            str(String(str(total_string))))

        # print(self.observation_table)
        self.compute_row_sums()
        self.normalize()

    def compute_row_sums(self) -> None:
        for prefix in self.S_Sigma:
            curr_sum = self.R.zero
            for suffix in self.E:
                curr_sum = curr_sum + self.observation_table[prefix][suffix]
            self.row_sums[prefix] = curr_sum

    def normalize(self) -> None:
        self.normalized_table = deepcopy(self.observation_table)
        for prefix in self.S_Sigma:
            for suffix in self.E:
                if self.row_sums[prefix] != self.R.zero:
                    # print(self.observation_table[String([ε])], self.normalized_table[String([ε])])
                    self.normalized_table[prefix][suffix] = self.normalized_table[prefix][suffix] / self.row_sums[prefix]
                    # print(self.observation_table[String([ε])], self.normalized_table[String([ε])])
                    # print("---------------------------")
                else:
                    self.normalized_table[prefix][suffix] = self.R.zero

    def add_prefix(self, prefix: str) -> None:
        self.S.add(String(str(prefix)))
        self.S_Sigma.add(String(str(prefix)))
        new_prefixes = set(String(str(prefix) + str(a))
                           for a in self.alphabet if a != ε)
        for new_prefix in new_prefixes:
            self.S_Sigma.add(new_prefix)

    def add_suffix(self, suffix: String) -> None:
        curr_str = ""
        for string in str(suffix):
            curr_str += string
            if String(curr_str) not in self.E:
                self.E.add(String(str(suffix)))
            elif String(string) not in self.E:
                self.E.add(String(str(string)))

    # For closedness (row-wise equality)
    def equals(self, row1: String, row2: String) -> bool:
        return self.normalized_table[row1] == self.normalized_table[row2]

    # For consistency (entry-wise equality)
    def single_equals(self, row1: String, row2: String, suffix: String) -> bool:
        if suffix == String(""): suffix = String([ε])
        if row1 == String(""): row1 = String([ε])
        if row2 == String(""): row2 = String([ε])
        return self.normalized_table[row1][suffix] == self.normalized_table[row2][suffix]

    def closed(self) -> bool and String:
        for prefix in self.S_Sigma - self.S:
            flag = False
            for s in self.S:
                if self.equals(prefix, s):
                    flag = True
                else:
                    example = prefix
            if not flag:
                return False, example
        return True, None

    def consistent1(self) -> bool and String:
        for prefix_id in range(len(self.S)):
            for other_prefix_id in range(1 + prefix_id, len(self.S)):
                prefix = list(self.S)[prefix_id]
                other_prefix = list(self.S)[other_prefix_id]
                if self.equals(prefix, other_prefix):
                    for symbol in list(self.alphabet):
                        symbol = String(str(symbol))
                        for suffix in self.E:
                            if suffix == "": suffix = String([ε])
                            if not self.single_equals(String(str(prefix) + str(symbol)),
                                                          String(str(other_prefix) + str(symbol)),
                                                          suffix):
                                return False, String(str(symbol) + str(suffix))
        return True, None
    
    # def consistent(self) -> bool and String:
    #     for prefix_id in range(len(self.S)):
    #         for other_prefix_id in range(1 + prefix_id, len(self.S)):
    #             prefix = list(self.S)[prefix_id]
    #             other_prefix = list(self.S)[other_prefix_id]
    #             if self.equals(prefix, other_prefix):
    #                 shuffled_alph = deepcopy(list(self.alphabet))
    #                 shuffle(shuffled_alph)
    #                 for symbol in shuffled_alph:
    #                     if symbol == ε:
    #                         continue
    #                     for suffix in self.E:
    #                         if suffix == "": suffix = String([ε])
    #                         if (prefix == String([ε]) and other_prefix == String([ε])):
    #                             if not self.single_equals(String(str(symbol)),
    #                                                       String(str(symbol)),
    #                                                       String(str(suffix))):
    #                                 if suffix == String([ε]) and symbol not in self.E:
    #                                     return False, symbol
    #                                 else:
    #                                     if String(str(symbol) + str(suffix)) not in self.E:
    #                                         return False, String(str(symbol) + str(suffix))
    #                         elif prefix == String([ε]):
    #                             if not self.single_equals(String(str(symbol)),
    #                                                       String(str(other_prefix) + str(symbol)),
    #                                                       String(str(suffix))):
    #                                 if suffix == String([ε]) and symbol not in self.E:
    #                                     return False, symbol
    #                                 else:
    #                                     if String(str(symbol) + str(suffix)) not in self.E:
    #                                         return False, String(str(symbol) + str(suffix))
    #                         elif other_prefix == String([ε]):
    #                             if not self.single_equals(String(str(prefix) + str(symbol)),
    #                                                       String([symbol]),
    #                                                       String(str(suffix))):
    #                                 if suffix == String([ε]) and symbol not in self.E:
    #                                     return False, symbol
    #                                 else:
    #                                     if String(str(symbol) + str(suffix)) not in self.E:
    #                                         return False, String(str(symbol) + str(suffix))
    #                         else:
    #                             if not self.single_equals(String(str(other_prefix) + str(symbol)),
    #                                                       String(str(prefix) + str(symbol)),
    #                                                       String(str(suffix))):
    #                                 if suffix == String([ε]) and symbol not in self.E:
    #                                     return False, symbol
    #                                 else:
    #                                     if String(str(symbol) + str(suffix)) not in self.E:
    #                                         return False, String(str(symbol) + str(suffix))
    #     return True, None
                    

    def guess(self) -> FSA:
        new_fsa = FSA(self.R)
        new_fsa.Sigma = set(self.alphabet)
        states = dict()
        symbols = dict()
        state_sums = dict()

        # Adding initial state
        states[0] = self.normalized_table[String([ε])]
        symbols[String([ε])] = 0
        new_fsa.add_state(State(0))
        new_fsa.set_I(State(0))
        if self.row_sums[String([ε])] != self.R.zero:
            new_fsa.set_I(State(0), self.row_sums[String([ε])])
        if self.normalized_table[String([ε])][String([ε])] != self.R.zero:
            new_fsa.set_F(State(0), self.normalized_table[String([ε])][String([ε])])
        state_sums[0] = self.row_sums[String([ε])]

        # Each additional state will receive incremented number
        no_states = 1
        for string in self.S:
            # # Checking if row is already in the rows
            if self.normalized_table[string] not in states.values():
                states[no_states] = self.normalized_table[string]
                new_fsa.add_state(State(no_states))
                # Checking if the epsilon value is 1 meaning acceptance
                if self.normalized_table[string][String([ε])] != self.R.zero:
                    new_fsa.set_F(State(no_states), self.normalized_table[string][String([ε])])
                no_states += 1
                # Add it to the symbols
            symbols[string] = list(states.keys())[list(states.values()).index(
                self.normalized_table[string])]

        # Going through all other rows and seeing which states they belong to
        # Rows have to be in states due to closedness
        for string in self.S_Sigma - self.S:
            symbols[string] = list(states.keys())[list(states.values()).index(
                self.normalized_table[string])]

        # Adding the arcs from the strings in S by adding symbols within the alphabet
        # to them and seeing which states the FSA will end up with after the transition,
        # if the weights are fit
        for string in self.S:
            for symbol in self.alphabet:
                if symbol != ε:
                    # solution 2
                    if self.check_override(symbols[string], 
                                           symbols[self.addition_eps_handler(string, symbol)], 
                                           symbol, 
                                           new_fsa): 
                        continue
                    new_fsa.Sigma = set(self.alphabet)
                    if string == String([ε]):
                        new_fsa.set_arc(symbols[string],
                                        symbol,
                                        symbols[String(str(symbol))],
                                        self.row_sums[String(str(symbol))] / self.row_sums[String([ε])] if self.row_sums[String([ε])] != self.R.zero else self.row_sums[String(str(symbol))])
                    else:
                        new_fsa.set_arc(symbols[string],
                                        symbol,
                                        symbols[String(str(string) + str(symbol))],
                                        self.row_sums[String(str(string) + str(symbol))] / self.row_sums[String(str(string))] if self.row_sums[String(str(string))] != self.R.zero else self.R.zero)
                else:
                    if self.normalized_table[string][String([ε])] != self.R.zero:
                        if string != String([ε]):
                            new_fsa.set_F(symbols[string], self.normalized_table[string][String([ε])])
                new_fsa.Sigma = self.alphabet
        return new_fsa

    def check_override(self, start, end, symbol, fsa) -> bool:
        arcs = fsa.arcs(State(int(start)))
        for arc in arcs:
            if String(str(arc[0])) == String(str(symbol)) and State(str(arc[1])) == State(str(end)):
                return True
        
        return False
    
    def addition_eps_handler(self, string, addition) -> String:
        if string == String([ε]) or string == "":
            if addition != String([ε]) or addition == "":
                return String(str(addition))
            else:
                return String([ε])
        else:
            if addition != String([ε]) or addition == "":
                return String(str(string) +str(addition))
            else:
                return String(str(string))
    
    def eps_removal_from_strings(self, string: String) -> String:
        trimmed_string = ""
        for symbol in str(string):
            if symbol != "ε":
                trimmed_string += symbol
        
        return String(trimmed_string)

    @property
    def allstrings(self) -> set:
        words = set()
        for prefix in self.S_Sigma:
            for suffix in self.E:
                if (prefix == String([ε]) and suffix == String([ε])):
                    words.add(String([ε,ε]))
                elif prefix == String([ε]):
                    words.add(String(str(suffix)))
                elif suffix == String([ε]):
                    words.add(String(str(prefix)))
                else:
                    words.add(String(str(prefix) + str(suffix)))
        return words
