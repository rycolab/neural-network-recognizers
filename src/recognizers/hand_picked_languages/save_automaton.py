import argparse
import re

import torch

from recognizers.hand_picked_languages.all_strings import all_strings_dfa
from recognizers.hand_picked_languages.empty_set import empty_set_dfa
from recognizers.hand_picked_languages.repeat_01 import repeat_01_dfa
from recognizers.hand_picked_languages.even_pairs import even_pairs_dfa
from recognizers.hand_picked_languages.modular_arithmetic_simple import (
    modular_arithmetic_simple_dfa
)
from recognizers.hand_picked_languages.parity import parity_dfa
from recognizers.hand_picked_languages.cycle_navigation import cycle_navigation_dfa
from recognizers.hand_picked_languages.first import first_dfa
from recognizers.hand_picked_languages.dyck_k_m import dyck_k_m_dfa

def get_automaton(name):
    match name:
        case 'all-strings':
            return all_strings_dfa()
        case 'empty-set':
            return empty_set_dfa()
        case 'repeat-01':
            return repeat_01_dfa()
        case 'even-pairs':
            return even_pairs_dfa()
        case 'modular-arithmetic-simple':
            return modular_arithmetic_simple_dfa()
        case 'parity':
            return parity_dfa()
        case 'cycle-navigation':
            return cycle_navigation_dfa()
        case 'first':
            return first_dfa()
        case _ if (match := re.match(r'^dyck-(\d+)-(\d+)$', name)):
            k = int(match.group(1))
            m = int(match.group(2))
            return dyck_k_m_dfa(k, m)
        case _:
            raise ValueError(f'invalid language name: {name}')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    automaton, alphabet = get_automaton(args.name)

    data = dict(automaton=automaton)
    if alphabet is not None:
        data['alphabet'] = alphabet
    torch.save(data, args.output)

if __name__ == '__main__':
    main()
