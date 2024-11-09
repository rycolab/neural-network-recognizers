import math

from recognizers.automata.reserved import ReservedSymbol
from recognizers.string_sampling.weighted_language import (
    WeightedLanguage,
    LengthRestrictedWeightedLanguage,
    Parse
)

PUSH = 2
POP = 3
MARKER = 4

class StackManipulation(WeightedLanguage):

    def alphabet_size(self):
        return 5

    def symbol_to_str(self, symbol):
        if symbol < 2:
            return str(symbol)
        elif symbol == PUSH:
            return 'PUSH'
        elif symbol == POP:
            return 'POP'
        elif symbol == MARKER:
            return '#'
        else:
            raise ValueError

    def with_length_range(self, length_range):
        return LengthRestrictedStackManipulation(length_range)

class LengthRestrictedStackManipulation(LengthRestrictedWeightedLanguage):

    def __init__(self, length_range):
        super().__init__()
        self.min_length, self.max_length = length_range

    def supports_log_probability(self):
        return False

    def supports_next_symbols(self):
        return True

    def supports_edit_distance(self):
        return False

    def sample(self, generator, include_log_probability, include_next_symbols):
        if include_next_symbols:
            next_symbols = []
        else:
            next_symbols = None
        n_stack = generator.randint(
            max(0, math.ceil((self.min_length - 1) / 2)),
            math.floor((self.max_length - 1) / 2)
        )
        n_push = generator.randint(
            max(0, math.ceil((self.min_length - 2 * n_stack - 1) / 3)),
            math.floor((self.max_length - 2 * n_stack - 1) / 3)
        )
        initial_stack = [generator.randrange(2) for _ in range(n_stack)]
        if include_next_symbols:
            next_symbol_set = [0, 1, POP, PUSH, MARKER]
            for _ in range(n_stack + 1):
                next_symbols.append(next_symbol_set)
        stack = initial_stack.copy()
        num_pushes_generated = 0
        stack_operations = []
        while True:
            allowed_operations = [PUSH]
            if stack:
                allowed_operations.append(POP)
            operation = generator.choice(allowed_operations)
            if operation == PUSH:
                if num_pushes_generated == n_push:
                    break
                pushed_symbol = generator.randrange(2)
                stack_operations.append(operation)
                stack_operations.append(pushed_symbol)
                stack.append(pushed_symbol)
                num_pushes_generated += 1
                if include_next_symbols:
                    next_symbols.append([0, 1])
            elif operation == POP:
                stack_operations.append(operation)
                stack.pop()
            else:
                raise ValueError
            if include_next_symbols:
                next_symbol_set = [MARKER]
                if num_pushes_generated < n_push:
                    next_symbol_set.append(PUSH)
                if stack:
                    next_symbol_set.append(POP)
                next_symbols.append(next_symbol_set)
        if include_next_symbols:
            for symbol in reversed(stack):
                next_symbols.append([symbol])
            next_symbols.append([ReservedSymbol.EOS])
        s = (*initial_stack, *stack_operations, MARKER, *reversed(stack))
        return s, Parse(None, next_symbols)

    def is_negative(self, s, include_edit_distance):
        return not self._is_positive(s), None

    def _is_positive(self, s):
        stack = []
        i = 0
        n = len(s)
        # Read in the initial stack.
        while i < n and s[i] in (0, 1):
            stack.append(s[i])
            i += 1
        # Read the sequence of stack operations and simulate them.
        while i < n and s[i] in (PUSH, POP):
            if s[i] == PUSH:
                i += 1
                if i < n and s[i] in (0, 1):
                    stack.append(s[i])
                    i += 1
                else:
                    return False
            elif s[i] == POP:
                # Reject pops on an empty stack.
                if not stack:
                    return False
                stack.pop()
                i += 1
            else:
                raise ValueError
        # Read the = symbol.
        if i < n and s[i] == MARKER:
            i += 1
        else:
            return False
        # Check that the rest of the string is exactly the resulting stack.
        return s[i:] == tuple(reversed(stack))
