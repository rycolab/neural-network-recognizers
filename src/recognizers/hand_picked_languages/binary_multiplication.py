import math

from .binary_util import (
    BinaryArithmeticOperation,
    LengthRestrictedBinaryArithmeticOperation
)

class BinaryMultiplication(BinaryArithmeticOperation):

    def operator_str(self):
        return '×'

    def with_length_range(self, length_range):
        return LengthRestrictedBinaryMultiplication(self, length_range)

class LengthRestrictedBinaryMultiplication(LengthRestrictedBinaryArithmeticOperation):

    def get_max_x(self, max_z):
        # If x * 0 <= max_z, then x can be infinite.
        return math.inf

    def get_max_y(self, max_z, x):
        # If x * y <= max_z,
        #   if x > 0, then y <= floor(max_z / x) <= max_z / x;
        #   if x = 0, then y can be infinite.
        if x == 0:
            return math.inf
        else:
            return math.floor(max_z / x)

    def compute_z(self, x, y):
        return x * y

    def get_length_weights(self):
        return (1, 1, 2)
