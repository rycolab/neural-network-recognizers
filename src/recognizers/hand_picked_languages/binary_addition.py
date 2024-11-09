from .binary_util import (
    BinaryArithmeticOperation,
    LengthRestrictedBinaryArithmeticOperation
)

class BinaryAddition(BinaryArithmeticOperation):

    def operator_str(self):
        return '+'

    def with_length_range(self, length_range):
        return LengthRestrictedBinaryAddition(self, length_range)

class LengthRestrictedBinaryAddition(LengthRestrictedBinaryArithmeticOperation):

    def get_max_x(self, max_z):
        return max_z

    def get_max_y(self, max_z, x):
        return max_z - x

    def compute_z(self, x, y):
        return x + y

    def get_length_weights(self):
        return (1, 1, 1)
