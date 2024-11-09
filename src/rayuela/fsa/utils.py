import numpy as np
import random
from itertools import permutations

EPS = 1e-8

def gram_schmidt(U, v):
    """Orthonormalises vector v with respect to set of orthonormal vectors U.

    Args:
        U: A set of orthonormal vectors
        v: The new vector to be orthonormalised

    Returns:
        a vector that is orthonormal to the given set of vectors.
    """
    def project(v, u):
        """projects v onto u"""
        if np.linalg.norm(u) == 0:
            return u
        return np.dot(v,u)/np.dot(u,u) * u

    for u in U:
        v = v - project(v, u)
        assert orthogonal(v, u)
    if np.linalg.norm(v) < EPS:
        return np.zeros_like(v)
    return v/np.linalg.norm(v)

def orthogonal(u, v):
    """Returns true if and only if two vectors are orthogonal."""
    return np.abs(np.dot(u, v)) < EPS

def span_contains(U, v):
    """Returns true if and only if a vector v lies in the span of a set of vectors U."""
    for u in U:
        if np.linalg.norm(u) != 0:
            u = u/np.linalg.norm(u)
    v = gram_schmidt(U, v)
    return orthogonal(v, v)

def swap_index(max, swap_history):
    """
    Args:
        max: max index -> lenght of the string we want to permute

    Returns: two random index numbers
    """

    rn1 = 0
    rn2 = 0
    while rn1 == rn2 or rn1 == rn2-1 or (rn1, rn2) in swap_history or (rn2, rn1) in swap_history:
        rn1 = random.randint(0, max)
        rn2 = random.randint(0, max)

    return rn1, rn2

def number_of_swaps(string):
    unique_letters = len(set(string))  # Count unique letters in the string
    return unique_letters * (unique_letters - 1) // 2


def findCeil(str1, first, l, h):
    """
    This function finds the index of the smallest character
    which is greater than 'first' and is present in str[l..h]
    """
    # initialize index of ceiling element
    ceilIndex = l

    # Now iterate through rest of the elements and find the
    # smallest character greater than 'first'
    for i in range(l + 1, 1 + h):
        if (str1[i] > first and str1[i] < str1[ceilIndex]):
            ceilIndex = i

    return ceilIndex

