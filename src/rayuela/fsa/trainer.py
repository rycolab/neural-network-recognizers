

class EM:

	pass


class EMFSA(EM):

	def __init__(self, fsa):
		self.fsa = fsa


class MLEFSA:

	def __init__(self, fsa):
		self.fsa = fsa


V = set(["a", "b", "c"])
n = 3

from rayuela.fsa.examples import NGram
fsa = NGram().ngram(V, n)
print(fsa)


	