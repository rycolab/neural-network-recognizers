
class Symbol:
	pass

class State:
	pass

class Configuration:
	pass

class PDA:

	def __init__(self, Sigma, cfg, semiring=None):

		# alphabet
		# assume that epsilon is 0
		self.Sigma = Sigma

		# underlying CFG
		self.cfg = cfg

		# stack
		self.stack = []

	def read(self, x):
		""" reads in the next input symbol """
		pass