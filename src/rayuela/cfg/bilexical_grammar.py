from aflt.cfg.grammar import Production, CFG

class LexicalNT:

	def __init__(self, name):
		self.name = name

	def __repr__(self):
		return self.name

	def __eq__(self, other):
		return self.name == other

	def __hash__(self):
		return hash(self.name)


class BilexicalGrammar(CFG):

	def __init__(self, Sigma, N):
		super().__init__(Sigma, N)


class TemplatedBilexicalGrammar:

	def __init__(self, Sigma):
		self.Sigma = Sigma


	def ground(self):
		pass


class Collins(TemplatedBilexicalGrammar):

	def __init__(self, Sigma):
		super().__init__(Sigma)

	def ground_mono(pair, x):
		nt = LexicalNT(x)
		p = Production(nt, x)

	def ground_pair(self, x, y):
		nt1 = LexicalNT(x)
		nt2 = LexicalNT(y)

		# right arc
		pr = Production(nt1, [nt1, nt2])
				
		# left arc
		pl = Production(nt2, [nt1, nt2])
		
		return (pr, pl)				



