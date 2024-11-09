from collections import defaultdict as dd

from rayuela.base.semiring import Boolean
from rayuela.base.misc import _random_weight as rw
from rayuela.base.misc import straight
from rayuela.base.symbol import Sym, ε

from rayuela.cfg.nonterminal import NT, S, Slash, Other, Triplet
from rayuela.cfg.production import Production
from rayuela.cfg.cfg import CFG
from rayuela.fsa.fsa import FSA
from rayuela.cfg.labeled_cfg import LabeledCFG
from rayuela.cfg.transformer import Transformer

class X(NT):
	def __init__(self, index):
		self.index = index
		super().__init__(f"X({index})")

class Xbar(NT):
	def __init__(self, index):
		self.index = index
		#super().__init__(f"X̿({index})")
		super().__init__(f"Y({index})")


class Dependency:
	pass


class ArcEager(Dependency):

	def __init__(self, N, R=Boolean):
		self.N, self.R = N, R
		self.T = Transformer()


		#print("CFG")
		self.cfg = self._make_cfg()

		ncfg = CFG(R=R)
		for p, w in self.cfg.P:
			ncfg.add(w, p.head, *p.body)
		self.cfg = ncfg

		s = straight(list(map(str,range(1, self.N+1))), R)
		self.cfg = self.cfg.intersect_fsa(s).trim()


		print("CFG")
		print(self.cfg)
		exit(0)

		self.cfg = self._speculate1(self.cfg).trim()
		self.cfg = self._elim_X(self.cfg)
		print("Speculated CFG")
		#self.cfg = self.T.unaryremove(self.cfg).trim()

		print(self.cfg)

		#print(); print()

		exit(0)

		self.cfg = self._speculate1(self.cfg)
		#self.cfg = self._speculate2(self.cfg)	

		self.cfg = self.T.unaryremove(self.cfg).trim()
		#P = self.remove_illicit(self.cfg)
		#for p in P:
		#	del self.cfg._P[p]

		#print("Speculated")
		#print(self.cfg)
		#print(); print()
		#exit(0)

		#self.cfg = self._unfold_other(self.cfg)
		#self.cfg = self._unfold()
		#self.cfg = self._elim_X(self.cfg)
		#self.cfg.make_unary_fsa()

		print("CFG")
		print(self.cfg)
		exit(0)

		ncfg = LabeledCFG(R=self.R)
		for p, w in self.cfg.P:
			ncfg.add(self.R.one, (0, 0, 0), p.head, *p.body)
		ncfg.make_unary_fsa()
		self.cfg = ncfg

		#print("CFG")
		#print(self.cfg)
		#exit(0)
		#self.cfg = self._unfold(self._unfold_other(self.cfg))
		#print(); print()
		#print("CFG 2")
		#print(self.cfg)
		#exit(0)
		#self.cfg = self._unfold_other(self.cfg)

		#print("Unfolded")
		#print(self.cfg)
		#print(); print()

		#self.cfg = self._unfold_shift(self.cfg)
		#print("Unfold Shift")
		#print(self.cfg)

		#exit(0)

		ncfg = LabeledCFG(R=self.R)
		for p, w in self.cfg.P:
			if len(p.body) == 2:
				one, two = p.body[0], p.body[1]

				# right arc
				if "/" not in str(one) and "/" not in str(two):
					n1 = str(one).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					n2 = str(two).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					ncfg.add(self.R.one,  (n2, n1, "r"), p.head, *p.body)

				# shift
				elif "/" in str(one) and "/" not in str(two):
					ncfg.add(self.R.one, None, p.head, *p.body)

				# left arc
				else:
					n1 = str(one).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					n2 = str(two).split("/")[0].replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					ncfg.add(self.R.one,  (n1, n2, "l"), p.head, *p.body)

			elif len(p.body) == 1:
				if "/" in str(p.head):
					n1 = str(p.head).split("/")[0].replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					n2 = str(p.head).split("/")[1].replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					n3 = str(p.body[0]).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")

					ncfg.add(self.R.one, (n1, n2, "l"), p.head, *p.body)

				elif str(p.head) == "S":
					n = str(p.body[0]).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					ncfg.add(self.R.one,  (-1, n, "l"), p.head, *p.body)

				else:
					ncfg.add(self.R.one, None, p.head, *p.body)

			elif len(p.body) == 3:
				one, two = p.body[1], p.body[2]

				# right arc
				if "/" not in str(one) and "/" not in str(two):
					n1 = str(one).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					n2 = str(two).replace("(", "").replace(")", "").replace("X", "").replace("Y", "")
					ncfg.add(self.R.one,  (n2, n1, "r"), p.head, *p.body)
				else:
					exit("AAH")


			else:
				ncfg.add(self.R.one,  None, p.head, *p.body)

		#self.cfg = ncfg
		#self.cfg.make_unary_fsa()

	def remove_illicit(self, cfg):
		R = set()
		for n in range(2, self.N+1):
			p = Production(X(n), (X(n)/X(n), ~X(n)))
			assert p in cfg._P
			R.add(p)
		return R

	def _elim_X(self, cfg):

		targets = set()
		for p, w in cfg.P:
			if len(p.body) == 1 and isinstance(p.body[0], NT) and (not isinstance(p.head, Slash) or p.head == S):
				if isinstance(p.body, Triplet) and not isinstance(p.body[0].X, Sym):
					continue
				targets.add(p)

		print(); print()
		print("targets")
		for p in targets:
			print(p)
		print(); print()
		#input()

		for p in targets:			
			cfg = self.T.elim(cfg, p)
		
		for p, w in cfg.P:
			assert p not in targets, f"production {p} is still in the grammar"
			print("target", p)
		print("grammar")
		print(cfg)
		exit(0)

		return cfg

	def _unfold_shift(self, cfg):

		targets = []
		for p, w in cfg.P:
			if isinstance(p.body[0], Slash):
				targets.append(p)

		for p in targets:			
			cfg = cfg.unfold(p, 1)

		return cfg

	def _unfold_other(self, cfg):

		heads = set([])
		for head in cfg.V:
			if isinstance(head, Other):
				heads.add(head)

		for p, w in cfg.P:
			if p.head in heads:
				cfg = self.T.elim(cfg, p)
		return cfg.trim()

	def _unfold(self, cfg):

		for n in range(1, self.N+1):
			p = Production(Xbar(n), tuple([X(n)]))
			cfg = self.T.elim(cfg, p)
		return cfg

	def _speculate1(self, cfg):

		sigma, Xs = {}, set([])
		for p, w in self.cfg.P:
			if len(p.body) == 2:
				T = p.body[1]
				if isinstance(T, Triplet):
					buffer = int(str(T.p))
					head = int(str(T.X).split("(")[1].split(")")[0])
					if buffer + 1 == head:
						sigma[p] = 1
						Xs.add(p.body[1])

			#p1 = Production(X(m), (Xbar(n), X(m)))
			#assert p1 in cfg._P
			#sigma[p1] = 1

			#p2 = Production(X(m), (X(n), X(m)))
			#print(p2)
			#assert p2 in cfg._P
			#sigma[p2] = 1
			
			#Xs.add(X(m))


		return self.T.nullaryremove(self.T.speculate(cfg, Xs, sigma).trim()).trim()

	def _speculate2(self, cfg):

		sigma, Xs = {}, set([])
		for n in range(1, self.N+1):
			#for m in range(n+1, self.N+1):
			if n == self.N:
				continue
			m = n + 1
			p1 = Production(X(m), (Xbar(n), X(m)))
			assert p1 in cfg._P
			sigma[p1] = 1

			p2 = Production(X(m), (X(n), X(m)))
			assert p2 in cfg._P
			sigma[p2] = 1


		return self.T.speculate(cfg, Xs, sigma).trim()

	def _make_cfg(self):

		zero, one  = self.R.zero, self.R.one
		cfg = LabeledCFG(R=self.R)
		add = cfg.add

		for n1 in range(1, self.N+1):
			add(one, None, NT(f"X({n1})", n=n1), Sym(str(n1)))
			add(one, None, NT(f"Y({n1})", n=n1), NT(f"X({n1})", n=n1))
			add(one, (-1, n1, 'l'), NT("S"), NT(f"Y({n1})", n=n1))

		for n1 in range(1, self.N+1):
			for n2 in range(n1+1, self.N+1):
					add(one, (n2, n1, "l"), NT(f"X({n2})", n=n2), NT(f"Y({n1})", n=n1), NT(f"X({n2})", n=n2))
					add(one, (n1, n2, "r"), NT(f"Y({n1})", n=n1), NT(f"Y({n1})", n=n1), NT(f"Y({n2})", n=n2))

		return cfg

class Line(Dependency):
	def __init__(self, N, R):
		self.N, self.R = N, R
		self.fsa = self._make_fsa()

	def _make_fsa(self):
		fsa = FSA(R=self.R)
		for n1 in range(0, self.N):
			# fsa.add_arc(NT(f"X({n1})"), n1, NT(f"X({n1+1})"))
			fsa.add_arc(n1, n1+1, n1+1)

		return fsa

class Collins(Dependency):

	def __init__(self, N, R):
		self.N, self.R  = N, R
		self.cfg = self._make_cfg()
		self.cfg.make_unary_fsa()

	def _make_cfg(self):

		zero, one = self.R.zero, self.R.one
		cfg = LabeledCFG(R=self.R)
		add = cfg.add

		for n1 in range(1, self.N+1):
			add(one, None, NT("S"), NT(f"X({n1})"))
			add(one, None, NT(f"X({n1})"), Sym(str(n1)))
			for n2 in range(n1+1, self.N+1):
				add(one, (n1, n2, "l"), NT(f"X({n1})"), NT(f"X({n1})"), NT(f"X({n2})"))
				add(one, (n2, n1, "r"), NT(f"X({n2})"), NT(f"X({n1})"), NT(f"X({n2})"))

		return cfg


class Eisner(Dependency):

	# See https://oeis.org/A006013

	def __init__(self, N, R):
		self.N, self.R = N, R
		self.cfg = self._make_cfg()
		self.cfg.make_unary_fsa()

	def _make_cfg(self):

		zero, one  = self.R.zero, self.R.one
		cfg = LabeledCFG(R=self.R)
		add = cfg.add

		for n1 in range(1, self.N+1):
			add(one, None, NT(f"X({n1})", n=n1), Sym(str(n1)))
			add(one, None, NT(f"Y({n1})", n=n1), NT(f"X({n1})", n=n1))
			add(one, (-1, n1, 'l'), NT("S"), NT(f"Y({n1})", n=n1))

		for n1 in range(1, self.N+1):
			for n2 in range(n1+1, self.N+1):
					add(one, (n2, n1, "l"), NT(f"X({n2})", n=n2), NT(f"Y({n1})", n=n1), NT(f"X({n2})", n=n2))
					add(one, (n1, n2, "r"), NT(f"Y({n1})", n=n1), NT(f"Y({n1})", n=n1), NT(f"Y({n2})", n=n2))

		return cfg