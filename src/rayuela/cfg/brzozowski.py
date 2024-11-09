import copy
import pdb

from rayuela.base.misc import symify
from rayuela.base.symbol import Sym, ε
from rayuela.base.semiring import Boolean, Real, Tropical

from rayuela.cfg.production import Production
from rayuela.cfg.nonterminal import Delta, NT, S
from rayuela.cfg.cfg import CFG
from rayuela.cfg.transformer import Transformer
from rayuela.cfg.treesum import Treesum
from rayuela.base.misc import straight

from rayuela.cfg.parser import Parser

from rayuela.base.misc import _random_weight as rw
from collections import defaultdict as dd

class Brzozowski:
    def __init__(self) -> None:
        self.counter=0


    def derive_S(self ,cfg, S ,idx):
        one = cfg.R.one
        zero = cfg.R.zero
        ncfg = cfg.spawn()

        ecfg, _=cfg.eps_partition()
        U= Treesum(ecfg).table()

        for (p,w) in cfg.P:
            if not p.head==cfg.S:
                ncfg.add(w, p.head, *p.body)

        for ch in S:

            ncfg.add(one , cfg.S , ch ,Δ(cfg.S, ch , idx))
            for (p, w) in cfg.P:
            
                head=Δ(p.head, ch, idx)
                if len(p.body) == 0:
                    pass
                
                elif p.body[0] == ch:
                    ncfg.add(w, head, *p.body[1:])
                    pass 
                
                delta=one 
                for k, X in enumerate(p.body):
                    if X in cfg.V:
                        ncfg.add(delta * w, head, Δ(p.body[k], ch, idx), *p.body[(k+1):])
                    if U[X]==zero:
                        break 
                    delta*=U[X]
        
        ncfg.make_unary_fsa()
        return ncfg


    def derive_a(self, cfg, a, idx):
        """computes selective derivative wrt a"""
        #TO DO: fix numerical issue a remove computation of useless rules
        
        one = cfg.R.one
        zero = cfg.R.zero
        ncfg = cfg.spawn()

        ecfg, _ = cfg.eps_partition()
        U = Treesum(ecfg).table()

        for (p, w) in cfg.P:
            ncfg.add(w, p.head, *p.body)

            
        for (p, w) in cfg.P:
            
            head=Δ(p.head, a, idx)
            delta=one 

            for k, X in enumerate(p.body):
                if X==ε:
                    continue
                elif X==a or X in cfg.V : 
                    ncfg.add(delta * w, head, Δ(p.body[k], a , idx), *p.body[(k+1):])
                if U[X]==zero:
                    break 
                delta*=U[X]


        ncfg.add(one, Δ(a,a,idx),ε)
        ncfg.S=Δ (cfg.S,a,idx)
        
        ncfg.make_unary_fsa()
        return ncfg


    def derive_a_optimized(self, cfg, a, idx):
        """computes selective derivative wrt a
        #This is the optimized version that only computes the derivative wrt accessible
        #terminals"""
        #TO DO: fix numerical issue and remove computation of useless rules
        
        one = cfg.R.one
        zero = cfg.R.zero
        ncfg = cfg.spawn()

        ecfg, _ = cfg.eps_partition()
        U = Treesum(ecfg).table()
        

        for (p, w) in cfg.P:
            ncfg.add(w, p.head, *p.body)

        stack={cfg.S}
        visited=[cfg.S]

        while stack:

            Y= stack.pop()    
            for (p, w) in cfg.P_byhead(Y):
                
                head=Δ(Y, a, idx)
                delta=one 

                for k, X in enumerate(p.body):
                    if X==ε:
                        continue
                    elif X==a or X in cfg.V : 
                        ncfg.add(delta * w, head, Δ(X, a , idx), *p.body[(k+1):])
                        if not X in visited and isinstance(X,NT):
                            stack.add(X)
                            visited.append(X)
                    if U[X]==zero:
                        break 
                    delta*=U[X]


        ncfg.add(one, Δ(a,a,idx),ε)
        ncfg.S=Δ (cfg.S,a,idx)

        ecfg, _ = ncfg.eps_partition()
        U = Treesum(ecfg).table()
        # pdb.set_trace()
        
        ncfg.make_unary_fsa()
        return ncfg



    
    
    def parse(self,cfg , string , return_tree=False):
        """applies Brozowski derivative and retrieves weight of parse"""
        #TO DO: optimize fixed-point

        
        R=cfg.R
        n_cfg=cfg.copy()
        i=1
        for character in string:
            sym=Sym(character)
            n_cfg=self.derive_a_optimized(n_cfg,sym,i)
            i+=1
        
        n_ecfg, _ = n_cfg.eps_partition()
        U = Treesum(n_ecfg).table()
        parse_weight=U[n_cfg.S]

        

        #build Parse Forest
        ζ=R.chart()


        if(return_tree):
            return  ζ , n_cfg
        else:
            return parse_weight 

    def parse_optimal(self,cfg , string , return_tree=False):
        """applies Brozowski derivative and retrieves weight of parse
        computaion of Treesum has been optimized"""
        #TO DO: optimize fixed-point

        
        R=cfg.R
        n_cfg=cfg.copy()
        
        ecfg , _ =n_cfg.eps_partition()
        U=Treesum(ecfg).table()

        i=1
        for character in string:
            sym=Sym(character)
            n_cfg=self.derive_a_memoized(n_cfg,U,sym,i)
            
            self._selective_top_down(n_cfg,U,i) #IN PLACE TO SPEED UP COMPUTATION 
            #pdb.set_trace()
            i+=1
        
        parse_weight=U[n_cfg.S]

        #build Parse Forest
        ζ=R.chart()
        


        if(return_tree):
            return  ζ , n_cfg
        else:
            return parse_weight 



    def derive_a_memoized(self, cfg, U ,a, idx):
        """optimized version with memoized and optimized version of U"""
        #TO DO: fix numerical issue and remove computation of useless rules
        
        one = cfg.R.one
        zero = cfg.R.zero
        ncfg = cfg.spawn()

        for (p, w) in cfg.P:
            ncfg.add(w, p.head, *p.body)

        stack={cfg.S}
        visited=[cfg.S]

        while stack:

            Y= stack.pop()    
            for (p, w) in cfg.P_byhead(Y):
                
                head=Δ(Y, a, idx)
                delta=one 

                for k, X in enumerate(p.body):
                    if X==ε:
                        continue
                    elif X==a or X in cfg.V : 
                        ncfg.add(delta * w, head, Δ(X, a , idx), *p.body[(k+1):])
                        if not X in visited and isinstance(X,NT):
                            stack.add(X)
                            visited.append(X)
                    if U[X]==zero:
                        break 
                    delta*=U[X]


        ncfg.add(one, Δ(a,a,idx),ε)
        ncfg.S=Δ (cfg.S,a,idx)
        
        ncfg.make_unary_fsa()
        return ncfg

    #TO DO: MOVE TO TREESUM

    def _selective_top_down(self,cfg,V,idx,nullable=True):
        """in place function to update the value of treesum (V) for
        non-terminals generated at round idx.Takes advantage of memoization 
        and optimized fixed point"""

        
        
        listeners=dd(set)
        forward={cfg.S}
        visited={cfg.S}
        backward=set()

        while forward:
            Y=forward.pop()
            backward.add(Y)
            for (p, w) in cfg.P_byhead(Y):
                
                for X in p.body:
                    if isinstance(X,Δ) and X.idx==idx and not X in visited:
                        forward.add(X)
                        visited.add(X)
                    if isinstance(X,Δ) and X.idx==idx :
                        listeners[X].add(Y)


        if nullable:
            n_cfg, _ =cfg.eps_partition()
            n_cfg.S=cfg.S #COULD AVOID THIS BY CHANGING eps_partition()
        else:
            n_cfg=cfg

        while backward:
            
            Y=backward.pop()
            _update=V[Y]
            for (p, w) in n_cfg.P_byhead(Y):
                
                _rule_update = w
                for X in p.body:
                    if isinstance(X, NT):
                        _rule_update *= V[X]
                _update+=_rule_update    
                   
            if _update!=V[Y]:
                # pdb.set_trace()
                backward.update(listeners[Y])
                V[Y] += _update
        


        
#TO DO: MOVE TO NON-TERMINAL
class Δ(NT):

	def __init__(self, X, a, idx):
		assert  isinstance(a, Sym)
		super().__init__((X, a))
		self._X, self._a, self._idx = X, a, idx

	@property
	def X(self):
		return self._X

	@property
	def a(self):
		return self._a

	@property
	def idx(self):
		return self._idx

	def _downstairs(self):
		if isinstance(self.X , Δ):
			return self.X._downstairs() + str(self.a)
		else:
			return str(self.a)
		

	def _upstairs(self):
		if isinstance(self.X , Δ):
			return self.X._upstairs()
		else: 
			return str(self.X)
		

	def __repr__(self):
		return f"{self._upstairs()}"+"/"+f"{self._downstairs()}"

	def __hash__(self):
		return hash((self.X, self.a))

    #Attention !! should you add 'idx' as well?
    #You have to make sure that you go through every non-terminal!
	def __eq__(self, other):
		return isinstance(other, Δ) \
			and self.X == other.X \
			and self.a == other.a \
            and self.idx== other.idx
        


def grammar_inspector(cfg):
    for p,w in cfg.P:
        if isinstance(p.head,Δ):
            print(f"{p.head.idx} {p.head} ->{p.body} {w}")


def treesum_inspector(U):
    for key in U.keys():
        if isinstance(key,Δ):
            print(f"{key.idx} {key} : {U[key]}  ")
        else:
            print(f"{key} : {U[key]}")

