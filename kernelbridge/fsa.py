from __future__ import annotations
from abc import abstractmethod

from collections import defaultdict as ddict
from collections import deque

from ply import lex
from ply import yacc

"""
FSA
"""

class RegExError(Exception): pass


class State():
    r"""
    State within Finite State Automaton.
    Hashable object for FSA set/dict attributes.
    Maintains whether or not State is accepting.
    """
    def __init__(self, accept:bool=False, name:str=None):
        self.accept = accept
        self.name = name
        
    def __repr__(self):
        if self.name:
            return self.name
        else:
            return str(hash(self))[-3:]
    
    def __str__(self):
        return repr(self)
    
    def __cmp__(self, other:State) -> bool:
        if self.accept != other.accept:
            return False
        if self.name != other.name:
            return False
        return True
    
    def __gt__(self, other:State) -> bool:
        return str(self) > str(other)
    
    def __lt__(self, other:State) -> bool:
        return str(self) < str(other)
    
    def __ge__(self, other:State) -> bool:
        return str(self) >= str(other)
    
    def __le__(self, other:State) -> bool:
        return str(self) <= str(other)

    def copy(self):
        return State(self.accept, self.name)


class FiniteAutomaton():
    def __init__(self, q0=State(), F=set(), Sigma=None):
        self._q0 = q0
        self._F = F.copy()
        if not Sigma:
            self._Sigma = ddict(lambda: ddict(set))
            self._Sigma[self.q0]
        else:
            self._Sigma = Sigma.copy()
    
    @abstractmethod
    def __call__(self, inp:str) -> bool:
        r"""
        Reads input string and returns resulting final state.
        """
        raise NotImplementedError(NotImplemented)
    
    def __str__(self):
        return str({k: dict(v) for k, v in self.Sigma.items()})
    
    @property
    def q0(self):
        return self._q0
    
    @q0.setter
    def q0(self, s:State):
        # self._Q.add(s)
        self._q0 = s
    
    @property
    def Q(self):
        r"""
        Enumerates all states of the FSA (source nodes of Sigma).
        """
        return self.Sigma.keys()
    
    @property
    def F(self):
        r"""
        Enumerates all accepting states of the FSA.
        """
        return self._F

        # ### Manual enumeration from Sigma
        # if not hasattr(self, "_F_cached"):
        #     self._F_cached = set()
        #     for state, d in self.Sigma:
        #         for t, ss in self.Sigma[state]:
        #             for s in ss:
        #                 if s.accept:
        #                     self._F_cached.add(s)
        # return self._F_cached
    
    @property
    def Sigma(self):
        r"""
        Dict-of-dicts for state transitions of the FSA.
        """
        return self._Sigma
    
    @abstractmethod
    def transition(self, s:State, c:str):
        r"""
        Evaluates a single character transition from a given state,
        returning a corresponding set of destination states.
        """
        raise NotImplementedError(NotImplemented)

    def update_accept(self, s:State, a:bool):
        r"""
        Updates acceptance of State and adds/removes from F accordingly.
        """
        s.accept = a
        if a:
            self._F.add(s)
        elif not a and s in self.F:
            self._F.remove(s)
    
    @abstractmethod
    def update_transition(self, s1:State, s2:State, c:str):
        r"""
        Updates/adds a single-char transition between two states.
        States are added to the FSA's F as necessary.
        """
        raise NotImplementedError(NotImplemented)
    
    @abstractmethod
    def accepts(self, inp:str) -> bool:
        r"""
        Determines whether or not the FSA accepts a given input.
        """
        raise NotImplementedError(NotImplemented)


class NFA(FiniteAutomaton):
    def __init__(self, q0=State(), F=set(), Sigma=None):
        super().__init__(q0.copy(), F, Sigma)

    def __call__(self, inp:str) -> bool:
        curr = self.epsilon_closure(self.q0, include=True)
        for c in inp:
            nxt = set()
            for s in curr:
                nxt = nxt.union(self.transition(s, c))
            curr = nxt
            if not nxt: break
        return curr
    
    def update_transition(self, s1:State, s2:State, c:str):
        if c in {"ϵ", "ep", "epsilon", "Epsilon"}:
            c = ""

        self._Sigma[s1][c].add(s2)
        if s1.accept: self._F.add(s1)
        if s2.accept: self._F.add(s2)
    
    def incorporate(self, other:NFA):
        for s, d in other.Sigma.items():
            for t, ss in d.items():
                self._Sigma[s][t] = self.Sigma[s][t].union(ss)
        self._F = self.F.union(other.F)

    @staticmethod
    def concat(*M):
        r"""
        Iteratively adds ϵ-transition from end state of prev to start state
        of curr, removing acceptance from end state of prev.
        """
        ret = M[0]
        for m in M[1:]:
            for qf in list(ret.F):
                m.update_transition(qf, m.q0, "")
                ret.update_accept(qf, False)
            ret.incorporate(m)
        return ret
    
    @staticmethod
    def union(*M):
        r"""
        Adds ϵ-transition from new start state to start states of each
        machine in input.
        """
        ret = NFA()
        for m in M:
            ret.update_transition(ret.q0, m.q0, "")
            ret.incorporate(m)
        return ret
    
    @staticmethod
    def kleene(m:NFA):
        r"""
        Adds ϵ-transition from end to start, and from new start to old start.
        Removes acceptance from end, adds acceptance to old start.
        """
        for qf in list(m.F):
            m.update_transition(qf, m.q0, "")
            m.update_accept(qf, False)
        m.update_accept(m.q0, True)
        start = State()
        m.update_transition(start, m.q0, "")
        m.q0 = start
        return m
    
    def epsilon_closure(self, s:State, include:bool=False) -> set:
        r"""
        Enumerates the ϵ-closure of a state (all states reachable via
        any number ϵ-transitions, i.e. ϵ*).
        """
        ret = self.Sigma.get(s, dict()).get("", set())
        for ec in [self.epsilon_closure(state) for state in ret]:
            ret = ret.union(ec)
        if include:
            ret = ret.union({s})
        return ret

# TODO: e closure serves transition
#       e closure is a dfs enumeration traversing e edges only
#       

    def transition(self, s:State, c:str):
        if s not in self.Sigma: return set()
        
        ret = set()
        if c in self.Sigma[s]:
            ret = ret.union(self.Sigma[s][c])
        if "" in self.Sigma[s]:
            for s2 in self.epsilon_closure(s):
                ret = ret.union(self.transition(s2, c))
        return ret
    
    def accepts(self, inp:str) -> bool:
        final = self(inp)
        if not final: return False
        for ec in [self.epsilon_closure(s, True) for s in final]:
            final = final.union(ec)
        return not final.isdisjoint(self.F)

    def to_dfa(self) -> DFA:
        nfa_eq0_set = self.epsilon_closure(self.q0, include=True)
        nfa_eq0_list = sorted(list(nfa_eq0_set))
        dfa_q0 = State(
            accept = nfa_eq0_set.intersection(self.F),
            name = "0"
        )
        dfa_F = set()
        dfa_Sigma = ddict(dict)
        dfa_states = {dfa_q0.name: [dfa_q0, False]}
        ct = 1

        state_queue = deque()
        state_queue.append((dfa_q0.name, nfa_eq0_list))
        while state_queue:
            source_key, source_list = state_queue.popleft()
            source_dfa, visited = dfa_states[source_key]
            
            if visited: continue
            
            for u in source_list:
                dest_set = set()
                for t, v_set in self.Sigma[u].items():
                    if t == "" or t in dfa_Sigma[source_dfa]: continue

                    for v in v_set:
                        dest_set = dest_set.union(self.epsilon_closure((v), include=True))
                    dest_list = sorted(list(dest_set))
                    dest_key = "_".join(map(lambda x: str(hash(x)), dest_list))
                    if dest_key not in dfa_states:
                        dfa_states[dest_key] = [
                            State(
                                accept = dest_set.intersection(self.F),
                                name = str(ct)
                            ),
                            False
                        ]
                        ct += 1
                    dest_dfa = dfa_states[dest_key][0]

                    if source_dfa.accept: # redundant to add state to F as source/dest?
                        dfa_F.add(source_dfa)
                    if dest_dfa.accept:
                        dfa_F.add(dest_dfa)
                    dfa_Sigma[source_dfa][t] = dest_dfa

                    state_queue.append((dest_key, dest_list))

            dfa_states[source_key][1] = True
        return DFA(dfa_q0, dfa_F, dfa_Sigma)


class DFA(FiniteAutomaton):
    def __init__(self, q0=State(), F=set(), Sigma=ddict(dict)):
        super().__init__(q0, F, Sigma)

    def __call__(self, inp:str) -> bool:
        curr = self.q0
        for c in inp:
            curr = self.transition(curr, c)
            if not curr: return
        return curr
    
    def __str__(self):
        return super().__str__()
    
    def update_transition(self, s1:State, s2:State, c:str):
        r"""
        Updates/adds a single-char transition between two states.
        States are added to the DFA's F as necessary.
        """
        self._Sigma[s1][c] = s2
        if s1.accept: self._F.add(s1)
        if s2.accept: self._F.add(s2)

    def transition(self, s:State, c:str):
        if s in self.Sigma:
            return self.Sigma[s].get(c, None)

    def accepts(self, inp:str) -> bool:
        return self(inp) in self.F

    @classmethod
    def from_nfa(cls, nfa:NFA) -> DFA:
        return nfa.to_dfa()


"""
LEX
"""

tokens = [
    "LPAREN",
    "RPAREN",
    "ONEPLUS",
    "ZEROONE",
    "KLEENE",
    "UNION",
    "CHAR"
]

t_LPAREN = r"\("
t_RPAREN = r"\)"
t_ONEPLUS = r"\+"
t_ZEROONE = r"\?"
t_KLEENE = r"\*"
t_UNION = r"\|"
t_CHAR = r"[a-zA-Z]"

def t_error(t):
    print(f"Invalid character found at line {t.lineno}, pos {t.lexpos}")
    t.lexer.skip(1)

lexer = lex.lex()   


"""
YACC
"""

precedence = (
    ("left", "UNION"),
    ("left", "KLEENE", "ONEPLUS", "ZEROONE"),
)

my_pattern = "c?c?(a|b)*c*"
my_test = "ccc"
# my_pattern = "c*"
# my_test = ""

def p_start(p):
    """
    start : expression
          | empty
    """
    ret = run(p[1])
    my_dfa = ret.to_dfa()

    print(ret.accepts(my_test))
    print(my_dfa.accepts(my_test))

def p_expression(p):
    """
    expression : CHAR
    """
    p[0] = p[1]

def p_expression_concat(p):
    """
    expression : expression expression
    """
    p[0] = ("concat", p[1], p[2])

def p_expression_union(p):
    """
    expression : expression UNION expression
    """
    p[0] = (p[2], p[1], p[3])

def p_expression_oneplus(p):
    """
    expression : expression ONEPLUS
    """
    # A+ -> AA*
    p[0] = ("concat", p[1], ("*", p[1]))

def p_expression_zeroone(p):
    """
    expression : expression ZEROONE
    """
    # A? -> A|ϵ
    p[0] = ("|", "ϵ", p[1])

def p_expression_kleene(p):
    """
    expression : expression KLEENE
    """
    p[0] = (p[2], p[1])

def p_expression_group(p):
    """
    expression : LPAREN expression RPAREN
    """
    p[0] = p[2]

def p_empty(p):
    "empty :"
    pass

def p_error(p):
    print("Syntax error")


""" AST """

def run(p):
    if isinstance(p, tuple):
        if p[0] == "*":
            return NFA.kleene(run(p[1]))
        elif p[0] == "|":
            return NFA.union(run(p[1]), run(p[2]))
        elif p[0] == "concat":
            return NFA.concat(run(p[1]), run(p[2]))
    else:
        ret = NFA()
        ret.update_transition(ret.q0, State(True), p)
        return ret

parser = yacc.yacc()

# # while True:
# #     try:
# #         s = input("regex: ")
# #     except EOFError:
# #         break
# #     parser.parse(s, lexer=lexer)

my_fsa = parser.parse(my_pattern, lexer=lexer)
