from __future__ import annotations
from collections import defaultdict as ddict

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
            return str(hash(self))[-2:]
    
    def __str__(self):
        return repr(self)

    def copy(self):
        return State(self.accept, self.name)


class FiniteAutomaton():

    def __init__(self, q0=State(), F=set(), Sigma=None):
        self._q0 = q0.copy()
        self._F = F.copy()
        if not Sigma:
            self._Sigma = ddict(lambda: ddict(set))
            self._Sigma[self.q0]
        else:
            self._Sigma = Sigma.copy()
    
    def __call__(self, inp:str) -> bool:
        r"""
        Reads input string and returns resulting final state.
        """

        curr = {self.q0}
        for c in inp:
            for s in curr:
                curr = self.transition(s, c)
                if curr: break
            if not curr: return None
        return curr
    
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
    
    def transition(self, s:State, c:str):
        r"""
        Evaluates a single character transition from a given state,
        returning a corresponding set of destination states.
        """
        if s in self.Sigma:
            if c in self.Sigma[s]:
                return self.Sigma[s][c]
            elif "" in self.Sigma[s]:
                ret = set()
                for s2 in self.transition(s, ""):
                    ret = ret.union(self.transition(s2, c))
                return ret
        return set()
    
    def epsilon_closure(self, s:State) -> set:
        r"""
        Enumerates the ϵ-closure of a state (all states reachable via
        any number ϵ-transitions, i.e. ϵ*).
        """
        ret = self.transition(s, "")
        if not ret: return {}
        for ss in [self.epsilon_closure(state) for state in ret]:
            ret = ret.union(ss)
        return ret

    def update_accept(self, s:State, a:bool):
        r"""
        Updates acceptance of State and adds/removes from F accordingly
        """
        s.accept = a
        if a:
            self._F.add(s)
        elif not a and s in self.F:
            self._F.remove(s)
    
    def update_transition(self, s1:State, s2:State, c:str):
        r"""
        Updates/adds a single-char transition between two states.
        States are added to the FSA's Q and F as necessary.
        """
        if c in {"ϵ", "ep", "epsilon", "Epsilon"}:
            c = ""

        self._Sigma[s1][c].add(s2)
        if s1.accept: self._F.add(s1)
        if s2.accept: self._F.add(s2)
    
    def accepts(self, inp:str) -> bool:
        final = self(inp)
        if not final: return False
        for ss in [self.epsilon_closure(s) for s in final]:
            final = final.union(ss)
        return False if not final.intersection(self.F) else True


class NFA(FiniteAutomaton):
    def __init__(self, q0=State(), F=set(), Sigma=None):
        super().__init__(q0, F, Sigma)
    
    def __str__(self):
        return super().__str__()
    
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

my_pattern = "c?c?(a|b)*"
my_test = "ccaba"

def p_start(p):
    """
    start : expression
          | empty
    """
    ret = run(p[1])
    # print(ret)
    # print(ret.F)
    # print(ret(my_test))
    print(ret.accepts(my_test))

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
