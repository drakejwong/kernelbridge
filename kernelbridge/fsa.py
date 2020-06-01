from __future__ import annotations
from collections import defaultdict

from ply import lex
from ply import yacc

"""
FSA
"""

class RegExError(Exception): pass


class State():
    r"""
    State for FSA.
    Houses transitions/edges to other States and whether or not State is accepting.
    Transition to another State by calling a State object on a character.
    """

    def __init__(self, accept:bool=False):
        self.trans = defaultdict(set)
        self.accept = accept
    
    def __call__(self, c:str):
        return self.trans.get(c, {})
    
    def __repr__(self):
        return str(hash(self))[-2:] + repr(list(self.trans.keys()))
    
    def __str__(self):
        # return repr(self)
        ret = ""
        for e, s1 in self.trans.items():
            ret += f"{repr(self)}-{e}->{s1}"
        return ret[1:]
    
    def add_trans(self, s:State, c:str):
        if len(c) > 1:
            raise Exception(f"Attempted to add ambiguous transition '{c}'")
        # elif c in self.trans:
        #     raise Exception(f"Transition {c} already exists for state {s}")
        else:
            self.trans[c].add(s)


class FiniteAutomaton():

    def __init__(self):
        self._q0 = State()
        self._Q = {self.q0}
        self._F = set()
        self._Sigma = set()
    
    def __call__(self, inp:str) -> bool:
        def dfs(s:State, p:str) -> bool:
            print(s, p)
            if not s: return False
            if not p:
                if s.accept: return True
                return any([child.accept for child in s("ϵ")])
            return any(
                [dfs(child, p[1:]) for child in s(p[0])] +
                [dfs(child, p) for child in s("ϵ")]
            )
        return dfs(self.q0, inp)
    
    def __repr__(self):
        # return repr(self.q0) + repr(self.Q) + repr(self.F)
        return "; ".join([str(s) for s in self.Q])
    
    @property
    def q0(self):
        return self._q0
    
    @q0.setter
    def q0(self, s:State):
        self._Q.add(s)
        self._q0 = s
    
    @property
    def Q(self):
        return self._Q
    
    @property
    def F(self):
        return self._F
    
    @property
    def Sigma(self):
        return self._Sigma

    def update_accept(self, s:State, a:bool):
        s.accept = a
        if a:
            self._F.add(s)
        elif not a and s in self.F:
            self._F.remove(s)
    
    def update_transition(self, s1:State, s2:State, c:str):
        if c in {"ep", "epsilon", "Epsilon"}:
            c = "ϵ"
        self._Q.add(s1)
        if s1.accept: self._F.add(s1)
        self._Q.add(s2)
        if s2.accept: self._F.add(s2)
        self._Sigma.add(c)
        s1.add_trans(s2, c)
    
    def incorporate(self, other:FiniteAutomaton):
        self._Q = self._Q.union(other.Q)
        self._Sigma = self._Sigma.union(other.Sigma)
        self._F = self._F.union(other.F)

    @staticmethod
    def concat(*M):
        r"""
        iteratively adds ϵ-transition from end state of prev to start state of curr,
        removing acceptance from end state of prev
        """
        ret = M[0]
        for m in M[1:]:
            for qf in list(ret.F):
                m.update_transition(qf, m.q0, "ep")
                ret.update_accept(qf, False)
            ret.incorporate(m) # union Q, F, Sigma
        return ret
    
    @staticmethod
    def union(*M):
        r"""
        adds ϵ-transition from new start state
        to start states of each machine in input
        """
        ret = FiniteAutomaton()
        for m in M:
            ret.update_transition(ret.q0, m.q0, "ep")
            ret.incorporate(m)
        return ret
    
    @staticmethod
    def kleene(m:FiniteAutomaton):
        r"""
        adds ϵ-transition from end to start, and from new start to old start.
        removes acceptance from end, adds acceptance to old start.
        """
        for qf in list(m.F):
            m.update_transition(qf, m.q0, "ep")
            m.update_accept(qf, False)
        m.update_accept(m.q0, True)
        start = State()
        m.update_transition(start, m.q0, "ep")
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
t_UNION = r"\|"
t_KLEENE = r"\*"
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

my_pattern = "ab*"
my_test = "a"

def p_start(p):
    """
    start : expression
          | empty
    """
    ret = run(p[1])
    print(ret)
    print(ret.F)
    print(ret(my_test))

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
            return FiniteAutomaton.kleene(run(p[1]))
        elif p[0] == "|":
            return FiniteAutomaton.union(run(p[1]), run(p[2]))
        elif p[0] == "concat":
            return FiniteAutomaton.concat(run(p[1]), run(p[2]))
    else:
        ret = FiniteAutomaton()
        ret.update_transition(ret.q0, State(True), p)
        return ret

parser = yacc.yacc()

# while True:
#     try:
#         s = input("regex: ")
#     except EOFError:
#         break
#     parser.parse(s, lexer=lexer)

my_fsa = parser.parse(my_pattern, lexer=lexer)
