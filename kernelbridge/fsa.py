from __future__ import annotations
from abc import abstractmethod

from collections import defaultdict as ddict, deque, Counter

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
    def __init__(self, q0=State(), F=set(), graph=None):
        self._q0 = q0
        self._F = F.copy()
        if not graph:
            self._graph = ddict(lambda: ddict(set))
            self._graph[self.q0]
        else:
            self._graph = graph.copy()
    
    @abstractmethod
    def __call__(self, inp:str) -> bool:
        r"""
        Reads input string and returns resulting final state.
        """
        raise NotImplementedError(NotImplemented)
    
    def __str__(self):
        return str({k: dict(v) for k, v in self.graph.items()})
    
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
        Enumerates all states of the FSA (source nodes of graph).
        """
        return frozenset(self.graph.keys())
    
    @property
    def F(self):
        r"""
        Enumerates all accepting states of the FSA.
        """
        return self._F
    
    @property
    def Sigma(self):
        if not hasattr(self, "_Sigma_cached"):
            self._Sigma_cached = set()
            for d in self.graph.values():
                for c in d:
                    self._Sigma_cached.add(c)
        return frozenset(self._Sigma_cached)

    
    @property
    def graph(self):
        r"""
        Dict-of-dicts for state transitions of the FSA.
        """
        return self._graph
    
    @abstractmethod
    def delta(self, s:State, c:str):
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
    def update_delta(self, s1:State, s2:State, c:str):
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
    def __init__(self, q0=State(), F=set(), graph=None):
        super().__init__(q0.copy(), F, graph)

    def __call__(self, inp:str) -> bool:
        curr = self.epsilon_closure(self.q0, include=True)
        for c in inp:
            nxt = set()
            for s in curr:
                nxt = nxt.union(self.delta(s, c))
            curr = nxt
            if not nxt: break
        return curr
    
    def update_delta(self, s1:State, s2:State, c:str):
        if c in {"ϵ", "ep", "epsilon", "Epsilon"}:
            c = ""

        self._graph[s1][c].add(s2)
        if s1.accept: self._F.add(s1)
        if s2.accept: self._F.add(s2)
    
    def incorporate(self, other:NFA):
        for s, d in other.graph.items():
            for t, ss in d.items():
                self._graph[s][t] = self.graph[s][t].union(ss)
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
                m.update_delta(qf, m.q0, "")
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
            ret.update_delta(ret.q0, m.q0, "")
            ret.incorporate(m)
        return ret
    
    @staticmethod
    def kleene(m:NFA):
        r"""
        Adds ϵ-transition from end to start, and from new start to old start.
        Removes acceptance from end, adds acceptance to old start.
        """
        for qf in list(m.F):
            m.update_delta(qf, m.q0, "")
            m.update_accept(qf, False)
        m.update_accept(m.q0, True)
        start = State()
        m.update_delta(start, m.q0, "")
        m.q0 = start
        return m
    
    def epsilon_closure(self, s:State, include:bool=False) -> set:
        r"""
        Enumerates the ϵ-closure of a state (all states reachable via
        any number ϵ-transitions, i.e. ϵ*).
        """
        ret = self.graph.get(s, dict()).get("", set())
        for ec in [self.epsilon_closure(state) for state in ret]:
            ret = ret.union(ec)
        if include:
            ret = ret.union({s})
        return ret

    def delta(self, s:State, c:str):
        if s not in self.graph: return set()
        
        ret = set()
        if c in self.graph[s]:
            ret = ret.union(self.graph[s][c])
        if "" in self.graph[s]:
            for s2 in self.epsilon_closure(s):
                ret = ret.union(self.delta(s2, c))
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
        dfa_graph = ddict(dict)
        dfa_states = {dfa_q0.name: [dfa_q0, False]}
        ct = 1

        state_queue = deque()
        state_queue.append((dfa_q0.name, nfa_eq0_list))
        while state_queue:
            src_key, src_list = state_queue.popleft()
            src_dfa, visited = dfa_states[src_key]
            
            if visited: continue
            
            for u in src_list:
                dest_set = set()
                for t, v_set in self.graph[u].items():
                    if t == "" or t in dfa_graph[src_dfa]: continue

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

                    if src_dfa.accept: # redundant to add state to F as src/dest?
                        dfa_F.add(src_dfa)
                    if dest_dfa.accept:
                        dfa_F.add(dest_dfa)
                    dfa_graph[src_dfa][t] = dest_dfa

                    state_queue.append((dest_key, dest_list))

            dfa_states[src_key][1] = True
        return DFA(dfa_q0, dfa_F, dfa_graph)


class DFA(FiniteAutomaton):
    def __init__(self, q0=State(), F=set(), graph=ddict(dict)):
        super().__init__(q0, F, graph)

    def __call__(self, inp:str) -> bool:
        curr = self.q0
        for c in inp:
            curr = self.delta(curr, c)
            if not curr: return
        return curr
    
    def update_delta(self, s1:State, s2:State, c:str):
        r"""
        Updates/adds a single-char transition between two states.
        States are added to the DFA's F as necessary.
        """
        self._graph[s1][c] = s2
        if s1.accept: self._F.add(s1)
        if s2.accept: self._F.add(s2)

    def delta(self, s:State, c:str):
        if s in self.graph:
            return self.graph[s].get(c, None)

    def accepts(self, inp:str) -> bool:
        return self(inp) in self.F

    def _prune_unreachable(self):
        reachable = set([self.q0])
        new_states = set([self.q0])
        while True:
            temp = set()
            for q in new_states:
                temp = temp.union(set(self.graph[q].values()))
            new_states = temp.difference(reachable)
            reachable = reachable.union(new_states)
            if not new_states: break
        unreachable = self.Q.difference(reachable)

        for s in unreachable:
            del self._graph[s]

        # inbound = ddict(int)
        # for out in self.graph.values():
        #     for dest in out.values():
        #         inbound[dest] += 1

        # prev = len(inbound)
        # while len(inbound) == prev:
        #     prev = len(inbound)
        #     for state in list(self.Q):
        #         if state not in inbound:
        #             out = self.graph.pop(state)
        #             for dest in out.values():
        #                 inbound[dest] -= 1
        #                 if inbound[dest] <= 0:
        #                     del inbound[dest]
    
    def minimize(self):
        r"""
        Minimizes DFA to an equivalent DFA with the smallest possible
        number of states via Hopcroft's algorithm.

        Hopcroft's algorithm is the fastest known algorithm for partition
        refinement in the automata context. Alternatives include Moore's
        algorithm, Brzozowski's algorithm, or the table-filling algorithm
        based on the Myhill-Nerode theorem.
        """
        self._prune_unreachable()

        """Hopcroft's partition refinement"""
        P = {frozenset(self.F)}
        complement = frozenset(self.Q.difference(self.F))
        if complement: P.add(complement)
        W = deque(P)
        while W:
            A = W.popleft()
            for c in self.Sigma:
                X = frozenset(s for s in self.Q if self.delta(s, c) in A)
                for Y in list(P):
                    inter = X.intersection(Y)
                    diff = Y.difference(X)
                    if not inter or not diff: continue
                    P.remove(Y)
                    P.add(inter)
                    P.add(diff)
                    try:
                        W.remove(Y)
                        W.append(inter)
                        W.append(diff)
                    except:
                        if len(inter) <= len(diff):
                            W.append(inter)
                        else:
                            W.append(diff)
        
        """Reconstruct DFA from refined supernodes of P"""
        label = 1
        min_q0, min_graph, min_F = -1, ddict(dict), set()
        P = list(P)
        reps, min_states = [], []
        for supernode in P:
            src = list(supernode)[0]
            reps.append(src)

            accept = src in self.F
            curr = State(accept=accept, name=str(label))
            label += 1
            if self.q0 in supernode:
                curr.name = "0"
                label -= 1
                min_q0 = curr
            if accept:
                min_F.add(curr)
            min_states.append(curr)

        supernode_to_minstate = dict(zip(P, min_states))
        container = dict(zip(reps, P))
        for s in self.Q:
            if s in container: continue
            for supernode in P:
                if s in supernode:
                    container[s] = supernode
        for curr, src, supernode in zip(min_states, reps, P):
            for c, dest in self.graph[src].items():
                min_graph[curr][c] = supernode_to_minstate[container[dest]]
        
        self._q0 = min_q0
        self._graph = min_graph
        self._F = min_F

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

# my_pattern = "c?c?d(a|b)*c*"
# my_test = "ccac"
my_pattern = "aa(b|c)*"
my_test = "a"

def p_start(p):
    """
    start : expression
          | empty
    """
    ret = run(p[1])
    my_dfa = ret.to_dfa()

    print(ret.accepts(my_test))
    print(my_dfa.accepts(my_test))
    my_dfa.minimize()
    print(my_dfa.accepts(my_test))
    # print(my_dfa.q0, my_dfa)
    # print(my_dfa.Q, my_dfa.Sigma, my_dfa.F)

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
        ret.update_delta(ret.q0, State(True), p)
        return ret

parser = yacc.yacc()

# # while True:
# #     try:
# #         s = input("regex: ")
# #     except EOFError:
# #         break
# #     parser.parse(s, lexer=lexer)

my_fsa = parser.parse(my_pattern, lexer=lexer)
