"""
This module defines the abstract BaseKernel class for kernel creation
and establishes a context-free grammar for kernel composition.

Heavily inspired by GraphDot (Yu-Hang Tang).
"""

from collections import namedtuple, OrderedDict
from collections.abc import Iterable
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
import abc
import re


class BaseKernel:

    @staticmethod
    def create(name, desc, expr, inputs, *hyperparameter_specs):
        r"""
        Wrapper for extensible Kernel class creation.
        Heavily inspired by GraphDot (Yu-Hang Tang).

        Parameters
        ----------
        name: str
            Name of kernel
        desc: str
            Description of kernel properties
        expr: str
            SymPy compatible mathematical expression
        *inputs: tuple(str)
            SymPy compatible input variable names (2)
        **hyperparameter_specs: dict
            `kwarag` specs for hyperparameters (symbol, bounds, docstr)

        Returns
        ----------
        class
            New Kernel class based on given parameters
        """

        """ Parse expression """
        if isinstance(expr, str):
            expr = sy.sympify(expr)

        """ Parse input variable definitions """
        if len(inputs) != 2:
            raise ValueError(
                f"""A kernel must have exactly two inputs
                (received {len(inputs)})."""
            )
        inputs = [sy.Symbol(s) if isinstance(s, str) else s for s in inputs]

        """ Parse hyperparameters + specifications """
        hypers = OrderedDict()
        for spec in hyperparameter_specs:
            if not hasattr(spec, '__iter__'):
                symbol = spec
                hypers[symbol] = dict(dtype=np.dtype(np.float32))
            if len(spec) == 1:
                symbol = spec[0]
                hypers[symbol] = dict(dtype=np.dtype(np.float32))
            if len(spec) == 2:
                symbol, dtype = spec
                hypers[symbol] = dict(dtype=np.dtype(dtype))
            if len(spec) == 3:
                symbol, dtype, doc = spec
                hypers[symbol] = dict(dtype=np.dtype(dtype), doc=doc)
            elif len(spec) == 4:
                symbol, dtype, lb, ub = spec
                hypers[symbol] = dict(dtype=np.dtype(dtype),
                                      bounds=(lb, ub))
            elif len(spec) == 5:
                symbol, dtype, lb, ub, doc = spec
                hypers[symbol] = dict(dtype=np.dtype(dtype),
                                      bounds=(lb, ub),
                                      doc=doc)
            else:
                raise ValueError(
                    'Invalid hyperparameter specification, '
                    'must be one of\n'
                    '(symbol)\n',
                    '(symbol, dtype)\n',
                    '(symbol, dtype, doc)\n',
                    '(symbol, dtype, lb, ub)\n',
                    '(symbol, dtype, lb, ub, doc)\n',
                )

        class Kernel(BaseKernel):

            __name__ = name

            _desc = desc
            _expr = expr
            _inputs = inputs
            _hypers = hypers
            _dtype = np.dtype([(k, v['dtype']) for k, v in hypers.items()],
                              align=True)

            def __init__(self, *args, **kwargs):

                self._theta_values = values = OrderedDict()
                self._theta_bounds = bounds = OrderedDict()

                for symbol, value in zip(self._hypers, args):
                    values[symbol] = value
                for symbol in self._hypers:
                    try:
                        values[symbol] = kwargs[symbol]
                    except KeyError:
                        if symbol not in values:
                            raise KeyError(
                                f"Must provide {symbol} for {self.__name__}"
                            )

                    try:
                        bounds[symbol] = kwargs['%s_bounds' % symbol]
                    except KeyError:
                        try:
                            bounds[symbol] = self._hypers[symbol]['bounds']
                        except KeyError:
                            raise KeyError(
                                f"Bounds for {symbol} of kernel"
                                f"{self.__name__} not set, and no"
                                "default bounds exist"
                            )

            @property
            def _inputs_hypers(self):
                if not hasattr(self, '_inputs_hypers_cached'):
                    self._inputs_hypers_cached = [
                        *self._inputs, *self._hypers.keys()
                    ]
                return self._inputs_hypers_cached

            @property
            def _K(self):
                if not hasattr(self, '_K_cached'):
                    self._K_cached = lambdify(
                        self._inputs_hypers,
                        self._expr
                    )
                return self._K_cached

            @property
            def _jac(self):
                if not hasattr(self, '_jac_cached'):
                    self._jac_cached = [
                        lambdify(self._inputs_hypers, sy.diff(expr, h))
                        for h in self._hypers
                    ]
                return self._jac_cached

            def __call__(self, x1, x2, jac=False):
                r"""
                Evaluates Kernel on pairwise input based on class' expr.
                Optionally returns Jacobian.
                """
                if x2 is None:
                    res = self._K(x1, x1, *self.theta)
                else:
                    res = self._K(x1, x2, *self.theta)

                if jac:
                    return (
                        res,
                        [j(x1, x2, *self.theta) for j in self._jac]
                    )
                else:
                    return res

            def __repr__(self):
                cls = self.__name__
                theta = [
                    f"{t}={v}"
                    for t, v in self._theta_values.items()
                ]
                if len(theta) == 1:
                    theta = theta[0]
                else:
                    theta = ", ".join(theta)

                bounds = [
                    f"{t}_bounds={v}"
                    for t, v in self._theta_bounds.items()
                ]
                if not bounds:
                    bounds = ""
                elif len(bounds) == 1:
                    bounds = ", " + bounds[0]
                else:
                    bounds = ", ".join([""] + bounds)

                return f"{cls}({theta + bounds})"

            def __str__(self):
                return repr(self)

            ###
            # gen_expr
            ###

            @property
            def dtype(self):
                return self._dtype

            @property
            def state(self):
                return tuple(self._theta_values.values())

            @property
            def theta(self):
                return namedtuple(
                    self.__name__ + 'Hyperparameters',
                    self._theta_values.keys()
                )(**self._theta_values)

            @theta.setter
            def theta(self, seq):
                assert(len(seq) == len(self._theta_values))
                for theta, value in zip(self._hypers, seq):
                    self._theta_values[theta] = value

            @property
            def bounds(self):
                return tuple(self._theta_bounds.values())

        doc_hypers = [
            f"""{symbol}: {hdef["dtype"]}
                {hdef["doc"]}
            {symbol}_bounds: pair of {hdef["dtype"]}
                Lower and upper bounds of {symbol}
            """ for symbol, hdef in hypers.items()
        ]

        Kernel.__doc__ = rf"""
            {desc}
            Parameters
            ----------
            {"".join(doc_hypers)}
        """

        return Kernel

    def __add__(self, other):
        r"""
        Python magic method for Kernel addition (k1 + k2).
        k_+(x, y) = k_1(x, y) + k_2(x, y)
        """
        return Sum(
            self,
            other if isinstance(other, BaseKernel) else Constant(other)
        )

    def __radd__(self, other):
        return Sum(
            other if isinstance(other, BaseKernel) else Constant(other),
            self
        )

    def __mul__(self, other):
        r"""
        Python magic method for Kernel multiplication (k1 * k2).
        k_\times(x_1, x_2) = k_1(x_1, x_2) \cdot k_2(x_1, x_2)
        """
        return Product(
            self,
            other if isinstance(other, BaseKernel) else Constant(other)
        )

    def __rmul__(self, other):
        return Product(
            other if isinstance(other, BaseKernel) else Constant(other),
            self
        )


class Composition(BaseKernel):
    r"""
    Parent class for kernel operations.
    Inspired by GPflow architecture.

    TODO: composing Composition kernels with other kernels--extend list
    """

    def __init__(self, *kernels):
        self._kernels = []
        for k in kernels:
            if isinstance(k, Iterable):
                self._kernels.extend(k)
            else:
                self._kernels.append(k)
    
    @property
    def theta(self):
        return tuple([k.theta for k in self._kernels])

    @theta.setter
    def theta(self, *seqs):
        if len(seqs) == 1:
            seqs = seqs[0]

        assert(len(seqs) == len(self._kernels))

        for k, seq in zip(self._kernels, seqs):
            k.theta = seq

    @property
    def bounds(self):
        return tuple([k.bounds for k in self._kernels])

    def __repr__(self):
        cls = self.__name__  # pylint: disable=no-member
        names = [f"{repr(k)}" for k in self._kernels]
        if len(names) > 1:
            names = ", ".join(names)

        return f"{cls}({names})"

    def __str__(self):
        if not hasattr(self, "_opstr"):
            return repr(self)

        names = [f"{repr(k)}" for k in self._kernels]
        if len(names) > 1:
            names = self._opstr.join(names) # pylint: disable=no-member

        return f"{names}" # pylint: disable=no-member

    def __call__(self, x1, x2, jac=False):
        ret = self._agg([ker(x1, x2) for ker in self._kernels])
        if jac:
            return ret, self._agg([ker(x1, x2, jac=True)[1] for ker in self._kernels])
        else:
            return ret


    @property
    @abc.abstractmethod
    def _agg(self):
        pass


class Sum(Composition):
    r"""
    Sum kernel based on input list of kernels.
    """

    def __init__(self, *kernels):
        self.__name__ = "Sum"
        self._desc = "Composition of kernels via addition"
        self._opstr = " + "
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.sum


class Product(Composition):
    r"""
    Product kernel based on input list of kernels.
    """

    def __init__(self, *kernels):
        self.__name__ = "Product"
        self._desc = "Composition of kernels via multiplication"
        self._opstr = " * "
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.prod


#
# DirectSum, TensorProduct, RConvolution
#


class RConvolution(Composition):
    r"""
    R-convolution kernel based on user-defined re decomposition.
    Default behavior: Kronecker delta applied to subsequences of inputs
    """

    def __init__(self, regex):
        self.__name__ = "RConvolution"
        self._desc = "R-convolution"
        self._regex = regex
        # TODO _kernels dict {feature:kernel} / port to Convolution
        super().__init__(KroneckerDelta())
    
    def __call__(self, x1, x2):
        x1ss = [
            x1[i:j + 1]
            for i in range(len(x1))
            for j in range(i, len(x1))
            if re.fullmatch(self._regex, x1[i:j + 1])
        ]
        x2ss = [
            x2[i:j + 1]
            for i in range(len(x2))
            for j in range(i, len(x2))
            if re.fullmatch(self._regex, x2[i:j + 1])
        ]

        return np.sum([
            self._kernels[0](x1s, x2s)
            for x1s in x1ss
            for x2s in x2ss
        ])
    
    # TODO
    def __repr__(self):
        pass

    # TODO
    def __str__(self):
        pass


"""
PREDEFINED KERNELS
"""


def Constant(c, c_bounds=(0, np.inf)):
    r"""
    Kernel that solely evaluates to a constant (often 1).

    Parameters
    ----------
    c: float > 0
        The value of the kernel

    Returns
    ----------
    BaseKernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(constant=np.float32)
    # @cpptype([('c', np.float32)])
    class ConstantKernel(BaseKernel):
        def __init__(self, c, c_bounds):
            self.__name__ = str(c)
            self.c = float(c)
            self.c_bounds = c_bounds

        def __call__(self, i, j, jac=False):
            if jac:
                return self.c, [1.0]
            else:
                return self.c

        def __repr__(self):
            return f'Constant({self.c})'

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            f = f'{theta_prefix}c'
            if jac:
                return f, ['1.0f']
            else:
                return f

        @property
        def theta(self):
            return namedtuple(
                'ConstantHyperparameters',
                ['c']
            )(self.c)

        @theta.setter
        def theta(self, seq):
            self.c = seq[0]

        @property
        def bounds(self):
            return (self.c_bounds,)

    return ConstantKernel(c, c_bounds)


def KroneckerDelta(h=0, h_bounds=(1e-3, 1)):
    r"""Creates a Kronecker delta kernel that returns either h or 1 depending
    on whether two objects compare equal, i.e. :math:`k_\delta(i, j) =
    \begin{cases} 1, i = j \\ h, otherwise \end{cases}`

    Parameters
    ----------
    h: float in (0, 1)
        The value of the kernel when two objects do not compare equal

    Returns
    -------
    BaseKernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(lo=np.float32, hi=np.float32)
    # @cpptype([('h', np.float32)])
    class KroneckerDeltaKernel(BaseKernel):

        def __init__(self, h, h_bounds):
            self.h = float(h)
            self.h_bounds = h_bounds

        def __call__(self, x1, x2, jac=False):
            ret = 1.0 if x1 == x2 else self.h
            if jac:
                return ret, [0.0 if x1 == x2 else 1.0]
            else:
                return ret

        def __repr__(self):
            # return f'KroneckerDelta({self.h})'
            return 'KroneckerDelta({})'.format(self.h)

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            # f = f'({x} == {y} ? 1.0f : {theta_prefix}h)'
            f = '({} == {} ? 1.0f : {}h)'.format(x, y, theta_prefix)
            if jac is True:
                # return f, [f'({x} == {y} ? 0.0f : 1.0f)']
                return f, ['({} == {} ? 0.0f : 1.0f)'.format(x, y)]
            else:
                return f

        @property
        def theta(self):
            return namedtuple(
                'KroneckerDeltaHyperparameters',
                ['h']
            )(self.h)

        @theta.setter
        def theta(self, seq):
            self.h = seq[0]

        @property
        def bounds(self):
            return (self.h_bounds,)

    return KroneckerDeltaKernel(h, h_bounds)


SquaredExponential = BaseKernel.create(
    "SquaredExponential",

    r"""A squared exponential kernel smoothly transitions from 1 to
    0 as the distance between two vectors increases from zero to infinity, i.e.
    :math:`k_\mathrm{se}(\mathbf{x}, \mathbf{y}) = \exp(-\frac{1}{2}
    \frac{\lVert \mathbf{x} - \mathbf{y} \rVert^2}{\sigma^2})`""",

    'exp(-0.5 * (x1 - x2)**2 * length_scale**-2)',

    ('x1', 'x2'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""Determines how quickly should the kernel decay to zero. The kernel has
     a value of approx. 0.606 at one length scale, 0.135 at two length
     scales, and 0.011 at three length scales.""")
)

RationalQuadratic = BaseKernel.create(
    'RationalQuadratic',

    r"""A rational quadratic kernel is equivalent to the sum of many squared
    exponential kernels with different length scales. The parameter `alpha`
    tunes the relative weights between large and small length scales. When
    alpha approaches infinity, the kernel is identical to the squared
    exponential kernel.""",

    '(1 + (x1 - x2)**2 / (2 * alpha * length_scale**2))**(-alpha)',

    ('x1', 'x2'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""The smallest length scale of the square exponential components."""),
    ('alpha', np.float32, 1e-3, np.inf,
     r"""The relative weights of large-scale square exponential components.
     Larger alpha values leads to a faster decay of the weights for larger
     length scales.""")
)
