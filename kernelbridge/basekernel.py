"""
This module defines the abstract BaseKernel class for kernel creation
and establishes a context-free grammar for kernel composition.

Heavily inspired by GraphDot (Yu-Hang Tang).
"""

from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
import abc

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

        TODO: __repr__ (Template?)
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
            r"""
            Template kernel interface.

            Constructs new Kernel class via BaseKernel.create.
            """

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
                        values[symbol] = kwargs['%s_bounds' % symbol]
                    except KeyError:
                        try:
                            bounds[symbol] = self._hypers[symbol]['bounds']
                        except KeyError:
                            raise KeyError(
                                f"Bounds for {symbol} of kernel {self.__name__} not set, and no default bounds exist"
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
                        lambdify(self._vars_and_hypers, sy.diff(expr, h))
                        for h in self._hypers
                    ]
                return self._jac_cached
            
            def __call__(self, x1, x2, jac=False):
                r"""
                Evaluates Kernel on pairwise input based on class' expr.
                Optionally returns Jacobian.
                """
                if jac:
                    return (
                        self._K(x1, x2, *self.theta),
                        [j(x1, x2, *self.theta) for j in self._jac]
                    )
                else:
                    return self._K(x1, x2, *self.theta)
            
            def __repr__(self):
                cls = self.__name__
                theta = [f'{t}={v}' for t, v in self._theta_values.items()]
                bounds = [f'{t}_bounds={v}' for t, v in self._theta_bounds.items()]

                return f"{cls}({theta}, {bounds})"
            
            ###
            ### gen_expr
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
                    self._theta_values[theta] = value # check bounds?

            @property
            def bounds(self):
                return tuple(self._theta_bounds.values())
        
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

    TODO: storing thetas (hypers) for each kernel
    TODO: composing Composition kernels with other kernels--extend list
    """

    def __init__(self, *kernels):
        self._kernels = list(kernels[0]) # need better parse *args

    def __repr__(self):
        cls = self.__name__
        names = [f'{ker.__name__}' for ker in self._kernels]

        return f"{cls}({names})"
    
    def __call__(self, x1, x2, jac=False):
        return self._agg([ker(x1, x2, jac) for ker in self._kernels])
    
    @property
    @abc.abstractmethod
    def _agg(self):
        pass


class Sum(Composition):
    r"""
    Sum kernel based on input list of kernels.
    """

    def __init__(self, *kernels):
        self.__name__ = "SumKernel"
        self._desc = "Composition of kernels via addition"
        
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.sum

class Product(Composition):
    r"""
    Product kernel based on input list of kernels.
    """

    def __init__(self, *kernels):
        self.__name__ = "ProductKernel"
        self._desc = "Composition of kernels via multiplication"
        
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.prod


###
### DirectSum, TensorProduct, RConvolution
###


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
            self.c = float(c)
            self.c_bounds = c_bounds

        def __call__(self, i, j, jac=False):
            if jac is True:
                return self.c, [1.0]
            else:
                return self.c

        def __repr__(self):
            # return f'Constant({self.c})'
            return 'Constant({})'.format(self.c)

        def gen_expr(self, x, y, jac=False, theta_prefix=''):
            # f = f'{theta_prefix}c'
            f = '{}c'.format(theta_prefix)
            if jac is True:
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


SquaredExponential = BaseKernel.create(
    "SquaredExponential",

    r"""A squared exponential kernel smoothly transitions from 1 to
    0 as the distance between two vectors increases from zero to infinity, i.e.
    :math:`k_\mathrm{se}(\mathbf{x}, \mathbf{y}) = \exp(-\frac{1}{2}
    \frac{\lVert \mathbf{x} - \mathbf{y} \rVert^2}{\sigma^2})`""",

    'exp(-0.5 * (x - y)**2 * length_scale**-2)',

    ('x', 'y'),

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

    '(1 + (x - y)**2 / (2 * alpha * length_scale**2))**(-alpha)',

    ('x', 'y'),

    ('length_scale', np.float32, 1e-6, np.inf,
     r"""The smallest length scale of the square exponential components."""),
    ('alpha', np.float32, 1e-3, np.inf,
     r"""The relative weights of large-scale square exponential components.
     Larger alpha values leads to a faster decay of the weights for larger
     length scales.""")
)
