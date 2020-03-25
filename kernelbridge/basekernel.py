"""
This module defines the base kernel and
rules for kernel creation and composition.
"""

from collections import namedtuple, OrderedDict
import numpy as np
import sympy as sy
from sympy.utilities.lambdify import lambdify
import abc

class BaseKernel:

    @staticmethod
    def create(name, desc, expr, *params, **hyperparameter_specs):
        """
        Wrapper for kernel creation
        """

        ''' Parse expression '''
        if isinstance(expr, str):
            expr = sy.sympify(expr)
        
        ''' Parse parameters '''
        if len(params) != 2:
            raise ValueError(f"A kernel takes in exactly two parameters (received {len(params)}).")
        params = [sy.Symbol(s) if isinstance(s, str) else s for s in params]

        ''' Parse hyperparameters + specs, input cases '''
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
            elif len(spec) == 4:
                symbol, dtype, lb, ub = spec
                hypers[symbol] = dict(dtype=np.dtype(dtype),
                                         bounds=(lb, ub))
            else:
                raise ValueError(
                    'Invalid hyperparameter specification, must be one of\n'
                    '(symbol)\n',
                    '(symbol, dtype)\n',
                    '(symbol, dtype, lbound, ubound)\n',
                )
            ### doc?

        class Kernel(BaseKernel):
            """
            Template kernel interface for user-defined expression, hyperparameters, etc.
            """

            __name__ = name

            _desc = desc
            _expr = expr
            _params = params
            _hypers = hypers
            _dtype = np.dtype([(k, v['dtype']) for k, v in hypers.items()],
                                align=True)

            def __init__(self, *args, **kwargs):

                self._theta_values = values = OrderedDict()
                self._theta_bounds = bounds = OrderedDict()

                for symbol, value in zip(self._hypers, args):
                    values[theta] = value
                
                for symbol in self._hypers:
                    try:
                        values[symbol] = kwargs[symbol]
                    except KeyError:
                        if symbol not in values:
                            raise KeyError(
                                f"Must provide {symbol} for {self.__name__}"
                            )

                    try:
                        values[symbol] = kwargs['%_bounds' % symbol]
                    except KeyError:
                        try:
                            bounds[symbol] = self._hypers[symbol]['bounds']
                        except KeyError:
                            raise KeyError(
                                f"Bounds for {symbol} of kernel {self.__name__} not set, and no default bounds exist"
                            )
            
            @property # cache/memoize?
            def _params_hypers(self):
                if not hasattr(self, '_params_hypers_cached'):
                    self._params_hypers_cached = [
                        *self._params, *self._hypers.keys()
                    ]
                return self._params_hypers_cached
            
            @property
            def _K(self):
                if not hasattr(self, '_K_cached'):
                    self._K_cached = lambdify(
                        self._params_hypers,
                        self._expr
                    )
                return self._K_cached
            
            @property
            def _jac(self):
                if not hasattr(self, '_jac_cached'):
                    self._jac_cached = [
                        lambdify(self._vars_and_hypers, sy.diff(expr, h))
                        for h in self._hyperdefs
                    ]
                return self._jac_cached
            
            def __call__(self, x1, x2, jac=False): # K(x1) -> K(x1,x1) vs. explicit
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
        """
        Kernel -> Kernel + Kernel
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
        return Product(
            self,
            other if isinstance(other, BaseKernel) else Constant(other)
        )

    def __rmul__(self, other):
        return Product(
            other if isinstance(other, BaseKernel) else Constant(other),
            self
        )


class Combination(BaseKernel):
    """
    Parent class for all kernel operations.
    Inspired by GPflow architecture.
    """

    def __init__(self, *kernels):
        self._kernels = []
        # self.kernels(kernels) # worth abstracting?
        self._kernels.extend(kernels)

    def __repr__(self):
        cls = self.__name__
        names = [f'{ker.__name__}' for ker in self.kernels]

        return f"{cls}({names})"
    
    # @property
    # def kernels(self):
    #     return self._kernels
    
    # @kernels.setter # supports self-combination
    # def kernels(self, *kernels):
    #     #add docstring
    #     for ker in kernels: # append combination
    #         if isinstance(k, self.__class__):
    #             self._kernels.extend(ker.kernels)
    #         else:
    #             self._kernels.append(ker)
    
    def __call__(self, x1, x2, jac=False):
        return self._agg([ker(x1, x2, jac) for ker in self.kernels])
    
    @property
    @abc.abstractmethod
    def _agg(self):
        pass


class Sum(Combination):
    def __init__(self, *kernels):
        self.__name__ = "SumKernel"
        self._desc = "Combination of kernels via addition"
        
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.sum

class Product(Combination):
    def __init__(self, *kernels):
        self.__name__ = "ProductKernel"
        self._desc = "Combination of kernels via multiplication"
        
        super().__init__(kernels)

    @property
    def _agg(self):
        return np.prod


###
### DirectSum, TensorProduct, RConvolution
###


def Constant(c, c_bounds=(0, np.inf)):
    """
    Creates a no-op kernel that returns a constant value (often being 1),
    i.e. :math:`k_\mathrm{c}(\cdot, \cdot) \equiv constant`

    Parameters
    ----------
    constant: float > 0
        The value of the kernel

    Returns
    -------
    BaseKernel
        A kernel instance of corresponding behavior
    """

    # only works with python >= 3.6
    # @cpptype(constant=np.float32)
    @cpptype([('c', np.float32)])
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