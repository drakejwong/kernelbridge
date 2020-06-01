"""
This module endows BaseKernels with GPflow interoperability via duck typing.
"""
from kernelbridge.basekernel import BaseKernel

import numpy as np
import tensorflow as tf


class GPflowKernel(BaseKernel):

    def __init__(self, kernel, active_dims=None):
        self._kernel = kernel
    
    def __call__(self, x1, x2=None, *, full_cov=True, presliced=False):
        """
        Wrapper for __call__ method interoperable with GPflow models
        """
        if (not full_cov) and (x2 is not None):
            raise ValueError(
                "Computing kernel on two elements requires full covariance.")

        if not hasattr(x1, "__len__"):
            return self._kernel(x1, x2)

        elif not full_cov:
            assert x2 is None
            return tf.fill(tf.shape(x1)[:-1], tf.squeeze(1.0))

        else:
            res = np.zeros((len(x1), len(x2)))
            for i, x1i in enumerate(x1):
                for j, x2j in enumerate(x2):
                    res[i][j] = self._kernel(x1i[0], x2j[0])
            return res


class BadGPflowKernel(BaseKernel):

    def __init__(self, kernel, active_dims=None):
        self._name = kernel.__name__
        self._kernel = kernel
        self._active_dims = self._normalize_active_dims(active_dims)

    @staticmethod
    def _normalize_active_dims(value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = np.array(value, dtype=int)
        return value

    @property
    def active_dims(self):
        if not hasattr(self, '_active_dims'):
            self._active_dims = self._normalize_active_dims(None)
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims = self._normalize_active_dims(value)

    def on_separate_dims(self, other):
        """
        GPflow method to check if two kernels share active dimensions.
        """
        if isinstance(self.active_dims, slice) or isinstance(other.active_dims, slice):
            return False
        elif self.active_dims is None or other.active_dims is None:
            return False
        else:
            this_dims = self.active_dims.reshape(-1, 1)
            other_dims = other.active_dims.reshape(1, -1)
            return not np.any(this_dims == other_dims)

    def slice(self, x1, x2):
        """
        GPflow method to select active dimensions of inputs via slicing.
        """
        dims = self.active_dims
        if isinstance(dims, slice):
            x1 = x1[..., dims]
            if x2 is not None:
                x2 = x2[..., dims]
        elif dims is not None:
            x1 = tf.gather(x1, dims, axis=-1)
            if x2 is not None:
                x2 = tf.gather(x2, dims, axis=-1)
        return x1, x2

    def slice_cov(self, cov):
        """
        GPflow method to slice active dims for covariance matrices
        """
        if cov.shape.ndims == 2:
            cov = tf.linalg.diag(cov)

        dims = self.active_dims

        if isinstance(dims, slice):
            return cov[..., dims, dims]
        elif dims is not None:
            nlast = tf.shape(cov)[-1]
            ndims = len(dims)

            cov_shape = tf.shape(cov)
            cov_reshaped = tf.reshape(cov, [-1, nlast, nlast])
            gather1 = tf.gather(tf.transpose(cov_reshaped, [2, 1, 0]), dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), dims)
            cov = tf.reshape(
                tf.transpose(gather2, [2, 0, 1]), tf.concat(
                    [cov_shape[:-2], [ndims, ndims]], 0)
            )

    def _validate_ard_active_dims(self, ard_parameter):
        """
        GPflow method to match automatic relevance determination with dims
        """
        if self.active_dims is None or isinstance(self.active_dims, slice):
            return

        if ard_parameter.shape.rank > 0 and ard_parameter.shape[0] != len(self.active_dims):
            raise ValueError(
                f"Size of `active_dims` {self.active_dims} does not match "
                f"size of ard parameter ({ard_parameter.shape[0]})"
            )

    @property
    def parameters(self):
        return self.state # pylint: disable=no-member

    @property
    def trainable_parameters(self):
        return self.state # pylint: disable=no-member

    def __call__(self, x1, x2=None, *, full_cov=True, presliced=False):
        """
        Wrapper for __call__ method interoperable with GPflow models
        """
        if (not full_cov) and (x2 is not None):
            raise ValueError(
                "Computing kernel on two elements requires full covariance.")

        if not presliced:
            x1, x2 = self.slice(x1, x2)

        if not full_cov:
            assert x2 is None
            return np.array([self._kernel(x1i, x1i) for x1i in x1])

        else:
            return np.array([self._kernel(xi[0], xi[1]) for xi in zip(x1, x2)])
