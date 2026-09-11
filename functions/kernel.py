"""Kernel definitions for the unsymmetric grid-based policy iteration.

The routines use product kernels of the form

    k(x, y) = phi(||x-y||) (y^T x)^case.

Use make_kernel to select the quadratic Matern or compactly supported
Wendland C4 radial kernel. The original Matern implementation is unchanged.
"""

from __future__ import annotations

import abc

import numpy as np


Array = np.ndarray


class Kernel(metaclass=abc.ABCMeta):
    """Abstract radial kernel used inside a product-kernel ansatz."""

    def __init__(self, gamma: float, case: int = 0) -> None:
        """Store the shape parameter and the product-kernel exponent."""
        self.gamma = gamma
        self.case  = case

    def set_gamma(self, gamma: float) -> None:
        """Change the shape parameter without rebuilding the kernel object."""
        self.gamma = gamma

    @abc.abstractmethod
    def phi(self, r: Array) -> Array:
        """Evaluate the radial basis function phi."""
        raise NotImplementedError

    @abc.abstractmethod
    def phiR(self, r: Array) -> Array:
        """Evaluate the scaled first radial derivative used by the gradient."""
        raise NotImplementedError

    @abc.abstractmethod
    def phiRR(self, r: Array) -> Array:
        """Evaluate the scaled second radial derivative."""
        raise NotImplementedError


class QuadMatern(Kernel):
    """Quadratic Matern-type radial kernel used in the experiments."""

    def phi(self, r: Array) -> Array:
        """Evaluate phi(r)."""
        gamma = self.gamma
        return np.exp(-gamma * r) * (3 + 3 * gamma * r + gamma**2 * r**2)

    def phiR(self, r: Array) -> Array:
        """Evaluate phi'(r) / r in the notation used by the PI update."""
        gamma = self.gamma
        return -np.exp(-gamma * r) * (1 + gamma * r) * gamma**2

    def phiRR(self, r: Array) -> Array:
        """Evaluate the second scaled radial derivative."""
        gamma = self.gamma
        return gamma**4 * np.exp(-gamma * r)


class WendlandC4(Kernel):
    """Compact support r < 1/gamma; valid in dimensions up to three."""

    def phi(self, r: Array) -> Array:
        s = self.gamma*r
        return np.maximum(1-s, 0)**6 * (35*s*s + 18*s + 3)

    def phiR(self, r: Array) -> Array:
        s = self.gamma*r
        return -56*self.gamma**2 * np.maximum(1-s, 0)**5 * (5*s + 1)

    def phiRR(self, r: Array) -> Array:
        return 1680*self.gamma**4 * np.maximum(1-self.gamma*r, 0)**4


def make_kernel(name: str = 'matern', gamma: float | None = None, case: int = 2) -> Kernel:
    """Choose a kernel; default widths suit this example, case=2 fixes v(0)=Dv(0)=0."""
    choices = {'matern': (QuadMatern, 2.0), 'wendland': (WendlandC4, 0.3)}
    if name not in choices:
        raise ValueError(f'Unknown kernel {name!r}; choose from {tuple(choices)}.')
    radial, default_gamma = choices[name]
    return radial(default_gamma if gamma is None else gamma, case)


def pairwise_distance(x: Array, y: Array) -> Array:
    """Return the pairwise Euclidean distances between column vectors."""
    x_norm = np.sum(x**2, axis=0, keepdims=True)
    y_norm = np.sum(y**2, axis=0, keepdims=True).T

    return np.sqrt(np.abs(x_norm + y_norm - 2 * y.T @ x))


def product_kernel_matrix(kernel: Kernel, centers: Array, points: Array) -> Array:
    """Evaluate the product-kernel Gram matrix k(centers, points)."""
    d        = kernel.case
    distance = pairwise_distance(centers, points)
    linear   = points.T @ centers

    return kernel.phi(distance) * linear**d


def product_kernel_gradient(kernel: Kernel, points: Array, alpha: Array, centers: Array) -> Array:
    """Evaluate the gradient of the product-kernel surrogate.

    The returned array has the same column layout as ``points``.  This is the
    derivative with respect to the evaluation argument in the ansatz
    ``sum_i alpha_i k(center_i, x)``.
    """
    d              = kernel.case
    distance_xc    = pairwise_distance(points, centers)
    distance_cx    = pairwise_distance(centers, points)
    linear_cx      = centers.T @ points
    linear_xc      = points.T @ centers
    radial_cx      = kernel.phi(distance_xc)
    radial_der_cx  = kernel.phiR(distance_xc)
    radial_der_xc  = kernel.phiR(distance_cx)

    center_term    = (centers * alpha) @ (
        -radial_der_cx * linear_cx**d
        + radial_cx * d * linear_cx**(d - 1)
    )
    point_term     = ((radial_der_xc * linear_xc**d) @ alpha) * points

    return center_term + point_term
