"""Kernel PI with/without shrinking; model and PMP references live in model.py.

Run: python counterexample.py
Dependencies: NumPy, SciPy, Matplotlib. Errors use the current domain in each run.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from functions import kernel, model

MODEL = model.EscapeCounterexample()
KERNEL = kernel.make_kernel('matern', case=2)  # Or 'wendland'; optional gamma=...
DT, STEPS = 5e-7, 9


def disk_grid(n, extent=.5):
    a = np.linspace(-extent, extent, n)
    x = np.array(np.meshgrid(a, a)).reshape(2, -1)
    radius2 = np.sum(x*x, axis=0)
    return x[:, (radius2 > 1e-20) & (radius2 < .25)]  # Omit the zero kernel row.


def km(centers, points):
    return kernel.product_kernel_matrix(KERNEL, centers, points)


def policy_iteration(training, tests, reference, shrink):
    x = training.copy()
    u = MODEL.stable_control(x)
    angles = np.linspace(0, 2*np.pi, 256, endpoint=False)
    rays = np.array([np.cos(angles), np.sin(angles)])
    radius, active, errors = np.full(256, .5), np.ones(tests.shape[1], dtype=bool), []
    for s in range(STEPS):
        gram = km(x, x)
        shifted = x + DT*MODEL.closed_loop_rhs(x, u)
        alpha = np.linalg.solve((gram-km(x, shifted))/DT, MODEL.running_cost(x, u))
        values = km(x, tests) @ alpha
        errors.append(np.linalg.norm((values-reference)[active])/np.linalg.norm(reference[active]))
        print(f'{"shrink" if shrink else "fixed ":6s} {s:2d}: N={x.shape[1]:4d}, error={errors[-1]:.3e}')
        if s == STEPS-1:
            break
        gradient = kernel.product_kernel_gradient(KERNEL, x, alpha, x)
        u = MODEL.control_from_value_gradient(x, gradient)
        if shrink:
            # Boundary minimum replaces the fixed Van-der-Pol level in the repo.
            level = .99**(1/(s+1)) * np.min(km(x, rays*radius) @ alpha)
            low, high = np.zeros_like(radius), radius.copy()
            for _ in range(25):
                middle = (low+high)/2
                inside = km(x, rays*middle) @ alpha < level
                low, high = np.where(inside, middle, low), np.where(inside, high, middle)
            radius = low
            active &= values < level
            keep = gram @ alpha < level
            if np.count_nonzero(keep) < 10 or not np.any(active):
                raise RuntimeError('Too few retained points; refine the training/test grid.')
            x, u = x[:, keep], u[:, keep]
    return np.array(errors), values, active, rays*radius


if __name__ == '__main__':
    training, tests = disk_grid(50), disk_grid(40, .49)
    reference = np.empty(tests.shape[1])
    for j, point in enumerate(tests.T):
        reference[j] = MODEL.solve_open_loop_bvp(point, 60., 101)[2]
        if (j+1) % 50 == 0:
            print(f'PMP references: {j+1}/{tests.shape[1]}', flush=True)
    fixed = policy_iteration(training, tests, reference, False)
    shrinking = policy_iteration(training, tests, reference, True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for result, label in [(fixed, 'Fixed initial disk'), (shrinking, 'Shrinking domain')]:
        axes[0].semilogy(result[0], '-o', label=label)
    axes[0].set(xlabel='Policy iteration', ylabel='Relative value error (PMP reference)')
    axes[0].grid(alpha=.25)
    axes[0].legend()
    scatter = axes[1].scatter(*tests, c=np.abs(fixed[1]-reference), s=18, cmap='magma')
    boundary = shrinking[3]
    axes[1].plot(*np.c_[boundary, boundary[:, :1]], color='deepskyblue',linewidth=3, label='Final reduced domain')
    axes[1].set(aspect='equal', xlabel='$x_1$', ylabel='$x_2$', title='Final pointwise absolute error: fixed domain')
    axes[1].legend(fontsize=10, loc="lower right")
    fig.colorbar(scatter, ax=axes[1], label='Error against PMP reference')
    Path('figures').mkdir(exist_ok=True)
    fig.savefig('figures/counterexample_pmp.pdf')
    plt.show()
