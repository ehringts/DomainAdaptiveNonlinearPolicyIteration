# Domain Adaptive Nonlinear Policy Iteration

Numerical experiments on the influence of the computational domain on policy iteration for infinite-horizon nonlinear optimal control. The main experiment compares a fixed computational domain with a sequence of contracted domains. 

The implementation accompanies:

> Tobias Ehring, Behzad Azmi, and Bernard Haasdonk. *On the Convergence of the Policy Iteration for Infinite-Horizon Nonlinear Optimal Control Problems*. arXiv:2507.09994, 2025. [Preprint](https://arxiv.org/abs/2507.09994) · [DOI](https://doi.org/10.48550/arXiv.2507.09994).

## Installation and execution

Use Python 3.10 or newer with NumPy, SciPy, and Matplotlib. Run the following commands from a terminal:

```bash
git clone https://github.com/ehringts/DomainAdaptiveNonlinearPolicyIteration.git   
cd DomainAdaptiveNonlinearPolicyIteration
python -m pip install numpy scipy matplotlib
python example.py
```

The script computes the PMP reference values, runs both policy-iteration variants, and saves `figures/counterexample_pmp.pdf`. It also displays the figure. For execution without a graphical display:

```bash
MPLBACKEND=Agg python example.py
```

## Main experiment

The model has two state and two control variables, with

$$
f(x)=0,\qquad g(x)=(I_2+xx^\top)B,\qquad
h(x)=\frac{x^\top Qx}{(1+\|x\|^2)^2},\qquad R=I_2,
$$

$$
B=\begin{pmatrix}1&0\\0&20\end{pmatrix},\qquad
Q=\begin{pmatrix}1&1/5\\1/5&17/400\end{pmatrix}.
$$

Both runs start on the disk $\Omega_{-1}=\{x:\|x\|<1/2\}$ with the feedback

$$
u_0(x)=-\frac{B^{-1}x}{1+\|x\|^2},
$$

which gives the initial closed loop $\dot x=-x$. The model also admits finite-cost escape trajectories, making it useful for investigating the role of domain contraction.

- **Fixed domain:** all initial training and test points are retained.
- **Shrinking domain:** after each policy improvement, training and test points are restricted to a sublevel set of the current value surrogate. Its level is the numerically approximated boundary minimum multiplied by $0.99^{1/(\eta+1)}$.

Both variants use the same unsymmetric kernel collocation: policy evaluation discretizes the dynamic programming identity with an explicit Euler step and solves a dense linear system. Policy improvement uses the gradient of the kernel surrogate. The domain boundary is sampled along 2048 radial directions.

### Experiment settings

To reproduce the refined-grid configuration, use the following line in the main block of `example.py`:

```python
training, tests = disk_grid(50), disk_grid(40, .49)
```

| Setting | Value |
| --- | --- |
| Initial training grid | Uniform 50 × 50 grid on `[-0.5, 0.5]²`, restricted to the disk: **1876 points** |
| Initial test grid | Uniform 40 × 40 grid on `[-0.49, 0.49]²`, restricted to the disk: **1240 points** |
| Euler step `DT` | `5e-7` |
| Policy evaluations `STEPS` | `9`, indexed from 0 to 8 |
| Main kernel | Matérn, `gamma=2.0`, `case=2` |

The grid sizes are Cartesian grid dimensions before restriction to the disk. The console reports the iteration number, current number of training points, and relative test error.

### PMP reference values

`EscapeCounterexample.solve_open_loop_bvp` computes numerical reference values independently of the kernel approximation using `scipy.integrate.solve_bvp`. It works in the transformed variables

$$
z=\frac{x}{\sqrt{1+\|x\|^2}},\qquad
\frac{\mathrm d\tau}{\mathrm dt}=1-\|z\|^2,\qquad
w=\frac{u}{\sqrt{1-\|z\|^2}}.
$$

The transformed dynamics are $z'=Bw$, with running cost $z^\top Qz+\|w\|^2$. The PMP state and adjoint equations are solved for two types of candidates:

- A stabilizing branch, approximated on the transformed horizon $\tau\in[0,60]$ with terminal adjoint $p(60)=0$.
- Escape branches reaching $\|z(T)\|=1$ at a free terminal time, with the corresponding transversality conditions. This boundary represents infinite-time escape in the original variables.

The least cost among the converged feasible candidates defines the numerical reference. The solver uses 101 initial mesh points, adaptive refinement, and tolerance `1e-8`. The main script recomputes these references on each run.

### Reading the figure

The **left panel** shows the relative discrete $\ell^2$ error at each iteration:

$$
E_\eta=
\left(
\frac{\sum_{x\in X_\eta^{\mathrm{Test}}}|\widehat v_\eta(x)-v_{\mathrm{ref}}(x)|^2}
{\sum_{x\in X_\eta^{\mathrm{Test}}}|v_{\mathrm{ref}}(x)|^2}
\right)^{1/2}.
$$

Here $X_\eta^{\mathrm{Test}}$ is the test set in the current domain of each run. Thus, the fixed-domain curve uses the full test set, while the shrinking-domain curve uses the retained points.

The **right panel** shows the final **pointwise absolute error of the fixed-domain run**:

$$
e_{\mathrm{abs}}(x)=|\widehat v_8^{\mathrm{fixed}}(x)-v_{\mathrm{ref}}(x)|.
$$

Each dot represents one test point, and its color indicates the error magnitude. The light-blue curve marks the final domain obtained by the separate shrinking run. All errors are measured against the numerical PMP reference.

## Code structure and kernel selection

The main experiment needs three source files:

| File | Purpose |
| --- | --- |
| [`counterexample.py`](counterexample.py) | Grids, both PI variants, error evaluation, and plotting |
| [`functions/kernel.py`](functions/kernel.py) | Selectable radial kernels, product-kernel matrices, and gradients |
| [`functions/model.py`](functions/model.py) | `EscapeCounterexample`, its PMP reference solver, and the `VanDerPol` model |

Select the kernel at the top of `counterexample.py`:

```python
KERNEL = kernel.make_kernel('matern', case=2)    # Default gamma: 2.0
# KERNEL = kernel.make_kernel('wendland', case=2)  # Default gamma: 0.3
```

Pass `gamma=...` to choose a different shape parameter. The available radial kernels are the Matérn kernel used in the original implementation and the compactly supported Wendland C4 kernel. Both are multiplied by $(x^\top y)^2$, enforcing $\widehat v(0)=0$ and $\nabla\widehat v(0)=0$.



These experiments use the shared routines in `functions/auxFunctions.py`, with `functions/observer.py` and `functions/plotStyle.py` providing iteration history and plot settings. The Van der Pol error experiment caches reference values and error histories in `data/`; remove the corresponding `.npz` files when recomputing after parameter changes.
