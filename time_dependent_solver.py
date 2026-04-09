import torch
from torch import Tensor
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy as sp
# TODO: figure out why this isn't working
# import helpers
# from helpers import sign_change, trace_2by2, inverse_2by2, collapse, clean_input

# NOTE: DO NOT USE THIS FILE RIGHT NOW

# necessary for linear algebra
torch.backends.cuda.preferred_linalg_library("magma")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# set default datatype of tensors
DTYPE = torch.complex128

class TimeDependentSolver:
    """Class to solve for the eigenvalues and eigenfunctions of a 1-D time-dependent periodic system.

    Attributes:
        TODO: fill this in
    """

    def __init__(
        self,
        V: Callable[[Tensor, int], Tensor],
        x_min: float = 0.0,
        x_max: float = 1.0,
        x_steps: int = 1024,
        f_steps_min: int = 1,
        omega: float = 1.0,
    ):
        """Initialize the class.

        Args:
            V (Callable[[Tensor, int], Tensor]): Function for the potential given as fourier coefficeints (TODO: consider better description here).
            x_min (float, optional): Minimum x value for one period. Defaults to 0.0.
            x_max (float, optional): Maximum x value for one period. Defaults to 1.0.
            x_steps (int, optional): Number of x steps between x_min and x_max (excluding x_min). Defaults to 1024.
            f_steps_min (int, optional): Minimum number of frequency steps. Defaults to 1.
            omega (float, optional): Frequency spacing between steps. Defaults to 1.0.
        """
        self.V = V
        self.f_steps_min = f_steps_min

        assert x_min == 0, "x_min must be zero (for now)."
        self.dx = (x_max + x_min) / x_steps
        self.x_vals_full = torch.linspace(
            x_min, x_max, x_steps + 1, device=DEVICE
        )
        # throwing out first element corresponding to identity
        self.x_vals = self.x_vals_full[1:]

        self.max_V = torch.sum(
            torch.abs(
                torch.max(V(self.x_vals, f_steps_min).real, dim=0).values
            )
        ).item()
        # TODO: is .item() really necessary?
        self.n_floquet = int(
            np.ceil(np.max([(4 * self.max_V) // omega, f_steps_min]))
        )
        self.f_steps = 2 * self.n_floquet + 1
        self.f_vals = torch.diag(
            torch.linspace(
                -self.n_floquet * omega,
                self.n_floquet * omega,
                self.f_steps,
                device=DEVICE,
                dtype=DTYPE,
            )
        )

    def delta_squared(self, E: Tensor) -> Tensor:
        """Generates the delta-squared object.

        Args:
            E (Tensor): Tensor of energies to compute delta-squared.

        Returns:
            delta_squared: Delta-squared object as given in write-up (TODO).
        """
        fourier_coeffs = self.V(self.x_vals, self.f_steps_min)
        V_shape = self.x_vals.shape + self.f_vals.shape
        V = torch.zeros(V_shape, device=DEVICE, dtype=DTYPE)

        V += fourier_coeffs[:, self.f_steps_min].unsqueeze(-1).unsqueeze(
            -1
        ).expand(V_shape) * torch.eye(
            self.f_steps, device=DEVICE, dtype=DTYPE
        ).expand(V_shape)

        for j in range(0, self.f_steps_min):
            i = j + 1
            diagonal_upper = fourier_coeffs[:, self.f_steps_min + i].unsqueeze(
                -1
            ).unsqueeze(-1).expand(V_shape) * torch.diag(
                torch.ones(self.f_steps - i, device=DEVICE, dtype=DTYPE),
                diagonal=i,
            ).expand(V_shape)
            diagonal_lower = fourier_coeffs[:, self.f_steps_min - i].unsqueeze(
                -1
            ).unsqueeze(-1).expand(V_shape) * torch.diag(
                torch.ones(self.f_steps - i, device=DEVICE, dtype=DTYPE),
                diagonal=-i,
            ).expand(V_shape)

            V += diagonal_lower + diagonal_upper

        V += self.f_vals.expand(V_shape)

        return 2 * (
            V.expand(E.shape + V.shape)
            - E.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(E.shape + V.shape)
            * torch.eye(self.f_steps, device=DEVICE, dtype=DTYPE).expand(
                E.shape + V_shape
            )
        )

    def a(self, E: Tensor) -> Tensor:
        delta_squared = self.delta_squared(E)
        identity = torch.eye(self.f_steps, device=DEVICE, dtype=DTYPE).expand(
            delta_squared.shape
        )

        a = torch.zeros(
            E.shape + self.x_vals.shape + (2 * self.f_steps, 2 * self.f_steps),
            device=DEVICE,
            dtype=DTYPE,
        )

        # NOTE: other terms 0
        a[..., 0 : self.f_steps, self.f_steps : (2 * self.f_steps)] = identity
        a[..., self.f_steps : (2 * self.f_steps), 0 : self.f_steps] = (
            delta_squared
        )

        return a

    def a_squared(self, E: Tensor) -> Tensor:
        delta_squared = self.delta_squared(E)

        a_squared = torch.zeros(
            E.shape + self.x_vals.shape + (2 * self.f_steps, 2 * self.f_steps),
            device=DEVICE,
            dtype=DTYPE,
        )

        # NOTE: other terms 0
        a_squared[..., 0 : self.f_steps, 0 : self.f_steps] = delta_squared
        a_squared[
            ...,
            self.f_steps : (2 * self.f_steps),
            self.f_steps : (2 * self.f_steps),
        ] = delta_squared

        return a_squared

    def lower_tri_a_exp(self, E: Tensor) -> Tensor:
        delta_squared = self.delta_squared(E)
        delta_squared_dx = delta_squared * self.dx**2
        identity = torch.eye(self.f_steps, device=DEVICE, dtype=DTYPE)
        a = self.a(E)

        # using gershegorin circle thm
        z_max = (
            2 * self.f_vals.real.max().item() + self.max_V * self.f_steps_min
        )
        # z_max ^n / n! < 10^-32 approx e^-64
        # w/ stirling approx n ln(z_max) - n ln(n) < -64
        # divide by n: ln(z_max/n) < -64/n w/ 0 < n < 64
        # => ln(n/z_max) > 1
        # => n \approx 3 * z_max
        # n_cutoff = int(np.min((3 * z_max, 64)))
        n_cutoff = 4

        tri_shape = (n_cutoff, n_cutoff)
        ones = torch.ones(tri_shape, device=DEVICE, dtype=DTYPE)
        lower_tri = torch.kron(
            torch.tril(ones).unsqueeze(0).unsqueeze(0), delta_squared_dx
        )
        upper_tri = torch.kron(
            torch.triu(ones, diagonal=1).unsqueeze(0).unsqueeze(0),
            identity.expand(delta_squared_dx.shape),
        )

        even_factors = torch.nan_to_num(
            torch.diag(1 / (2 * torch.arange(n_cutoff, device=DEVICE))),
            nan=1,
            posinf=1,
            neginf=1,
        )
        odd_factors = torch.diag(
            1 / (2 * torch.arange(n_cutoff, device=DEVICE) + 1)
        )
        factors = torch.matmul(even_factors, odd_factors)
        block_factors = torch.kron(
            factors.unsqueeze(0).unsqueeze(0),
            identity.expand(delta_squared_dx.shape),
        )

        odd_factors_plus_one = 1 / (
            2 * torch.arange(1, n_cutoff + 1, device=DEVICE) + 1
        )

        delta_squared_tri = torch.matmul(lower_tri, block_factors) + upper_tri

        delta_squared_blocks = torch.stack(
            torch.stack(
                delta_squared_tri.tensor_split(n_cutoff, dim=-2), dim=-3
            ).tensor_split(n_cutoff, dim=-1),
            dim=0,
        )

        delta_squared_collapsed = collapse(delta_squared_blocks)
        delta_squared_collapsed_odd = (
            delta_squared_collapsed
            * odd_factors_plus_one.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(delta_squared_collapsed.shape)
            * self.dx
        )

        a_exp_even = torch.kron(
            torch.eye(2, device=DEVICE, dtype=DTYPE).unsqueeze(0).unsqueeze(0),
            torch.sum(delta_squared_collapsed, dim=-3),
        )
        a_exp_odd = torch.kron(
            torch.eye(2, device=DEVICE, dtype=DTYPE).unsqueeze(0).unsqueeze(0),
            torch.sum(delta_squared_collapsed_odd, dim=-3),
        )

        a_exp = a_exp_even + torch.matmul(a, a_exp_odd)

        norm = (
            torch.exp(
                -torch.sqrt(
                    delta_squared.real.max(dim=-1).values.max(dim=-1).values
                )
                * self.dx
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(a_exp.shape)
        )

        return a_exp, norm

    def matrix_a_exp(self, E: Tensor) -> Tensor:
        a = self.a(E)
        a_exp = torch.linalg.matrix_exp(a * self.dx)
        delta_squared = self.delta_squared(E)

        norm = (
            torch.exp(
                -torch.sqrt(
                    delta_squared.real.max(dim=-1)
                    .values.max(dim=-1)
                    .values.to(DTYPE)
                )
                * self.dx
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(a_exp.shape)
        )

        return a_exp, norm

    def a_exp_inv(self):
        pass

    def collapse_a_exp(
        self, E: Tensor, exp_method: str = "matrix_exp"
    ) -> Tensor:
        match exp_method:
            case "lower_tri":
                a_exp, norm = self.lower_tri_a_exp(E)
            case "matrix_exp":
                a_exp, norm = self.matrix_a_exp(E)
            case _:
                raise ValueError(
                    "Method must be one of 'lower_tri' or 'matrix_exp'"
                )
        a_exp = a_exp.movedim(0, 1)
        norm = norm.movedim(0, 1)

        return collapse(a_exp)  # / collapse(norm)

    def loss(
        self, E: Tensor, k: Tensor, exp_method: str = "matrix_exp"
    ) -> Tensor:
        a_exp_collapsed = self.collapse_a_exp(E, exp_method)
        identity = torch.eye(2 * self.f_steps, device=DEVICE, dtype=DTYPE)
        k = clean_input(k)

        monodromy_matrix = a_exp_collapsed.unsqueeze(0).expand(
            k.shape + a_exp_collapsed.shape
        )  # - torch.exp(1j * k) * identity.expand(
        #     k.shape + a_exp_collapsed.shape
        # )

        # loss = torch.min(torch.linalg.svdvals(monodromy_matrix), dim=-1).values
        evals, evects = torch.linalg.eig(monodromy_matrix)
        evects = evects.squeeze()
        loss = evals + 1 / evals

        dc_overlap = (
            (
                torch.abs(
                    evects[
                        :, [self.n_floquet, self.n_floquet + self.f_steps], :
                    ]
                )
                ** 2
            )
            .sum(-2)
            .squeeze()
        )

        return loss.squeeze(), dc_overlap

    def plot_loss(
        self,
        E: Tensor,
        k: Tensor | NDArray | float | int,
        exp_method: str = "matrix_exp",
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        log_scale: bool = False,
    ) -> None:
        """Plots the loss function given energies and k values.

        Arguments:
            k (Tensor, NDArray, float, int): Tensor/array of k values, or a single number.
            E (Tensor): Tensor of energies.
            x_min (float, None): Minimum x value for the plot.
            x_max (float, None): Maximum x value for the plot.
            y_min (float, None): Minimum y value for the plot.
            y_max (float, None): Maximum y value for the plot.
            log_scale (bool): Whether to plot y on a log scale.
        """
        k = clean_input(k)
        E_cpu = E.cpu()

        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(len(k)):
            # not doing k's in parallel for memory savings
            # surely nobody will plot 1024 k curves anyways, right?
            loss = self.loss(E, k[i].item(), exp_method).cpu()
            ax.plot(E_cpu, loss, label=f"k = {k[i].item():.2f} Loss")

        ax.legend()
        ax.grid()
        ax.set_title("Losses vs. Energy Plot")

        ax.set_xlabel("Energy")
        ax.set_ylabel("Loss")

        ax.set_xlim(xmin=x_min, xmax=x_max)
        ax.set_ylim(ymin=y_min, ymax=y_max)

        if log_scale:
            ax.set_yscale("log")

        plt.show()