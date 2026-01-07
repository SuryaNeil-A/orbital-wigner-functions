import torch
from collections.abc import Callable
import matplotlib.pyplot as plt

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
DTYPE = torch.complex64


def sign_change(matrices: torch.Tensor) -> torch.Tensor:
    """Computes indices of a tensor where the sign changes.

    Leading dimensions are treated as batch dimensions.

    Arguments:
        matrices (Tensor): Tensor of matrices to evalute.

    Returns:
        change_index (Tensor): Tensor of indices where the matrix changes sign.
    """
    signed_matrices = torch.sign(matrices)
    # takes adjacent elements and multiplies, if <= 0 then a sign change occured
    change_index = torch.argwhere(
        signed_matrices[:-1] * signed_matrices[1:] <= 0
    ).squeeze(dim=1)
    return change_index


def trace_2by2(matrices: torch.Tensor) -> torch.Tensor:
    """Computes the batched trace of a 2x2 matrix.

    Leading dimensions are treated as batch dimensions.

    Arguments:
        matrices (Tensor): Tensor of matrices, where the last two dimensions must be 2x2.

    Returns:
        trace (Tensor): Trace of the input matrices.
    """
    assert matrices.shape[-2:] == (2, 2), (
        "Matrix must be a 2x2 matrix in final two dimensions."
    )
    return matrices[..., 0, 0] + matrices[..., 1, 1]


def inverse_2by2(matrices: torch.Tensor) -> torch.Tensor:
    """Computes the batched inverse of a 2x2 matrix.

    Leading dimensions are treated as batch dimensions.

    Arguments:
        matrices (Tensor): Tensor of matrices, where the last two dimensions must be 2x2.

    Returns:
        inverse (Tensor): Inverse of matrices

    """
    assert matrices.shape[-2:] == (2, 2), (
        "Matrix must by a 2x2 matrix in the final two dimensions"
    )
    inverse = torch.zeros_like(
        matrices, dtype=matrices.dtype, device=matrices.device
    )

    # analytical formula for 2x2 inverse
    inverse[..., 0, 0] = matrices[..., 1, 1]
    inverse[..., 1, 1] = matrices[..., 0, 0]
    inverse[..., 0, 1] = -1 * matrices[..., 0, 1]
    inverse[..., 1, 0] = -1 * matrices[..., 1, 0]

    return inverse


def collapse(matrices: torch.Tensor) -> torch.Tensor:
    """Collapses the first dimension of a tensor by matrix multiplying pairwise.

    Following the behavior of torch.matmul, all but the first and (possibly) last two dimensions are treated as batch dimensions.
    Assumes that the matrix to multiply is square.

    Arguments:
        matrices (Tensor): A tensor of matrices to collapse.

    Returns:
        collapsed_matrices (Tensor): Matrices with first dimension collapsed.
    """
    assert matrices.shape[-1] == matrices.shape[-2], "Matrix must be square."

    while matrices.shape[0] > 1:
        if matrices.shape[0] % 2 == 1:
            # splits and multiplies last two matrices
            last_prod = torch.matmul(matrices[-2], matrices[-1])
            # and reforms matrices with the product replacing the last two
            matrices = torch.cat(
                (matrices[:-2], last_prod.unsqueeze(0)), dim=0
            )

        num_slices = matrices.shape[0]
        # split into pairs
        matrices = matrices.view((num_slices // 2, 2) + matrices.shape[1:])
        # and multiply
        matrices = torch.matmul(matrices[:, 0], matrices[:, 1])

    return matrices.squeeze(0)


class ContinuumSolver:
    """
    Class to solve for the eigenvalues and eigenfunctions of a 1-D period system.

    Attributes:
        TODO: fill this in
    """

    def __init__(
        self,
        V: Callable[[torch.Tensor], torch.Tensor],
        x_min: float = 0,
        x_max: float = 1,
        x_steps: float = 1024,
    ):
        """Initialize the class.

        Arguments:
            V: function of the potential (for one period).
            x_min: x value of the left-hand side of one period.
            x_min: x value of the right-hand side of one period.
            x_steps: number of steps to take up to and including x_max (but not x_min).
        """

        self.V = V
        self.dx = (x_max + x_min) / (x_steps + 1)

        self.x_vals_full = torch.linspace(
            x_min, x_max, x_steps + 1, device=DEVICE
        )
        # throwing out first element corresponding to identity
        self.x_vals = self.x_vals_full[1:]

    def a(self, k: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Constructs the A matrix.

        Arguments:
            k (Tensor): Tensor of k values.
            E (Tensor): Tensor of energies.

        Returns:
            a (Tensor): The A matrix with the shape of (x, k, E, 2, 2).
        """
        a = torch.zeros(
            self.x_vals.shape + k.shape + E.shape + (2, 2),
            dtype=DTYPE,
            device=DEVICE,
        )
        V_expanded = (
            self.V(self.x_vals)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )
        # converting to complex numbers so roots of negatives are evaluated correctly
        E_expanded = (
            E.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.x_vals.shape + k.shape + E.shape)
            .to(DTYPE)
        )
        k_expanded = (
            k.unsqueeze(0)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )

        # using analytical formula to create A matrix
        a[..., 0, 0] = 0
        a[..., 0, 1] = 1
        # can't use delta here due to numerical errors
        a[..., 1, 0] = 2 * (V_expanded - E_expanded) + k_expanded**2
        a[..., 1, 1] = -2j * k_expanded

        # moving dimensions so output is (x, k, E, 2, 2) to make leading dimensions batch dimensions
        return a
    
    def delta(self, k: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Creates the delta quantity which is $\sqrt(2(V - E))$.

        Arguments:
            k (Tensor): Tensor of k values to find delta at (so it returns with the correct shape for later functions).
            E (Tensor): Tensor of energies to evaluate at.

        Returns:
            delta (Tensor): $\sqrt(2*(V - E))$ with the shape of (x, k, E).
        """
        V_expanded = (
            self.V(self.x_vals)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )
        # converting to complex numbers so roots of negatives are evaluated correctly
        E_expanded = (
            E.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.x_vals.shape + k.shape + E.shape)
            .to(DTYPE)
        )

        return torch.sqrt(2 * (V_expanded - E_expanded))

    def a_exp(self, k: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Constructs the matrix exponential of A using the analytical formula.

        Arguments:
            k (Tensor): Tensor of k values.
            E (Tensor): Tensor of energies.

        Returns:
            a_exp (Tensor): Matrix exponential of A with the shape of (x, k, E, 2, 2).
        """
        delta = self.delta(k, E)
        a_exp = torch.zeros(
            self.x_vals.shape + k.shape + E.shape + (2, 2),
            device=DEVICE,
            dtype=DTYPE,
        )
        k_expanded = (
            k.unsqueeze(0)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )

        # using analytical formula to construct matrix exponential of A
        # when x->0 in sinh(ax)/x the limit evaluates to a, so wrapping sinh(delta * dx)/ delta in nan_to_num
        a_exp[..., 0, 0] = torch.cosh(delta * self.dx) + (
            1j
            * k_expanded
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )
        a_exp[..., 0, 1] = torch.nan_to_num(
            torch.sinh(delta * self.dx) / delta, nan=self.dx
        )
        a_exp[..., 1, 0] = delta * torch.sinh(delta * self.dx) + (
            (k_expanded**2)
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )
        a_exp[..., 1, 1] = torch.cosh(delta * self.dx) + (
            -1j
            * k_expanded
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )

        # normalzing a_exp by multiplying by e^-delta
        norm = (
            torch.abs(torch.exp(-1 * delta * self.dx))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(a_exp.shape)
        )
        return a_exp * norm

    def a_exp_inv(
        self, k: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """Constructs the inverse of the matrix exponential of A using the known inverse fomrula.

        Instead of computing a_exp first and inverting, this inverts first and then computes a_exp_inv directly.

        Arguments:
            k (Tensor): Tensor of k values.
            E (Tensor): Tensor of energies.

        Returns:
            a (Tensor): Inverse matrix exponential of A with the shape of (x, k, E, 2, 2).
        """
        # TODO: figure out if this or other inverse is better
        delta = self.delta(k, E)
        a_exp_inv = torch.zeros(
            self.x_vals.shape + k.shape + E.shape + (2, 2),
            device=DEVICE,
            dtype=DTYPE,
        )
        k_expanded = (
            k.unsqueeze(0)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )

        # using analytical formula for the inverse of a 2x2 matrix to construct inverse
        # when x->0 in sinh(ax)/x the limit evaluates to a, so wrapping sinh(delta * dx)/ delta in nan_to_num
        a_exp_inv[..., 1, 1] = torch.cosh(delta * self.dx) + (
            1j
            * k_expanded
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )
        a_exp_inv[..., 0, 1] = -1 * torch.nan_to_num(
            torch.sinh(delta * self.dx) / delta, nan=self.dx
        )
        a_exp_inv[..., 1, 0] = -1 * delta * torch.sinh(delta * self.dx) - (
            (k_expanded**2)
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )
        a_exp_inv[..., 0, 0] = torch.cosh(delta * self.dx) + (
            -1j
            * k_expanded
            * torch.nan_to_num(
                torch.sinh(delta * self.dx) / delta, nan=self.dx
            )
        )

        # normalzing a_exp_inv by multiplying by e^delta
        # positive so it cancels with a_exp
        norm = (
            torch.abs(torch.exp(delta * self.dx))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(a_exp_inv.shape)
        )
        return a_exp_inv * norm

    def derivative_a_exp(self, k, E):
        """TODO"""
        pass

    def second_derivative_a_exp(self, k, E):
        """TODO"""
        pass

    def collapse_a_exp(self, k: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """Collapse the x steps on the a_exp matrices into one single matrix.

        Arguments:
            k (Tensor): Tensor of k values.
            E (Tensor): Tensor of energies.

        Returns:
            collapsed_matrices (Tensor): Collapsed matrices with the shape of (k, E, 2, 2).
        """
        return collapse(self.a_exp(k, E))

    def collapse_derivative_a_exp(self, k, E):
        """TODO"""
        pass

    def collapse_second_derivative_a_exp(self, k, E):
        """TODO"""
        pass

    def loss(
        self, k: torch.Tensor | float | int, E: torch.Tensor | float | int
    ) -> torch.Tensor:
        """Compputes the loss function for the system given k and E values.

        Arguments:
            k (Tensor, float, int): Tensor of k values (or a single number, which it will convert to a Tensor).
            E (Tensor, float, int): Tensor of energies (or a single number, which it will convert to a Tensor).

        Returns:
            loss (Tensor): Tensor of loss values for the given k and E, with shape (k, E).
        """
        # TODO: consider if there is an error with k/E as numpy array
        if type(E) is not torch.Tensor:
            print("converting")
            E = torch.tensor([E], device=DEVICE)
        if type(k) is not torch.Tensor:
            k = torch.tensor([k], device=DEVICE)

        norm = torch.sum(torch.real(self.delta(k, E) * self.dx), dim=0)
        matrices = self.collapse_a_exp(k, E)

        loss = torch.real(trace_2by2(matrices)) * torch.exp(norm)
        return loss.squeeze()

    def im_loss(self, k, E):
        """TODO"""
        pass

    def derivative_loss(self, k, E):
        """TODO"""
        pass

    def second_derivative_loss(self, k, E):
        """TODO"""
        pass

    def eigenvals(self):
        """Finds the eigenvalues for the given system."""
        pass

    def eigenstate_symmetric(self):
        pass

    def eigenstate_asymmetric(self):
        pass

    def plot_loss(
        self,
        k: torch.Tensor | float | int,
        E: torch.Tensor,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        log_scale: bool = False,
    ) -> None:
        """Plots the loss function given a Tensor of energies and k values.

        Arguments:
            k (Tensor, float, int): Tensor or number of k values.
            E (Tensor): Tensor of energies.
            x_min (float, None): Minimum x value for the plot.
            x_max (float, None): Maximum x value for the plot.
            y_min (float, None): Minimum y value for the plot.
            y_max (float, None): Maximum y value for the plot.
            log_scale (bool): Whether to plot y on a log scale.
        """
        loss = self.loss(k, E).cpu()
        E_cpu = E.cpu()

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(E_cpu, loss, label="Loss", color="tab:orange")

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

    def plot_band_structure(self):
        pass

    def plot_eigenstate(self):
        pass
