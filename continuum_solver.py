import torch
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
from numpy import float64, ndarray
from numpy.typing import NDArray
import scipy as sp

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
        signed_matrices[..., :-1] * signed_matrices[..., 1:] <= 0
    ).squeeze(-1)
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


def clean_input(input: torch.Tensor | NDArray | float | int) -> torch.Tensor:
    """Cleans up input value for processing by the solver.

    Example: If input is a float, converts it to a tensor.
    Example 2: If input is a single element tensor (with empty shape), it converts it to a tensor with shape 1.

    Arguments:
        input (Tensor, NDArray, float, int): Input value to clean up.

    Returns:
        input (Tensor): Cleaned up input.
    """
    match input:
        case float() | int() | float64():
            input = torch.tensor([input], device=DEVICE)
        case ndarray():
            input = torch.from_numpy(input).to(device=DEVICE)
        case torch.Tensor():
            if input.shape == torch.Size([]):
                input = input.unsqueeze(0)
        case _:
            raise TypeError(
                "Type not supported. Try converting to float, Numpy Array, or PyTorch Tensor."
            )

    return input


def false_position(
    func: Callable[[torch.Tensor], torch.Tensor],
    x_min: float,
    x_max: float,
    tol: float = 1e-6,
    x_steps: int = 1024,
):
    """Solves for the roots of a function using the false position method.
    Leading dimensions of the function output are treated as batch dimensions.

    Arguments:
        func (Callabale): Function to find the roots of in given interval.
        x_min (float): Minimum x for interval.
        x_max (float): Maximum x for interval.
        tol (float): Tolerance for the error on the roots.
        x_steps (int): Number of x points to generate by default.
    """
    # first creates initial grid of points to find roots at
    x_vals_base = torch.linspace(x_min, x_max, x_steps, device=DEVICE)
    y_vals_base = func(x_vals_base)
    sign_change_y_base = sign_change(y_vals_base)

    for i in range(len(sign_change_y_base)):
        # copying data from base values into new tensors
        # this is so when restarting at a new root, the original linspace is unaffected
        x_vals = x_vals_base.detach().clone()
        y_vals = func(x_vals)
        sign_change_y = sign_change(y_vals)

        # initializing root value to left edge
        root = func(x_vals[sign_change_y[i]])
        while func(root) <= tol:
            x_1 = x_vals[sign_change_y[i]]
            x_2 = x_vals[sign_change_y[i] + 1]
            # calculates root by linear interpolation between sign crossing points
            root = (
                (x_1 * func(x_2) -  x_2 * func(x_1)) / (func(x_2) - func(x_1))
            )

            x_vals = torch.linspace(x_1, x_2, x_steps, device=DEVICE)
            y_vals = func(x_vals)
            sign_change_y = sign_change(y_vals)


def symmetric(ya: NDArray, yb: NDArray) -> NDArray:
    """Function to evaluate for symmetric period boundary conditions (derivative opposite sign).
    
    Arguments:
        ya (NDArray): (2,) array of y and its derivative values at the left edge.
        yb (NDArray): (2,) array of y and its derivative at the right edge.

    Returns:
        residuals (NDArray): (2,) array containing the residuals.
    """
    return np.array([ya[0] - yb[0], ya[1] + yb[1]])

def asymmetric(ya: NDArray, yb: NDArray) -> NDArray:
    """Function to evaluate for symmetric period boundary conditions (derivative opposite sign).
    
    Arguments:
        ya (NDArray): (2,) array of y and its derivative values at the left edge.
        yb (NDArray): (2,) array of y and its derivative at the right edge.

    Returns:
        residuals (NDArray): (2,) array containing the residuals.
    """
    return np.array([ya[0] - yb[0], ya[1] - yb[1]])

class ContinuumSolver:
    """Class to solve for the eigenvalues and eigenfunctions of a 1-D period system.

    Attributes:
        TODO: fill this in
    """

    def __init__(
        self,
        V: Callable[[torch.Tensor], torch.Tensor],
        x_min: float = 0.,
        x_max: float = 1,
        x_steps: int = 1024,
    ):
        """Initialize the class.

        Arguments:
            V: function of the potential (for one period).
            x_min: x value of the left-hand side of one period.
            x_min: x value of the right-hand side of one period.
            x_steps: number of steps to take up to and including x_max (but not x_min).
        """

        self.V = V
        assert x_min == 0, "x_min must be zero (for now)."
        self.dx = (x_max + x_min) / x_steps

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

    def a_exp_inv(self, k: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
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
        self,
        k: torch.Tensor | NDArray | float | int,
        E: torch.Tensor | NDArray | float | int,
    ) -> torch.Tensor:
        """Compputes the loss function for the system given k and E values.

        Arguments:
            k (Tensor, NDArray, float, int): Tensor/array of k values (or a single number, which it will convert to a Tensor).
            E (Tensor, NDArray, float, int): Tensor/array of energies (or a single number, which it will convert to a Tensor).

        Returns:
            loss (Tensor): Tensor of loss values for the given k and E, with shape (k, E).
        """
        # TODO: consider if there is an error with k/E as numpy array
        E = clean_input(E)
        # E = torch.tensor(E, device=DEVICE)
        k = clean_input(k)

        norm = torch.sum(
            torch.real(self.delta(clean_input(0.0), E) * self.dx), dim=0
        )
        matrices = self.collapse_a_exp(clean_input(0.0), E)

        loss = (torch.real(trace_2by2(matrices)) * torch.exp(norm)).squeeze()

        return (
            loss.unsqueeze(0).expand(k.shape + E.shape)
            - (2 * torch.cos(k)).unsqueeze(-1).expand(k.shape + E.shape)
        ).squeeze()

    def im_loss(self, k, E):
        """TODO"""
        pass

    def derivative_loss(self, k, E):
        """TODO"""
        pass

    def second_derivative_loss(self, k, E):
        """TODO"""
        pass

    def solve_eigenvals(
        self,
        k: torch.Tensor | NDArray | float | int,
        E_min: float = 0,
        E_max: float = 10,
        E_tol: float = 1e-6,
        E_steps: int = 1024,
        save_eigenvals: bool = True,
    ) -> torch.Tensor:
        """Finds the eigenvalues for the given system using the false-point method.

        Arguments:
            k (Tensor, NDAarray, float): Tensor/array of k values (or a single number, which it will convert to a Tensor).
            E_min (float): Minimum energy for interval.
            E_max (float): Maximum energy for interval.
            E_tol (float): Tolerance for the error on the eigenvalues.
            E_steps (int): Number of E points to generate for loss function.
        Returns:
            eigenvals (Tensor): Tensor of eigenvalues for the given k.
        """
        k = clean_input(k)

        eigenvals = false_position(
            lambda E: self.loss(k, E), E_min, E_max, E_tol, E_steps
        )

        if save_eigenvals:
            self.eigenvals = eigenvals
            self.eigenvals_k = k

        return eigenvals

    def transfer_matrix(
        self, k: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """Constructs the transfer matrix using the analytical formula.

        Arguments:
            k (Tensor): Tensor of k values.
            E (Tensor): Tensor of energies.

        Returns:
            transfer_matrix (Tensor): Transfer matrix of A with the shape of (x, k, E, 2, 2).
        """
        delta = self.delta(k, E)
        transfer_matrix = torch.zeros(
            self.x_vals.shape + k.shape + E.shape + (2, 2),
            device=DEVICE,
            dtype=DTYPE,
        )
        V_expanded = (
            self.V(self.x_vals)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(self.x_vals.shape + k.shape + E.shape)
        )

        # using analytical formula to construct transfer matrix
        transfer_matrix[..., 0, 0] = (
            2 * torch.cosh(delta * self.dx)
            - 2 * V_expanded * self.dx * torch.sinh(delta * self.dx) / delta
        )
        transfer_matrix[..., 0, 1] = -1
        transfer_matrix[..., 1, 0] = 1
        transfer_matrix[..., 1, 1] = 0

        # normalzing a_exp by multiplying by e^-delta
        norm = (
            torch.abs(torch.exp(-1 * delta * self.dx))
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(transfer_matrix.shape)
        )
        return transfer_matrix * norm

    def solve_eigenstate_matrix(
        self,
        E: torch.Tensor | NDArray | float | int,
        initial_condit: str = "symmetric",
    ) -> torch.Tensor:
        """Computes the eigenstate for the system given k and E values.

        Arguments:
            E (Tensor, NDArray, float, int): Tensor/array of energies (or a single number, which it will convert to a Tensor).
            initial_condit (str): Initial conditions to apply, either symmetric or antisymmetric.

        Returns:
            eigenstate (Tensor): Tensor of eigenstate for given k and e, with shape (x).
        """
        E = clean_input(E)

        eigenstate = torch.zeros(
            self.x_vals_full.shape + E.shape + (2, 1),
            device=DEVICE,
            dtype=DTYPE,
        )
        deviation = torch.zeros(self.x_vals.shape)
        norm_list = torch.zeros(self.x_vals.shape)
        norm_total = torch.ones(self.x_vals_full.shape)
        norm_total[0] = 1

        transfer_matrix = self.transfer_matrix(clean_input(0.0), E).squeeze(1)

        match initial_condit:
            case "symmetric":
                initial_conditions = (
                    torch.tensor([[1e-3], [0]], device=DEVICE)
                    .unsqueeze(0)
                    .expand(E.shape + (2, 1))
                )
            case "antisymmetric":
                initial_conditions = (
                    torch.tensor([[0], [1]], device=DEVICE)
                    .unsqueeze(0)
                    .expand(E.shape + (2, 1))
                )
            case _:
                raise ValueError(
                    "initial_condit must be either 'symmetric' or 'antisymmetric'."
                )

        eigenstate[0] = initial_conditions

        for i in range(len(self.x_vals)):
            eigenstate[i + 1] = torch.matmul(transfer_matrix[i], eigenstate[i])
            norm = eigenstate[i + 1, ..., 0, 0]
            # print(f"Norm = {norm}")
            norm_list[i] = norm
            # print(norm_total[i + 1])

            # if i > 512:
            norm_total[i + 1] = norm * norm_total[i]

            eigenstate[i + 1] *= torch.nan_to_num(1 / norm, nan=1.0)
            deviation[i] = torch.abs(
                eigenstate[i + 1, ..., 1, 0] - eigenstate[i, ..., 0, 0]
            )
            # assert deviation <= 0.1, "Deviation too high, something went wrong."

        return eigenstate.squeeze(1), norm_total

    def a_ode(
        self,
        t: NDArray | float | int,
        y: NDArray,
        k: float | int,
        E: float | int,
    ) -> NDArray:
        """ODE to solve the eigenstate given a particular k and E value.

        Returns a vector of the derivatives of the y vector.
        Everything is done in numpy because scipy cannot take in torch Tensors.

        Arguments:
            t (NDArray, float, int): Array (or number) of x values to evaluate the derivative at (ODE solver treats x as time or t).
            y (NDarray): A (2,) array input y vector.
            k: (float, int): The k value to find the derivative at.
            E: (float, int): The energy to find the derivative at.

        Returns:
            derivative (NDarray): Derivative of the input evaluated according to the A matrix.
        """

        a = np.zeros((2, 2), dtype=np.complex128)

        a[0, 0] = 0
        a[0, 1] = 1
        a[1, 0] = 2 * (self.V(clean_input(t).numpy(force=True)) - E) + k**2
        a[1, 1] = -2j * k

        return np.matmul(y, a.T)

    def solve_eigenstate_ode(
        self,
        k: float | int,
        E: float | int,
        solution_type: str = "auto"
    ) -> NDArray:
        """Solves the coupled ODEs to find the eigenstate at a particular E and k value.

        Arguments:
            k (float, int): k value to use when solving the ODE.
            E (float, int): E value to use when solving  the ODE.

        Returns:
            eigenstate (NDarray): Eigenstate evaluated at the class' initialized x values.

        """
        right_half_x_vals = self.x_vals_full[(len(self.x_vals_full) // 2):]
        match solution_type:
            case "auto":
                try:
                    right_sol = sp.integrate.solve_ivp(
                        lambda t, y: self.a_ode(t, y, 0, E),
                        t_span=(right_half_x_vals[0].item(), right_half_x_vals[-1].item()),
                        y0=np.array([1, 0], dtype=np.complex64),
                        t_eval=right_half_x_vals.numpy(force=True),
                        method="DOP853"
                    )
                    left_sol = right_sol.y[0][::-1]
                    assert np.abs(right_sol.y[1]).max() != np.abs(right_sol.y[1][-1])
                except AssertionError:
                    right_sol = sp.integrate.solve_ivp(
                        lambda t, y: self.a_ode(t, y, 0, E),
                        t_span=(right_half_x_vals[0].item(), right_half_x_vals[-1].item()),
                        y0=np.array([0, 1], dtype=np.complex64),
                        t_eval=right_half_x_vals.numpy(force=True),
                        method="DOP853"
                    )
                    left_sol = -1*right_sol.y[0][::-1]
                    assert np.abs(right_sol.y[0]).max() != np.abs(right_sol.y[0][-1]), "Solver could not find a valid eigenstate."
            case "symmetric":
                right_sol = sp.integrate.solve_ivp(
                        lambda t, y: self.a_ode(t, y, 0, E),
                        t_span=(right_half_x_vals[0].item(), right_half_x_vals[-1].item()),
                        y0=np.array([1, 0], dtype=np.complex64),
                        t_eval=right_half_x_vals.numpy(force=True),
                        method="DOP853"
                    )
                left_sol = right_sol.y[0][::-1]
            case "antisymmetric":
                right_sol = sp.integrate.solve_ivp(
                        lambda t, y: self.a_ode(t, y, 0, E),
                        t_span=(right_half_x_vals[0].item(), right_half_x_vals[-1].item()),
                        y0=np.array([0, 1], dtype=np.complex64),
                        t_eval=right_half_x_vals.numpy(force=True),
                        method="DOP853"
                    )
                left_sol = -1*right_sol.y[0][::-1]
            case _:
                raise ValueError("Solution type must be either 'auto', 'symmetric', or 'antisymmetric'.")

        sol = np.concat((left_sol, right_sol.y[0][1:]))

        return sol / np.sum(np.abs(sol)**2 * self.dx)

    def eigenstate_symmetric(self):
        pass

    def eigenstate_asymmetric(self):
        pass

    def plot_loss(
        self,
        k: torch.Tensor | NDArray | float | int,
        E: torch.Tensor,
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
            loss = self.loss(k[i].item(), E).cpu()
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

    def plot_band_structure(self):
        pass

    def plot_eigenstate(
        self,
        k: torch.Tensor | NDArray | float | int,
        E: float | int,
        solution_type: str = "auto",
        energy_guess: float | int = 0.51,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        plot_real: bool = True,
        plot_imag: bool = False,
        plot_prob: bool = False,
        log_scale: bool = False,
    ) -> None:
        """Plots the eigenstate given energies and k values.

        Arguments:
            k (Tensor, NDArray, float, int): Tensor/array of k values, or a single number.
            E (float, int): Single number for energy to find eigenstate at.
            x_min (float, None): Minimum x value for the plot.
            x_max (float, None): Maximum x value for the plot.
            y_min (float, None): Minimum y value for the plot.
            y_max (float, None): Maximum y value for the plot.
            plot_real (bool): Whether to plot the real part of the eigenstate.
            plot_imag (bool): Whether to plot the imaginary part of the eigenstate.
            plot_prob (bool): Whether to plot the probability density of the eigenstate
            log_scale (bool): Whether to plot y on a log scale.
        """
        k = clean_input(k)
        x_cpu = self.x_vals_full.cpu()
        eigenvals = np.zeros(len(k))

        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(len(k)):
            # not doing k's in parallel for memory savings
            # surely nobody will plot 1024 k curves anyways, right?
            eigenvals[i] = sp.optimize.newton(
                lambda E: self.loss(k[i], E).item(), energy_guess, maxiter=20
            )
            
            eigenstate = self.solve_eigenstate_ode(
                k[i].item(), eigenvals[i], solution_type
            )
            if plot_real:
                ax.plot(
                    x_cpu,
                    np.real(eigenstate),
                    label=f"k = {k[i].item():.2f} Real Eigenstate",
                )
            if plot_imag:
                ax.plot(
                    x_cpu,
                    np.imag(eigenstate),
                    label=f"k = {k[i].item():.2f} Imaginary Eigenstate",
                )
            if plot_prob:
                ax.plot(
                    x_cpu,
                    np.abs(eigenstate) ** 2,
                    label=f"k = {k[i].item():.2f} Probability Density",
                )

        ax.legend()
        ax.grid()
        ax.set_title("Eigenstate Plot")

        ax.set_xlabel("x")
        ax.set_ylabel("Value")

        ax.set_xlim(xmin=x_min, xmax=x_max)
        ax.set_ylim(ymin=y_min, ymax=y_max)

        if log_scale:
            ax.set_yscale("log")
        
        plt.show()

        # fig2, ax2 = plt.subplots(figsize = (10, 8))

        # ax2.plot(k.cpu(), eigenvals, label="Band Structure")
        # print(eigenvals)

        # ax2.set_title("Band Structure Plot")
        # ax2.legend()
        # ax2.grid()

        # ax2.set_xlabel("k")
        # ax2.set_ylabel("E")

        # plt.show()


class TimeDependentSolver:
    """Class to solve for the eigenvalues and eigenfunctions of a 1-D time-dependent periodic system.

    Attributes:
        TODO: fill this in
    """

    def __init__(
        self,
        V: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_min: float = 0.,
        x_max: float = 1.,
        x_steps: int = 1024,
        t_min: float = 0.,
        t_max: float = 1.,
        t_steps: int = 100
    ):
        """Initialize the class.

        Arguments:
            V (Callable): Time-dependent function of the potential (for one period).
            x_min (float): X value of the left-hand side of one period.
            x_min (float): X value of the right-hand side of one period.
            x_steps (int): Number of steps to take up to and including x_max (but not x_min).
            t_min (float): Starting time value.
            t_max (float): Ending time value.
            t_steps (int): Number of time steps to take up to and including t_max/t_min.
        """
        pass

    def delta(self):
        pass

    def delta_squared(self):
        pass

    def a(self):
        pass

    def a_exp(self):
        pass

    def a_exp_inv(self):
        pass

    def collapse_a_exp(self):
        pass

    def loss(self):
        pass

    def plot_loss(self):
        pass