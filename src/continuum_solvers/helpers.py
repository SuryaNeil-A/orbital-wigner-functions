import torch
from torch import Tensor
from numpy import float64, ndarray
from numpy.typing import NDArray

# necessary for linear algebra
# torch.backends.cuda.preferred_linalg_library("magma")
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# set default datatype of tensors
DTYPE = torch.complex128


def sign_change(matrices: Tensor) -> Tensor:
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


def trace_2by2(matrices: Tensor) -> Tensor:
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


def inverse_2by2(matrices: Tensor) -> Tensor:
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


def collapse(matrices: Tensor) -> Tensor:
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


def clean_input(input: Tensor | NDArray | float | int) -> Tensor:
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
        case Tensor():
            if input.shape == torch.Size([]):
                input = input.unsqueeze(0)
        case _:
            raise TypeError(
                "Type not supported. Try converting to float, Numpy Array, or PyTorch Tensor."
            )

    return input
