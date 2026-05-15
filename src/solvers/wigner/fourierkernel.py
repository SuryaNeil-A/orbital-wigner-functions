import torch
import numpy as np
from solvers.__init__ import DEVICE
from solvers.wigner.__init__ import DTYPE
from solvers.wigner.hoppings import Hoppings


class BlochOperators:
    def __init__(self, model, params, t, t_params, k_size):
        # initializing all variables for the class
        self.params = params
        self.dim = len(k_size)
        self.model = model
        hoppings = Hoppings(params)
        # moving Total_steps to first dimension to make next operations easier
        self.inter_hoppings = getattr(hoppings, self.model)(t)[0].movedim(
            -1, 0
        )
        self.intra_hoppings = getattr(hoppings, self.model)(t)[1].movedim(
            -1, 0
        )
        self.system_size_k = k_size
        self.unit_cell = self.intra_hoppings.shape[1:]
        self.t = t
        self.t_params = t_params
        self.dt = t_params["total_t"] / len(t)

    def construct_H(self):
        # currently only handling 1d case
        if self.dim == 1:
            k = torch.linspace(
                -3 * torch.pi / 2,
                3 * torch.pi / 2,
                self.system_size_k[0],
                dtype=DTYPE,
                device=DEVICE,
            )
            # H_0, H_plus, H_minus dims: [total_steps, unit_cell]
            """H_0 = torch.zeros(self.t.shape + self.unit_cell, dtype = DTYPE, device = DEVICE)
            H_plus = torch.zeros(self.t.shape + self.unit_cell, dtype = DTYPE, device = DEVICE)
            H_minus = torch.zeros(self.t.shape + self.unit_cell, dtype = DTYPE, device = DEVICE)"""

            # sum hoppings to construct hamiltonians, H_0 for intra cell, H_plus/minus for inter cell hoppings
            """for i in range(self.unit_cell[0]):
                for j in range(self.unit_cell[0]):
                    H_0[:, i, j] = (self.intra_hoppings[:, i, j] + torch.conj(self.intra_hoppings[:, j, i])) / 2
                    H_plus[:, i, j] = torch.conj(self.inter_hoppings[:, j, i])
                    H_minus[:, i, j] = self.inter_hoppings[:, i, j]"""
            # initialize H's w/ shape [total_steps, unit_cell]
            H_0 = (
                self.intra_hoppings
                + self.intra_hoppings.conj().transpose(1, 2)
            ) / 2
            H_plus = self.inter_hoppings.conj().transpose(1, 2)
            H_minus = self.inter_hoppings

            # expand the H's into appropriate dims: [total_steps, unit_cell, k_size]
            # and multiply H_plus/minus by e^+-ik respectively)
            H_0 = H_0.unsqueeze(-1).expand(H_0.shape + self.system_size_k)
            H_plus = H_plus.unsqueeze(-1).expand(
                H_plus.shape + self.system_size_k
            ) * torch.exp(1j * k).to("cuda")
            H_minus = H_minus.unsqueeze(-1).expand(
                H_minus.shape + self.system_size_k
            ) * torch.exp(-1j * k).to("cuda")

            # sum H's and exchange dimensions, final result has dims [total_steps, k_size, unit_cell]
            H_total = H_0 + H_plus + H_minus
            return H_total.movedim(-1, 1)
        else:
            pass

    # def construct_H(self):
    #   # currently only handling 1d case
    #   if self.dim == 1:
    #     k_real = torch.linspace(
    #         -3 * torch.pi / 2,
    #          3 * torch.pi / 2,
    #         self.system_size_k[0],
    #         dtype = torch.float32,
    #         device = device
    #     )

    #     k = k_real.to(DTYPE)

    #     # initialize H's
    #     H_0 = (self.intra_hoppings + self.intra_hoppings.conj().transpose(1, 2)) / 2
    #     H_plus = self.inter_hoppings.conj().transpose(1, 2)
    #     H_minus = self.inter_hoppings

    #     # expand the H's into appropriate dims: [total_steps, unit_cell, unit_cell, k_size]
    #     H_0 = H_0.unsqueeze(-1).expand(H_0.shape + self.system_size_k)
    #     H_plus = H_plus.unsqueeze(-1).expand(H_plus.shape + self.system_size_k) * torch.exp(1j * k)
    #     H_minus = H_minus.unsqueeze(-1).expand(H_minus.shape + self.system_size_k) * torch.exp(-1j * k)

    #     # final result has dims [total_steps, k_size, unit_cell, unit_cell]
    #     H_total = H_0 + H_plus + H_minus
    #     H_total = H_total.movedim(-1, 1)

    #     if self.model == "c2_4band_atomic_lower" and self.params.get("flatten", True):

    #       m = float(self.params["mass"])
    #       period = self.params["period"]

    #       if self.params.get("atomic_lower_trivial", True):
    #         lam = float(np.sqrt(max(m + 4.0, 0.0)))
    #       else:
    #         lam = 1.0

    #       omega1 = 2 * np.pi / period
    #       omega2 = 2 * np.pi * np.sqrt(2) / period
    #       omega3 = 2 * np.pi * (np.sqrt(5) + 1) / period

    #       t_real = self.t.to(dtype = torch.float32, device = DEVICE)

    #       phi1 = omega1 * t_real
    #       phi2 = omega2 * t_real
    #       phi3 = omega3 * t_real

    #       sin_k = torch.sin(k_real)[None, :]
    #       cos_k = torch.cos(k_real)[None, :]

    #       sin_1 = torch.sin(phi1)[:, None]
    #       sin_2 = torch.sin(phi2)[:, None]
    #       sin_3 = torch.sin(phi3)[:, None]

    #       cos_1 = torch.cos(phi1)[:, None]
    #       cos_2 = torch.cos(phi2)[:, None]
    #       cos_3 = torch.cos(phi3)[:, None]

    #       d5 = m + cos_k + cos_1 + cos_2 + cos_3

    #       d_norm = torch.sqrt(
    #           lam**2 * (
    #               sin_k**2
    #             + sin_1**2
    #             + sin_2**2
    #             + sin_3**2
    #           )
    #         + d5**2
    #       ).clamp_min(1e-10)

    #       H_total = H_total / d_norm[..., None, None]

    #   return H_total

    def construct_U(self):
        # create unitaries by finding the eigenvalues and eigenvectors H_t
        # the builtin torch.matrix_exp() operation had issues, thus this was done instead
        H_t = self.construct_H()
        L, Q = torch.linalg.eigh(H_t)

        # compute exponential of eigenvalues
        Lexp = torch.exp(-1j * L * self.dt)

        # multiply eigenvects by exp of eigenvalues
        Q_Lexp = Q * Lexp.unsqueeze(-2)

        # compute U
        U = torch.matmul(Q_Lexp, Q.conj().transpose(-2, -1))

        # necessary to prevent garbage data being left in memory, on crash function left these variables behind in memory and runtime restart was needed
        del L, Lexp, Q, Q_Lexp
        torch.cuda.empty_cache()

        # U dims: [total_steps, k_size, unit_cell]
        return U

    def collapse_U(self):
        # collapse all unitaries pairwise for time discretization
        # this procedure requires the discretization of each time step to be a power of 2
        matrices = self.construct_U()
        while matrices.shape[0] > self.t_params["time_steps"]:
            # computer number of steps_per_t currently in unitaries
            n_slices = matrices.shape[0]

            # separate unitaries into pairs, view has dims [total_steps // 2, 2, k_size, unit_cell]
            matrices = matrices.view((n_slices // 2, 2) + matrices.shape[1:])

            # multiply pairs until number of unitaries reduced below time_steps
            matrices = torch.matmul(matrices[:, 0], matrices[:, 1])

        # collapse unitaries down into dims [time_steps, k_size, unit_cell]
        for i in range(1, self.t_params["time_steps"]):
            matrices[i] = torch.matmul(matrices[i], matrices[i - 1])

        # necessary to prevent garbage accumulation in memory
        torch.cuda.empty_cache()

        return matrices

    # used to plot band structure for Bloch Hamiltonian
    def bandStructure(self):
        if self.dim == 1:
            k = torch.linspace(
                -3 * torch.pi / 2,
                3 * torch.pi / 2,
                self.system_size_k[0],
                dtype=DTYPE,
                device=DEVICE,
            )
            H_t = self.construct_H()

            evals, evects = torch.linalg.eigh(H_t)

            return evals, evects, k
        else:
            pass


class State:
    def __init__(self, params, size, H_0):
        self.dim = len(size)
        self.params = params
        self.system_size = size
        self.H_0 = H_0
        self.unit_cell = H_0.shape[-1]

    def initialize(self):
        if self.dim == 1:
            phase_A_tensor = torch.zeros(
                self.system_size[0], dtype=DTYPE, device=DEVICE
            )
            phase_B_tensor = torch.zeros(
                self.system_size[0], dtype=DTYPE, device=DEVICE
            )

            # creating phase difference tensors for initial state
            for i in range(self.system_size[0]):
                phase_A_tensor[i] = (
                    self.params["phase"]
                    / 2
                    * np.exp(
                        -((i - self.params["pos_x"]) ** 2)
                        / (self.params["width"] ** 2)
                    )
                )
                phase_B_tensor[i] = (
                    -self.params["phase"]
                    / 2
                    * np.exp(
                        -((i - self.params["pos_x"]) ** 2)
                        / (self.params["width"] ** 2)
                    )
                )
            phase_diff = phase_A_tensor - phase_B_tensor
            phase_tensor = torch.exp(1j * phase_diff)

            # creating beta(x) as defined in the paper
            beta_tensor = torch.zeros(
                self.system_size[0], dtype=DTYPE, device=DEVICE
            )
            for i in range(self.system_size[0]):
                beta_tensor[i] = self.params["beta"] + (
                    self.params["hs_beta"] - self.params["beta"]
                ) * np.exp(
                    -((i - self.params["pos_x"]) ** 2)
                    / (self.params["width"] ** 2)
                )

            # creating mu(x) as defined in the paper
            mu_tensor = torch.zeros(
                self.system_size[0], dtype=DTYPE, device=DEVICE
            )
            for i in range(self.system_size[0]):
                mu_tensor[i] = self.params["mu"] + (
                    self.params["mu_hs"] - self.params["mu"]
                ) * np.exp(
                    -(
                        (i - self.params["pos_x"]) ** 2
                        / (self.params["width"] ** 2)
                    )
                )

            return phase_tensor, beta_tensor, mu_tensor
        else:
            pass

    def constructGamma(self):
        phase_tensor, beta_tensor, mu_tensor = self.initialize()
        # H_0 has dims [k_size, unit_cell, unit_cell] (time dimension squeezed out as we are only looking at H at time 0 for initial state)
        H_expanded = self.H_0
        # reflects dimensionality of the system (i.e. dim_H_0 = 1 if we want to compute for a 1d system)
        dim_H_0 = len(self.H_0.shape) - 2

        # expanding identity to match dimensionality
        identity = torch.eye(self.unit_cell, dtype=DTYPE, device=DEVICE)
        for i in range(dim_H_0):
            identity = identity.unsqueeze(0)
        identity_expanded = identity.expand(self.H_0.shape)

        # unsqueezing out appropriate dimensions and multiplying elementwise
        mu_mat = (
            mu_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * identity_expanded
        )

        # expanding mu_tensor into appropriate shape
        dim_mu_tensor = len(mu_tensor.shape)
        for i in range(dim_mu_tensor):
            H_expanded = H_expanded.unsqueeze(0)
        H_expanded = H_expanded.expand(mu_mat.shape)

        H_mu = H_expanded - mu_mat

        gamma = beta_tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * H_mu

        return gamma, phase_tensor

    def tensorExp(self, matrices):
        # computing matrix exp by finding eigenvalues and eigenvectors
        # torch.matrix_exp() was having issues, see construct_U() in BlochOperators for details
        dims = matrices.shape
        L, Q = torch.linalg.eigh(matrices)

        # compute the exponential of the eigenvalues
        Lexp = torch.exp(L).to(device=DEVICE)

        # rearrange to match final dims of [..., unit_cell, unit_cell]
        Lexpdiag = torch.zeros(dims, dtype=DTYPE, device=DEVICE)
        for j in range(self.unit_cell):
            Lexpdiag[:, :, j, j] = Lexp[:, :, j]

        # compute expanded identity and multiply
        eye = torch.eye(self.unit_cell, dtype=DTYPE, device=DEVICE)
        eye_full = eye.unsqueeze(0).unsqueeze(0).expand(dims)
        Linv = torch.linalg.inv(eye_full + Lexpdiag)

        # multiply eigenvectors by inverse of identiy + exp of eigenvals
        Q_Linv = Q @ Linv

        # compute gammaExp
        gammaExp = torch.matmul(Q_Linv, Q.conj().transpose(-2, -1))

        # necessary to prevent garbage data left in memory on crash, runtime restart otherwise needed
        # see construct_U() in BlochOperators for details
        del L, Linv, Q, Q_Linv, Lexp
        torch.cuda.empty_cache()
        return gammaExp

    def constructState_pos(self):
        # creating initial state in position basis
        gamma, phase_tensor = self.constructGamma()
        rho_beta = self.tensorExp(gamma).transpose(-2, -1)

        # unsqueezing to match dimensionality of H_0
        dim_H_0 = len(self.H_0.shape) - 2
        for i in range(dim_H_0):
            phase_tensor = phase_tensor.unsqueeze(-1)
        phase_tensor = phase_tensor.expand(
            self.system_size + self.H_0.shape[:-2]
        )

        # NOT SURE HOW TO FIX, but needs to be for arbitrary unit cell (or is phase texture only important for 2 sites?)!
        # explicitly arranging rho_beta as in the paper for a unit_cell size of 2
        if self.unit_cell == 2:
            # moving dims such that rho_beta is [unit_cell, unit_cell, ...] for easy handling of higher dimensional systems
            rho_beta = rho_beta.movedim((-2, -1), (0, 1))
            rho_beta[0, 1, :] = torch.mul(rho_beta[0, 1, :], phase_tensor)
            rho_beta[1, 0, :] = torch.mul(
                rho_beta[1, 0, :], phase_tensor.conj()
            )

            # rearranging back into the familiar structure of [..., unit_cell, unit_cell]
            rho_beta = rho_beta.movedim((0, 1), (-2, -1))

            # necessary to prevent garbage accumulation in memory
            torch.cuda.empty_cache()

            # rho_beta has dims [system_size, system_size, unit_cell, unit_cell]
            return rho_beta
        else:
            torch.cuda.empty_cache()
            return rho_beta

    def stateFT(self):
        # fourier transform position state into momentum space
        if self.dim == 1:
            rho_beta = self.constructState_pos()

            # saving initial shape to reshape back into
            initial_shape = rho_beta.shape

            # create position and momentum grid, p_size is identical to system_size
            x_values = torch.arange(self.system_size[0], device=DEVICE)
            p_values_x = torch.linspace(
                -torch.pi, torch.pi, self.system_size[0], device=DEVICE
            )

            # compute the Fourier matrix with phase adjustments
            # F has dims [p_size, system_size]
            F = torch.exp(-1j * torch.outer(p_values_x, x_values))

            # reshape rho_beta for multiplication, folding all extra dimensions into N
            N = rho_beta.shape[1] * self.unit_cell * self.unit_cell
            rho_beta_reshaped = rho_beta.reshape(self.system_size[0], N)

            # perform fourier transform via matrix mult
            # rho_beta_p has dims [p_size, N]
            rho_beta_p = torch.matmul(F, rho_beta_reshaped)

            # reshape rho_beta_p back to original dims [system_size, system_size, unit_cell, unit_cell] (now in momentum space)
            rho_beta_p = rho_beta_p.reshape(initial_shape)
            return rho_beta_p
        else:
            pass


class FourierSolver:
    def __init__(
        self, params, t, t_params, size, k_size, batch_total, model, hop_params
    ):
        self.params = params
        self.t_params = t_params
        self.t = t
        self.system_size = size
        self.k_size = k_size
        self.model = model
        self.batch_total = batch_total
        # defining number of k points per batch in x direction
        self.k_pts_perbatch = (2 * k_size[0]) // (3 * batch_total[0])
        # initializing U and H_0
        self.U = BlochOperators(
            self.model, hop_params, self.t, self.t_params, self.k_size
        ).collapse_U()
        self.H_0 = (
            BlochOperators(
                self.model,
                hop_params,
                torch.tensor([0]),
                self.t_params,
                self.k_size,
            )
            .construct_H()
            .squeeze(0)
        )
        self.unit_cell = self.H_0.shape[-1]

    def timeEvolve(self, batch_index):
        if len(self.k_size) == 1:
            # taking the FT of slice of H_0 according to k batching
            batch_state_0 = State(
                self.params,
                self.system_size,
                self.H_0[
                    (
                        ((self.system_size[0]) // 2)
                        + self.k_pts_perbatch * batch_index
                    ) : (
                        (self.system_size[0] // 2)
                        + self.k_pts_perbatch * (batch_index + 1)
                    )
                ],
            ).stateFT()
            # creating initital batch state and expanding into dims [time_steps, system_size (p), system_size (k), unit_cell, unit_cell]
            batch_state = batch_state_0.unsqueeze(0).expand(
                [self.U.shape[0]] + list(batch_state_0.shape)
            )
            batch_state_out = torch.empty(
                batch_state.shape, dtype=DTYPE, device=DEVICE
            )

            # creating the unitaries to act on the batched state by taking slices of U
            batch_U = torch.empty(
                batch_state.shape, dtype=DTYPE, device=DEVICE
            )
            for i in range(self.k_pts_perbatch):
                batch_U[:, :, i, :, :] = self.U[
                    :,
                    (i + self.k_pts_perbatch * batch_index) : (
                        i
                        + self.system_size[0]
                        + self.k_pts_perbatch * batch_index
                    ),
                    :,
                    :,
                ]

            # computing adjoint and flipping p to -p for inverse
            batch_U_inv = torch.flip(batch_U.adjoint(), dims=[1])

            # multiplying unitaries for time evolution
            batch_state_out = torch.matmul(
                torch.matmul(batch_U, batch_state), batch_U_inv
            )

            # appending initial state to batch_state
            batch_state_out = torch.cat(
                (batch_state_0.unsqueeze(0), batch_state_out), dim=0
            )

            # output state has dims [time_steps + 1, system_size, system_size, unit_cell, unit_cell]
            return batch_state_out
        else:
            pass

    def stateIFT(self, matrices):
        # inverse fourier transform back to position space of time-evolved state
        if len(self.k_size) == 1:
            # create position grid
            x_values = torch.arange(self.system_size[0], device=DEVICE)
            p_values_x = torch.linspace(
                -torch.pi, torch.pi, self.system_size[0], device=DEVICE
            )

            # compute the IFT matrix with phase adjustment
            # F has dims [time_steps + 1, system_size, system_size]
            F = (
                (
                    torch.exp(1j * torch.outer(p_values_x, x_values))
                    / self.system_size[0]
                )
                .unsqueeze(0)
                .expand(
                    self.t_params["time_steps"] + 1,
                    self.system_size[0],
                    self.system_size[0],
                )
            )

            # reshape rho_beta for matrix mult
            N = self.unit_cell**2
            matrices_reshaped = matrices.view(
                self.t_params["time_steps"] + 1, self.system_size[0], N
            )

            # perform IFT by matrix mult
            # rho_Out has dims [time_steps + 1, system_size, N]
            rho_out = torch.matmul(F.transpose(-1, -2), matrices_reshaped)

            # reshape rho_out into original dims of [time_steps + 1, system_size, unit_cell, unit_cell]
            rho_out = rho_out.view(
                self.t_params["time_steps"] + 1,
                self.system_size[0],
                self.unit_cell,
                self.unit_cell,
            )

            return rho_out
        else:
            pass

    def batching(self):
        # apply time evolution over all batches
        if len(self.k_size) == 1:
            # evaluating rho and combining for each batch
            rho_total = torch.zeros(
                (
                    self.t_params["time_steps"] + 1,
                    self.system_size[0],
                    self.unit_cell,
                    self.unit_cell,
                ),
                dtype=DTYPE,
                device=DEVICE,
            )
            for i in range(self.batch_total[0]):
                batch_rho = self.timeEvolve(i)

                # summing over k points
                rho_total += batch_rho.sum(dim=2) / (
                    self.batch_total[0] * self.k_pts_perbatch
                )

            # IFT of output state into position space
            rho_total_IFT = self.stateIFT(rho_total)

            # rho_total_IFT has dims [time_steps + 1, system_size, unit_cell, unit_cell]
            return rho_total_IFT
