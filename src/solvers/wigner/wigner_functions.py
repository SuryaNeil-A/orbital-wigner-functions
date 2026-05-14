import torch
import numpy as np
from continuum_solvers.__init__ import DEVICE, DTYPE


class Hoppings:
    def __init__(self, params):
        self.params = params

    # custom model to input various hopping terms
    # intra-hoppings are forced hermitian later (this can be removed), but inter-hoppings should be non-hermitian in general
    # inter-Upper triangle corresponds to e^{-ik} and lower triangular have e^{ik}???
    def custom(self, t):
        # inter (between) cell hoppings
        inter_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # set these terms
        inter_hoppings[0, 0] = (1 / 2) * torch.exp(-1j * t)
        inter_hoppings[1, 0] = (1 / np.sqrt(2)) * torch.exp(
            -1j * t * self.params["Elec"]
        )

        # intra (within) cell hoppings
        intra_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # set these terms
        intra_hoppings[0, 0] = 1 / 2  # rho_aa term
        intra_hoppings[0, 1] = 1 / np.sqrt(2)  # rho_ab term
        intra_hoppings[1, 0] = 1 / np.sqrt(2)  # rho_ba term
        intra_hoppings[1, 1] = 0  # rho_bb term

        return inter_hoppings, intra_hoppings

    # prebuilt SSH model from the paper
    def ssh(self, t):
        # inter cell hoppings
        inter_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[0, 1] = 1
        inter_hoppings[1, 0] = 1

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        intra_hoppings[0, 1] = self.params["mass"]
        intra_hoppings[1, 0] = self.params["mass"]

        return inter_hoppings, intra_hoppings

    # prebuilt stroboscopic SSH model from the paper
    def stroboscopic_ssh_c2(self, t):
        # inter cell hoppings
        inter_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # +0.001 to ensure ceiling function creates intended output
        inter_hoppings[1, 0] = (
            self.params["mass"] / 2
            - 1
            * self.params["mass"]
            * (-1)
            ** (
                torch.ceil(
                    torch.real(
                        -torch.sin(
                            2 * torch.pi * t / self.params["period"] + 0.001
                        )
                    )
                )
            )
            / 2
        )
        inter_hoppings[0, 1] = (
            self.params["mass"] / 2
            + self.params["mass"]
            * (-1)
            ** (
                torch.ceil(
                    torch.real(
                        -torch.sin(
                            2 * torch.pi * t / self.params["period"] + 0.001
                        )
                    )
                )
            )
            / 2
        )

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # intra_hoppings[0, 1] = self.params["mass"] * torch.ceil(torch.real(torch.sin(2*torch.pi*t / self.params["period"] + 0.001)))
        # intra_hoppings[1, 0] = self.params["mass"] * torch.ceil(torch.real(torch.sin(2*torch.pi*t / self.params["period"] + 0.001)))

        return inter_hoppings, intra_hoppings

    def stroboscopic_ssh_c1(self, t):
        # inter cell hoppings
        inter_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # +0.001 to ensure ceiling function creates intended output
        inter_hoppings[1, 0] = self.params["mass"] * torch.ceil(
            torch.real(
                -torch.sin(2 * torch.pi * t / self.params["period"] + 0.001)
            )
        )
        # inter_hoppings[0, 1] += self.params["mass"] * torch.ceil(torch.real(-torch.sin(2*torch.pi*t / self.params["period"] + 0.001)))
        # inter_hoppings[1, 0] += self.params["mass"] / 2
        # inter_hoppings[0, 1] += self.params["mass"] / 2

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (2, 2) + t.shape, dtype=DTYPE, device=DEVICE
        )

        intra_hoppings[0, 1] = self.params["mass"] * torch.ceil(
            torch.real(
                torch.sin(2 * torch.pi * t / self.params["period"] + 0.001)
            )
        )
        intra_hoppings[1, 0] = self.params["mass"] * torch.ceil(
            torch.real(
                torch.sin(2 * torch.pi * t / self.params["period"] + 0.001)
            )
        )

        return inter_hoppings, intra_hoppings

    def doublestrobe(self, t):
        omega1 = 2 * np.pi * np.sqrt(2) / self.params["period"]
        omega2 = 2 * np.pi * (np.sqrt(5) + 1) / self.params["period"]
        # t *= -1

        inter_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )
        inter_hoppings[1, 0] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(-torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                - 1
                * self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            -torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )
        inter_hoppings[0, 1] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(-torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                + self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            -torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )

        inter_hoppings[2, 3] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(-torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                - 1
                * self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )
        inter_hoppings[3, 2] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(-torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                + self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )

        inter_hoppings[0, 3] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                - 1
                * self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )
        inter_hoppings[1, 2] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                + self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )

        inter_hoppings[2, 1] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                - 1
                * self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )
        inter_hoppings[3, 0] = (
            (torch.ceil(torch.real(torch.sin(omega2 * t + 0.0007))))
            * (torch.ceil(torch.real(torch.sin(omega1 * t + 0.0011))))
            * (
                self.params["mass"] / 2
                + self.params["mass"]
                * (-1)
                ** (
                    torch.ceil(
                        torch.real(
                            torch.sin(
                                2 * torch.pi * t / self.params["period"]
                                + 0.001
                            )
                        )
                    )
                )
                / 2
            )
        )

        intra_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[2, 0] = self.params["mass"] * torch.ceil(
            -torch.real(torch.sin(omega2 * t + 0.0007))
        )
        inter_hoppings[3, 1] = self.params["mass"] * torch.ceil(
            -torch.real(torch.sin(omega2 * t + 0.0007))
        )

        # inter_hoppings[2, 0] = self.params["mass"] * torch.ceil(torch.real(torch.sin(omega2*t + 0.0007)))/2
        # inter_hoppings[3, 1] = self.params["mass"] * torch.ceil(torch.real(torch.sin(omega2*t + 0.0007)))/2
        return inter_hoppings, intra_hoppings

    # Lieb lattice thouless pumping
    def lieb_pump(self, t):
        phi0 = 0.3
        # inter cell hoppings
        inter_hoppings = torch.zeros(
            (3, 3) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[0, 1] = -1

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (3, 3) + t.shape, dtype=DTYPE, device=DEVICE
        )

        intra_hoppings[0, 1] = 1
        intra_hoppings[1, 0] = 1

        intra_hoppings[1, 2] = 0 * 1 + 0 * torch.exp(
            1j * 2 * torch.pi * t / self.params["period"] + 1j * phi0
        )
        intra_hoppings[2, 1] = 0 * 1 + 0 * torch.exp(
            -1j * 2 * torch.pi * t / self.params["period"] - 1j * phi0
        )

        return inter_hoppings, intra_hoppings

    def offDiag_AAH(self, t):
        # inter cell hoppings
        a = 0.5
        b = 0.13
        omega = 2 * torch.pi / self.params["period"]

        inter_hoppings = torch.zeros(
            (5, 5) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[0, 4] = -a * torch.cos(omega * t - 6 * torch.pi / 5) + b
        # inter_hoppings[4, 0] = -a * torch.cos(omega * t - 6 * torch.pi / 5) + b

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (5, 5) + t.shape, dtype=DTYPE, device=DEVICE
        )

        intra_hoppings[0, 1] = -a * torch.cos(omega * t - 2 * torch.pi / 5) + b
        intra_hoppings[1, 2] = -a * torch.cos(omega * t - 8 * torch.pi / 5) + b
        intra_hoppings[2, 3] = -a * torch.cos(omega * t - 4 * torch.pi / 5) + b
        intra_hoppings[3, 4] = -a * torch.cos(omega * t) + b
        intra_hoppings[1, 0] = -a * torch.cos(omega * t - 2 * torch.pi / 5) + b
        intra_hoppings[2, 1] = -a * torch.cos(omega * t - 8 * torch.pi / 5) + b
        intra_hoppings[3, 2] = -a * torch.cos(omega * t - 4 * torch.pi / 5) + b
        intra_hoppings[4, 3] = -a * torch.cos(omega * t) + b

        return inter_hoppings, intra_hoppings

    def diag_AAH(self, t):
        # inter cell hoppings
        a = 2
        omega = 2 * torch.pi / self.params["period"]

        inter_hoppings = torch.zeros(
            (5, 5) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[0, 4] = 1
        # inter_hoppings[4, 0] = 1

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (5, 5) + t.shape, dtype=DTYPE, device=DEVICE
        )

        intra_hoppings[0, 1] = 1
        intra_hoppings[1, 2] = 1
        intra_hoppings[2, 3] = 1
        intra_hoppings[3, 4] = 1
        intra_hoppings[1, 0] = 1
        intra_hoppings[2, 1] = 1
        intra_hoppings[3, 2] = 1
        intra_hoppings[4, 3] = 1

        for k in range(5):
            intra_hoppings[k, k] = (
                a * 2 * torch.cos(omega * t + 2 * torch.pi * k * (np.sqrt(2)))
            )

        return inter_hoppings, intra_hoppings

    def band13_AAH(self, t):
        # inter cell hoppings
        J = 1
        delta = 0.95
        omega = 2 * torch.pi / self.params["period"]

        inter_hoppings = torch.zeros(
            (13, 13) + t.shape, dtype=DTYPE, device=DEVICE
        )

        inter_hoppings[0, 12] = -J - delta * torch.cos(
            omega * t + (2 / 13) * torch.pi
        )
        # inter_hoppings[4, 0] = 1

        # intra cell hoppings
        intra_hoppings = torch.zeros(
            (13, 13) + t.shape, dtype=DTYPE, device=DEVICE
        )

        for i in range(12):
            intra_hoppings[i, i + 1] = -J - delta * torch.cos(
                omega * t
                + (10 / 13) * torch.pi * (i + 1)
                + (2 / 13) * torch.pi
            )
        for i in range(1, 13):
            intra_hoppings[i, i - 1] = -J - delta * torch.cos(
                omega * t + (10 / 13) * torch.pi * i + (2 / 13) * torch.pi
            )

        return inter_hoppings, intra_hoppings

    def c2_4band(self, t):
        """
        Four-band, 1D + 3-frequency second-Chern pump.

        Coordinates:
          k, phi1(t), phi2(t), phi3(t)

        mass = -3 gives C2 = +1
        mass =  5 gives C2 =  0

        This uses your notebook's convention:
          H(k,t) = intra(t) + inter(t)^dagger exp(+ik) + inter(t) exp(-ik)
        """

        t = t.to(device=DEVICE)

        amp = self.params.get("amp", 1.0)
        m = self.params["mass"]

        omega1 = 2 * np.pi / self.params["period"]
        omega2 = 2 * np.pi * np.sqrt(2) / self.params["period"]
        omega3 = 2 * np.pi * (np.sqrt(5) + 1) / self.params["period"]

        phi1 = omega1 * t
        phi2 = omega2 * t
        phi3 = omega3 * t

        s1 = torch.sin(phi1)
        s2 = torch.sin(phi2)
        s3 = torch.sin(phi3)

        c1 = torch.cos(phi1)
        c2 = torch.cos(phi2)
        c3 = torch.cos(phi3)

        d5 = m + c1 + c2 + c3

        inter_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )
        intra_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # ------------------------------------------------------------------
        # Intercell part:
        #
        # inter = amp * ( i Gamma1 / 2 + Gamma5 / 2 )
        #
        # Your construct_H then makes:
        # inter^dagger e^{+ik} + inter e^{-ik}
        # = amp * ( sin(k) Gamma1 + cos(k) Gamma5 )
        # ------------------------------------------------------------------

        inter_hoppings[0, 0] = amp * 0.5
        inter_hoppings[1, 1] = amp * 0.5
        inter_hoppings[2, 2] = -amp * 0.5
        inter_hoppings[3, 3] = -amp * 0.5

        inter_hoppings[0, 3] = amp * 0.5j
        inter_hoppings[1, 2] = amp * 0.5j
        inter_hoppings[2, 1] = amp * 0.5j
        inter_hoppings[3, 0] = amp * 0.5j

        # ------------------------------------------------------------------
        # Onsite synthetic-frequency part:
        #
        # intra = amp * [
        #     sin(phi1) Gamma2
        #   + sin(phi2) Gamma3
        #   + sin(phi3) Gamma4
        #   + (mass + cos(phi1) + cos(phi2) + cos(phi3)) Gamma5
        # ]
        # ------------------------------------------------------------------

        # d5 * Gamma5
        intra_hoppings[0, 0] += amp * d5
        intra_hoppings[1, 1] += amp * d5
        intra_hoppings[2, 2] += -amp * d5
        intra_hoppings[3, 3] += -amp * d5

        # sin(phi1) * Gamma2
        intra_hoppings[0, 3] += -1j * amp * s1
        intra_hoppings[1, 2] += 1j * amp * s1
        intra_hoppings[2, 1] += -1j * amp * s1
        intra_hoppings[3, 0] += 1j * amp * s1

        # sin(phi2) * Gamma3
        intra_hoppings[0, 2] += amp * s2
        intra_hoppings[1, 3] += -amp * s2
        intra_hoppings[2, 0] += amp * s2
        intra_hoppings[3, 1] += -amp * s2

        # sin(phi3) * Gamma4
        intra_hoppings[0, 2] += -1j * amp * s3
        intra_hoppings[1, 3] += -1j * amp * s3
        intra_hoppings[2, 0] += 1j * amp * s3
        intra_hoppings[3, 1] += 1j * amp * s3

        return inter_hoppings, intra_hoppings

    def c2_4band_atomic_lower(self, t):
        """
        Four-band second-Chern model with an atomic lower-trivial phase.

        For m < -4:
          lambda = 0
          H_flat = -amp * Gamma5
          C2 = 0
          no k-dependence, no t-dependence, no transport

        For -4 < m < -2:
          lambda > 0
          same C2 = +1 as the Wilson-Dirac model

        Avoid m = -4 exactly.
        """

        t = t.to(device=DEVICE, dtype=torch.float32)

        amp = self.params.get("amp", 1.0)
        m = float(self.params["mass"])
        period = self.params["period"]

        # Atomicize the entire lower trivial phase.
        # At m = -3, lambda = 1, so this is the original model.
        # At m = -5, lambda = 0, so this is exactly atomic.
        if self.params.get("atomic_lower_trivial", True):
            lam = float(np.sqrt(max(m + 4.0, 0.0)))
        else:
            lam = 1.0

        omega1 = 2 * np.pi / period
        omega2 = 2 * np.pi * np.sqrt(2) / period
        omega3 = 2 * np.pi * (np.sqrt(5) + 1) / period

        phi1 = omega1 * t
        phi2 = omega2 * t
        phi3 = omega3 * t

        s1 = torch.sin(phi1)
        s2 = torch.sin(phi2)
        s3 = torch.sin(phi3)

        c1 = torch.cos(phi1)
        c2 = torch.cos(phi2)
        c3 = torch.cos(phi3)

        d5 = m + c1 + c2 + c3

        inter_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )
        intra_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=DTYPE, device=DEVICE
        )

        # ------------------------------------------------------------------
        # Intercell part:
        #
        # inter = amp * ( i lambda Gamma1 / 2 + Gamma5 / 2 )
        #
        # construct_H gives:
        # inter^dagger e^{+ik} + inter e^{-ik}
        # = amp * ( lambda sin(k) Gamma1 + cos(k) Gamma5 )
        # ------------------------------------------------------------------

        # Gamma5 / 2
        inter_hoppings[0, 0] = amp * 0.5
        inter_hoppings[1, 1] = amp * 0.5
        inter_hoppings[2, 2] = -amp * 0.5
        inter_hoppings[3, 3] = -amp * 0.5

        # i lambda Gamma1 / 2
        inter_hoppings[0, 3] = amp * 0.5j * lam
        inter_hoppings[1, 2] = amp * 0.5j * lam
        inter_hoppings[2, 1] = amp * 0.5j * lam
        inter_hoppings[3, 0] = amp * 0.5j * lam

        # ------------------------------------------------------------------
        # Onsite part:
        #
        # amp * [
        #   lambda sin(phi1) Gamma2
        # + lambda sin(phi2) Gamma3
        # + lambda sin(phi3) Gamma4
        # + d5 Gamma5
        # ]
        # ------------------------------------------------------------------

        # d5 * Gamma5
        intra_hoppings[0, 0] += amp * d5
        intra_hoppings[1, 1] += amp * d5
        intra_hoppings[2, 2] += -amp * d5
        intra_hoppings[3, 3] += -amp * d5

        # lambda sin(phi1) * Gamma2
        intra_hoppings[0, 3] += -1j * amp * lam * s1
        intra_hoppings[1, 2] += 1j * amp * lam * s1
        intra_hoppings[2, 1] += -1j * amp * lam * s1
        intra_hoppings[3, 0] += 1j * amp * lam * s1

        # lambda sin(phi2) * Gamma3
        intra_hoppings[0, 2] += amp * lam * s2
        intra_hoppings[1, 3] += -amp * lam * s2
        intra_hoppings[2, 0] += amp * lam * s2
        intra_hoppings[3, 1] += -amp * lam * s2

        # lambda sin(phi3) * Gamma4
        intra_hoppings[0, 2] += -1j * amp * lam * s3
        intra_hoppings[1, 3] += -1j * amp * lam * s3
        intra_hoppings[2, 0] += 1j * amp * lam * s3
        intra_hoppings[3, 1] += 1j * amp * lam * s3

        return inter_hoppings, intra_hoppings

    def c2_commutator_diffusive_strobe(self, t):
        """
        Controlled commutator loop with a separate broadening knob.

        Basis:
          0 = A up
          1 = A down
          2 = B up
          3 = B down

        One cycle:
          1. I        : intracell identity dimer
          2. Q        : charge-neutral conditional intercell dimer
          3. G_n      : intracell SU(2) dimer
          4. G_n Q^-1 : inverse conditional dimer in the G_n frame

        Three knobs:
          mass          -> controls how non-diagonal G_n is
          pulse_detune  -> detunes the dimer pulse area from pi/2
          phase_jitter  -> cycle-to-cycle jitter/broadening for S_vv(0)

        Good reference point:
          |mass| > 4, pulse_detune = 0, any phase_jitter
          => G_n diagonal, [G_n, Q] = 0, cycle freezes at strobe boundaries.

        Good topology/noncommutation point:
          |mass| = 3, pulse_detune = 0, phase_jitter > 0
          => noncommuting cycle-to-cycle SU(2) mismatch with a broadened spectrum.
        """

        t = t.to(device=DEVICE, dtype=torch.float32)
        dtype = DTYPE

        amp = float(self.params.get("amp", 1.0))

        pulse_detune = float(self.params.get("pulse_detune", 0.0))

        # Default pulse area is pi/2 * (1 + detune)
        if "slot_time" in self.params:
            slot_time = float(self.params["slot_time"])
            pulse_area = amp * slot_time
        else:
            pulse_area = float(
                self.params.get(
                    "pulse_area", 0.5 * np.pi * (1.0 + pulse_detune)
                )
            )
            slot_time = pulse_area / amp

        cycle_time = 4.0 * slot_time

        mass = float(self.params.get("mass", -3.0))
        abs_mass = abs(mass)

        # ------------------------------------------------------------
        # Four slot gates
        # ------------------------------------------------------------

        tau = torch.remainder(t, cycle_time)

        gI = ((tau >= 0.0 * slot_time) & (tau < 1.0 * slot_time)).to(
            torch.float32
        )
        gQ = ((tau >= 1.0 * slot_time) & (tau < 2.0 * slot_time)).to(
            torch.float32
        )
        gG = ((tau >= 2.0 * slot_time) & (tau < 3.0 * slot_time)).to(
            torch.float32
        )
        gR = ((tau >= 3.0 * slot_time) & (tau < 4.0 * slot_time)).to(
            torch.float32
        )

        # ------------------------------------------------------------
        # Synthetic phases sampled once per cycle
        # ------------------------------------------------------------

        ncycle = torch.floor(t / cycle_time)

        phase1 = float(self.params.get("phase1", 0.17))
        phase2 = float(self.params.get("phase2", 1.31))
        phase3 = float(self.params.get("phase3", 2.47))

        phase_step1 = float(
            self.params.get("phase_step1", 2.0 * np.pi * np.sqrt(2.0))
        )
        phase_step2 = float(
            self.params.get("phase_step2", 2.0 * np.pi * np.sqrt(3.0))
        )
        phase_step3 = float(
            self.params.get("phase_step3", 2.0 * np.pi * np.sqrt(5.0))
        )

        phase_jitter = float(self.params.get("phase_jitter", 0.0))
        jitter_mode = self.params.get("jitter_mode", "hash")

        def hash_noise(x, a, b, c):
            # reproducible pseudo-random number in [-1, 1]
            return 2.0 * torch.remainder(torch.sin(a * x + b) * c, 1.0) - 1.0

        if phase_jitter == 0.0:
            xi1 = torch.zeros_like(t)
            xi2 = torch.zeros_like(t)
            xi3 = torch.zeros_like(t)

        elif jitter_mode == "qp":
            # deterministic extra incommensurate modulation
            xi1 = torch.sin(np.sqrt(7.0) * ncycle + 0.37)
            xi2 = torch.sin(np.sqrt(11.0) * ncycle + 1.19)
            xi3 = torch.sin(np.sqrt(13.0) * ncycle + 2.41)

        else:
            # default: pseudo-random cycle-to-cycle jitter
            xi1 = hash_noise(ncycle, 12.9898, 0.31, 43758.5453)
            xi2 = hash_noise(ncycle, 78.2330, 1.27, 24634.6345)
            xi3 = hash_noise(ncycle, 45.1640, 2.53, 19341.1942)

        phi1 = phase1 + phase_step1 * ncycle + phase_jitter * xi1
        phi2 = phase2 + phase_step2 * ncycle + phase_jitter * xi2
        phi3 = phase3 + phase_step3 * ncycle + phase_jitter * xi3

        inter_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=dtype, device=DEVICE
        )
        intra_hoppings = torch.zeros(
            (4, 4) + t.shape, dtype=dtype, device=DEVICE
        )

        # ------------------------------------------------------------
        # Build G_n in SU(2)
        #
        # |mass| > 4  -> diagonal commuting regime
        # |mass| = 3  -> non-Abelian off-diagonal regime
        # ------------------------------------------------------------

        mu = 1.0 - abs_mass

        if "lambda_offdiag" in self.params:
            lambda_offdiag = float(self.params["lambda_offdiag"])
        else:
            lambda_offdiag = float(np.sqrt(max(0.0, min(1.0, 4.0 - abs_mass))))

        diag_scale = float(self.params.get("diag_scale", 1.0))

        x0 = mu + torch.cos(phi1) + torch.cos(phi2) + torch.cos(phi3)
        x1 = lambda_offdiag * torch.sin(phi1)
        x2 = lambda_offdiag * torch.sin(phi2)
        x3 = diag_scale * torch.sin(phi3)

        R = torch.sqrt(x0**2 + x1**2 + x2**2 + x3**2).clamp_min(1e-10)

        a = (x0.to(dtype) + 1j * x3.to(dtype)) / R.to(dtype)
        b = (x2.to(dtype) + 1j * x1.to(dtype)) / R.to(dtype)

        # G =
        # [[ a,      b  ],
        #  [-b^*,   a^*]]
        g00 = a
        g01 = b
        g10 = -torch.conj(b)
        g11 = torch.conj(a)

        one = torch.ones_like(t, dtype=dtype)
        zero = torch.zeros_like(t, dtype=dtype)

        # ------------------------------------------------------------
        # Helper: intracell dimer with 2x2 matrix q
        #
        # H = amp * [ A_j^\dagger q B_j + h.c. ]
        # ------------------------------------------------------------

        def add_intra_q(gate, q00, q01, q10, q11):
            c = (amp * gate).to(dtype)

            # A -> B block
            intra_hoppings[0, 2] += c * q00
            intra_hoppings[0, 3] += c * q01
            intra_hoppings[1, 2] += c * q10
            intra_hoppings[1, 3] += c * q11

            # B -> A block = q^\dagger
            intra_hoppings[2, 0] += c * torch.conj(q00)
            intra_hoppings[3, 0] += c * torch.conj(q01)
            intra_hoppings[2, 1] += c * torch.conj(q10)
            intra_hoppings[3, 1] += c * torch.conj(q11)

        # ------------------------------------------------------------
        # Slot 1: I
        #   A_{j,up}   <-> B_{j,up}
        #   A_{j,down} <-> B_{j,down}
        # ------------------------------------------------------------

        add_intra_q(gI, one, zero, zero, one)

        # ------------------------------------------------------------
        # Slot 2: Q
        #
        # q_Q(k) = diag(e^{-ik}, e^{+ik})
        #
        # Bonds:
        #   A_{j,up}   <-> B_{j+1,up}
        #   A_{j,down} <-> B_{j-1,down}
        # ------------------------------------------------------------

        cQ = (amp * gQ).to(dtype)

        # up: H[A_up, B_up] = e^{-ik}
        inter_hoppings[0, 2] += cQ

        # down: H[A_down, B_down] = e^{+ik}
        inter_hoppings[3, 1] += cQ

        # ------------------------------------------------------------
        # Slot 3: G_n
        # ------------------------------------------------------------

        add_intra_q(gG, g00, g01, g10, g11)

        # ------------------------------------------------------------
        # Slot 4: G_n Q^{-1}
        #
        # q_4(k) = G_n diag(e^{+ik}, e^{-ik})
        # ------------------------------------------------------------

        cR = (amp * gR).to(dtype)

        # Column B_up carries e^{+ik}
        inter_hoppings[2, 0] += cR * torch.conj(g00)
        inter_hoppings[2, 1] += cR * torch.conj(g10)

        # Column B_down carries e^{-ik}
        inter_hoppings[0, 3] += cR * g01
        inter_hoppings[1, 3] += cR * g11

        return inter_hoppings, intra_hoppings


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
