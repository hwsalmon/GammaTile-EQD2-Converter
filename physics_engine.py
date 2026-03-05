"""
physics_engine.py — Phase 2: Radiobiological Engine

Implements voxel-wise conversion of Cs-131 GammaTile physical dose to EQD2
using the Linear-Quadratic (LQ) model with the Lea-Catcheside G-factor for a
permanently implanted, exponentially decaying brachytherapy source.

Physics background
------------------
For a permanent implant the dose rate decays as:

    Ḋ(t) = Ḋ₀ · exp(−λt)

where λ = ln(2)/T½ is the source decay constant.  Sublethal DNA damage
accumulates throughout the entire irradiation period and is simultaneously
repaired with rate μ = ln(2)/T_rep.

The Lea-Catcheside factor G integrates this competition (Dale 1985):

    G = λ / (λ + μ)

The full dose D is delivered over an infinite time (permanent implant), so
the LQ cell survival exponent simplifies to:

    -ln(S) = α·D + β·G·D²

Rearranging into BED and EQD2 for clinical use:

    BED  = D · [1 + G·D / (α/β)]
    EQD2 = BED / (1 + 2/(α/β))     [normalised to 2 Gy/fraction external beam]

References
----------
- Dale RG (1985). The application of the linear-quadratic dose-effect
  equation to fractionated and protracted radiotherapy.
  Br J Radiol 58:515–528.  doi:10.1259/0007-1285-58-690-515
- Brenner DJ, Hall EJ (1991). Conditions for the equivalence of continuous
  to pulsed low dose rate brachytherapy.
  Int J Radiat Oncol Biol Phys 20(1):181–190.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass
class RadiobiologyParameters:
    """Physical and radiobiological parameters for Cs-131 EQD2 conversion.

    Attributes:
        half_life_days: Cs-131 physical half-life [days].  Fixed physical
            constant — must NOT be exposed as a user-editable field in the UI.
        t_rep_hours:    Sublethal damage repair half-time [h].  Tissue- and
            endpoint-dependent; user-adjustable.
        alpha_beta:     α/β ratio [Gy] of the tissue of interest.
            Default 2 Gy is appropriate for late-responding tissue / prostate.
    """

    half_life_days: float = 9.7      # Cs-131 physical constant
    t_rep_hours: float = 1.5         # Default: 1.5 h (common for prostate)
    alpha_beta: float = 2.0          # Default: 2 Gy (late tissue / prostate)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError(f"half_life_days must be > 0, got {self.half_life_days}")
        if self.t_rep_hours <= 0:
            raise ValueError(f"t_rep_hours must be > 0, got {self.t_rep_hours}")
        if self.alpha_beta <= 0:
            raise ValueError(f"alpha_beta must be > 0, got {self.alpha_beta}")


# ---------------------------------------------------------------------------
# Computation result container
# ---------------------------------------------------------------------------


@dataclass
class EQD2Result:
    """Output of a single EQD2 computation run.

    Attributes:
        eqd2_gy:        EQD2 array [Gy], same shape as input dose.
        bed_gy:         BED array [Gy], same shape as input dose.
        params:         Parameter snapshot used to produce this result.
        lambda_per_h:   Source decay constant [h⁻¹].
        mu_per_h:       Repair rate constant [h⁻¹].
        G_factor:       Lea-Catcheside G factor (dimensionless, 0 < G < 1).
    """

    eqd2_gy: np.ndarray
    bed_gy: np.ndarray
    params: RadiobiologyParameters
    lambda_per_h: float
    mu_per_h: float
    G_factor: float

    @property
    def max_eqd2_gy(self) -> float:
        return float(self.eqd2_gy.max())

    @property
    def mean_eqd2_gy(self) -> float:
        return float(self.eqd2_gy.mean())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class PhysicsEngine:
    """Voxel-wise converter from Cs-131 physical dose to EQD2.

    The engine is designed for interactive use in the viewer: parameter changes
    update only the derived constants (λ, μ, G), not the loaded dose data.
    Re-computation of EQD2 for any slice or volume is then O(n_voxels) with no
    I/O overhead.

    Usage::

        engine = PhysicsEngine()                        # defaults
        engine = PhysicsEngine(RadiobiologyParameters(t_rep_hours=1.0, alpha_beta=3.0))

        result = engine.compute_eqd2_volume(dose_gy)   # full 3-D volume
        result = engine.compute_eqd2_slice(dose_gy[k]) # single axial slice

        # Live parameter update (triggers constant recomputation):
        engine.update_parameters(t_rep_hours=2.0)
        result = engine.compute_eqd2_volume(dose_gy)   # uses new G-factor
    """

    #: Reference dose per fraction used in EQD2 normalisation [Gy]
    D_REF: float = 2.0

    def __init__(self, params: Optional[RadiobiologyParameters] = None) -> None:
        """Initialise the engine.

        Args:
            params: Radiobiological parameters.  Defaults to Cs-131 prostate
                defaults (T½=9.7 d, T_rep=1.5 h, α/β=2 Gy).
        """
        self.params: RadiobiologyParameters = params or RadiobiologyParameters()
        self._lambda: float
        self._mu: float
        self._G: float
        self._recompute_constants()

    # ------------------------------------------------------------------
    # Public API — parameter management
    # ------------------------------------------------------------------

    def update_parameters(
        self,
        t_rep_hours: Optional[float] = None,
        alpha_beta: Optional[float] = None,
    ) -> None:
        """Update user-adjustable parameters and recompute G.

        Passing ``None`` for a parameter leaves its current value unchanged.

        Args:
            t_rep_hours: New repair half-time [h].
            alpha_beta:  New α/β ratio [Gy].

        Raises:
            ValueError: If the new parameter value is non-positive.
        """
        if t_rep_hours is not None:
            t_rep_hours = float(t_rep_hours)
            if t_rep_hours <= 0:
                raise ValueError(f"t_rep_hours must be > 0, got {t_rep_hours}")
            self.params.t_rep_hours = t_rep_hours

        if alpha_beta is not None:
            alpha_beta = float(alpha_beta)
            if alpha_beta <= 0:
                raise ValueError(f"alpha_beta must be > 0, got {alpha_beta}")
            self.params.alpha_beta = alpha_beta

        self._recompute_constants()
        logger.info(
            "Parameters updated | T_rep=%.3f h | α/β=%.2f Gy | G=%.6f",
            self.params.t_rep_hours,
            self.params.alpha_beta,
            self._G,
        )

    # ------------------------------------------------------------------
    # Public API — computation
    # ------------------------------------------------------------------

    def compute_eqd2_volume(self, dose_gy: np.ndarray) -> EQD2Result:
        """Convert a 3-D physical dose volume to EQD2.

        Args:
            dose_gy: Physical dose array [Gy], shape (n_slices, n_rows, n_cols).

        Returns:
            :class:`EQD2Result` with ``eqd2_gy`` and ``bed_gy`` arrays.
        """
        return self._compute(dose_gy)

    def compute_eqd2_slice(self, dose_slice_gy: np.ndarray) -> EQD2Result:
        """Convert a single 2-D dose slice to EQD2.

        Suitable for real-time viewer updates — only one slice needs
        recomputing when the user scrolls or changes parameters.

        Args:
            dose_slice_gy: Physical dose array [Gy], shape (n_rows, n_cols).

        Returns:
            :class:`EQD2Result` with 2-D ``eqd2_gy`` and ``bed_gy`` arrays.
        """
        return self._compute(dose_slice_gy)

    # ------------------------------------------------------------------
    # Convenience: compute with temporary parameters (non-destructive)
    # ------------------------------------------------------------------

    def compute_eqd2_with(
        self,
        dose_gy: np.ndarray,
        t_rep_hours: float,
        alpha_beta: float,
    ) -> EQD2Result:
        """Compute EQD2 with specific parameters without mutating engine state.

        Useful for parameter sweeps or background computation in worker threads.

        Args:
            dose_gy:     Physical dose array [Gy].
            t_rep_hours: Repair half-time [h] to use for this computation.
            alpha_beta:  α/β ratio [Gy] to use for this computation.

        Returns:
            :class:`EQD2Result`.
        """
        tmp_params = RadiobiologyParameters(
            half_life_days=self.params.half_life_days,
            t_rep_hours=t_rep_hours,
            alpha_beta=alpha_beta,
        )
        lam, mu, G = self._derive_constants(tmp_params)
        return self._compute_with_constants(dose_gy, tmp_params, lam, mu, G)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def decay_constant_per_hour(self) -> float:
        """Source decay constant λ [h⁻¹]."""
        return self._lambda

    @property
    def repair_constant_per_hour(self) -> float:
        """DNA repair rate constant μ [h⁻¹]."""
        return self._mu

    @property
    def G_factor(self) -> float:
        """Lea-Catcheside G factor (dimensionless, 0 < G < 1).

        G → 1 when repair is very fast relative to decay (μ ≫ λ), meaning
        sublethal damage accumulates with full effectiveness — the LQ β-term
        is unreduced.  G → 0 when decay is very fast (λ ≫ μ), meaning dose
        is delivered so quickly that repair cannot occur — equivalent to acute
        irradiation with full quadratic killing.

        For Cs-131 (T½=9.7 d) with T_rep=1.5 h: G ≈ 0.014.
        """
        return self._G

    def summary(self) -> dict:
        """Return all current parameters and derived constants as a plain dict."""
        return {
            "isotope": "Cs-131",
            "T_half_days": self.params.half_life_days,
            "T_rep_hours": self.params.t_rep_hours,
            "alpha_beta_gy": self.params.alpha_beta,
            "lambda_per_hour": self._lambda,
            "mu_per_hour": self._mu,
            "G_factor": self._G,
            "D_ref_gy": self.D_REF,
        }

    def __repr__(self) -> str:
        return (
            f"PhysicsEngine("
            f"T½={self.params.half_life_days} d, "
            f"T_rep={self.params.t_rep_hours} h, "
            f"α/β={self.params.alpha_beta} Gy, "
            f"G={self._G:.6f})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute_constants(self) -> None:
        """Recompute λ, μ, G from current params and cache them."""
        self._lambda, self._mu, self._G = self._derive_constants(self.params)
        logger.debug(
            "Constants: λ=%.6e h⁻¹ | μ=%.6e h⁻¹ | G=%.6f",
            self._lambda, self._mu, self._G,
        )

    @staticmethod
    def _derive_constants(
        params: RadiobiologyParameters,
    ) -> tuple[float, float, float]:
        """Compute (λ, μ, G) from a parameter set.

        Formulae:
            λ = ln(2) / (T½_days × 24)   [h⁻¹]
            μ = ln(2) / T_rep_hours        [h⁻¹]
            G = λ / (λ + μ)               [dimensionless]

        Returns:
            Tuple ``(lambda_h, mu_h, G)``.
        """
        LN2 = np.log(2.0)
        lambda_h = LN2 / (params.half_life_days * 24.0)
        mu_h = LN2 / params.t_rep_hours
        G = lambda_h / (lambda_h + mu_h)
        return float(lambda_h), float(mu_h), float(G)

    def _compute(self, dose_gy: np.ndarray) -> EQD2Result:
        """Run the EQD2 calculation using the engine's current constants."""
        return self._compute_with_constants(
            dose_gy, self.params, self._lambda, self._mu, self._G
        )

    def _compute_with_constants(
        self,
        dose_gy: np.ndarray,
        params: RadiobiologyParameters,
        lam: float,
        mu: float,
        G: float,
    ) -> EQD2Result:
        """Core voxel-wise EQD2 calculation.

        BED  = D × (1 + G·D / (α/β))
        EQD2 = BED / (1 + D_REF / (α/β))

        All arithmetic is float64 to maintain precision across the full dose
        range (0 to ~300 Gy in high-dose brachytherapy regions).

        Args:
            dose_gy:  Physical dose array in Gy (any shape).
            params:   Parameter set used (captured in result for provenance).
            lam:      Source decay constant λ [h⁻¹].
            mu:       Repair rate constant μ [h⁻¹].
            G:        Lea-Catcheside factor.

        Returns:
            :class:`EQD2Result`.
        """
        dose = np.asarray(dose_gy, dtype=np.float64)
        ab = params.alpha_beta

        # BED = D · (1 + G·D / α/β)
        bed = dose * (1.0 + (G * dose) / ab)

        # EQD2 = BED / (1 + D_ref / α/β)
        eqd2_denominator = 1.0 + self.D_REF / ab
        eqd2 = bed / eqd2_denominator

        return EQD2Result(
            eqd2_gy=eqd2,
            bed_gy=bed,
            params=params,
            lambda_per_h=lam,
            mu_per_h=mu,
            G_factor=G,
        )
