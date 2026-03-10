from __future__ import annotations

from dataclasses import asdict, dataclass

from .legacy_comparison import LegacyComparisonConfig
from .survival import SynthConfig


STANDARDIZED_DEFAULTS = {
    "n": 2000,
    "p_x": 5,
    "seed": 42,
    "target_censor_rate": 0.35,
}


@dataclass(frozen=True)
class SyntheticScenario:
    slug: str
    family: str
    source: str
    title: str
    config: dict[str, object]
    notes: str = ""


def _proxy_corr(a: float, sigma: float) -> float:
    return abs(a) / ((a**2 + sigma**2) ** 0.5)


def _proxy_label(a_exposure: float, sigma_exposure: float, a_outcome: float, sigma_outcome: float) -> str:
    mean_corr = 0.5 * (_proxy_corr(a_exposure, sigma_exposure) + _proxy_corr(a_outcome, sigma_outcome))
    if mean_corr >= 0.65:
        return "informative"
    return "non-informative"


def _confounding_label(gamma: float, beta: float) -> str:
    level = max(abs(gamma), abs(beta))
    if level < 0.35:
        return "weak"
    if level < 0.8:
        return "moderate"
    return "strong"


def _effect_label(tau_log_hr: float) -> str:
    magnitude = abs(tau_log_hr)
    if magnitude <= 0.15:
        size = "near-null"
    elif magnitude <= 0.35:
        size = "small"
    elif magnitude <= 0.6:
        size = "moderate"
    else:
        size = "large"

    if tau_log_hr < 0:
        return f"beneficial {size}"
    if tau_log_hr > 0:
        return f"harmful {size}"
    return "null"


def _linearity_label(linear_treatment: bool, linear_outcome: bool) -> str:
    if linear_treatment and linear_outcome:
        return "Linear treatment / linear outcome"
    if linear_treatment and not linear_outcome:
        return "Linear treatment / nonlinear outcome"
    if not linear_treatment and linear_outcome:
        return "Nonlinear treatment / linear outcome"
    return "Nonlinear treatment / nonlinear outcome"


def _survival_title(config: SynthConfig) -> str:
    return (
        f"{_linearity_label(config.linear_treatment, config.linear_outcome)}, "
        f"{_proxy_label(config.aZ, config.sigma_z, config.aW, config.sigma_w)} proxies, "
        f"{_confounding_label(config.gamma_u_in_a, config.beta_u_in_t)} confounding, "
        f"{_effect_label(config.tau_log_hr)} treatment effect"
    )


def _legacy_title(config: LegacyComparisonConfig) -> str:
    return (
        "Legacy comparison DGP, "
        f"{_proxy_label(config.az, config.sigma_z, config.av, config.sigma_v)} proxies, "
        f"{_confounding_label(config.gamma_u_in_w, config.beta_u_in_t)} confounding, "
        f"{_effect_label(config.tau_log_hr)} treatment effect"
    )


def standardized_survival_config(**overrides: object) -> SynthConfig:
    values = {**STANDARDIZED_DEFAULTS, **overrides}
    return SynthConfig(**values)


def standardized_legacy_comparison_config(**overrides: object) -> LegacyComparisonConfig:
    values = {**STANDARDIZED_DEFAULTS, **overrides}
    return LegacyComparisonConfig(**values)


def standardized_synthetic_scenarios() -> list[SyntheticScenario]:
    raw_survival_scenarios = [
        (
            "survival_default",
            "data_generation.py",
            standardized_survival_config(),
            "Canonical NC survival generator from nc_csf/data_generation.py with standardized defaults.",
        ),
        (
            "notebook_linear_informative_strong_beneficial_large",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=-0.7,
                linear_treatment=True,
                linear_outcome=True,
            ),
            "Ported from the notebook's linear/linear informative-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_linear_informative_weak_harmful_small",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.2,
                beta_u_in_t=0.2,
                tau_log_hr=0.25,
                linear_treatment=True,
                linear_outcome=True,
            ),
            "Ported from the notebook's linear/linear weak-confounding case; censoring standardized to 35%.",
        ),
        (
            "notebook_linear_noninformative_strong_harmful_moderate",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=0.5,
                linear_treatment=True,
                linear_outcome=True,
            ),
            "Ported from the notebook's linear/linear weak-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_linear_noninformative_weak_harmful_nearnull",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.25,
                beta_u_in_t=0.2,
                tau_log_hr=0.12,
                linear_treatment=True,
                linear_outcome=True,
            ),
            "Ported from the notebook's linear/linear near-null case; censoring standardized to 35%.",
        ),
        (
            "notebook_outcome_nonlinear_informative_strong_beneficial_large",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=-0.7,
                linear_treatment=True,
                linear_outcome=False,
            ),
            "Ported from the notebook's nonlinear-outcome informative-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_outcome_nonlinear_informative_weak_harmful_small",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.2,
                beta_u_in_t=0.2,
                tau_log_hr=0.3,
                linear_treatment=True,
                linear_outcome=False,
            ),
            "Ported from the notebook's nonlinear-outcome weak-confounding case; censoring standardized to 35%.",
        ),
        (
            "notebook_outcome_nonlinear_noninformative_strong_harmful_moderate",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=0.5,
                linear_treatment=True,
                linear_outcome=False,
            ),
            "Ported from the notebook's nonlinear-outcome weak-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_outcome_nonlinear_noninformative_weak_harmful_nearnull",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.25,
                beta_u_in_t=0.2,
                tau_log_hr=0.12,
                linear_treatment=True,
                linear_outcome=False,
            ),
            "Ported from the notebook's nonlinear-outcome near-null case; censoring standardized to 35%.",
        ),
        (
            "notebook_fully_nonlinear_informative_strong_beneficial_large",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=-0.7,
                linear_treatment=False,
                linear_outcome=False,
            ),
            "Ported from the notebook's fully nonlinear informative-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_fully_nonlinear_informative_weak_harmful_small",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=1.125,
                aW=1.5,
                sigma_w=1.53,
                gamma_u_in_a=0.2,
                beta_u_in_t=0.2,
                tau_log_hr=0.3,
                linear_treatment=False,
                linear_outcome=False,
            ),
            "Ported from the notebook's fully nonlinear weak-confounding case; censoring standardized to 35%.",
        ),
        (
            "notebook_fully_nonlinear_noninformative_strong_harmful_moderate",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.9,
                beta_u_in_t=1.1,
                tau_log_hr=0.5,
                linear_treatment=False,
                linear_outcome=False,
            ),
            "Ported from the notebook's fully nonlinear weak-proxy case; censoring standardized to 35%.",
        ),
        (
            "notebook_fully_nonlinear_noninformative_weak_harmful_nearnull",
            "non_censoring_testing.ipynb",
            standardized_survival_config(
                aZ=1.5,
                sigma_z=7.35,
                aW=1.5,
                sigma_w=4.77,
                gamma_u_in_a=0.25,
                beta_u_in_t=0.2,
                tau_log_hr=0.12,
                linear_treatment=False,
                linear_outcome=False,
            ),
            "Ported from the notebook's fully nonlinear near-null case; censoring standardized to 35%.",
        ),
    ]

    scenarios = [
        SyntheticScenario(
            slug=slug,
            family="survival",
            source=source,
            title=_survival_title(config),
            config=asdict(config),
            notes=notes,
        )
        for slug, source, config, notes in raw_survival_scenarios
    ]

    legacy = standardized_legacy_comparison_config()
    scenarios.append(
        SyntheticScenario(
            slug="legacy_comparison_default",
            family="legacy_comparison",
            source="NC_CSF_comparison.ipynb",
            title=_legacy_title(legacy),
            config=asdict(legacy),
            notes="Legacy notebook comparison generator with Eq.7/Eq.9 augmentation, standardized to n=2000, p=5, seed=42, censoring=35%.",
        )
    )
    return scenarios
