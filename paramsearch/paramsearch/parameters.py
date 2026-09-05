"""Parameter space definition.

Each searchable parameter maps 1:1 to a HOCON key under particle-agent.config,
overridden at launch via -Dparticle-agent.config.<name>=<value>.

Parameters are grouped by which agent model uses them; the search space for a
given study is the shared group plus the groups of the chosen model
(particleAgentFactory). `bounds` are the search box for both Sobol screening
and Optuna. `log=True` samples on a log scale.
"""

import math
from dataclasses import dataclass

SPP = "SPPAgentFactory"
SPIN = "SpinSystemAgentFactory"
NEURAL_FIELD = "NeuralFieldAgentFactory"
MODELS = (SPP, SPIN, NEURAL_FIELD)

# Parameter groups. "shared" applies to every model; "ring" to both
# ring-attractor implementations (spin system and neural field).
GROUP_MODELS = {
    "shared": {SPP, SPIN, NEURAL_FIELD},
    "spp": {SPP},
    "ring": {SPIN, NEURAL_FIELD},
    "neural-field": {NEURAL_FIELD},
}


@dataclass(frozen=True)
class Parameter:
    name: str
    group: str
    bounds: tuple[float, float]
    default: float
    log: bool = False
    integer: bool = False


PARAMETERS = [
    # --- shared: kinematics, hopping, marching intermittency, perception ---
    Parameter("averageSpeed", "shared", (0.001, 0.05), 0.01),
    Parameter("hopProbability", "shared", (0.0, 0.05), 0.01),
    Parameter("crowdedHopProbability", "shared", (0.0, 0.5), 0.2),
    Parameter("hopDuration", "shared", (0.1, 1.0), 0.3),
    Parameter("hopSpeed", "shared", (0.05, 0.3), 0.1),
    Parameter("activityPeriod", "shared", (30.0, 2700.0), 270.0, log=True),
    Parameter("minimalInactivityPeriod", "shared", (10.0, 2700.0), 90.0, log=True),
    Parameter("resumeMarchProbabilityPerSecond", "shared", (1e-4, 0.1), 0.01, log=True),
    Parameter("occlusionThreshold", "shared", (5, 50), 25, integer=True),
    # --- SPP three-zone model ---
    Parameter("previousDirectionWeight", "spp", (0.0, 0.95), 0.6),
    Parameter("randomComponentWeight", "spp", (0.0, 0.3), 0.05),
    Parameter("repulsionRange", "spp", (0.01, 0.1), 0.035),
    Parameter("alignmentRange", "spp", (0.05, 0.3), 0.135),
    # Upper bound capped at the search Scenario's agentContainerSize (0.3 m):
    # perception only reaches the 8 neighbouring containers, so larger ranges
    # would be silently truncated.
    Parameter("attractionRange", "spp", (0.15, 0.3), 0.3),
    Parameter("repulsionWeight", "spp", (0.1, 5.0), 1.5, log=True),
    Parameter("alignmentWeight", "spp", (0.1, 10.0), 3.0, log=True),
    Parameter("attractionWeight", "spp", (1e-4, 0.1), 0.001, log=True),
    # --- ring attractor (spin system + neural field) ---
    Parameter("receptiveFieldStd", "ring", (0.05, 1.5), 0.4, log=True),
    Parameter("synapticConnectivityCoefficient", "ring", (0.1, 2.0), 0.5),
    Parameter("inverseTemperatureCoefficient", "ring", (1.0, 1e4), 1000.0, log=True),
    Parameter("neuralInhibitionCoefficient", "ring", (0.0, 1.0), 0.0),
    Parameter("totalSocialAttraction", "ring", (0.01, 2.0), 0.24, log=True),
    # --- neural field escape / anti-goal mechanism ---
    Parameter("antiGoalOverrideRange", "neural-field", (0.035, 0.5), 0.3),
    # Escape gain. Bounds from the 2026-09-05 strength sweep: marching peaks
    # near 0.72 and degrades by 2.0; a strength of 0 does NOT disable the
    # mechanism (flagged pursuers are silenced, not attractive).
    Parameter("antiGoalStimulusStrength", "neural-field", (0.01, 2.0), 0.72, log=True),
    Parameter("antiGoalAngleRangeStart", "neural-field", (0.785, 2.75), 1.507),
    Parameter("pursuerHeadingAngleEnd", "neural-field", (0.393, 3.1415), 1.507),
]


def active_parameters(model: str) -> list[Parameter]:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
    return [p for p in PARAMETERS if model in GROUP_MODELS[p.group]]


def by_name(name: str) -> Parameter:
    return next(p for p in PARAMETERS if p.name == name)


def salib_problem(params: list[Parameter]) -> dict:
    """SALib problem dict. Log parameters are sampled in log10 space."""
    bounds = [
        [math.log10(p.bounds[0]), math.log10(p.bounds[1])] if p.log else list(p.bounds)
        for p in params
    ]
    return {
        "num_vars": len(params),
        "names": [p.name for p in params],
        "bounds": bounds,
    }


def decode_sample(row, params: list[Parameter]) -> dict[str, float]:
    """Map a SALib sample row back to parameter values (undo log10, round ints)."""
    values = {}
    for p, v in zip(params, row):
        if p.log:
            v = 10.0**v
        if p.integer:
            v = int(round(v))
        values[p.name] = v
    return values
