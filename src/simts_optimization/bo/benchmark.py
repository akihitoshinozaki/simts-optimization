"""C5 benchmark: BO versus random and Sobol search over repeated seeds.

The referee (the hidden landscape) is consulted only here, after campaigns end,
to decide when each campaign first measured a setting in the good region.

Run from the command line::

    python -m simts_optimization.bo --seeds 20 --budget 60
"""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from ..sim.landscape import HiddenLandscape
from .evaluation import SimulatedExperiment, _json_value
from .loop import BASELINE_CONFIG, STRATEGIES, CampaignConfig, CampaignResult, run_campaign


@dataclass(frozen=True)
class CampaignOutcome:
    strategy: str
    seed: int
    evaluations_to_discovery: int | None
    """1-based count of measurements (all kinds) up to the first good-region one."""
    recommendation_in_region: bool
    best_quality_curve: tuple[float, ...]
    """Best referee quality among settings measured so far, per measurement."""
    failed_count: int
    constraint_violation_count: int
    repeat_count: int
    reference_count: int
    noise_std: float | None


@dataclass(frozen=True)
class StrategySummary:
    strategy: str
    campaigns: int
    discovery_rate: float
    median_evaluations_to_discovery: float
    """Undiscovered campaigns count as budget + 1 (right-censored)."""
    mean_evaluations_to_discovery: float
    recommendation_in_region_rate: float


@dataclass(frozen=True)
class BenchmarkResult:
    budget: int
    seeds: tuple[int, ...]
    good_region_volume_fraction: float
    outcomes: tuple[CampaignOutcome, ...]
    summaries: dict[str, StrategySummary]
    bo_wins_against: dict[str, float]
    """Fraction of seeds where BO discovered strictly sooner (ties count half)."""
    complete: bool

    def to_dict(self) -> dict:
        return _json_value(asdict(self))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")


def summarize_campaign(result: CampaignResult, landscape: HiddenLandscape) -> CampaignOutcome:
    quality = [landscape.quality(r.settings) for r in result.records]
    hits = [i for i, r in enumerate(result.records) if landscape.in_good_region(r.settings)]
    kinds = [r.kind for r in result.records]
    return CampaignOutcome(
        strategy=result.strategy,
        seed=result.seed,
        evaluations_to_discovery=hits[0] + 1 if hits else None,
        recommendation_in_region=(result.recommendation is not None
                                  and landscape.in_good_region(result.recommendation)),
        best_quality_curve=tuple(float(q) for q in np.maximum.accumulate(quality)),
        failed_count=sum(r.failed for r in result.records),
        constraint_violation_count=sum(r.status == "constraint_violation" for r in result.records),
        repeat_count=kinds.count("repeat"),
        reference_count=kinds.count("reference"),
        noise_std=result.noise.pooled_std,
    )


def good_region_volume_fraction(landscape: HiddenLandscape, experiment: SimulatedExperiment,
                                samples: int = 200_000, seed: int = 0) -> float:
    space = experiment.space
    unit = np.random.default_rng(seed).random((samples, space.dimension))
    x = space.lower + unit * (space.upper - space.lower)
    settings = dict(zip(space.names, x.T))
    return float(np.mean(landscape._raw_quality(*(settings[n] for n in space.names))
                         / landscape._best_raw_quality >= landscape.good_region_fraction_of_best))


def _discovery(outcome: CampaignOutcome, budget: int) -> int:
    return outcome.evaluations_to_discovery or budget + 1


def run_benchmark(
    seeds: Sequence[int] = tuple(range(20)),
    budget: int = 60,
    landscape: HiddenLandscape | None = None,
    bo_config: CampaignConfig | None = None,
    baseline_config: CampaignConfig | None = None,
    progress: Callable[[CampaignOutcome], None] | None = None,
    campaign_dir: str | Path | None = None,
) -> BenchmarkResult:
    """Run every strategy on every seed with the same measurement budget.

    Each seed fixes the Sobol scramble, measurement noise seeds, and the random
    stream, so strategies face the same apparatus. BO's budget includes its
    repeats and references; baselines spend everything on new settings.
    """
    landscape = landscape or HiddenLandscape()
    bo_config = bo_config or CampaignConfig(budget=budget)
    baseline_config = baseline_config or CampaignConfig(**{
        **asdict(BASELINE_CONFIG), "budget": budget})
    if bo_config.budget != budget or baseline_config.budget != budget:
        raise ValueError("all strategies must share the same budget")

    def factory():
        return SimulatedExperiment(landscape)

    outcomes = []
    for seed in seeds:
        for strategy in STRATEGIES:
            config = bo_config if strategy == "bo" else baseline_config
            result = run_campaign(strategy, seed, config, factory)
            if campaign_dir is not None:
                Path(campaign_dir).mkdir(parents=True, exist_ok=True)
                result.save(Path(campaign_dir) / f"{strategy}_seed{seed:03d}.json")
            outcome = summarize_campaign(result, landscape)
            outcomes.append(outcome)
            if progress:
                progress(outcome)

    summaries = {}
    for strategy in STRATEGIES:
        own = [o for o in outcomes if o.strategy == strategy]
        steps = np.array([_discovery(o, budget) for o in own], dtype=float)
        summaries[strategy] = StrategySummary(
            strategy, len(own),
            float(np.mean([o.evaluations_to_discovery is not None for o in own])),
            float(np.median(steps)), float(np.mean(steps)),
            float(np.mean([o.recommendation_in_region for o in own])),
        )
    by_key = {(o.strategy, o.seed): o for o in outcomes}
    wins = {}
    for baseline in STRATEGIES[1:]:
        scores = []
        for seed in seeds:
            bo, other = (_discovery(by_key[(s, seed)], budget) for s in ("bo", baseline))
            scores.append(1.0 if bo < other else 0.5 if bo == other else 0.0)
        wins[baseline] = float(np.mean(scores))
    bo = summaries["bo"]
    complete = all(
        bo.median_evaluations_to_discovery < summaries[b].median_evaluations_to_discovery
        and bo.mean_evaluations_to_discovery < summaries[b].mean_evaluations_to_discovery
        and bo.discovery_rate >= summaries[b].discovery_rate
        and wins[b] > 0.5
        for b in STRATEGIES[1:]
    )
    return BenchmarkResult(budget, tuple(seeds),
                           good_region_volume_fraction(landscape, factory()),
                           tuple(outcomes), summaries, wins, complete)


def plot_benchmark(result: BenchmarkResult, path: str | Path, landscape: HiddenLandscape) -> None:
    """Discovery curves and best-quality traces for each strategy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"bo": "#2a6fdb", "random": "#d9822b", "sobol": "#3f9e5a"}
    labels = {"bo": "Bayesian optimization", "random": "Random search", "sobol": "Sobol search"}
    steps = np.arange(1, result.budget + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for strategy in STRATEGIES:
        own = [o for o in result.outcomes if o.strategy == strategy]
        found = np.array([[o.evaluations_to_discovery is not None
                           and o.evaluations_to_discovery <= k for k in steps] for o in own])
        left.plot(steps, found.mean(axis=0), color=colors[strategy], lw=2, label=labels[strategy])
        curves = np.array([o.best_quality_curve for o in own])
        right.plot(steps, np.median(curves, axis=0), color=colors[strategy], lw=2,
                   label=labels[strategy])
        right.fill_between(steps, *np.percentile(curves, [25, 75], axis=0),
                           color=colors[strategy], alpha=0.15, lw=0)
    left.set(xlabel="Measurements used (incl. repeats and references)",
             ylabel="Fraction of seeds that found the good region", ylim=(0, 1.02),
             title=f"Discovery ({len(result.seeds)} seeds)")
    right.axhline(landscape.good_region_fraction_of_best, color="0.4", ls="--", lw=1)
    right.set(xlabel="Measurements used", ylabel="Best hidden quality measured so far",
              ylim=(0, 1.02), title="Median best quality (IQR shaded)")
    for ax in (left, right):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> BenchmarkResult:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--output-dir", default="outputs/c5")
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    landscape = HiddenLandscape()

    def progress(o: CampaignOutcome) -> None:
        print(f"seed {o.seed:3d} {o.strategy:6s} discovery={o.evaluations_to_discovery} "
              f"recommendation_in_region={o.recommendation_in_region}", flush=True)

    result = run_benchmark(range(args.seeds), args.budget, landscape, progress=progress,
                           campaign_dir=output / "campaigns")
    result.save(output / "benchmark.json")
    plot_benchmark(result, output / "benchmark.png", landscape)
    print(f"good-region volume fraction: {result.good_region_volume_fraction:.4f}")
    for s in result.summaries.values():
        print(f"{s.strategy:6s} discovered={s.discovery_rate:.2f} "
              f"median={s.median_evaluations_to_discovery:.1f} "
              f"mean={s.mean_evaluations_to_discovery:.1f} "
              f"recommendation_in_region={s.recommendation_in_region_rate:.2f}")
    print("BO win rate:", result.bo_wins_against)
    print("C5 complete:", result.complete)
    return result


if __name__ == "__main__":
    main()
