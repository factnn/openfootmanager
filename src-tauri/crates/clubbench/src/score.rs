//! Paired-seed, reference-relative scoring — the ClubBench evaluation protocol.
//!
//! Protocol (see docs/scenarios.md):
//!   1. Every agent plays the SAME set of evaluation seeds (paired design).
//!   2. On each seed, a frozen reference policy also plays the same world.
//!   3. Per dimension we report:
//!        - raw mean (candidate and reference),
//!        - paired Δ = mean over seeds of (candidate_i − reference_i), with a
//!          CI, and
//!        - Z = (candidate_mean − reference_mean) / reference_std, signed so
//!          that higher is always better.
//!   4. Raw football/finance numbers are always shown alongside Z so the
//!      leaderboard stays interpretable ("took 46 points, spent £2M less than
//!      the reference manager").

use crate::env::{ClubPick, WorldSize};
use crate::episode_agents::{AutoManager, Policy};
use crate::run::{run_episode_cadence_for_world, CadenceResult, ClubMetrics};
use domain::team::PlayStyle;

/// The frozen reference policy: ClubBench-Heuristic-v1. Code is fixed and
/// public; the leaderboard is anchored on this, never on the current SOTA.
pub fn reference_v1() -> AutoManager {
    AutoManager::new(PlayStyle::Attacking)
}

/// A named evaluation dimension with its direction.
pub struct Dimension {
    pub name: &'static str,
    pub higher_better: bool,
}

/// One seed's candidate vs reference value for a dimension.
#[derive(Clone, Copy)]
pub struct PairedSample {
    pub seed: u64,
    pub candidate: f64,
    pub reference: f64,
}

/// Per-dimension report.
pub struct DimReport {
    pub name: String,
    pub candidate_mean: f64,
    pub reference_mean: f64,
    pub delta_mean: f64,
    pub delta_ci: f64,
    pub z: f64,
    pub higher_better: bool,
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn std(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

fn t95(n: usize) -> f64 {
    // Approximate two-sided 95% t critical values for df = n-1.
    match n {
        2 => 12.71,
        3 => 4.30,
        4 => 3.18,
        5 => 2.78,
        6 => 2.57,
        7 => 2.45,
        8 => 2.37,
        9 => 2.31,
        10 => 2.26,
        12 => 2.20,
        15 => 2.13,
        20 => 2.09,
        30 => 2.04,
        _ => 1.96,
    }
}

/// Score one dimension over paired seeds.
pub fn score_dimension(samples: &[PairedSample], dim: &Dimension) -> DimReport {
    let cand: Vec<f64> = samples.iter().map(|s| s.candidate).collect();
    let refs: Vec<f64> = samples.iter().map(|s| s.reference).collect();
    let deltas: Vec<f64> = samples.iter().map(|s| s.candidate - s.reference).collect();

    let cand_mean = mean(&cand);
    let ref_mean = mean(&refs);
    let ref_std = std(&refs);
    let delta_mean = mean(&deltas);
    let delta_std = std(&deltas);
    let delta_ci = t95(samples.len()) * delta_std / (samples.len() as f64).sqrt();

    let z_raw = if ref_std > 1e-9 {
        (cand_mean - ref_mean) / ref_std
    } else {
        if delta_mean.abs() < 1e-9 { 0.0 } else { delta_mean.signum() }
    };
    let z = if dim.higher_better { z_raw } else { -z_raw };

    DimReport {
        name: dim.name.to_string(),
        candidate_mean: cand_mean,
        reference_mean: ref_mean,
        delta_mean,
        delta_ci,
        z,
        higher_better: dim.higher_better,
    }
}

/// The six dimensions scored by ClubBench v0.1 (sport + finance + squad).
/// `squad_size` is reported so that selling (players leaving) and buying
/// (players joining) are directly visible; its "better" direction is
/// scenario-dependent, so it is scored with `higher_better: false` (a trimmed
/// squad reads as the manager having moved players on).
pub const DIMENSIONS: [Dimension; 6] = [
    Dimension { name: "points", higher_better: true },
    Dimension { name: "balance", higher_better: true },
    Dimension { name: "wage_bill", higher_better: false },
    Dimension { name: "squad_value", higher_better: true },
    Dimension { name: "avg_age", higher_better: false },
    Dimension { name: "squad_size", higher_better: false },
];

fn metric_value(m: &ClubMetrics, name: &str) -> f64 {
    match name {
        "points" => m.points as f64,
        "balance" => m.balance as f64,
        "wage_bill" => m.wage_bill as f64,
        "squad_value" => m.squad_value as f64,
        "avg_age" => m.avg_age,
        "squad_size" => m.squad_size as f64,
        _ => 0.0,
    }
}

/// Collect paired (candidate, reference) samples for every dimension over a
/// fixed set of seeds. The reference and candidate both play each seed.
pub fn collect_paired(
    seeds: &[u64],
    horizon_days: u64,
    candidate: &mut dyn Policy,
) -> (Vec<PairedSample>, Vec<DimReport>) {
    collect_paired_for(&ClubPick::Index(0), seeds, horizon_days, candidate)
}

/// As [`collect_paired`], managing the club selected by `pick`.
pub fn collect_paired_for(
    pick: &ClubPick,
    seeds: &[u64],
    horizon_days: u64,
    candidate: &mut dyn Policy,
) -> (Vec<PairedSample>, Vec<DimReport>) {
    collect_paired_for_world(pick, WorldSize::Medium, seeds, horizon_days, candidate)
}

/// As [`collect_paired_for`], with an explicit world size.
pub fn collect_paired_for_world(
    pick: &ClubPick,
    world: WorldSize,
    seeds: &[u64],
    horizon_days: u64,
    candidate: &mut dyn Policy,
) -> (Vec<PairedSample>, Vec<DimReport>) {
    let mut reference = reference_v1();
    // per dimension -> Vec<PairedSample>
    let mut per_dim: Vec<Vec<PairedSample>> = DIMENSIONS.iter().map(|_| Vec::new()).collect();

    for &seed in seeds {
        let ref_res: CadenceResult = run_episode_cadence_for_world(seed, pick, world, horizon_days, &mut reference);
        let cand_res: CadenceResult = run_episode_cadence_for_world(seed, pick, world, horizon_days, candidate);
        for (i, dim) in DIMENSIONS.iter().enumerate() {
            per_dim[i].push(PairedSample {
                seed,
                candidate: metric_value(&cand_res.metrics, dim.name),
                reference: metric_value(&ref_res.metrics, dim.name),
            });
        }
    }

    let reports = DIMENSIONS
        .iter()
        .enumerate()
        .map(|(i, dim)| score_dimension(&per_dim[i], dim))
        .collect();
    let samples: Vec<PairedSample> = per_dim.into_iter().next().unwrap_or_default();
    (samples, reports)
}

/// Render the dimension reports as a compact table.
pub fn render_reports(reports: &[DimReport]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<12} {:>10} {:>10} {:>10} {:>8} {:>7}\n",
        "dim", "cand μ", "ref μ", "Δ", "Δ±CI", "Z"
    ));
    for r in reports {
        out.push_str(&format!(
            "{:<12} {:>10.1} {:>10.1} {:>10.1} {:>8.1} {:>7.2}\n",
            r.name,
            r.candidate_mean,
            r.reference_mean,
            r.delta_mean,
            r.delta_ci,
            r.z
        ));
    }
    out
}
