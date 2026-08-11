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

use crate::env::{AgentMode, ClubPick, ScenarioBudget, WorldSize};
use crate::episode_agents::{AutoManager, Policy};
use crate::run::{run_episode_cadence_with_mode, CadenceResult, ClubMetrics};
use domain::team::PlayStyle;

/// The frozen reference policy: ClubBench-Heuristic-v1. Code is fixed and
/// public; the leaderboard is anchored on this, never on the current SOTA.
pub fn reference_v1() -> AutoManager {
    AutoManager::new(PlayStyle::Attacking)
}

/// The mode-appropriate reference: the Manager track anchors on AutoManager
/// (handles the market), the Coach track on a pure best-XI + tactics coach.
pub fn reference_for(mode: AgentMode) -> Box<dyn crate::episode_agents::Policy> {
    match mode {
        AgentMode::Manager => Box::new(reference_v1()),
        AgentMode::Coach => Box::new(crate::episode_agents::CoachBestXI {
            play_style: PlayStyle::Attacking,
        }),
    }
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

/// The dimensions scored by ClubBench (sport + finance + squad).
///
/// Finance is scored on **net_value** (total wealth created: squad value +
/// balance change over the episode) and **net_spend** (transfer outlay), NOT
/// on raw ending balance — a rich club that burns £80M must not outscore a
/// frugal club that achieves the same with £10M. `squad_size`'s "better"
/// direction is scenario-dependent; it is scored lower-is-better (a trimmed
/// squad reads as the manager having moved players on).
pub const DIMENSIONS: [Dimension; 7] = [
    Dimension { name: "points", higher_better: true },
    Dimension { name: "net_value", higher_better: true },
    Dimension { name: "net_spend", higher_better: false },
    Dimension { name: "wage_bill", higher_better: false },
    Dimension { name: "squad_value", higher_better: true },
    Dimension { name: "avg_age", higher_better: false },
    Dimension { name: "squad_size", higher_better: false },
];

fn metric_value(m: &ClubMetrics, name: &str) -> f64 {
    match name {
        "points" => m.points as f64,
        "net_value" => m.net_value as f64,
        "net_spend" => m.net_spend as f64,
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
    collect_paired_for_world(pick, WorldSize::Medium, &ScenarioBudget::default(), seeds, horizon_days, candidate)
}

/// As [`collect_paired_for`], with an explicit world size and budget.
pub fn collect_paired_for_world(
    pick: &ClubPick,
    world: WorldSize,
    budget: &ScenarioBudget,
    seeds: &[u64],
    horizon_days: u64,
    candidate: &mut dyn Policy,
) -> (Vec<PairedSample>, Vec<DimReport>) {
    collect_paired_for_mode(pick, world, budget, AgentMode::Manager, seeds, horizon_days, candidate)
}

/// As [`collect_paired_for_world`], with an explicit agent mode.
pub fn collect_paired_for_mode(
    pick: &ClubPick,
    world: WorldSize,
    budget: &ScenarioBudget,
    mode: AgentMode,
    seeds: &[u64],
    horizon_days: u64,
    candidate: &mut dyn Policy,
) -> (Vec<PairedSample>, Vec<DimReport>) {
    let mut reference = reference_for(mode);
    // per dimension -> Vec<PairedSample>
    let mut per_dim: Vec<Vec<PairedSample>> = DIMENSIONS.iter().map(|_| Vec::new()).collect();

    for &seed in seeds {
        let ref_res: CadenceResult = run_episode_cadence_with_mode(seed, pick, world, budget, mode, horizon_days, reference.as_mut());
        let cand_res: CadenceResult = run_episode_cadence_with_mode(seed, pick, world, budget, mode, horizon_days, candidate);
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

/// Mean raw metrics of the reference policy over `seeds` — the difficulty
/// anchor for a scenario cell (its points, ending balance, squad, etc.).
pub fn reference_mean_metrics(
    pick: &ClubPick,
    world: WorldSize,
    budget: &ScenarioBudget,
    mode: AgentMode,
    seeds: &[u64],
    horizon_days: u64,
) -> ClubMetrics {
    let mut reference = reference_for(mode);
    let mut sum: Option<ClubMetrics> = None;
    for &seed in seeds {
        let r = run_episode_cadence_with_mode(seed, pick, world, budget, mode, horizon_days, reference.as_mut());
        let m = r.metrics;
        sum = Some(match sum {
            None => m.clone(),
            Some(s) => ClubMetrics {
                points: s.points + m.points,
                position: s.position + m.position,
                goal_difference: s.goal_difference + m.goal_difference,
                balance: s.balance + m.balance,
                transfer_budget: s.transfer_budget + m.transfer_budget,
                wage_bill: s.wage_bill + m.wage_bill,
                squad_value: s.squad_value + m.squad_value,
                avg_age: s.avg_age + m.avg_age,
                squad_size: s.squad_size + m.squad_size,
                net_value: s.net_value + m.net_value,
                net_spend: s.net_spend + m.net_spend,
            },
        });
    }
    let s = sum.unwrap_or_default();
    let n = seeds.len().max(1) as f64;
    ClubMetrics {
        points: (s.points as f64 / n) as u32,
        position: (s.position as f64 / n) as usize,
        goal_difference: (s.goal_difference as f64 / n) as i32,
        balance: (s.balance as f64 / n) as i64,
        transfer_budget: (s.transfer_budget as f64 / n) as i64,
        wage_bill: (s.wage_bill as f64 / n) as u64,
        squad_value: (s.squad_value as f64 / n) as u64,
        avg_age: s.avg_age / n,
        squad_size: (s.squad_size as f64 / n) as usize,
        net_value: (s.net_value as f64 / n) as i64,
        net_spend: (s.net_spend as f64 / n) as i64,
    }
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
