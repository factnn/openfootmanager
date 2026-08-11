//! ClubBench CLI.
//!
//! `gate0`    — the original lineup/tactics experiment (single-action baselines).
//! `cadence`  — the decision-cadence experiment: the agent is consulted at every
//!              matchday and transfer offer, producing a long decision trajectory.

use clubbench::agents::{Agent, BestXIAgent, NoopAgent, RandomXIAgent, StyleProbe, WorstXIAgent};
use clubbench::episode_agents::{
    AutoManager, OffersOnlyManager, PassiveManager, Policy, ProactiveManager, SellingManager,
};
use clubbench::run::{run_episode, run_episode_cadence_for_world};
use clubbench::score;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "clubbench", about = "ClubBench environment experiments")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Original lineup/tactics experiment
    Gate0 {
        #[arg(long, default_value = "42,43,44")]
        seeds: String,
        #[arg(long, default_value_t = 400)]
        days: u64,
    },
    /// Decision-cadence experiment (matchdays + transfer offers)
    Cadence {
        #[arg(long, default_value = "42,43,44")]
        seeds: String,
        #[arg(long, default_value_t = 400)]
        days: u64,
        /// World size: medium (default) | standard | compact
        #[arg(long, default_value = "medium")]
        world: String,
        /// Scenario budget: crisis | moneyball | rebuild (default) | title
        #[arg(long, default_value = "rebuild")]
        scenario: String,
    },
    /// Paired-seed, reference-relative scoring (the benchmark evaluation protocol)
    Score {
        #[arg(long, default_value = "42,43,44,45,46,47,48,49")]
        seeds: String,
        #[arg(long, default_value_t = 400)]
        days: u64,
        /// Managed club by squad-strength rank (0 = weakest)
        #[arg(long)]
        club: Option<usize>,
        /// World size: medium (default, ~120 clubs) | standard (~440) | compact (~16)
        #[arg(long, default_value = "medium")]
        world: String,
        /// Scenario budget: crisis | moneyball | rebuild (default) | title
        #[arg(long, default_value = "rebuild")]
        scenario: String,
    },
    /// Run the full benchmark suite (scenario grid × all baselines) — one command
    Run {
        #[arg(long, default_value = "42,43")]
        seeds: String,
        #[arg(long, default_value_t = 300)]
        days: u64,
        /// World size: medium (default) | standard | compact
        #[arg(long, default_value = "medium")]
        world: String,
        /// Managed-club strength ranks (comma-separated); 0 = weakest
        #[arg(long, default_value = "0")]
        clubs: String,
        /// Scenario budgets (comma-separated): crisis,moneyball,rebuild,title
        #[arg(long, default_value = "crisis,rebuild")]
        scenarios: String,
        /// Track: manager (transfers+scouting+finance) | coach (lineup/tactics only)
        #[arg(long, default_value = "manager")]
        mode: String,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Gate0 { seeds, days } => gate0(&seeds, days),
        Commands::Cadence { seeds, days, world, scenario } => cadence(&seeds, days, &world, &scenario),
        Commands::Score { seeds, days, club, world, scenario } => score_cmd(&seeds, days, club, &world, &scenario),
        Commands::Run { seeds, days, world, clubs, scenarios, mode } => {
            run_benchmark(&seeds, days, &world, &clubs, &scenarios, &mode)
        }
    }
}

/// Run the full benchmark: for every (club, scenario) cell, score every
/// baseline candidate vs the frozen reference, and print a consolidated
/// leaderboard (per-cell Z table + overall mean Z per dimension).
fn run_benchmark(
    seeds_str: &str,
    days: u64,
    world_str: &str,
    clubs_str: &str,
    scenarios_str: &str,
    mode_str: &str,
) {
    use clubbench::env::{AgentMode, ClubPick, ScenarioBudget};
    use clubbench::score::{collect_paired_for_mode, DimReport};
    use std::collections::BTreeMap;

    let mode = match mode_str.trim().to_lowercase().as_str() {
        "coach" => AgentMode::Coach,
        _ => AgentMode::Manager,
    };
    let seeds: Vec<u64> = seeds_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let world = world_size(world_str);
    let clubs: Vec<usize> = clubs_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    let scenarios: Vec<&str> = scenarios_str.split(',').map(str::trim).collect();

    println!(
        "ClubBench Benchmark — mode={:?}, world={:?}, {} seeds, clubs={:?}, scenarios={:?}",
        mode,
        world,
        seeds.len(),
        clubs,
        scenarios
    );
    println!(
        "reference = {}",
        if mode == AgentMode::Coach { "CoachBestXI (frozen, Attacking)" } else { "ClubBench-Heuristic-v1 (frozen AutoManager, Attacking)" }
    );
    if seeds.len() < 8 {
        println!("note: Z is noisy with <8 seeds (reference σ is estimated from few samples); use 8+ seeds for stable leaderboards.\n");
    } else {
        println!();
    }

    let dims = ["points", "net_value", "net_spend", "wage_bill", "squad_value", "avg_age", "squad_size"];
    let mut candidates: Vec<Box<dyn Policy>> = if mode == AgentMode::Coach {
        vec![
            Box::new(clubbench::episode_agents::CoachBestXI { play_style: domain::team::PlayStyle::Attacking }),
            Box::new(clubbench::episode_agents::CoachBestXI { play_style: domain::team::PlayStyle::Balanced }),
            Box::new(clubbench::episode_agents::CoachBestXI { play_style: domain::team::PlayStyle::Defensive }),
            Box::new(clubbench::episode_agents::CoachRandom),
            Box::new(clubbench::episode_agents::CoachWorst),
        ]
    } else {
        vec![
            Box::new(ProactiveManager::new(domain::team::PlayStyle::Attacking)),
            Box::new(SellingManager::new(domain::team::PlayStyle::Attacking)),
            Box::new(AutoManager::new(domain::team::PlayStyle::Balanced)),
            Box::new(OffersOnlyManager),
            Box::new(PassiveManager),
        ]
    };

    // overall: candidate -> dim -> sum of Z (for the mean)
    let mut overall: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut cells = 0usize;

    for scenario in &scenarios {
        let budget = ScenarioBudget::by_name(scenario);
        for &club in &clubs {
            let pick = ClubPick::Strength(club);
            println!("=== scenario={}  club-rank={} ===", scenario, club);

            // Reference raw baseline (difficulty anchor): its own mean metrics.
            let ref_m = clubbench::score::reference_mean_metrics(&pick, world, &budget, mode, &seeds, days);
            println!(
                "  reference raw: pts={:.1}  balance={:.0}  net_value={:.0}  squad_value={:.0}  squad_size={:.1}",
                ref_m.points, ref_m.balance, ref_m.net_value, ref_m.squad_value, ref_m.squad_size
            );

            println!("  {:<12} {}", "candidate", dims.iter().map(|d| format!("{:>9}", format!("{}(Z)", d))).collect::<Vec<_>>().join(" "));
            for candidate in candidates.iter_mut() {
                let (_, reports) = collect_paired_for_mode(&pick, world, &budget, mode, &seeds, days, candidate.as_mut());
                let zs: Vec<f64> = reports.iter().map(|r: &DimReport| r.z).collect();
                println!(
                    "  {:<12} {}",
                    candidate.name(),
                    zs.iter().map(|z| format!("{:>9.2}", z)).collect::<Vec<_>>().join(" ")
                );
                let e = overall.entry(candidate.name().to_string()).or_insert_with(|| vec![0.0; dims.len()]);
                for (i, z) in zs.iter().enumerate() {
                    e[i] += z;
                }
            }
            cells += 1;
            println!();
        }
    }

    println!("=== overall: mean Z across {} cells ===", cells);
    println!("{:<12} {}", "candidate", dims.iter().map(|d| format!("{:>9}", format!("{}(Z)", d))).collect::<Vec<_>>().join(" "));
    for (name, sums) in &overall {
        let means: Vec<f64> = sums.iter().map(|s| s / cells as f64).collect();
        println!("{:<12} {}", name, means.iter().map(|m| format!("{:>9.2}", m)).collect::<Vec<_>>().join(" "));
    }
}

fn world_size(s: &str) -> clubbench::env::WorldSize {
    match s.trim().to_lowercase().as_str() {
        "compact" => clubbench::env::WorldSize::Compact,
        "standard" => clubbench::env::WorldSize::Standard,
        _ => clubbench::env::WorldSize::Medium,
    }
}

fn score_cmd(seeds_str: &str, days: u64, club: Option<usize>, world: &str, scenario: &str) {
    let seeds: Vec<u64> = seeds_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let pick = club.map(|rank| clubbench::env::ClubPick::Strength(rank));
    let world = world_size(world);
    let budget = clubbench::env::ScenarioBudget::by_name(scenario);
    println!(
        "ClubBench Score — paired-seed, reference-relative ({} seeds, {} days, club={:?}, world={:?}, scenario={})",
        seeds.len(),
        days,
        club,
        world,
        scenario
    );
    println!("reference = ClubBench-Heuristic-v1 (frozen AutoManager, Attacking)\n");

    let mut candidates: Vec<Box<dyn Policy>> = vec![
        Box::new(AutoManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(AutoManager::new(domain::team::PlayStyle::Balanced)),
        Box::new(ProactiveManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(SellingManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(OffersOnlyManager),
        Box::new(PassiveManager),
    ];

    for candidate in candidates.iter_mut() {
        let (_, reports) = match &pick {
            Some(p) => score::collect_paired_for_world(p, world, &budget, &seeds, days, candidate.as_mut()),
            None => score::collect_paired(&seeds, days, candidate.as_mut()),
        };
        println!("=== {} vs reference ===", candidate.name());
        print!("{}", score::render_reports(&reports));
        println!();
    }
}

fn gate0(seeds_str: &str, days: u64) {
    let seeds: Vec<u64> = seeds_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let agents: Vec<Box<dyn Agent>> = vec![
        Box::new(BestXIAgent::new(55)),
        Box::new(StyleProbe { style: domain::team::PlayStyle::Balanced }),
        Box::new(StyleProbe { style: domain::team::PlayStyle::Attacking }),
        Box::new(StyleProbe { style: domain::team::PlayStyle::HighPress }),
        Box::new(StyleProbe { style: domain::team::PlayStyle::Defensive }),
        Box::new(NoopAgent),
        Box::new(RandomXIAgent),
        Box::new(WorstXIAgent),
    ];

    println!("ClubBench Gate 0 — lineup/tactics effectiveness ({} days/episode)", days);
    println!("{:<18} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>8}", "agent", "seed", "pos", "P", "W", "D", "pts", "GF:GA");

    let mut agg: std::collections::BTreeMap<String, (f64, f64, f64)> = std::collections::BTreeMap::new();
    for seed in &seeds {
        for agent in &agents {
            let r = run_episode(*seed, days, agent.as_ref());
            println!(
                "{:<18} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>3}:{:<3}",
                agent.name(), seed, r.metrics.position, r.played, r.won, r.drawn, r.metrics.points, r.goals_for, r.goals_against
            );
            let e = agg.entry(agent.name().to_string()).or_insert((0.0, 0.0, 0.0));
            e.0 += r.metrics.position as f64;
            e.1 += r.metrics.points as f64;
            e.2 += 1.0;
        }
        println!();
    }
    println!("=== averages over {} seeds ===", seeds.len());
    for (name, (pos, pts, n)) in &agg {
        println!("{:<18} avg_pos {:5.2}   avg_pts {:5.2}", name, pos / n, pts / n);
    }
}

fn cadence(seeds_str: &str, days: u64, world: &str, scenario: &str) {
    let seeds: Vec<u64> = seeds_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let world = world_size(world);
    let budget = clubbench::env::ScenarioBudget::by_name(scenario);
    let mut policies: Vec<Box<dyn Policy>> = vec![
        Box::new(AutoManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(OffersOnlyManager),
        Box::new(PassiveManager),
    ];

    println!("ClubBench Cadence — long decision trajectory ({} days/episode, world={:?}, scenario={})", days, world, scenario);
    println!(
        "{:<14} {:>5} {:>6} {:>4} {:>4} {:>4} {:>4} {:>5} {:>8}",
        "policy", "seed", "steps", "pos", "P", "W", "D", "pts", "GF:GA"
    );

    let mut agg: std::collections::BTreeMap<String, (f64, f64, f64, f64)> = std::collections::BTreeMap::new();
    for seed in &seeds {
        for policy in policies.iter_mut() {
            let r = run_episode_cadence_for_world(*seed, &clubbench::env::ClubPick::Index(0), world, &budget, days, policy.as_mut());
            println!(
                "{:<14} {:>5} {:>6} {:>4} {:>4} {:>4} {:>4} {:>5} {:>3}:{:<3}",
                policy.name(), seed, r.steps, r.metrics.position, r.played, r.won, r.drawn, r.metrics.points, r.goals_for, r.goals_against
            );
            let e = agg.entry(policy.name().to_string()).or_insert((0.0, 0.0, 0.0, 0.0));
            e.0 += r.steps as f64;
            e.1 += r.metrics.position as f64;
            e.2 += r.metrics.points as f64;
            e.3 += 1.0;
        }
        println!();
    }
    println!("=== averages over {} seeds ===", seeds.len());
    for (name, (steps, pos, pts, n)) in &agg {
        println!(
            "{:<14} avg_steps {:7.1}   avg_pos {:5.2}   avg_pts {:5.2}",
            name, steps / n, pos / n, pts / n
        );
    }
}
