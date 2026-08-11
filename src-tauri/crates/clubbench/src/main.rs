//! ClubBench CLI.
//!
//! `gate0`    — the original lineup/tactics experiment (single-action baselines).
//! `cadence`  — the decision-cadence experiment: the agent is consulted at every
//!              matchday and transfer offer, producing a long decision trajectory.

use clubbench::agents::{Agent, BestXIAgent, NoopAgent, RandomXIAgent, StyleProbe, WorstXIAgent};
use clubbench::episode_agents::{AutoManager, OffersOnlyManager, PassiveManager, Policy};
use clubbench::run::{run_episode, run_episode_cadence};
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
    },
    /// Paired-seed, reference-relative scoring (the benchmark evaluation protocol)
    Score {
        #[arg(long, default_value = "42,43,44,45,46,47,48,49")]
        seeds: String,
        #[arg(long, default_value_t = 400)]
        days: u64,
        /// Managed club by squad-strength rank (0 = weakest, 15 = strongest)
        #[arg(long)]
        club: Option<usize>,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Gate0 { seeds, days } => gate0(&seeds, days),
        Commands::Cadence { seeds, days } => cadence(&seeds, days),
        Commands::Score { seeds, days, club } => score_cmd(&seeds, days, club),
    }
}

fn score_cmd(seeds_str: &str, days: u64, club: Option<usize>) {
    let seeds: Vec<u64> = seeds_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let pick = club.map(|rank| clubbench::env::ClubPick::Strength(rank));
    println!(
        "ClubBench Score — paired-seed, reference-relative ({} seeds, {} days, club={:?})",
        seeds.len(),
        days,
        club
    );
    println!("reference = ClubBench-Heuristic-v1 (frozen AutoManager, Attacking)\n");

    let mut candidates: Vec<Box<dyn Policy>> = vec![
        Box::new(AutoManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(AutoManager::new(domain::team::PlayStyle::Balanced)),
        Box::new(OffersOnlyManager),
        Box::new(PassiveManager),
    ];

    for candidate in candidates.iter_mut() {
        let (_, reports) = match &pick {
            Some(p) => score::collect_paired_for(p, &seeds, days, candidate.as_mut()),
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

fn cadence(seeds_str: &str, days: u64) {
    let seeds: Vec<u64> = seeds_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let mut policies: Vec<Box<dyn Policy>> = vec![
        Box::new(AutoManager::new(domain::team::PlayStyle::Attacking)),
        Box::new(OffersOnlyManager),
        Box::new(PassiveManager),
    ];

    println!("ClubBench Cadence — long decision trajectory ({} days/episode)", days);
    println!(
        "{:<14} {:>5} {:>6} {:>4} {:>4} {:>4} {:>4} {:>5} {:>8}",
        "policy", "seed", "steps", "pos", "P", "W", "D", "pts", "GF:GA"
    );

    let mut agg: std::collections::BTreeMap<String, (f64, f64, f64, f64)> = std::collections::BTreeMap::new();
    for seed in &seeds {
        for policy in policies.iter_mut() {
            let r = run_episode_cadence(*seed, days, policy.as_mut());
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
