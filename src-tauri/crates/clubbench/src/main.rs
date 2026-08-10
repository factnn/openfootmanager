//! ClubBench Gate 0: does the environment reward lineup decisions?
//!
//! Runs the same season from several seeds under four baselines — greedy best
//! XI, the AI default (noop), a random XI and a worst XI — and prints the
//! user's final league position / points. If lineup matters, we expect
//! `BestXI ≈ Noop ≥ Random ≫ Worst`.

use clubbench::agents::{Agent, BestXIAgent, NoopAgent, RandomXIAgent, StyleProbe, WorstXIAgent};
use clubbench::run::run_episode;
use clap::Parser;
use std::collections::BTreeMap;

#[derive(Parser)]
#[command(name = "clubbench", about = "ClubBench Gate 0: lineup-effectiveness experiment")]
struct Cli {
    /// Comma-separated episode seeds
    #[arg(long, default_value = "42,43,44")]
    seeds: String,
    /// Game days per episode (default ≈ one full season)
    #[arg(long, default_value_t = 400)]
    days: u64,
}

fn main() {
    let cli = Cli::parse();
    let seeds: Vec<u64> = cli
        .seeds
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

    println!("ClubBench Gate 0 — lineup effectiveness ({} days/episode)", cli.days);
    println!("{:<10} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>8}", "agent", "seed", "pos", "P", "W", "D", "pts", "GF:GA");

    // agent name -> (sum_position, sum_points, n)
    let mut agg: BTreeMap<String, (f64, f64, f64)> = BTreeMap::new();

    for seed in &seeds {
        for agent in &agents {
            let r = run_episode(*seed, cli.days, agent.as_ref());
            println!(
                "{:<10} {:>5} {:>4} {:>4} {:>4} {:>4} {:>5} {:>3}:{:<3}",
                agent.name(),
                seed,
                r.position,
                r.played,
                r.won,
                r.drawn,
                r.points,
                r.goals_for,
                r.goals_against
            );
            let e = agg.entry(agent.name().to_string()).or_insert((0.0, 0.0, 0.0));
            e.0 += r.position as f64;
            e.1 += r.points as f64;
            e.2 += 1.0;
        }
        println!();
    }

    println!("=== averages over {} seeds ===", seeds.len());
    for (name, (pos, pts, n)) in &agg {
        println!("{:<10} avg_pos {:5.2}   avg_pts {:5.2}", name, pos / n, pts / n);
    }
}
