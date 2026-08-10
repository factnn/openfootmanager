//! Headless season runner — the first step of the ClubBench environment probe.
//!
//! What it proves:
//!   1. a full season can be driven headlessly (no Tauri/GUI) via
//!      `ofm_core::turn::process_day`;
//!   2. with `ofm_core::rng::set_seed`, the same seed reproduces the same
//!      initial world *and* the same season trajectory (content-level; entity
//!      UUIDs still differ).
//!
//! Usage:
//!   ofm-headless --seed 42 --days 400
//!   ofm-headless --seed 42 --days 400 --check-determinism

use chrono::{TimeZone, Utc};
use clap::Parser;
use domain::manager::Manager;
use ofm_core::clock::GameClock;
use ofm_core::game::Game;
use ofm_core::generator::{
    generate_world_data_seeded_with, repair_opening_youth_academies, WorldGenConfig,
};
use ofm_core::turn;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ofm-headless", about = "Headless season runner / ClubBench environment probe")]
struct Cli {
    /// Episode seed — same seed ⇒ same initial world + same trajectory
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of game days to advance (default ≈ one full season: Jul → next summer)
    #[arg(long, default_value_t = 400)]
    days: u64,
    /// Build two games from the same seed and compare initial state + trajectory
    #[arg(long)]
    check_determinism: bool,
}

/// Build a fully playable game from a seeded compact world (see
/// `ofm_core/tests/scenario_tests.rs::make_scenario_game` for the canonical pattern).
fn build_game(seed: u64) -> Game {
    let world = generate_world_data_seeded_with(seed, &WorldGenConfig::compact(), None);
    let start = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap();
    let clock = GameClock::new(start);

    let first_team = world.teams.first().expect("generated world has at least one team");
    let mut manager = Manager::new(
        "headless-mgr".to_string(),
        "Headless".to_string(),
        "Manager".to_string(),
        "1980-01-01".to_string(),
        "England".to_string(),
    );
    manager.hire(first_team.id.clone());

    let team_ids: Vec<String> = world.teams.iter().map(|t| t.id.clone()).collect();

    let mut game = Game::new(clock, manager, world.teams, world.players, world.staff, vec![]);
    game.available_staff_market_last_activity_date = Some(start.format("%Y-%m-%d").to_string());
    repair_opening_youth_academies(&mut game);
    game.league = Some(ofm_core::schedule::generate_league(
        "Scenario League",
        2026,
        &team_ids,
        start,
    ));
    ofm_core::season_context::refresh_game_context(&mut game);
    game
}

/// Content-level fingerprint that *ignores* entity IDs — sorted multiset of
/// (position, ovr) per player plus team names. Tells us whether two worlds are
/// the same in substance even when randomly assigned UUIDs differ.
fn content_fingerprint(game: &Game) -> String {
    let mut players: Vec<String> = game
        .players
        .iter()
        .map(|p| format!("{:?}:{}", p.position, p.ovr))
        .collect();
    players.sort();
    let mut teams: Vec<String> = game.teams.iter().map(|t| t.name.clone()).collect();
    teams.sort();
    format!(
        "teams={} players={}\n{}\n{}",
        game.teams.len(),
        game.players.len(),
        teams.join("\n"),
        players.join("\n")
    )
}

fn print_standings(game: &Game, label: &str) {
    println!("== {label} == date {}", game.clock.current_date.format("%Y-%m-%d"));
    let Some(league) = &game.league else {
        println!("  (no league)");
        return;
    };
    let mut st = league.standings.clone();
    st.sort_by(|a, b| b.points.cmp(&a.points).then_with(|| b.goal_difference().cmp(&a.goal_difference())));
    println!("  {:24} P  W  D  L  GF GA Pts", "Team");
    for e in st.iter().take(6) {
        let name = game.teams.iter().find(|t| t.id == e.team_id).map(|t| t.name.as_str()).unwrap_or("?");
        println!("  {:24} {:2} {:2} {:2} {:2} {:2} {:2} {:3}",
            name, e.played, e.won, e.drawn, e.lost, e.goals_for, e.goals_against, e.points);
    }
}

fn advance(game: &mut Game, days: u64) -> std::time::Duration {
    let start = Instant::now();
    for day in 0..days {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            turn::process_day(game);
        }));
        if result.is_err() {
            eprintln!("process_day panicked on day {day}");
            std::process::exit(2);
        }
    }
    start.elapsed()
}

fn main() {
    let cli = Cli::parse();
    println!("== ofm-headless: seed={} days={} ==", cli.seed, cli.days);

    // A single seeded stream drives build + season for run A.
    ofm_core::rng::set_seed(cli.seed);
    let mut game_a = build_game(cli.seed);
    let initial_a = content_fingerprint(&game_a);
    println!(
        "initial state: teams={}, players={}, staff={}",
        game_a.teams.len(),
        game_a.players.len(),
        game_a.staff.len()
    );

    let elapsed = advance(&mut game_a, cli.days);
    print_standings(&game_a, "run A");
    println!(
        "advance {} days in {:.3}s ({:.3}ms/day)",
        cli.days,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / cli.days as f64
    );

    if cli.check_determinism {
        // Fresh identical stream for run B → should reproduce build + trajectory.
        ofm_core::rng::set_seed(cli.seed);
        let mut game_b = build_game(cli.seed);
        println!("\n-- determinism check --");
        println!(
            "initial identical (content, ignoring ids): {}",
            initial_a == content_fingerprint(&game_b)
        );

        advance(&mut game_b, cli.days);
        print_standings(&game_b, "run B (same seed)");
        let same_content = content_fingerprint(&game_a) == content_fingerprint(&game_b);
        println!("trajectory identical (content): {}", same_content);
        println!(
            "=> findings: {}. ClubBench can reset an episode with `rng::set_seed(scenario_seed)`.",
            if same_content { "initial world AND season trajectory are seed-reproducible" }
            else { "initial world is seed-reproducible, but the season trajectory diverges" }
        );
    }

    ofm_core::rng::reset_random();
}
