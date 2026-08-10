//! Headless season runner — the first step of the ClubBench environment probe.
//!
//! What it proves:
//!   1. a full season can be driven headlessly (no Tauri/GUI) via
//!      `ofm_core::turn::process_day`;
//!   2. the same seed produces the same *initial* world (world-gen determinism);
//!   3. whether the *trajectory* is reproducible (it is expected NOT to be,
//!      because the match engine / turn subsystems draw from ambient RNG —
//!      see the note in `ofm_core/tests/scenario_tests.rs`).
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
    /// World seed — same seed ⇒ same initial world
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

/// Fingerprint of a raw generated world (before any game build steps), used to
/// isolate whether non-determinism lives in *world generation* itself or in the
/// *game build* glue (`repair_opening_youth_academies` etc.).
fn world_content_fingerprint(world: &ofm_core::generator::WorldData) -> String {
    let mut players: Vec<String> = world
        .players
        .iter()
        .map(|p| format!("{:?}:{}", p.position, p.ovr))
        .collect();
    players.sort();
    let mut teams: Vec<String> = world.teams.iter().map(|t| t.name.clone()).collect();
    teams.sort();
    format!(
        "teams={} players={} staff={}\n{}\n{}",
        world.teams.len(),
        world.players.len(),
        world.staff.len(),
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

    let mut game_a = build_game(cli.seed);
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
        let mut game_b = build_game(cli.seed);
        println!("\n-- determinism check --");

        // (a) Is the *raw generated world* reproducible from the seed?
        let w1 = generate_world_data_seeded_with(cli.seed, &WorldGenConfig::compact(), None);
        let w2 = generate_world_data_seeded_with(cli.seed, &WorldGenConfig::compact(), None);
        println!(
            "raw world reproducible (same seed): {}",
            world_content_fingerprint(&w1) == world_content_fingerprint(&w2)
        );

        // (b) Is the *built game* reproducible (world + game-build glue)?
        println!(
            "built game identical (content, ignoring ids): {}",
            content_fingerprint(&game_a) == content_fingerprint(&game_b)
        );

        advance(&mut game_b, cli.days);
        print_standings(&game_b, "run B (same seed)");
        let same_content = content_fingerprint(&game_a) == content_fingerprint(&game_b);
        println!("trajectory identical (content): {}", same_content);
        println!(
            "=> findings: {}. ClubBench needs a deterministic RNG strategy (game-owned \
             seeded RNG threaded through the engine/turn loop) for exact per-episode reset.",
            if same_content { "world content is seed-reproducible AND season trajectory is reproducible" }
            else { "initial world content is seed-reproducible, but the season trajectory diverges (ambient RNG)" }
        );
    }
}
