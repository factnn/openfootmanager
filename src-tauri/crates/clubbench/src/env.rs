//! The environment: build a game, observe it, act on it, advance time.

use chrono::{DateTime, TimeZone, Utc};
use domain::league::{CompetitionFormat, CompetitionScope, CompetitionType, FixtureStatus, League};
use domain::manager::Manager;
use domain::player::Position;
use ofm_core::clock::GameClock;
use ofm_core::game::Game;
use ofm_core::generator::{
    build_explicit_competition, generate_world_data_seeded_with, repair_opening_youth_academies,
    CompetitionDefinition, FormatDef, ParticipantSpec, WorldGenConfig,
};
use ofm_core::turn;
use serde::Serialize;
use std::collections::BTreeMap;

/// A player as the agent sees it (the visible information set — no hidden
/// potential, no memory access).
#[derive(Serialize, Clone, Debug)]
pub struct PlayerView {
    pub id: String,
    pub name: String,
    pub position: Position,
    pub group_position: Position,
    pub ovr: u8,
    pub age: u8,
    pub condition: u8,
    pub fitness: u8,
    pub morale: u8,
    pub injured: bool,
    pub transfer_listed: bool,
    pub wage: u32,
    pub market_value: u64,
}

/// The observation delivered to an agent on a decision step.
#[derive(Serialize, Clone, Debug)]
pub struct Observation {
    pub date: String,
    pub team_name: String,
    pub formation: String,
    pub league_position: usize,
    pub points: u32,
    pub next_fixture: Option<String>,
    pub squad: Vec<PlayerView>,
}

/// How the managed club is chosen from the generated world.
#[derive(Debug, Clone)]
pub enum ClubPick {
    /// `world.teams[idx]` (world-order, deterministic per seed).
    Index(usize),
    /// The club whose average squad strength ranks `rank`-th when all clubs
    /// are sorted by mean player ovr (0 = weakest, len-1 = strongest).
    Strength(usize),
}

/// Build the world's club competitions: one league per nation, split into
/// divisions of up to 20 clubs ordered by squad strength (an approximation of
/// the app's foundation-competition plan; continental cups come later).
fn build_foundation_competitions(
    world: &ofm_core::generator::WorldData,
    start: DateTime<Utc>,
) -> Vec<League> {
    let mut team_ovr: BTreeMap<String, f64> = BTreeMap::new();
    let mut sums: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for p in &world.players {
        if let Some(tid) = &p.team_id {
            let e = sums.entry(tid.clone()).or_insert((0.0, 0));
            e.0 += p.ovr as f64;
            e.1 += 1;
        }
    }
    for (k, (s, n)) in sums {
        team_ovr.insert(k, if n == 0 { 0.0 } else { s / n as f64 });
    }

    let mut by_nation: BTreeMap<String, Vec<&domain::team::Team>> = BTreeMap::new();
    for team in &world.teams {
        by_nation
            .entry(team.football_nation.clone())
            .or_default()
            .push(team);
    }

    let mut competitions = Vec::new();
    for (nation, mut teams) in by_nation {
        teams.sort_by(|a, b| {
            team_ovr
                .get(&b.id)
                .unwrap_or(&0.0)
                .partial_cmp(team_ovr.get(&a.id).unwrap_or(&0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (div, chunk) in teams.chunks(20).enumerate() {
            let def = CompetitionDefinition {
                id: format!("{nation}-div{}", div + 1),
                name: format!("{nation} Division {}", div + 1),
                r#type: CompetitionType::League,
                scope: CompetitionScope::Domestic,
                region_id: None,
                country_id: Some(nation.clone()),
                required_region_ids: vec![],
                priority: 0,
                format: FormatDef {
                    kind: CompetitionFormat::LeagueTable,
                    legs: Some(2),
                    group_size: None,
                    qualifiers_per_group: None,
                    best_third_qualifiers: None,
                },
                participants: ParticipantSpec {
                    explicit: Some(chunk.iter().map(|t| t.id.clone()).collect()),
                    selector: None,
                },
                berths: vec![],
                season_start_month: None,
                season_start_day: None,
                name_key: None,
                logo: None,
            };
            if let Some(league) = build_explicit_competition(&def, 2026, start) {
                competitions.push(league);
            }
        }
    }
    competitions
}

/// World size for the benchmark env. `Standard` (~440 clubs) is the most
/// realistic but ≈43 s per season; `Medium` (~120 clubs) is a practical
/// balance; `Compact` is the fast toy world (fake economics).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorldSize {
    Compact,
    Medium,
    Standard,
}

/// Which track the episode exercises. `Coach` restricts the agent to matchday
/// decisions (lineup/tactics) — transfers/scouting are frozen, so the env only
/// stops at matchdays and hides the market. `Manager` unlocks the full market.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgentMode {
    Coach,
    Manager,
}

/// The managed club's financial starting state — a scenario lever. The
/// generated worlds' budgets scale with reputation and are unrealistically
/// small, so each scenario defines the club's finances explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScenarioBudget {
    pub finance: i64,
    pub transfer_budget: i64,
    pub wage_budget: i64,
}

impl Default for ScenarioBudget {
    /// The big-budget default (keeps the transfer market usable).
    fn default() -> Self {
        Self::rebuild()
    }
}

impl ScenarioBudget {
    /// Relegation fight — a tiny budget to work with.
    pub fn crisis() -> Self {
        Self { finance: 20_000_000, transfer_budget: 5_000_000, wage_budget: 2_000_000 }
    }
    /// Top-half on a tight budget — spend smart or not at all.
    pub fn moneyball() -> Self {
        Self { finance: 40_000_000, transfer_budget: 15_000_000, wage_budget: 2_500_000 }
    }
    /// Big-budget rebuild — room to buy, pressure to deliver.
    pub fn rebuild() -> Self {
        Self { finance: 60_000_000, transfer_budget: 50_000_000, wage_budget: 3_000_000 }
    }
    /// Title contender — top-tier finances.
    pub fn title() -> Self {
        Self { finance: 100_000_000, transfer_budget: 60_000_000, wage_budget: 3_500_000 }
    }

    pub fn by_name(name: &str) -> ScenarioBudget {
        match name.trim().to_lowercase().as_str() {
            "crisis" => Self::crisis(),
            "moneyball" => Self::moneyball(),
            "title" => Self::title(),
            _ => Self::rebuild(),
        }
    }
}

fn world_config(size: WorldSize) -> WorldGenConfig {
    match size {
        WorldSize::Compact => WorldGenConfig::compact(),
        WorldSize::Medium => WorldGenConfig {
            clubs_per_division: 20,
            nations: ofm_core::generator::clubs::STANDARD_NATIONS[..3].to_vec(),
        },
        WorldSize::Standard => WorldGenConfig::standard(),
    }
}

/// Build a fully playable game with proper multi-competition structure,
/// managing the club chosen by `pick` in the given world size. The user's club
/// plays in its own nation's division; other divisions simulate in the
/// background, keeping the world economy and transfer market alive.
pub fn build_game_for_club_with(
    seed: u64,
    pick: &ClubPick,
    world: WorldSize,
    budget: &ScenarioBudget,
) -> Game {
    let world = generate_world_data_seeded_with(seed, &world_config(world), None);
    let start = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap();
    let clock = GameClock::new(start);

    let team_ovr: std::collections::HashMap<String, f64> = {
        let mut sums: std::collections::HashMap<String, (f64, usize)> = Default::default();
        for p in &world.players {
            if let Some(tid) = &p.team_id {
                let e = sums.entry(tid.clone()).or_insert((0.0, 0));
                e.0 += p.ovr as f64;
                e.1 += 1;
            }
        }
        sums.into_iter()
            .map(|(k, (s, n))| (k, if n == 0 { 0.0 } else { s / n as f64 }))
            .collect()
    };
    let mut ranked: Vec<&domain::team::Team> = world.teams.iter().collect();
    ranked.sort_by(|a, b| {
        team_ovr
            .get(&a.id)
            .unwrap_or(&0.0)
            .partial_cmp(team_ovr.get(&b.id).unwrap_or(&0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let team_id = match pick {
        ClubPick::Index(i) => world
            .teams
            .get(*i)
            .map(|t| t.id.clone())
            .unwrap_or_else(|| world.teams[0].id.clone()),
        ClubPick::Strength(rank) => ranked
            .get(*rank)
            .map(|t| t.id.clone())
            .unwrap_or_else(|| ranked[0].id.clone()),
    };
    let team = world.teams.iter().find(|t| t.id == team_id).expect("team exists");
    let mut manager = Manager::new(
        "headless-mgr".to_string(),
        "Headless".to_string(),
        "Manager".to_string(),
        "1980-01-01".to_string(),
        "England".to_string(),
    );
    manager.hire(team.id.clone());

    let competitions = build_foundation_competitions(&world, start);
    let mut game = Game::new(clock, manager, world.teams, world.players, world.staff, vec![]);
    game.available_staff_market_last_activity_date = Some(start.format("%Y-%m-%d").to_string());
    repair_opening_youth_academies(&mut game);
    game.competitions = competitions;
    game.active_competition_ids = game.competitions.iter().map(|c| c.id.clone()).collect();
    game.sync_legacy_league();
    ofm_core::season_context::refresh_game_context(&mut game);
    // The scenario's financial starting state overrides the generated budgets.
    if let Some(tid) = game.manager.team_id.clone() {
        if let Some(team) = game.teams.iter_mut().find(|t| t.id == tid) {
            team.finance = budget.finance;
            team.transfer_budget = budget.transfer_budget;
            team.wage_budget = budget.wage_budget;
        }
    }
    game
}

/// Build a fully playable game in the Medium world (default) with the default
/// scenario budget, managing the club chosen by `pick`.
pub fn build_game_for_club(seed: u64, pick: &ClubPick) -> Game {
    build_game_for_club_with(seed, pick, WorldSize::Medium, &ScenarioBudget::default())
}

/// Compact single-league world, managing the first team — used by the fast
/// Coach-track experiment (`gate0`). Not used by the decision-cadence env.
pub fn build_game(seed: u64) -> Game {
    let world = generate_world_data_seeded_with(seed, &WorldGenConfig::compact(), None);
    let start = Utc.with_ymd_and_hms(2026, 7, 1, 0, 0, 0).unwrap();
    let clock = GameClock::new(start);
    let team = world.teams.first().expect("world has teams");
    let mut manager = Manager::new(
        "headless-mgr".to_string(),
        "Headless".to_string(),
        "Manager".to_string(),
        "1980-01-01".to_string(),
        "England".to_string(),
    );
    manager.hire(team.id.clone());
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

/// The club's total net worth: squad market value + bank balance.
pub fn net_worth(game: &Game) -> i64 {
    let team_id = game.manager.team_id.as_deref().unwrap_or_default();
    let balance = game
        .teams
        .iter()
        .find(|t| t.id == team_id)
        .map(|t| t.finance)
        .unwrap_or(0);
    let squad_value: i64 = game
        .players
        .iter()
        .filter(|p| p.team_id.as_deref() == Some(team_id))
        .map(|p| p.market_value as i64)
        .sum();
    balance + squad_value
}

/// The agent's information set at the current date.
pub fn observe(game: &Game) -> Observation {
    let user_team_id = game.manager.team_id.as_deref().unwrap_or_default();
    let team = game.teams.iter().find(|t| t.id == user_team_id);
    let today = game.clock.current_date.format("%Y-%m-%d").to_string();

    let squad = game
        .players
        .iter()
        .filter(|p| p.team_id.as_deref() == Some(user_team_id))
        .map(|p| PlayerView {
            id: p.id.clone(),
            name: p.match_name.clone(),
            position: p.position.clone(),
            group_position: p.position.to_group_position(),
            ovr: p.ovr,
            age: age_from_dob(&p.date_of_birth, &today),
            condition: p.condition,
            fitness: p.fitness,
            morale: p.morale,
            injured: p.injury.is_some(),
            transfer_listed: p.transfer_listed,
            wage: p.wage,
            market_value: p.market_value,
        })
        .collect();

    let next_fixture = game
        .league
        .as_ref()
        .and_then(|league| {
            league.fixtures.iter().find(|f| {
                f.status == FixtureStatus::Scheduled
                    && (f.home_team_id == user_team_id || f.away_team_id == user_team_id)
            })
        })
        .map(|f| {
            let home = game
                .teams
                .iter()
                .find(|t| t.id == f.home_team_id)
                .map(|t| t.name.clone())
                .unwrap_or_default();
            let away = game
                .teams
                .iter()
                .find(|t| t.id == f.away_team_id)
                .map(|t| t.name.clone())
                .unwrap_or_default();
            format!("{} {} vs {} {}", f.date, home, away, if f.home_team_id == user_team_id { "(H)" } else { "(A)" })
        });

    let (league_position, points) = game
        .league
        .as_ref()
        .map(|league| {
            let mut st = league.standings.clone();
            st.sort_by(|a, b| b.points.cmp(&a.points).then_with(|| b.goal_difference().cmp(&a.goal_difference())));
            let pos = st
                .iter()
                .position(|s| s.team_id == user_team_id)
                .map(|i| i + 1)
                .unwrap_or(0);
            let pts = st
                .iter()
                .find(|s| s.team_id == user_team_id)
                .map(|s| s.points)
                .unwrap_or(0);
            (pos, pts)
        })
        .unwrap_or((0, 0));

    Observation {
        date: today,
        team_name: team.map(|t| t.name.clone()).unwrap_or_default(),
        formation: team.map(|t| t.formation.clone()).unwrap_or_else(|| "4-4-2".into()),
        league_position,
        points,
        next_fixture,
        squad,
    }
}

fn age_from_dob(dob: &str, today: &str) -> u8 {
    let Ok(b) = chrono::NaiveDate::parse_from_str(dob, "%Y-%m-%d") else {
        return 0;
    };
    let Ok(t) = chrono::NaiveDate::parse_from_str(today, "%Y-%m-%d") else {
        return 0;
    };
    ((t - b).num_days() / 365) as u8
}

/// Set the user's starting XI (slot-aligned). Empty vec → leave to the AI
/// default (best-fit per formation slot).
pub fn apply_lineup(game: &mut Game, lineup: &[String]) {
    if let Some(team_id) = game.manager.team_id.as_ref() {
        if let Some(team) = game.teams.iter_mut().find(|t| &t.id == team_id) {
            team.starting_xi_ids = lineup.to_vec();
        }
    }
}

/// Set the user's play style (e.g. Attacking / HighPress).
pub fn apply_play_style(game: &mut Game, style: domain::team::PlayStyle) {
    if let Some(team_id) = game.manager.team_id.as_ref() {
        if let Some(team) = game.teams.iter_mut().find(|t| &t.id == team_id) {
            team.play_style = style;
        }
    }
}

/// Index of the user's scheduled fixture today, if any.
pub fn user_fixture_index(game: &Game) -> Option<usize> {
    let user_team_id = game.manager.team_id.as_ref()?;
    let today = game.clock.current_date.format("%Y-%m-%d").to_string();
    game.league.as_ref()?.fixtures.iter().enumerate().find_map(|(i, f)| {
        if f.date == today
            && f.status == FixtureStatus::Scheduled
            && (f.home_team_id == *user_team_id || f.away_team_id == *user_team_id)
        {
            Some(i)
        } else {
            None
        }
    })
}

/// Advance one game day through the standard turn loop. The simple match
/// engine is XI-aware (see `ofm_core::turn::select_starting_xi_ids`), so the
/// user's lineup decisions are respected.
pub fn advance_day(game: &mut Game) {
    turn::process_day(game);
}
