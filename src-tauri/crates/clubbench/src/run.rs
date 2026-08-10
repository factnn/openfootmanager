//! Episode runner: drive a season with an agent and collect the result.

use crate::agents::Agent;
use crate::env;
use ofm_core::game::Game;

#[derive(Debug, Clone)]
pub struct EpisodeResult {
    pub seed: u64,
    pub position: usize,
    pub played: u32,
    pub won: u32,
    pub drawn: u32,
    pub lost: u32,
    pub points: u32,
    pub goals_for: u32,
    pub goals_against: u32,
}

/// Run one deterministic episode: `seed` fixes both the world and the entire
/// season trajectory. The agent is consulted on every user matchday.
pub fn run_episode(seed: u64, days: u64, agent: &dyn Agent) -> EpisodeResult {
    ofm_core::rng::set_seed(seed);
    let mut game = env::build_game(seed);

    for _ in 0..days {
        if env::user_fixture_index(&game).is_some() {
            agent.decide_tactics(&mut game);
            let lineup = agent.decide_lineup(&game);
            env::apply_lineup(&mut game, &lineup);
        }
        env::advance_day(&mut game);
    }

    ofm_core::rng::reset_random();
    result_of(&game, seed)
}

fn result_of(game: &Game, seed: u64) -> EpisodeResult {
    let user_team_id = game.manager.team_id.as_deref().unwrap_or_default();
    let mut st = game
        .league
        .as_ref()
        .map(|l| l.standings.clone())
        .unwrap_or_default();
    st.sort_by(|a, b| {
        b.points
            .cmp(&a.points)
            .then_with(|| b.goal_difference().cmp(&a.goal_difference()))
    });
    let position = st
        .iter()
        .position(|s| s.team_id == user_team_id)
        .map(|i| i + 1)
        .unwrap_or(0);
    let e = st.iter().find(|s| s.team_id == user_team_id);
    EpisodeResult {
        seed,
        position,
        played: e.map(|e| e.played).unwrap_or(0),
        won: e.map(|e| e.won).unwrap_or(0),
        drawn: e.map(|e| e.drawn).unwrap_or(0),
        lost: e.map(|e| e.lost).unwrap_or(0),
        points: e.map(|e| e.points).unwrap_or(0),
        goals_for: e.map(|e| e.goals_for).unwrap_or(0),
        goals_against: e.map(|e| e.goals_against).unwrap_or(0),
    }
}
