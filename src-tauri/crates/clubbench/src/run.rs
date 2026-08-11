//! Episode runner: drive a season with an agent and collect the result.

use crate::agents::Agent;
use crate::env;
use crate::episode::Episode;
use crate::episode_agents::Policy;
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

/// Result of a decision-cadence episode.
#[derive(Debug, Clone)]
pub struct CadenceResult {
    pub seed: u64,
    pub steps: u64,
    pub position: usize,
    pub played: u32,
    pub won: u32,
    pub drawn: u32,
    pub lost: u32,
    pub points: u32,
    pub goals_for: u32,
    pub goals_against: u32,
}

/// Run the decision-cadence episode with a policy. The agent is consulted at
/// every decision point (matchdays + transfer offers); `steps` is the number of
/// decisions made over the horizon.
pub fn run_episode_cadence(
    seed: u64,
    horizon_days: u64,
    policy: &mut dyn Policy,
) -> CadenceResult {
    let mut ep = Episode::new(seed, horizon_days);
    let mut obs = ep.observe();
    let mut guard = 0u64;
    while !obs.done && guard < 200_000 {
        let action = policy.act(&obs);
        obs = ep.step(action);
        guard += 1;
    }

    let r = result_of(&ep.game, seed);
    CadenceResult {
        seed,
        steps: ep.step_count(),
        position: r.position,
        played: r.played,
        won: r.won,
        drawn: r.drawn,
        lost: r.lost,
        points: r.points,
        goals_for: r.goals_for,
        goals_against: r.goals_against,
    }
}
