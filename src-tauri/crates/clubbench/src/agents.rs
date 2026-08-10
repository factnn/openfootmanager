//! Baseline agents for the ClubBench environment.

use domain::player::{Player, Position};
use ofm_core::game::Game;
use ofm_core::player_rating::{effective_rating_for_assignment, formation_slots};

/// An agent that makes decisions for the user's team each matchday.
pub trait Agent {
    fn name(&self) -> &str;
    /// Return a slot-aligned starting XI (player ids), or an empty vec to defer
    /// to the game's AI default (best-fit per formation slot).
    fn decide_lineup(&self, game: &Game) -> Vec<String>;
    /// Optional: adjust the user's tactics (play style / formation) before the
    /// match. Called before [`Agent::decide_lineup`] each matchday.
    fn decide_tactics(&self, _game: &mut Game) {}
}

fn user_team_id(game: &Game) -> String {
    game.manager.team_id.clone().unwrap_or_default()
}

/// Non-injured squad members of a team.
fn available_players<'a>(game: &'a Game, team_id: &str) -> Vec<&'a Player> {
    game.players
        .iter()
        .filter(|p| p.team_id.as_deref() == Some(team_id) && p.injury.is_none())
        .collect()
}

/// Build a slot-aligned XI by picking, for each formation slot, the available
/// player maximising `score`. Slot-aligned means entry `i` of the result plays
/// formation slot `i`, exactly what the live-match engine expects.
fn build_xi<F>(game: &Game, team_id: &str, score: F) -> Vec<String>
where
    F: Fn(&Player, &Position) -> f64,
{
    let formation = game
        .teams
        .iter()
        .find(|t| t.id == team_id)
        .map(|t| t.formation.clone())
        .unwrap_or_else(|| "4-4-2".into());
    let slots = formation_slots(&formation);
    let mut pool = available_players(game, team_id);
    let mut xi = Vec::new();
    for slot in slots.iter().take(11) {
        let best = pool
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                score(a, slot)
                    .partial_cmp(&score(b, slot))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        if let Some(i) = best {
            xi.push(pool.remove(i).id.clone());
        }
    }
    xi
}

/// Greedy "best XI" — maximise effective slot rating, but bench badly fatigued
/// players in favour of any fresher alternative. The strong, stable baseline.
#[derive(Default)]
pub struct BestXIAgent {
    /// Players below this condition (0-100) are penalised in selection.
    pub condition_min: u8,
}

impl BestXIAgent {
    pub fn new(condition_min: u8) -> Self {
        Self { condition_min }
    }
}

impl Agent for BestXIAgent {
    fn name(&self) -> &str {
        "BestXI"
    }
    fn decide_lineup(&self, game: &Game) -> Vec<String> {
        let team_id = user_team_id(game);
        build_xi(game, &team_id, |p, slot| {
            let base = effective_rating_for_assignment(p, slot);
            if p.condition < self.condition_min {
                base - 25.0
            } else {
                base
            }
        })
    }
}

/// Antagonist baseline — pick the *worst* player for every slot.
pub struct WorstXIAgent;
impl Agent for WorstXIAgent {
    fn name(&self) -> &str {
        "WorstXI"
    }
    fn decide_lineup(&self, game: &Game) -> Vec<String> {
        let team_id = user_team_id(game);
        build_xi(game, &team_id, |p, slot| -effective_rating_for_assignment(p, slot))
    }
}

/// Random baseline — a uniformly random valid XI.
pub struct RandomXIAgent;
impl Agent for RandomXIAgent {
    fn name(&self) -> &str {
        "RandomXI"
    }
    fn decide_lineup(&self, game: &Game) -> Vec<String> {
        use rand::seq::SliceRandom;
        let team_id = user_team_id(game);
        let formation = game
            .teams
            .iter()
            .find(|t| t.id == team_id)
            .map(|t| t.formation.clone())
            .unwrap_or_else(|| "4-4-2".into());
        let slots = formation_slots(&formation);
        let mut pool = available_players(game, &team_id);
        pool.shuffle(&mut ofm_core::rng::rng());
        let mut xi = Vec::new();
        for slot in slots.iter().take(11) {
            // Prefer the slot's own group when possible; otherwise any leftover.
            let group = slot.to_group_position();
            let pos = pool
                .iter()
                .position(|p| p.position.to_group_position() == group)
                .or_else(|| pool.iter().position(|_| true));
            if let Some(i) = pos {
                xi.push(pool.remove(i).id.clone());
            }
        }
        xi
    }
}

/// Best XI plus a fixed play style — a probe for whether tactics are a bigger
/// lever than lineup selection.
pub struct StyleProbe {
    pub style: domain::team::PlayStyle,
}
impl Agent for StyleProbe {
    fn name(&self) -> &str {
        match self.style {
            domain::team::PlayStyle::Balanced => "BestXI+Balanced",
            domain::team::PlayStyle::Attacking => "BestXI+Attacking",
            domain::team::PlayStyle::Defensive => "BestXI+Defensive",
            domain::team::PlayStyle::Possession => "BestXI+Possession",
            domain::team::PlayStyle::Counter => "BestXI+Counter",
            domain::team::PlayStyle::HighPress => "BestXI+HighPress",
        }
    }
    fn decide_lineup(&self, game: &Game) -> Vec<String> {
        let team_id = user_team_id(game);
        build_xi(game, &team_id, |p, slot| effective_rating_for_assignment(p, slot))
    }
    fn decide_tactics(&self, game: &mut Game) {
        crate::env::apply_play_style(game, self.style.clone());
    }
}

/// Do-nothing baseline — never touch the lineup (the AI default applies).
pub struct NoopAgent;
impl Agent for NoopAgent {
    fn name(&self) -> &str {
        "Noop"
    }
    fn decide_lineup(&self, _game: &Game) -> Vec<String> {
        Vec::new()
    }
}
