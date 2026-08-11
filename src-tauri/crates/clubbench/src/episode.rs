//! The decision-cadence environment.
//!
//! Unlike the raw season runner, this is a *trajectory*: the environment stops
//! at every decision point — a user matchday or a pending transfer offer — and
//! each stop is exactly one agent step. A full season therefore produces
//! hundreds of steps, and every decision's consequences compound over time
//! (injuries, morale, finances, squad building) — the long-horizon property
//! that ClubBench exists to measure.
//!
//! The whole episode is reproducible: `Episode::new(seed, horizon)` seeds the
//! world AND the entire trajectory.

use crate::env;
use domain::league::FixtureStatus;
use domain::player::{Position, TransferOfferStatus};
use ofm_core::game::Game;
use ofm_core::transfers;
use serde::Serialize;

/// One decision the agent can make at a decision point.
#[derive(Debug, Clone)]
pub enum Action {
    /// Act on nothing and advance to the next decision point (the AI default
    /// applies to anything left unhandled).
    Continue,
    /// Set the starting XI for the upcoming matchday (slot-aligned).
    SetLineup { player_ids: Vec<String> },
    /// Set the play style for the upcoming matchday.
    SetTactics { play_style: domain::team::PlayStyle },
    /// Accept an incoming transfer offer for one of our players.
    AcceptOffer { player_id: String, offer_id: String },
    /// Reject an incoming transfer offer.
    RejectOffer { player_id: String, offer_id: String },
    /// Counter an incoming offer with our own requested fee.
    CounterOffer { player_id: String, offer_id: String, fee: u64 },
    /// Bid for a transfer-listed player.
    MakeBid { player_id: String, fee: u64 },
}

/// An incoming transfer offer for one of our players (status `Pending`).
#[derive(Serialize, Clone, Debug)]
pub struct OfferView {
    pub offer_id: String,
    pub player_id: String,
    pub player_name: String,
    pub from_team: String,
    pub fee: u64,
    pub round: u8,
    pub suggested_counter: Option<u64>,
}

/// A transfer-listed player we could bid on.
#[derive(Serialize, Clone, Debug)]
pub struct MarketView {
    pub player_id: String,
    pub player_name: String,
    pub position: Position,
    pub ovr: u8,
    pub age: u8,
    pub market_value: u64,
    pub team: String,
}

/// The full decision-point observation.
#[derive(Serialize, Clone, Debug)]
pub struct EpisodeObservation {
    pub step: u64,
    pub date: String,
    pub team_name: String,
    pub formation: String,
    pub league_position: usize,
    pub points: u32,
    pub budget: i64,
    pub is_matchday: bool,
    pub next_fixture: Option<String>,
    pub squad: Vec<env::PlayerView>,
    pub offers: Vec<OfferView>,
    pub market: Vec<MarketView>,
    pub done: bool,
}

/// The decision-cadence episode.
pub struct Episode {
    pub game: Game,
    pub step: u64,
    horizon_days: u64,
    advanced_days: u64,
}

impl Episode {
    /// Start a reproducible episode: `seed` fixes the world and trajectory;
    /// `horizon_days` bounds the episode length.
    pub fn new(seed: u64, horizon_days: u64) -> Self {
        ofm_core::rng::set_seed(seed);
        let game = env::build_game(seed);
        Self {
            game,
            step: 0,
            horizon_days,
            advanced_days: 0,
        }
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn observe(&self) -> EpisodeObservation {
        let user_team_id = self.game.manager.team_id.as_deref().unwrap_or_default();
        let team = self.game.teams.iter().find(|t| t.id == user_team_id);
        let today = self.game.clock.current_date.format("%Y-%m-%d").to_string();

        let is_matchday = self.user_fixture_index().is_some();
        let next_fixture = self
            .game
            .league
            .as_ref()
            .and_then(|league| {
                league
                    .fixtures
                    .iter()
                    .find(|f| {
                        f.status == FixtureStatus::Scheduled
                            && (f.home_team_id == user_team_id || f.away_team_id == user_team_id)
                    })
            })
            .map(|f| {
                let home = self
                    .game
                    .teams
                    .iter()
                    .find(|t| t.id == f.home_team_id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                let away = self
                    .game
                    .teams
                    .iter()
                    .find(|t| t.id == f.away_team_id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                format!(
                    "{} {} vs {} {}",
                    f.date,
                    home,
                    away,
                    if f.home_team_id == user_team_id { "(H)" } else { "(A)" }
                )
            });

        let (league_position, points) = self
            .game
            .league
            .as_ref()
            .map(|league| {
                let mut st = league.standings.clone();
                st.sort_by(|a, b| {
                    b.points
                        .cmp(&a.points)
                        .then_with(|| b.goal_difference().cmp(&a.goal_difference()))
                });
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

        EpisodeObservation {
            step: self.step,
            date: today,
            team_name: team.map(|t| t.name.clone()).unwrap_or_default(),
            formation: team.map(|t| t.formation.clone()).unwrap_or_else(|| "4-4-2".into()),
            league_position,
            points,
            budget: team.map(|t| t.transfer_budget).unwrap_or(0),
            is_matchday,
            next_fixture,
            squad: self.squad_view(user_team_id),
            offers: self.pending_offers(),
            market: self.market_view(user_team_id),
            done: self.advanced_days >= self.horizon_days,
        }
    }

    /// Apply one action and advance to the next decision point.
    pub fn step(&mut self, action: Action) -> EpisodeObservation {
        self.step += 1;
        // A `Continue` at an offer decision point means "don't act on these
        // offers": advance past them (they will expire over time) instead of
        // re-stopping at the same offers forever.
        let skip_offers = matches!(action, Action::Continue);
        self.apply(action);
        self.advance_to_next_decision(skip_offers);
        self.observe()
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::Continue => {}
            Action::SetLineup { player_ids } => env::apply_lineup(&mut self.game, &player_ids),
            Action::SetTactics { play_style } => env::apply_play_style(&mut self.game, play_style),
            Action::AcceptOffer { player_id, offer_id } => {
                let _ = transfers::respond_to_offer(&mut self.game, &player_id, &offer_id, true);
            }
            Action::RejectOffer { player_id, offer_id } => {
                let _ = transfers::respond_to_offer(&mut self.game, &player_id, &offer_id, false);
            }
            Action::CounterOffer { player_id, offer_id, fee } => {
                let _ = transfers::counter_offer(&mut self.game, &player_id, &offer_id, fee);
            }
            Action::MakeBid { player_id, fee } => {
                let _ = transfers::make_transfer_bid(&mut self.game, &player_id, fee);
            }
        }
    }

    /// Advance day by day, playing matches, until the next decision point
    /// (a user matchday, a pending offer, or the horizon). When `skip_offers`
    /// is set (the agent chose `Continue`), the currently pending offers are
    /// advanced past rather than re-stopped at; stale offers expire over time.
    fn advance_to_next_decision(&mut self, skip_offers: bool) {
        let mut first = true;
        while self.advanced_days < self.horizon_days {
            if self.user_fixture_index().is_some() {
                // Today is a matchday: the agent's action (lineup/tactics) is in.
                // process_day plays it (the engine is XI-aware), then we continue.
                ofm_core::turn::process_day(&mut self.game);
                self.advanced_days += 1;
                first = false;
                continue;
            }
            if !(skip_offers && first) && !self.pending_offers().is_empty() {
                // Decision point: pending transfer offers.
                break;
            }
            first = false;
            transfers::expire_stale_transfer_offers(&mut self.game);
            ofm_core::turn::process_day(&mut self.game);
            self.advanced_days += 1;
        }
    }

    fn user_fixture_index(&self) -> Option<usize> {
        env::user_fixture_index(&self.game)
    }

    fn squad_view(&self, team_id: &str) -> Vec<env::PlayerView> {
        self.game
            .players
            .iter()
            .filter(|p| p.team_id.as_deref() == Some(team_id))
            .map(|p| env::PlayerView {
                id: p.id.clone(),
                name: p.match_name.clone(),
                position: p.position.clone(),
                group_position: p.position.to_group_position(),
                ovr: p.ovr,
                age: age_from_dob(&p.date_of_birth, &self.game.clock.current_date.format("%Y-%m-%d").to_string()),
                condition: p.condition,
                fitness: p.fitness,
                morale: p.morale,
                injured: p.injury.is_some(),
                wage: p.wage,
                market_value: p.market_value,
            })
            .collect()
    }

    fn pending_offers(&self) -> Vec<OfferView> {
        let team_id = self.game.manager.team_id.as_deref().unwrap_or_default();
        let mut out = Vec::new();
        for player in self.game.players.iter().filter(|p| p.team_id.as_deref() == Some(team_id)) {
            for offer in &player.transfer_offers {
                if offer.status != TransferOfferStatus::Pending {
                    continue;
                }
                let from_team = self
                    .game
                    .teams
                    .iter()
                    .find(|t| t.id == offer.from_team_id)
                    .map(|t| t.name.clone())
                    .unwrap_or_default();
                out.push(OfferView {
                    offer_id: offer.id.clone(),
                    player_id: player.id.clone(),
                    player_name: player.match_name.clone(),
                    from_team,
                    fee: offer.fee,
                    round: offer.negotiation_round,
                    suggested_counter: offer.suggested_counter_fee,
                });
            }
        }
        out
    }

    fn market_view(&self, own_team_id: &str) -> Vec<MarketView> {
        let mut out: Vec<MarketView> = self
            .game
            .players
            .iter()
            .filter(|p| {
                p.transfer_listed
                    && p.team_id.as_deref() != Some(own_team_id)
                    && p.injury.is_none()
            })
            .map(|p| MarketView {
                player_id: p.id.clone(),
                player_name: p.match_name.clone(),
                position: p.position.clone(),
                ovr: p.ovr,
                age: age_from_dob(&p.date_of_birth, &self.game.clock.current_date.format("%Y-%m-%d").to_string()),
                market_value: p.market_value,
                team: self
                    .game
                    .teams
                    .iter()
                    .find(|t| Some(&t.id) == p.team_id.as_ref())
                    .map(|t| t.name.clone())
                    .unwrap_or_default(),
            })
            .collect();
        out.sort_by(|a, b| b.ovr.cmp(&a.ovr));
        out.truncate(30);
        out
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
