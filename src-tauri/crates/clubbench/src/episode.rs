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
    /// Set lineup AND play style in one action (match preparation).
    SetMatchPlan { player_ids: Vec<String>, play_style: domain::team::PlayStyle },
    /// Accept an incoming transfer offer for one of our players.
    AcceptOffer { player_id: String, offer_id: String },
    /// Reject an incoming transfer offer.
    RejectOffer { player_id: String, offer_id: String },
    /// Counter an incoming offer with our own requested fee.
    CounterOffer { player_id: String, offer_id: String, fee: u64 },
    /// Bid for a player.
    MakeBid { player_id: String, fee: u64 },
    /// Send a scout to report on a player (reveals fuzzed rating + potential band).
    Scout { player_id: String },
    /// Transfer-list one of our players for sale — attracts incoming offers.
    ListPlayer { player_id: String },
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

/// A player available on the market we could bid on. True attributes are NOT
/// shown (partial observability); only a scout report reveals a fuzzed rating
/// and a coarse potential band.
#[derive(Serialize, Clone, Debug)]
pub struct MarketView {
    pub player_id: String,
    pub player_name: String,
    pub position: Position,
    pub age: u8,
    pub market_value: u64,
    pub team: String,
    /// Fuzzed overall rating if a scout report exists, else None.
    pub reported_ovr: Option<u8>,
    /// Coarse potential band ("worldClass"/"strong"/"moderate"/"unclear") if scouted.
    pub potential: Option<String>,
    /// A scout is currently watching this player.
    pub scouting: bool,
}

/// A completed scout report (from the game's message inbox).
#[derive(Serialize, Clone, Debug)]
pub struct ScoutReportView {
    pub player_id: String,
    pub player_name: String,
    pub team: String,
    /// Fuzzed overall rating (1-99).
    pub avg_rating: Option<u32>,
    pub rating_desc: String,
    pub potential: String,
    pub confidence: String,
}

/// A scouting assignment in progress.
#[derive(Serialize, Clone, Debug)]
pub struct ScoutingView {
    pub player_id: String,
    pub player_name: String,
    pub days_remaining: u32,
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
    pub scout_reports: Vec<ScoutReportView>,
    pub scouting_in_progress: Vec<ScoutingView>,
    pub transfer_window_open: bool,
    pub done: bool,
}

/// The decision-cadence episode.
pub struct Episode {
    pub game: Game,
    pub step: u64,
    horizon_days: u64,
    advanced_days: u64,
    /// Offer ids already shown to the agent — new offers (never-seen ids) are
    /// the only offer decision points, so `Continue` doesn't re-stop forever.
    seen_offers: std::collections::HashSet<String>,
}

impl Episode {
    /// Start a reproducible episode managing the first club: `seed` fixes the
    /// world and trajectory; `horizon_days` bounds the episode length.
    pub fn new(seed: u64, horizon_days: u64) -> Self {
        Self::new_with_pick(seed, &env::ClubPick::Index(0), horizon_days)
    }

    /// Start a reproducible episode managing the club selected by `pick`.
    pub fn new_with_pick(seed: u64, pick: &env::ClubPick, horizon_days: u64) -> Self {
        Self::new_with_pick_and_world(seed, pick, env::WorldSize::Medium, &env::ScenarioBudget::default(), horizon_days)
    }

    /// Start a reproducible episode with an explicit world size and scenario
    /// budget.
    pub fn new_with_pick_and_world(
        seed: u64,
        pick: &env::ClubPick,
        world: env::WorldSize,
        budget: &env::ScenarioBudget,
        horizon_days: u64,
    ) -> Self {
        ofm_core::rng::set_seed(seed);
        let game = env::build_game_for_club_with(seed, pick, world, budget);
        Self {
            game,
            step: 0,
            horizon_days,
            advanced_days: 0,
            seen_offers: Default::default(),
        }
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn observe(&mut self) -> EpisodeObservation {
        let user_team_id = self.game.manager.team_id.as_deref().unwrap_or_default();
        let team = self.game.teams.iter().find(|t| t.id == user_team_id);
        let today = self.game.clock.current_date.format("%Y-%m-%d").to_string();

        // Mark the offers we're about to show as seen, so `Continue` doesn't
        // make the env re-stop on the same offers forever.
        let offers = self.pending_offers();
        for o in &offers {
            self.seen_offers.insert(o.offer_id.clone());
        }

        let is_matchday = self.user_matchday();
        let next_fixture = self
            .game
            .competitions
            .iter()
            .find_map(|league| {
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
            offers,
            market: self.market_view(user_team_id),
            scout_reports: self.scout_report_views(),
            scouting_in_progress: self.scouting_views(),
            transfer_window_open: transfers::transfer_window_is_open(&self.game),
            done: self.advanced_days >= self.horizon_days,
        }
    }

    /// Apply one action and advance to the next decision point.
    pub fn step(&mut self, action: Action) -> EpisodeObservation {
        self.step += 1;
        self.apply(action);
        // Every action addresses the current decision point, so the world
        // always moves forward at least one day (which plays any match today).
        self.advance_to_next_decision();
        self.observe()
    }

    fn apply(&mut self, action: Action) {
        match action {
            Action::Continue => {}
            Action::SetLineup { player_ids } => env::apply_lineup(&mut self.game, &player_ids),
            Action::SetTactics { play_style } => env::apply_play_style(&mut self.game, play_style),
            Action::SetMatchPlan { player_ids, play_style } => {
                env::apply_lineup(&mut self.game, &player_ids);
                env::apply_play_style(&mut self.game, play_style);
            }
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
            Action::Scout { player_id } => {
                if let Some(scout_id) = self.user_scout_id() {
                    let _ = ofm_core::scouting::send_scout(&mut self.game, &scout_id, &player_id);
                }
            }
            Action::ListPlayer { player_id } => {
                if let Some(team_id) = self.game.manager.team_id.as_ref() {
                    if let Some(player) = self
                        .game
                        .players
                        .iter_mut()
                        .find(|p| p.id == player_id && p.team_id.as_ref() == Some(team_id))
                    {
                        player.transfer_listed = true;
                    }
                }
            }
        }
    }

    /// Move the world forward one day (expiring stale offers, running the turn
    /// loop, and playing any user matchday via the XI-aware engine).
    fn advance_one_day(&mut self) {
        transfers::expire_stale_transfer_offers(&mut self.game);
        ofm_core::turn::process_day(&mut self.game);
        self.advanced_days += 1;
    }

    /// A transfer-window "market day": the agent gets a chance to scout and bid
    /// roughly every three days while the window is open.
    fn market_day(&self) -> bool {
        transfers::transfer_window_is_open(&self.game) && self.advanced_days % 3 == 0
    }

    /// Pending offers the agent has never been shown (never-seen ids).
    fn fresh_offers(&self) -> Vec<OfferView> {
        self.pending_offers()
            .into_iter()
            .filter(|o| !self.seen_offers.contains(&o.offer_id))
            .collect()
    }

    /// Advance until the next decision point (a user matchday, fresh pending
    /// offers, a transfer-window market day, or the horizon), always moving
    /// forward at least one day first.
    fn advance_to_next_decision(&mut self) {
        self.advance_one_day();
        loop {
            if self.advanced_days >= self.horizon_days {
                break;
            }
            if self.user_matchday() {
                break; // user matchday — lineup/tactics decision
            }
            if !self.fresh_offers().is_empty() {
                break; // fresh transfer offer — accept/reject/counter decision
            }
            if self.market_day() {
                break; // transfer-window market — scout/bid decision
            }
            self.advance_one_day();
        }
    }

    /// Does the user's club have a scheduled match today (any competition)?
    fn user_matchday(&self) -> bool {
        let today = self.game.clock.current_date.format("%Y-%m-%d").to_string();
        self.game.user_has_scheduled_match_on(&today)
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
                transfer_listed: p.transfer_listed,
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
        // Scout reports keyed by player id, for the market view.
        let reports: std::collections::HashMap<&str, &domain::message::ScoutReportData> = self
            .game
            .messages
            .iter()
            .filter_map(|m| m.context.scout_report.as_ref())
            .map(|r| (r.player_id.as_str(), r))
            .collect();
        let scouting: std::collections::HashSet<&str> = self
            .game
            .scouting_assignments
            .iter()
            .map(|a| a.player_id.as_str())
            .collect();

        // Market = a scouted shortlist of targets outside the squad: the most
        // valuable players in the world (transfer-listed or not). Ratings are
        // hidden until scouted (partial observability).
        let mut out: Vec<MarketView> = self
            .game
            .players
            .iter()
            .filter(|p| p.team_id.as_deref() != Some(own_team_id) && p.injury.is_none())
            .map(|p| {
                let report = reports.get(p.id.as_str());
                MarketView {
                    player_id: p.id.clone(),
                    player_name: p.match_name.clone(),
                    position: p.position.clone(),
                    age: age_from_dob(&p.date_of_birth, &self.game.clock.current_date.format("%Y-%m-%d").to_string()),
                    market_value: p.market_value,
                    team: self
                        .game
                        .teams
                        .iter()
                        .find(|t| Some(&t.id) == p.team_id.as_ref())
                        .map(|t| t.name.clone())
                        .unwrap_or_default(),
                    reported_ovr: report.and_then(|r| r.avg_rating).map(|v| v as u8),
                    potential: report.map(|r| r.potential_key.clone()),
                    scouting: scouting.contains(p.id.as_str()),
                }
            })
            .collect();
        // The market shortlist = the most valuable targets outside the squad,
        // so unscouted high-value players remain visible and scoutable.
        out.sort_by(|a, b| b.market_value.cmp(&a.market_value));
        out.truncate(30);
        out
    }

    /// Completed scout reports pulled from the game's message inbox.
    fn scout_report_views(&self) -> Vec<ScoutReportView> {
        self.game
            .messages
            .iter()
            .filter_map(|m| m.context.scout_report.as_ref())
            .map(|r| {
                let team = r
                    .team_name
                    .clone()
                    .or_else(|| {
                        self.game.players.iter().find(|p| p.id == r.player_id)
                            .and_then(|p| p.team_id.as_ref())
                            .and_then(|tid| self.game.teams.iter().find(|t| &t.id == tid))
                            .map(|t| t.name.clone())
                    })
                    .unwrap_or_default();
                ScoutReportView {
                    player_id: r.player_id.clone(),
                    player_name: r.player_name.clone(),
                    team,
                    avg_rating: r.avg_rating,
                    rating_desc: r.rating_key.clone(),
                    potential: r.potential_key.clone(),
                    confidence: r.confidence_key.clone(),
                }
            })
            .collect()
    }

    /// Scouting assignments in progress.
    fn scouting_views(&self) -> Vec<ScoutingView> {
        self.game
            .scouting_assignments
            .iter()
            .map(|a| ScoutingView {
                player_id: a.player_id.clone(),
                player_name: self
                    .game
                    .players
                    .iter()
                    .find(|p| p.id == a.player_id)
                    .map(|p| p.match_name.clone())
                    .unwrap_or_default(),
                days_remaining: a.days_remaining,
            })
            .collect()
    }

    /// The user's first scout (needed by `send_scout`).
    fn user_scout_id(&self) -> Option<String> {
        let user_team_id = self.game.manager.team_id.as_deref()?;
        self.game
            .staff
            .iter()
            .find(|s| s.team_id.as_deref() == Some(user_team_id) && s.role == domain::staff::StaffRole::Scout)
            .map(|s| s.id.clone())
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
