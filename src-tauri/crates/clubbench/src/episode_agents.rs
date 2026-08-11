//! Baseline agents for the decision-cadence environment (`episode::Episode`).

use crate::episode::{Action, EpisodeObservation};
use domain::team::PlayStyle;

/// A decision policy: given the current observation, pick one action.
pub trait Policy {
    fn name(&self) -> &str;
    fn act(&mut self, obs: &EpisodeObservation) -> Action;
}

/// A sensible automated manager:
/// - on a matchday: best-fit XI + a fixed play style;
/// - on offers: accept any bid at/above 1.2× market value, otherwise reject;
/// - otherwise: occasionally bid for a strong transfer-listed player we can afford.
pub struct AutoManager {
    pub play_style: PlayStyle,
}

impl AutoManager {
    pub fn new(play_style: PlayStyle) -> Self {
        Self { play_style }
    }

    fn best_lineup(&self, obs: &EpisodeObservation) -> Vec<String> {
        // Greedy per-formation-slot best fit, skipping the injured/low-condition.
        let slots = ofm_core::player_rating::formation_slots(&obs.formation);
        let mut pool: Vec<&crate::env::PlayerView> = obs.squad.iter().filter(|p| !p.injured).collect();
        let mut xi = Vec::new();
        for slot in slots.iter().take(11) {
            let group = slot.to_group_position();
            let mut best_idx = None;
            let mut best_rating: f64 = f64::MIN;
            for (i, p) in pool.iter().enumerate() {
                let rating = if p.group_position == group {
                    p.ovr as f64
                } else {
                    p.ovr as f64 - 8.0
                };
                let rating = if p.condition < 55 { rating - 25.0 } else { rating };
                if rating > best_rating {
                    best_rating = rating;
                    best_idx = Some(i);
                }
            }
            if let Some(i) = best_idx {
                xi.push(pool.remove(i).id.clone());
            }
        }
        xi
    }
}

impl Policy for AutoManager {
    fn name(&self) -> &str {
        "AutoManager"
    }
    fn act(&mut self, obs: &EpisodeObservation) -> Action {
        if obs.is_matchday {
            return Action::SetLineup {
                player_ids: self.best_lineup(obs),
            };
        }
        if !obs.offers.is_empty() {
            // Accept anything ≥ 1.2× market value, reject the rest.
            let offer = &obs.offers[0];
            let value = obs
                .squad
                .iter()
                .find(|p| p.id == offer.player_id)
                .map(|p| p.market_value as u64)
                .unwrap_or(0);
            if offer.fee >= value * 12 / 10 {
                return Action::AcceptOffer {
                    player_id: offer.player_id.clone(),
                    offer_id: offer.offer_id.clone(),
                };
            }
            return Action::RejectOffer {
                player_id: offer.player_id.clone(),
                offer_id: offer.offer_id.clone(),
            };
        }
        Action::Continue
    }
}

/// Never manage anything — the control baseline (everything is AI default).
pub struct PassiveManager;
impl Policy for PassiveManager {
    fn name(&self) -> &str {
        "Passive"
    }
    fn act(&mut self, _obs: &EpisodeObservation) -> Action {
        Action::Continue
    }
}

/// Resolve every pending offer, accepting above 1.2× value — but never touch
/// the lineup or the market. Isolates the transfer-decision contribution.
pub struct OffersOnlyManager;
impl Policy for OffersOnlyManager {
    fn name(&self) -> &str {
        "OffersOnly"
    }
    fn act(&mut self, obs: &EpisodeObservation) -> Action {
        if !obs.offers.is_empty() {
            let offer = &obs.offers[0];
            let value = obs
                .squad
                .iter()
                .find(|p| p.id == offer.player_id)
                .map(|p| p.market_value as u64)
                .unwrap_or(0);
            if offer.fee >= value * 12 / 10 {
                return Action::AcceptOffer {
                    player_id: offer.player_id.clone(),
                    offer_id: offer.offer_id.clone(),
                };
            }
            return Action::RejectOffer {
                player_id: offer.player_id.clone(),
                offer_id: offer.offer_id.clone(),
            };
        }
        Action::Continue
    }
}
