//! ClubBench — the headless, deterministic football-management environment and
//! its baseline agents.
//!
//! The environment mirrors the game's own "delegate" match path: the user's
//! fixture is simulated with the XI-aware live-match engine (so lineup choices
//! matter), while the rest of the world advances on the standard turn loop.
//! Every episode is reproducible via `ofm_core::rng::set_seed(seed)`.

pub mod agents;
pub mod env;
pub mod episode;
pub mod episode_agents;
pub mod run;
pub mod score;
