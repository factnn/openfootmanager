//! Deterministic, seedable per-thread RNG for reproducible benchmark episodes.
//!
//! Normal gameplay draws from OS entropy (random, as before). For a reproducible
//! ClubBench episode, call [`set_seed`] once at the start of the episode; every
//! later [`rng()`] call on the same thread then draws sequentially from a single
//! seeded [`StdRng`], so the whole trajectory is reproducible for a given seed.
//!
//! [`rng()`] is meant as a drop-in replacement for `rand::rng()` at call sites
//! that previously used ambient randomness.

use std::cell::RefCell;
use std::convert::Infallible;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{SeedableRng, TryRng};

/// A fresh OS-entropy-seeded [`StdRng`] (the default, non-deterministic mode).
fn entropy_seeded() -> StdRng {
    StdRng::try_from_rng(&mut rand::rng()).expect("failed to seed RNG from OS entropy")
}

thread_local! {
    static GLOBAL_RNG: Rc<RefCell<StdRng>> = Rc::new(RefCell::new(entropy_seeded()));
}

/// Switch this thread's RNG to a deterministic stream for `seed`. Call once per
/// episode (before the season loop); all subsequent [`rng()`] draws on this
/// thread become reproducible for the same `seed`.
pub fn set_seed(seed: u64) {
    GLOBAL_RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Revert to entropy-backed randomness (default gameplay behaviour).
pub fn reset_random() {
    GLOBAL_RNG.with(|r| *r.borrow_mut() = entropy_seeded());
}

/// A handle to this thread's shared RNG. Multiple handles share one underlying
/// [`StdRng`], so draws are strictly sequential and — after [`set_seed`] —
/// deterministic.
#[derive(Clone)]
pub struct GameRng(Rc<RefCell<StdRng>>);

impl TryRng for GameRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.0.borrow_mut().try_next_u32().unwrap())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.0.borrow_mut().try_next_u64().unwrap())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.0.borrow_mut().try_fill_bytes(dst)
    }
}

/// A handle to this thread's shared RNG (drop-in replacement for `rand::rng()`).
pub fn rng() -> GameRng {
    GLOBAL_RNG.with(|r| GameRng(Rc::clone(r)))
}
