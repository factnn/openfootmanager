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
    /// The active episode seed (None = normal gameplay, entropy-random).
    static GLOBAL_SEED: Rc<RefCell<Option<u64>>> = Rc::new(RefCell::new(None));
}

/// FNV-1a 64-bit — stable across Rust versions (unlike `std::hash::DefaultHasher`,
/// whose algorithm is not guaranteed). Used to derive semantic sub-stream seeds.
fn stable_hash(input: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in input {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Switch this thread's RNG to a deterministic stream for `seed`. Call once per
/// episode (before the season loop); all subsequent [`rng()`] draws on this
/// thread become reproducible for the same `seed`.
pub fn set_seed(seed: u64) {
    GLOBAL_SEED.with(|s| *s.borrow_mut() = Some(seed));
    GLOBAL_RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(seed));
}

/// Revert to entropy-backed randomness (default gameplay behaviour).
pub fn reset_random() {
    GLOBAL_SEED.with(|s| *s.borrow_mut() = None);
    GLOBAL_RNG.with(|r| *r.borrow_mut() = entropy_seeded());
}

/// Re-seed this thread's RNG from a deterministic semantic sub-stream keyed by
/// `(global_seed, domain, key)`. This makes exogenous randomness action-invariant
/// (ref_gpt §4): e.g. the stream for match #N depends only on (seed, "match", N),
/// never on what the agent did in earlier matches / scouts / transfers. One
/// decision no longer shifts the randomness of future, unrelated matches.
///
/// No-op when no episode seed is active (normal gameplay stays entropy-random).
pub fn set_domain(domain: &str, key: &[u8]) {
    let base = match GLOBAL_SEED.with(|s| *s.borrow()) {
        Some(seed) => seed,
        None => return,
    };
    let mut buf = Vec::with_capacity(domain.len() + key.len() + 8);
    buf.extend_from_slice(domain.as_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&base.to_le_bytes());
    let stream_seed = stable_hash(&buf);
    GLOBAL_RNG.with(|r| *r.borrow_mut() = StdRng::seed_from_u64(stream_seed));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn next_u64() -> u64 {
        rng().try_next_u64().unwrap()
    }

    #[test]
    fn set_domain_is_reproducible_and_action_invariant() {
        set_seed(42);
        // Same (domain, key) → same stream, every time.
        set_domain("match", &1u64.to_le_bytes());
        let x = next_u64();
        set_domain("match", &1u64.to_le_bytes());
        assert_eq!(x, next_u64(), "same key must give the same stream");

        // Different key → different stream.
        set_domain("match", &2u64.to_le_bytes());
        assert_ne!(x, next_u64(), "different key must give a different stream");

        // Action-invariance: consuming the "match" stream must not disturb the
        // "day" stream (one semantic sub-stream never pollutes another).
        set_domain("day", b"2026-07-01");
        let day_first = next_u64();
        set_domain("match", &7u64.to_le_bytes());
        let _ = next_u64(); // consume a draw from the match stream
        set_domain("day", b"2026-07-01");
        assert_eq!(day_first, next_u64(), "day stream must ignore match draws");

        reset_random();
    }
}
