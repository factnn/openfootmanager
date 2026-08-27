//! Deterministic A/B validation harness: simulate N matches between two
//! synthetic 4-4-2 teams with flat attributes, Standard roles, default tactics.
use engine::{simulate_with_rng, MatchConfig, PlayStyle, PlayerData, PlayerRole, Position, TeamData};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::env;

fn team(id: &str, rating: u8, style: PlayStyle) -> TeamData {
    let mk = |pos: Position, n: u8| -> Vec<PlayerData> {
        (1..=n)
            .map(|i| {
                let a = rating as u8;
                PlayerData {
                    id: format!("{}_{}_{}", id, format!("{:?}", pos), i),
                    name: format!("{:?}{}", pos, i),
                    position: pos,
                    ovr: rating,
                    condition: 100,
                    fitness: 75,
                    pace: a, stamina: a, strength: a, agility: a,
                    passing: a, shooting: a, tackling: a, dribbling: a,
                    defending: a, positioning: a, vision: a, decisions: a,
                    composure: a, aggression: a, teamwork: a, leadership: a,
                    handling: a, reflexes: a, aerial: a,
                    traits: vec![],
                    role: PlayerRole::Standard,
                }
            })
            .collect()
    };
    let mut players = mk(Position::Goalkeeper, 1);
    players.extend(mk(Position::Defender, 4));
    players.extend(mk(Position::Midfielder, 4));
    players.extend(mk(Position::Forward, 2));
    TeamData {
        id: id.to_string(),
        name: id.to_string(),
        formation: "4-4-2".to_string(),
        play_style: style,
        tactics: Default::default(),
        players,
    }
}

fn parse_style(s: &str) -> PlayStyle {
    match s {
        "Attacking" => PlayStyle::Attacking,
        "Defensive" => PlayStyle::Defensive,
        "Possession" => PlayStyle::Possession,
        "Counter" => PlayStyle::Counter,
        "HighPress" => PlayStyle::HighPress,
        _ => PlayStyle::Balanced,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // mc_validate <n> <home_rating> <away_rating> <home_style> <away_style> <seed>
    let n: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let hr: u8 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(75);
    let ar: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(75);
    let hs = parse_style(args.get(4).map(|s| s.as_str()).unwrap_or("Balanced"));
    let as_ = parse_style(args.get(5).map(|s| s.as_str()).unwrap_or("Balanced"));
    let seed: u64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(1);

    let home = team("H", hr, hs);
    let away = team("A", ar, as_);
    let cfg = MatchConfig::default();
    let mut tot_h = 0u64;
    let mut tot_a = 0u64;
    let mut w = 0u64;
    let mut d = 0u64;
    let mut l = 0u64;
    for i in 0..n {
        let mut rng = StdRng::seed_from_u64(seed + i);
        let rep = simulate_with_rng(&home, &away, &cfg, &mut rng);
        tot_h += rep.home_goals as u64;
        tot_a += rep.away_goals as u64;
        if rep.home_goals > rep.away_goals { w += 1 }
        else if rep.home_goals == rep.away_goals { d += 1 }
        else { l += 1 }
    }
    println!("{:.3} {:.3} W{:.4} D{:.4} L{:.4}",
        tot_h as f64 / n as f64, tot_a as f64 / n as f64,
        w as f64 / n as f64, d as f64 / n as f64, l as f64 / n as f64);
}
