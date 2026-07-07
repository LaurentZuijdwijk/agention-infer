//! Token sampling from a logits vector.
//!
//! Supports greedy (argmax) and stochastic sampling with temperature,
//! top-k, and top-p (nucleus) filtering. A small self-contained RNG keeps
//! the crate dependency-free and gives reproducible output from a seed.

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature. `0.0` (or less) means greedy argmax.
    pub temperature: f32,
    /// Keep only the top `k` logits before sampling. `0` disables.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds `top_p`. `>= 1.0` disables.
    pub top_p: f32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            seed: 0xD1CE_5EED_u64,
        }
    }
}

impl SamplingConfig {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Stateful sampler (owns the RNG so sequential draws differ).
pub struct Sampler {
    cfg: SamplingConfig,
    rng: Rng,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        let seed = cfg.seed;
        Self {
            cfg,
            rng: Rng::new(seed),
        }
    }

    /// Sample the next token id from raw logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        debug_assert!(!logits.is_empty());

        if self.cfg.temperature <= 0.0 {
            return argmax(logits) as u32;
        }

        // (id, logit) pairs, so we can filter/sort while remembering ids.
        let mut candidates: Vec<(usize, f32)> =
            logits.iter().copied().enumerate().collect();

        // Sort by logit descending.
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        // top-k truncation.
        if self.cfg.top_k > 0 && self.cfg.top_k < candidates.len() {
            candidates.truncate(self.cfg.top_k);
        }

        // Softmax with temperature over the (already sorted) candidates.
        let max_logit = candidates[0].1;
        let inv_temp = 1.0 / self.cfg.temperature;
        let mut probs: Vec<f32> = candidates
            .iter()
            .map(|&(_, l)| ((l - max_logit) * inv_temp).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // top-p (nucleus) truncation over the descending-sorted probs.
        let mut cutoff = probs.len();
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.cfg.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
        }

        // Draw from the surviving nucleus, renormalizing on the fly.
        let nucleus_sum: f32 = probs[..cutoff].iter().sum();
        let r = self.rng.next_f32() * nucleus_sum;
        let mut acc = 0.0f32;
        for i in 0..cutoff {
            acc += probs[i];
            if r <= acc {
                return candidates[i].0 as u32;
            }
        }
        // Numerical fallback: return the most probable candidate.
        candidates[0].0 as u32
    }
}

/// Index of the maximum element.
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// SplitMix64 → xorshift128+ style PRNG. Small, fast, good enough for sampling.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid a zero state.
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    fn next_f32(&mut self) -> f32 {
        // Use the top 24 bits for a uniform mantissa.
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        bits as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        let logits = [0.1, 3.0, -1.0, 2.9];
        assert_eq!(s.sample(&logits), 1);
    }

    #[test]
    fn top_k_1_is_deterministic() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: 42,
        };
        let mut s = Sampler::new(cfg);
        let logits = [0.1, 5.0, 0.2, 0.3];
        for _ in 0..20 {
            assert_eq!(s.sample(&logits), 1);
        }
    }

    #[test]
    fn distribution_is_reproducible() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 7,
        };
        let logits = [1.0f32, 2.0, 0.5, 1.5, 3.0];
        let a: Vec<u32> = {
            let mut s = Sampler::new(cfg.clone());
            (0..10).map(|_| s.sample(&logits)).collect()
        };
        let b: Vec<u32> = {
            let mut s = Sampler::new(cfg);
            (0..10).map(|_| s.sample(&logits)).collect()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rng_stays_in_unit_interval() {
        let mut rng = Rng::new(123);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
