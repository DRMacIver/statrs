//! Provides functions for calculating
//! [harmonic](https://en.wikipedia.org/wiki/Harmonic_number)
//! numbers

use crate::consts;
use crate::function::gamma;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Computes the `t`-th harmonic number
///
/// # Remarks
///
/// Returns `1` as a special case when `t == 0`
pub fn harmonic(t: u64) -> f64 {
    match t {
        0 => 1.0,
        _ => consts::EULER_MASCHERONI + gamma::digamma(t as f64 + 1.0),
    }
}

/// Computes the generalized harmonic number of  order `n` of `m`
/// e.g. `(1 + 1/2^m + 1/3^m + ... + 1/n^m)`
///
/// # Remarks
///
/// Returns `1` as a special case when `n == 0`
pub fn gen_harmonic(n: u64, m: f64) -> f64 {
    match n {
        0 => 1.0,
        _ => (0..n).fold(0.0, |acc, x| acc + (x as f64 + 1.0).powf(-m)),
    }
}

#[rustfmt::skip]
#[cfg(test)]
mod tests {
    use crate::prec;
    use super::*;
    use hegel::generators;

    /// `harmonic` goes through `digamma`; `gen_harmonic` sums the series
    /// directly. At order 1 they are the same number, so each is an independent
    /// oracle for the other.
    ///
    /// `n <= 1e5` bounds the runtime of the summing side, not the domain of
    /// either function. The tolerance covers the naive summation's accumulated
    /// rounding (worst observed 7.8e-15 relative, at n = 1e5).
    #[hegel::test]
    fn harmonic_matches_the_summed_series_at_order_one(tc: hegel::TestCase) {
        let n = tc.draw(generators::integers::<u64>().max_value(100_000));
        prec::assert_relative_eq!(harmonic(n), gen_harmonic(n, 1.0), max_relative = 1e-13);
    }

    /// At order 0 every term of `1 + 1/2^m + ... + 1/n^m` is 1, so the sum
    /// counts its own terms. Exact, and pins the number of terms against an
    /// off-by-one in the summation range.
    #[hegel::test]
    fn gen_harmonic_at_order_zero_counts_its_terms(tc: hegel::TestCase) {
        let n = tc.draw(generators::integers::<u64>().min_value(1).max_value(100_000));
        assert_eq!(gen_harmonic(n, 0.0), n as f64);
    }

    #[test]
    fn test_harmonic() {
        prec::assert_ulps_eq!(harmonic(0), 1.0, max_ulps = 0);
        prec::assert_abs_diff_eq!(harmonic(1), 1.0, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(2), 1.5, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(4), 2.083333333333333333333, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(8), 2.717857142857142857143, epsilon = 1e-14);
        prec::assert_abs_diff_eq!(harmonic(16), 3.380728993228993228993, epsilon = 1e-14);
    }

    #[test]
    fn test_gen_harmonic() {
        assert_eq!(gen_harmonic(0, 0.0), 1.0);
        assert_eq!(gen_harmonic(0, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(0, f64::NEG_INFINITY), 1.0);
        assert_eq!(gen_harmonic(1, 0.0), 1.0);
        assert_eq!(gen_harmonic(1, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(1, f64::NEG_INFINITY), 1.0);
        assert_eq!(gen_harmonic(2, 1.0), 1.5);
        assert_eq!(gen_harmonic(2, 3.0), 1.125);
        assert_eq!(gen_harmonic(2, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(2, f64::NEG_INFINITY), f64::INFINITY);
        prec::assert_abs_diff_eq!(gen_harmonic(4, 1.0), 2.083333333333333333333, epsilon = 1e-14);
        assert_eq!(gen_harmonic(4, 3.0), 1.177662037037037037037);
        assert_eq!(gen_harmonic(4, f64::INFINITY), 1.0);
        assert_eq!(gen_harmonic(4, f64::NEG_INFINITY), f64::INFINITY);
    }
}
