use num_traits::Num;

/// Implements univariate function bisection searching for criteria
/// ```text
/// smallest k such that f(k) >= z
/// ```
/// Evaluates to `None` if
/// - provided interval has lower bound greater than upper bound
/// - function found not semi-monotone on the provided interval containing `z`
///
/// Evaluates to `Some(k)`, where `k` satisfies the search criteria
pub fn integral_bisection_search<K: Num + Clone, T: Num + PartialOrd>(
    f: impl Fn(&K) -> T,
    z: T,
    lb: K,
    ub: K,
) -> Option<K> {
    if !(f(&lb)..=f(&ub)).contains(&z) {
        return None;
    }
    let two = K::one() + K::one();
    let mut lb = lb;
    let mut ub = ub;
    loop {
        let mid = (lb.clone() + ub.clone()) / two.clone();
        if !(f(&lb)..=f(&ub)).contains(&f(&mid)) {
            return None; // f found not monotone on interval
        } else if f(&lb) == z {
            return Some(lb);
        } else if f(&ub) == z || (lb.clone() + K::one()) == ub {
            return Some(ub); // found or no more integers between
        } else if f(&mid) >= z {
            ub = mid;
        } else {
            lb = mid;
        }
    }
}

/// Quantile `F^{-1}(p)` for a continuous distribution supported on `(0, ∞)`
/// whose cdf has no closed-form inverse, found with a safeguarded Newton–Raphson
/// step (Numerical Recipes' `rtsafe`); `cdf`, `sf` and `pdf` are the
/// distribution's own functions and `p` must lie strictly inside `(0, 1)`.
///
/// The bracket `[low, high]` is kept as an invariant and only ever tightened, so
/// a Newton step that is non-finite, would leave the bracket, or is not shrinking
/// at least as fast as a bisection falls back to bisection; the iterate can never
/// escape the support — the guard that keeps a quantile sitting against a
/// boundary from diverging to NaN (cf. Gamma in #382) — and the bracket is halved
/// often enough to reach the root within the iteration budget. In the upper half
/// the survival function is inverted rather than the cdf: as `cdf` saturates to
/// one it can no longer resolve a deep upper-tail quantile, whereas `sf` stays
/// well conditioned there.
pub fn newton_raphson_quantile(
    p: f64,
    cdf: impl Fn(f64) -> f64,
    sf: impl Fn(f64) -> f64,
    pdf: impl Fn(f64) -> f64,
) -> f64 {
    // Bracket the quantile within a factor of two, `cdf(low) <= p <= cdf(high)`,
    // by walking a unit interval out toward it. Moving *both* ends keeps the
    // bracket tight when the quantile is far from 1 (deep in either tail), so the
    // Newton phase starts close and converges in a handful of steps rather than
    // bisecting the whole span.
    let mut low = 1.0;
    let mut high = 2.0;
    while cdf(low) > p {
        high = low;
        low /= 2.0;
    }
    while cdf(high) < p {
        low = high;
        high *= 2.0;
    }

    // Solve `sf(x) = 1 - p` in the upper half and `cdf(x) = p` otherwise; either
    // way the residual is increasing in `x` with derivative `pdf(x)`.
    let upper = p > 0.5;
    let target = if upper { 1.0 - p } else { p };

    // A *relative* accuracy target: quantiles span many orders of magnitude, so
    // an absolute tolerance (as in `prec::convergence`) is meaningless deep in a
    // tail where the quantile itself is far smaller than that tolerance.
    let accuracy = crate::prec::DEFAULT_RELATIVE_ACC;
    const MAX_ITERATIONS: usize = 100;
    let mut x = (low + high) / 2.0;
    let mut last_step = high - low;
    for _ in 0..MAX_ITERATIONS {
        let residual = if upper {
            target - sf(x)
        } else {
            cdf(x) - target
        };
        let newton = x - residual / pdf(x);
        // A full Newton step below the relative tolerance means we have
        // converged; accept it before the bracket bookkeeping below, which would
        // otherwise reject a converged step that has rounded onto the very
        // endpoint we are about to move to `x`.
        if (newton - x).abs() <= accuracy * x.abs() {
            return newton;
        }
        // Tighten the bracket by the sign of the (increasing) residual.
        if residual >= 0.0 {
            high = x;
        } else {
            low = x;
        }
        // Take the Newton step only while it stays strictly inside the bracket
        // *and* shrinks at least as fast as a bisection would, else bisect. The
        // second half of that test is what bounds the iteration count: a tail
        // where the step is merely linear — the inverse gamma's `exp(-b/x)`
        // advances `b/x` by one per step regardless of how far the root is —
        // otherwise creeps toward the root and runs out of iterations short of
        // it, e.g. `InverseGamma(1, 1).inverse_cdf(1e-200)` off by 4%.
        let step = (newton - x).abs();
        if newton.is_finite() && newton > low && newton < high && 2.0 * step <= last_step {
            x = newton;
            last_step = step;
        } else {
            last_step = (high - low) / 2.0;
            x = low + last_step;
        }
    }
    x
}

#[cfg(test)]
macro_rules! testing_boiler {
    ($($arg_name:ident: $arg_ty:ty),+; $dist:ty; $dist_err:ty) => {
        #[cfg(not(feature = "std"))]
        #[allow(unused)]
        fn make_param_text($($arg_name: $arg_ty),+) -> &'static str {
            "(params N/A)"
        }

        #[cfg(feature = "std")]
        #[allow(unused)]
        fn make_param_text($($arg_name: $arg_ty),+) -> String {
            // ""
            let mut param_text = String::new();

            // "shape=10.0, rate=NaN, "
            $(
                param_text.push_str(
                    &format!(
                        "{}={:?}, ",
                        stringify!($arg_name),
                        $arg_name,
                    )
                );
            )+

            // "shape=10.0, rate=NaN" (removes trailing comma and whitespace)
            param_text.pop();
            param_text.pop();

            param_text
        }

        /// Creates and returns a distribution with the given parameters,
        /// panicking if `::new` fails.
        fn create_ok($($arg_name: $arg_ty),+) -> $dist {
            match <$dist>::new($($arg_name),+) {
                Ok(d) => d,
                Err(e) => panic!(
                    "{}::new was expected to succeed, but failed for {} with error: '{}'",
                    stringify!($dist),
                    make_param_text($($arg_name),+),
                    e
                )
            }
        }

        /// Returns the error when creating a distribution with the given parameters,
        /// panicking if `::new` succeeds.
        #[allow(dead_code)]
        fn create_err($($arg_name: $arg_ty),+) -> $dist_err {
            match <$dist>::new($($arg_name),+) {
                Err(e) => e,
                Ok(d) => panic!(
                    "{}::new was expected to fail, but succeeded for {} with result: {:?}",
                    stringify!($dist),
                    make_param_text($($arg_name),+),
                    d
                )
            }
        }

        /// Creates a distribution with the given parameters, calls the `get_fn`
        /// function with the new distribution and returns the result of `get_fn`.
        ///
        /// Panics if `::new` fails.
        fn create_and_get<F, T>($($arg_name: $arg_ty),+, get_fn: F) -> T
        where
            F: Fn($dist) -> T,
        {
            let n = create_ok($($arg_name),+);
            get_fn(n)
        }

        /// Creates a distribution with the given parameters, calls the `get_fn`
        /// function with the new distribution and compares the result of `get_fn`
        /// to `expected` exactly.
        ///
        /// Panics if `::new` fails.
        #[allow(dead_code)]
        fn test_exact<F, T>($($arg_name: $arg_ty),+, expected: T, get_fn: F)
        where
            F: Fn($dist) -> T,
            T: ::core::cmp::PartialEq + ::core::fmt::Debug
        {
            let x = create_and_get($($arg_name),+, get_fn);
            if x != expected {
                panic!(
                    "Expected {:?}, got {:?} for {}",
                    expected,
                    x,
                    make_param_text($($arg_name),+)
                );
            }
        }

        /// Gets a value for the given parameters by calling `create_and_get`
        /// and compares it to `expected`.
        ///
        /// Allows relative error of up to [`crate::prec::DEFAULT_RELATIVE_ACC`].
        ///
        /// Panics if `::new` fails.
        #[allow(dead_code)]
        fn test_relative<F>($($arg_name: $arg_ty),+, expected: f64, get_fn: F)
        where
            F: Fn($dist) -> f64,
        {
            let x = create_and_get($($arg_name),+, get_fn);
            let max_relative = $crate::prec::DEFAULT_RELATIVE_ACC;

            if !$crate::prec::relative_eq!(expected, x, max_relative = max_relative) {
                panic!(
                    "Expected {:?} to be almost equal to {:?} (max. relative error of {:?}), but wasn't for {}",
                    x,
                    expected,
                    max_relative,
                    make_param_text($($arg_name),+)
                );
            }
        }

        /// Gets a value for the given parameters by calling `create_and_get`
        /// and compares it to `expected`.
        ///
        /// Allows absolute error of up to `acc`.
        ///
        /// Panics if `::new` fails.
        #[allow(dead_code)]
        fn test_absolute<F>($($arg_name: $arg_ty),+, expected: f64, acc: f64, get_fn: F)
        where
            F: Fn($dist) -> f64,
        {
            let x = create_and_get($($arg_name),+, get_fn);

            // abs_diff_eq! cannot handle infinities, so we manually accept them here
            if expected.is_infinite() && x == expected {
                return;
            }

            if !$crate::prec::abs_diff_eq!(expected, x, epsilon = acc) {
                panic!(
                    "Expected {:?} to be almost equal to {:?} (max. absolute error of {:?}), but wasn't for {}",
                    x,
                    expected,
                    acc,
                    make_param_text($($arg_name),+)
                );
            }
        }

        /// Purposely fails creating a distribution with the given
        /// parameters and compares the returned error to `expected`.
        ///
        /// Panics if `::new` succeeds.
        #[allow(dead_code)]
        fn test_create_err($($arg_name: $arg_ty),+, expected: $dist_err)
        {
            let err = create_err($($arg_name),+);
            if err != expected {
                panic!(
                    "{}::new was expected to fail with error {:?}, but failed with error {:?} for {}",
                    stringify!($dist),
                    expected,
                    err,
                    make_param_text($($arg_name),+)
                )
            }
        }

        /// Gets a value for the given parameters by calling `create_and_get`
        /// and asserts that it is [`NAN`].
        ///
        /// Panics if `::new` fails.
        #[allow(dead_code)]
        fn test_is_nan<F>($($arg_name: $arg_ty),+, get_fn: F)
        where
            F: Fn($dist) -> f64
        {
            let x = create_and_get($($arg_name),+, get_fn);
            assert!(x.is_nan());
        }

        /// Gets a value for the given parameters by calling `create_and_get`
        /// and asserts that it is [`None`].
        ///
        /// Panics if `::new` fails.
        #[allow(dead_code)]
        fn test_none<F, T>($($arg_name: $arg_ty),+, get_fn: F)
        where
            F: Fn($dist) -> Option<T>,
            T: ::core::fmt::Debug,
        {
            let x = create_and_get($($arg_name),+, get_fn);

            if let Some(inner) = x {
                panic!(
                    "Expected None, got {:?} for {}",
                    inner,
                    make_param_text($($arg_name),+)
                )
            }
        }

        /// Asserts that associated error type is Send and Sync
        #[test]
        fn test_error_is_sync_send() {
            fn assert_sync_send<T: Sync + Send>() {}
            assert_sync_send::<$dist_err>();
        }
    };
}

#[cfg(test)]
pub(super) use testing_boiler;

/// Utility functions for testing implementations of distributions.
#[cfg(test)]
pub(super) mod density_util {
    use crate::distribution::{Continuous, ContinuousCDF, Discrete, DiscreteCDF};
    use crate::prec;

    enum CheckContinuousError {
        LnPdf { x: f64 },
        PdfStep { x: f64, _sum: f64 },
    }

    /// cdf should be the integral of the pdf
    fn check_integrate_pdf_is_cdf<D: ContinuousCDF<f64, f64> + Continuous<f64, f64>>(
        dist: &D,
        x_min: f64,
        x_max: f64,
        step: f64,
    ) -> Result<f64, CheckContinuousError> {
        let mut prev_x = x_min;
        let mut prev_density = dist.pdf(x_min);
        let mut sum = 0.0;

        loop {
            let x = prev_x + step;
            let density = dist.pdf(x);

            assert!(density >= 0.0);

            let ln_density = dist.ln_pdf(x);
            if ln_density.is_finite()
                && !prec::abs_diff_eq!(density.ln(), ln_density, epsilon = 1e-10)
            {
                return Err(CheckContinuousError::LnPdf { x });
            }

            // trapezoidal rule
            sum += (prev_density + density) * step / 2.0;

            let cdf = dist.cdf(x);
            if !prec::abs_diff_eq!(sum, cdf, epsilon = 1e-3) {
                return Err(CheckContinuousError::PdfStep { x, _sum: sum });
            }

            if x >= x_max {
                break;
            } else {
                prev_x = x;
                prev_density = density;
            }
        }

        Ok(sum)
    }

    /// cdf should be the sum of the pmf
    fn check_sum_pmf_is_cdf<D: DiscreteCDF<u64, f64> + Discrete<u64, f64>>(dist: &D, x_max: u64) {
        let mut sum = 0.0;

        // go slightly beyond x_max to test for off-by-one errors
        for i in 0..x_max + 3 {
            let prob = dist.pmf(i);

            assert!(prob >= 0.0);
            assert!(prob <= 1.0);

            sum += prob;

            if i == x_max {
                assert!(sum > 0.99);
            }

            prec::assert_abs_diff_eq!(sum, dist.cdf(i), epsilon = 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.1), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.5), 1e-10);
            // assert_almost_eq!(sum, dist.cdf(i as f64 + 0.9), 1e-10);
        }

        assert!(
            sum > 0.99 && sum <= 1.001,
            "sum should be close to 1, but is {sum}"
        );
    }

    /// pdf should be derivative of cdf
    fn check_derivative_of_cdf_is_pdf<D: ContinuousCDF<f64, f64> + Continuous<f64, f64>>(
        dist: &D,
        x_min: f64,
        x_max: f64,
        step: f64,
    ) -> Result<(), f64> {
        const DELTA: f64 = 1e-12;
        const DX: f64 = 2.0 * DELTA;
        let mut prev_x = x_min;

        loop {
            let x = prev_x + step;
            let x_ahead = x + DELTA;
            let x_behind = x - DELTA;
            let density = dist.pdf(x);

            let d_cdf = dist.cdf(x_ahead) - dist.cdf(x_behind);

            if !prec::abs_diff_eq!(d_cdf, DX * density, epsilon = 1e-11) {
                return Err(x);
            }

            if x >= x_max {
                break;
            } else {
                prev_x = x;
            }
        }

        Ok(())
    }

    /// Does a series of checks that all continuous distributions must obey.
    /// 99% of the probability mass should be between x_min and x_max or the finite
    /// difference of cdf should be near to the pdf for much of the support.
    pub fn check_continuous_distribution<D: ContinuousCDF<f64, f64> + Continuous<f64, f64>>(
        dist: &D,
        x_min: f64,
        x_max: f64,
    ) {
        assert_eq!(dist.pdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(dist.pdf(f64::INFINITY), 0.0);
        assert_eq!(dist.ln_pdf(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(dist.ln_pdf(f64::INFINITY), f64::NEG_INFINITY);
        assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
        assert_eq!(dist.cdf(f64::INFINITY), 1.0);

        let integrate_res =
            check_integrate_pdf_is_cdf(dist, x_min, x_max, (x_max - x_min) / 100000.0);
        let diff_res =
            check_derivative_of_cdf_is_pdf(dist, x_min, x_max, (x_max - x_min) / 100000.0);
        match integrate_res {
            // if integration failed along the way...
            Err(CheckContinuousError::LnPdf { x })
            | Err(CheckContinuousError::PdfStep { x, .. }) => {
                // then check if differentiation fails along the way
                if let Err(diff_err_x) = diff_res {
                    panic!(
                        "integration mismatched around x = {x} \n\
                        and derivative insufficiently close at x={diff_err_x}"
                    )
                }
                // otherwise: passes differentiation test
            }
            // if integration is close but not at endpoint (surprising!)...
            Ok(sum) if !prec::abs_diff_eq!(sum, 1.0, epsilon = 1e-3) => {
                // then check if differentiation fails along the way
                if let Err(diff_err_x) = diff_res {
                    panic!(
                        "integration summed to {sum}, insufficiently close to 1.0 \n\
                        and derivative insufficiently close at x={diff_err_x}"
                    )
                }
            }
            // passes integration test
            _ => (),
        }
    }

    /// Does a series of checks that all positive discrete distributions must
    /// obey.
    /// 99% of the probability mass should be between 0 and x_max (inclusive).
    pub fn check_discrete_distribution<D: DiscreteCDF<u64, f64> + Discrete<u64, f64>>(
        dist: &D,
        x_max: u64,
    ) {
        // assert_eq!(dist.cdf(f64::NEG_INFINITY), 0.0);
        // assert_eq!(dist.cdf(-10.0), 0.0);
        // assert_eq!(dist.cdf(-1.0), 0.0);
        // assert_eq!(dist.cdf(-0.01), 0.0);
        // assert_eq!(dist.cdf(f64::INFINITY), 1.0);

        check_sum_pmf_is_cdf(dist, x_max);
    }
}

/// Property-based tests for the contracts every univariate distribution in the
/// crate shares. They live here, beside `density_util`, because they are
/// cross-distribution checks rather than tests of any one distribution's
/// numerics: each property is written once and run against every family the
/// crate ships.
#[cfg(all(test, feature = "std"))]
mod hegel_props {
    use crate::distribution::{
        Bernoulli, Beta, Binomial, Categorical, Cauchy, Chi, ChiSquared, Continuous, ContinuousCDF,
        Discrete, DiscreteCDF, Erlang, Exp, FisherSnedecor, Gamma, Geometric, Gumbel,
        Hypergeometric, InverseGamma, Laplace, Levy, LogNormal, NegativeBinomial, Normal, Pareto,
        Poisson, StudentsT, Triangular, Uniform, Weibull,
    };
    use crate::prec;
    use crate::statistics::Distribution;
    use hegel::generators;

    /// Everything the generic continuous properties need from a distribution,
    /// bundled into one object-safe trait so a property can be written once
    /// and run against a boxed instance of any family.
    ///
    /// `sample_one` exists because `rand::distr::Distribution` has generic
    /// methods and cannot be a supertrait of a `dyn` type; the trait and its
    /// blanket impl are spelled out twice because the extra bound the
    /// sampler needs cannot be made conditional in place.
    #[cfg(feature = "rand")]
    trait Univariate:
        ContinuousCDF<f64, f64> + Continuous<f64, f64> + Distribution<f64> + core::fmt::Debug
    {
        fn sample_one(&self, rng: &mut hegel::extras::rand::HegelRandom) -> f64;
    }

    #[cfg(feature = "rand")]
    impl<D> Univariate for D
    where
        D: ContinuousCDF<f64, f64>
            + Continuous<f64, f64>
            + Distribution<f64>
            + ::rand::distr::Distribution<f64>
            + core::fmt::Debug,
    {
        fn sample_one(&self, rng: &mut hegel::extras::rand::HegelRandom) -> f64 {
            ::rand::distr::Distribution::sample(self, rng)
        }
    }

    #[cfg(not(feature = "rand"))]
    trait Univariate:
        ContinuousCDF<f64, f64> + Continuous<f64, f64> + Distribution<f64> + core::fmt::Debug
    {
    }

    #[cfg(not(feature = "rand"))]
    impl<D> Univariate for D where
        D: ContinuousCDF<f64, f64> + Continuous<f64, f64> + Distribution<f64> + core::fmt::Debug
    {
    }

    /// The discrete counterpart. Moments are left out of the bound:
    /// `NegativeBinomial` implements `DiscreteDistribution` where the other six
    /// implement `Distribution`, so no single bound covers all seven.
    trait UnivariateDiscrete: DiscreteCDF<u64, f64> + Discrete<u64, f64> + core::fmt::Debug {}

    impl<D> UnivariateDiscrete for D where
        D: DiscreteCDF<u64, f64> + Discrete<u64, f64> + core::fmt::Debug
    {
    }

    /// A drawn continuous distribution, together with the arguments at which
    /// this particular instance is known to break.
    struct Drawn {
        dist: Box<dyn Univariate>,
        /// True for an `x` where `cdf`/`pdf` returns NaN or panics instead of
        /// a probability or a density.
        bad_x: Box<dyn Fn(f64) -> bool>,
        /// True for a `p` where `inverse_cdf` panics, fails to terminate, or
        /// comes back somewhere other than the quantile.
        bad_p: Box<dyn Fn(f64) -> bool>,
    }

    impl Drawn {
        fn new(dist: impl Univariate + 'static) -> Self {
            Drawn {
                dist: Box::new(dist),
                bad_x: Box::new(|_| false),
                bad_p: Box::new(|_| false),
            }
        }

        fn bad_x(mut self, predicate: impl Fn(f64) -> bool + 'static) -> Self {
            self.bad_x = Box::new(predicate);
            self
        }

        fn bad_p(mut self, predicate: impl Fn(f64) -> bool + 'static) -> Self {
            self.bad_p = Box::new(predicate);
            self
        }
    }

    /// `Gamma::cdf` and its delegates hand `x * rate` straight to `gamma_lr`,
    /// which rejects a second argument outside `(0, inf)`, so a positive
    /// in-support `x` whose product with the rate under- or overflows panics.
    /// Pinned by `gamma::tests::cdf_panics_when_x_times_rate_underflows`.
    fn gamma_scaling_panics(rate: f64) -> impl Fn(f64) -> bool {
        move |x| x > 0.0 && x.is_finite() && (x * rate == 0.0 || !(x * rate).is_finite())
    }

    /// `inverse_cdf` zones excluded for the gamma family. Its Newton loop stops
    /// via `prec::convergence`, whose epsilon is 1e-9 *absolute*, and it gives
    /// up after 100 iterations without saying so:
    ///
    /// - a quantile below ~1e-8 is not resolved at all, pinned by
    ///   `gamma::tests::inverse_cdf_is_far_from_a_tiny_quantile`;
    /// - a tail probability below ~1e-100 exhausts the iteration budget,
    ///   pinned by `gamma::tests::inverse_cdf_exhausts_its_iterations`;
    /// - the bracket search walks `low` down by halves and hits the panic in
    ///   `gamma_scaling_panics` when the quantile is very small.
    ///
    /// The 1e-30 cut leaves seventy decades of margin over the iteration
    /// exhaustion, and the shape floor keeps the quantile clear of the
    /// absolute stop.
    fn gamma_inverse_is_broken(shape: f64) -> impl Fn(f64) -> bool {
        move |p| shape < 1.0 || p.min(1.0 - p) < 1e-30
    }

    /// `inverse_cdf` zones excluded for the families that invert through
    /// `inv_beta_reg` (`Beta`, `StudentsT`, `FisherSnedecor`). Its AS 64
    /// iteration stalls far from the quantile for a shape near or below 1, and
    /// for a deep tail probability it either panics or spins forever; all three
    /// are pinned in `crate::function::beta::tests`. The 1e-10 cut on the tail
    /// leaves thirty decades of margin over the non-terminating region, which
    /// the shrinker would otherwise walk straight back into.
    fn beta_inverse_is_broken(smallest_shape: f64) -> impl Fn(f64) -> bool {
        move |p| smallest_shape < 2.0 || p.min(1.0 - p) < 1e-10
    }

    /// Every univariate continuous distribution the crate ships that has both
    /// a cdf and a density. `Dirac` is left out: it is a point mass and
    /// implements no density.
    fn continuous_families() -> Vec<&'static str> {
        vec![
            "Beta",
            "Cauchy",
            "Chi",
            "ChiSquared",
            "Erlang",
            "Exp",
            "FisherSnedecor",
            "Gamma",
            "Gumbel",
            "InverseGamma",
            "Laplace",
            "Levy",
            "LogNormal",
            "Normal",
            "Pareto",
            "StudentsT",
            "Triangular",
            "Uniform",
            "Weibull",
        ]
    }

    /// Every univariate discrete distribution the crate ships over `u64`.
    /// `DiscreteUniform` is indexed by `i64` and so cannot join this list.
    fn discrete_families() -> Vec<&'static str> {
        vec![
            "Bernoulli",
            "Binomial",
            "Categorical",
            "Geometric",
            "Hypergeometric",
            "NegativeBinomial",
            "Poisson",
        ]
    }

    /// A magnitude drawn log-uniformly from `[10^lo, 10^hi]`. Scale and shape
    /// parameters span many decades, and a uniform float generator would spend
    /// nearly every draw in the top one.
    fn log_uniform(tc: &hegel::TestCase, lo: f64, hi: f64) -> f64 {
        10f64.powf(tc.draw(generators::floats::<f64>().min_value(lo).max_value(hi)))
    }

    /// A location parameter: zero some of the time, otherwise a signed
    /// log-uniform magnitude.
    fn location(tc: &hegel::TestCase, lo: f64, hi: f64) -> f64 {
        if tc.draw(generators::integers::<u8>().max_value(7)) == 0 {
            return 0.0;
        }
        let m = log_uniform(tc, lo, hi);
        if tc.draw(generators::booleans()) {
            m
        } else {
            -m
        }
    }

    /// Any real argument a caller might pass to `cdf`, `pdf` or `sf`: zero,
    /// either infinity, or a signed magnitude across the whole finite range.
    fn any_real(tc: &hegel::TestCase) -> f64 {
        // Weighted rather than a `one_of!`: seven of the ten indices fall
        // through to a finite magnitude, so the special values stay a minority
        // of the draws.
        match tc.draw(generators::integers::<u8>().max_value(9)) {
            0 => 0.0,
            1 => f64::NEG_INFINITY,
            2 => f64::INFINITY,
            _ => {
                let m = log_uniform(tc, -300.0, 300.0);
                if tc.draw(generators::booleans()) {
                    m
                } else {
                    -m
                }
            }
        }
    }

    /// A probability whose smaller tail is at least ~1e-10, and so inside the
    /// region where the crate's iterative inverses work at all. See
    /// `beta_inverse_is_broken` and `gamma_inverse_is_broken` for what lies
    /// below it, and the reproducers those name.
    fn well_conditioned_probability(tc: &hegel::TestCase) -> f64 {
        let tail = log_uniform(tc, -9.5, -0.302);
        if tc.draw(generators::booleans()) {
            tail
        } else {
            1.0 - tail
        }
    }

    /// Builds one instance of the named continuous family with parameter
    /// magnitudes log-uniform in `[10^lo, 10^hi]`.
    ///
    /// Several families come back carrying a `bad_x` or `bad_p` predicate.
    /// Those mark upstream defects, not contract boundaries; each names the
    /// ignored reproducer that pins it.
    fn build_continuous(tc: &hegel::TestCase, family: &str, lo: f64, hi: f64) -> Drawn {
        // Degrees of freedom that the constructor requires to be a positive
        // integer.
        let freedom = || {
            tc.draw(
                generators::integers::<u64>()
                    .min_value(1)
                    .max_value(1_000_000),
            )
        };
        match family {
            "Beta" => {
                let (a, b) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                Drawn::new(Beta::new(a, b).unwrap()).bad_p(beta_inverse_is_broken(a.min(b)))
            }
            "Cauchy" => {
                Drawn::new(Cauchy::new(location(tc, lo, hi), log_uniform(tc, lo, hi)).unwrap())
            }
            "Chi" => {
                let k = freedom();
                // `Chi` squares its argument before calling into the
                // incomplete gamma functions, so an `x` whose square under- or
                // overflows panics there; and its density forms
                // `x^(k-1) * exp(-x^2/2)`, which is `inf * 0` for a large `x`.
                // Pinned by `chi::tests::cdf_panics_for_a_tiny_argument`.
                Drawn::new(Chi::new(k).unwrap())
                    .bad_x(move |x| {
                        let squared = x * x / 2.0;
                        x > 0.0
                            && x.is_finite()
                            && (squared == 0.0
                                || !squared.is_finite()
                                || (x.powf(k as f64 - 1.0) * (-squared).exp()).is_nan())
                    })
                    .bad_p(gamma_inverse_is_broken(k as f64 / 2.0))
            }
            "ChiSquared" => {
                let k = log_uniform(tc, lo, hi);
                Drawn::new(ChiSquared::new(k).unwrap())
                    .bad_x(gamma_scaling_panics(0.5))
                    .bad_p(gamma_inverse_is_broken(k / 2.0))
            }
            "Erlang" => {
                let (shape, rate) = (freedom(), log_uniform(tc, lo, hi));
                Drawn::new(Erlang::new(shape, rate).unwrap())
                    .bad_x(gamma_scaling_panics(rate))
                    .bad_p(gamma_inverse_is_broken(shape as f64))
            }
            "Exp" => Drawn::new(Exp::new(log_uniform(tc, lo, hi)).unwrap()),
            "FisherSnedecor" => {
                let (d1, d2) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                // `FisherSnedecor::sf` forms `1 - d1*x/(d1*x + d2)` by
                // subtraction and loses the complement to rounding, so
                // `cdf + sf` reaches 1.98 for a small first freedom. Pinned by
                // `fisher_snedecor::tests::cdf_and_sf_do_not_sum_to_one`. The
                // loss only matters where that ratio is small and the first
                // freedom is below ~10.
                Drawn::new(FisherSnedecor::new(d1, d2).unwrap())
                    .bad_x(move |x| {
                        let ratio = d1 * x / (d1 * x + d2);
                        d1 < 10.0 && ratio > 0.0 && ratio < 1e-6
                    })
                    .bad_p(beta_inverse_is_broken(d1.min(d2)))
            }
            "Gamma" => {
                let (shape, rate) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                Drawn::new(Gamma::new(shape, rate).unwrap())
                    .bad_x(gamma_scaling_panics(rate))
                    .bad_p(gamma_inverse_is_broken(shape))
            }
            "Gumbel" => {
                let (mu, beta) = (location(tc, lo, hi), log_uniform(tc, lo, hi));
                // `Gumbel::pdf` multiplies `exp(z)` by `exp(-exp(z))` with
                // `z = (mu - x)/beta`; once `exp(z)` overflows the product is
                // `inf * 0`. Pinned by
                // `gumbel::tests::pdf_is_nan_in_the_left_tail`.
                Drawn::new(Gumbel::new(mu, beta).unwrap())
                    .bad_x(move |x| ((mu - x) / beta).exp().is_infinite())
            }
            "InverseGamma" => {
                let (shape, rate) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                // `InverseGamma::pdf` multiplies out
                // `rate^shape * x^(-shape-1) * exp(-rate/x) / Gamma(shape)`,
                // and any infinity meeting a zero among those four factors
                // gives NaN. Pinned by
                // `inverse_gamma::tests::pdf_is_nan_near_zero`. The predicate
                // mirrors the product to catch exactly those arguments.
                Drawn::new(InverseGamma::new(shape, rate).unwrap())
                    .bad_x(move |x| {
                        let product = if shape == 1.0 {
                            rate / (x * x) * (-rate / x).exp()
                        } else {
                            rate.powf(shape) * x.powf(-shape - 1.0) * (-rate / x).exp()
                                / crate::function::gamma::gamma(shape)
                        };
                        x > 0.0 && x.is_finite() && product.is_nan()
                    })
                    .bad_p(move |p| {
                        // `InverseGamma::pdf` multiplies the four factors
                        // above before dividing by `Gamma(shape)`, so the
                        // numerator overflows and the density comes back
                        // infinite for a shape of a few hundred. The Newton
                        // step in `newton_raphson_quantile` divides by it, so
                        // `inverse_cdf` then stops wherever the bracket search
                        // left it. Pinned by
                        // `inverse_gamma::tests::pdf_overflows_for_a_moderate_shape`.
                        let mode = rate / (shape + 1.0);
                        !InverseGamma::new(shape, rate)
                            .unwrap()
                            .pdf(mode)
                            .is_finite()
                            || gamma_inverse_is_broken(shape)(p)
                    })
            }
            "Laplace" => {
                Drawn::new(Laplace::new(location(tc, lo, hi), log_uniform(tc, lo, hi)).unwrap())
            }
            "Levy" => {
                let (mu, c) = (location(tc, lo, hi), log_uniform(tc, lo, hi));
                // `Levy::pdf` divides by `(x - mu)^1.5`, which underflows to
                // zero while its numerator has already underflowed. Pinned by
                // `levy::tests::pdf_is_nan_just_above_the_location`.
                Drawn::new(Levy::new(mu, c).unwrap())
                    .bad_x(move |x| x > mu && (x - mu).powf(1.5) == 0.0)
            }
            "LogNormal" => {
                Drawn::new(LogNormal::new(location(tc, lo, hi), log_uniform(tc, lo, hi)).unwrap())
            }
            "Normal" => {
                Drawn::new(Normal::new(location(tc, lo, hi), log_uniform(tc, lo, hi)).unwrap())
            }
            "Pareto" => {
                let (scale, shape) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                // `Pareto::pdf` forms `shape * scale^shape / x^(shape+1)`,
                // which is `inf / inf` once both powers overflow. Pinned by
                // `pareto::tests::pdf_is_nan_for_a_large_argument`.
                Drawn::new(Pareto::new(scale, shape).unwrap()).bad_x(move |x| {
                    x >= scale && (shape * scale.powf(shape) / x.powf(shape + 1.0)).is_nan()
                })
            }
            "StudentsT" => {
                let (mu, sigma, nu) = (
                    location(tc, lo, hi),
                    log_uniform(tc, lo, hi),
                    log_uniform(tc, lo, hi),
                );
                // The cdf routes through `beta_reg(nu/2, 1/2, .)`, so the
                // inverse inherits `inv_beta_reg`'s small-shape stall via the
                // degrees of freedom.
                Drawn::new(StudentsT::new(mu, sigma, nu).unwrap()).bad_p(beta_inverse_is_broken(nu))
            }
            "Triangular" => {
                let min = location(tc, lo, hi);
                let width = log_uniform(tc, lo, hi);
                let offset = tc.draw(generators::floats::<f64>().min_value(0.0).max_value(1.0));
                tc.assume(min + width > min && (min + width).is_finite());
                let mode = (min + width * offset).clamp(min, min + width);
                Drawn::new(Triangular::new(min, min + width, mode).unwrap())
            }
            "Uniform" => {
                let min = location(tc, lo, hi);
                let width = log_uniform(tc, lo, hi);
                tc.assume(min + width > min && (min + width).is_finite());
                Drawn::new(Uniform::new(min, min + width).unwrap())
            }
            _ => {
                let (shape, scale) = (log_uniform(tc, lo, hi), log_uniform(tc, lo, hi));
                // `Weibull::new` precomputes `scale^-shape`; once that
                // overflows or underflows, the cdf and the density are NaN
                // across the whole support. The density separately goes NaN
                // for a large `x`, where `(x/scale)^(shape-1)` overflows while
                // `exp(-x^shape / scale^shape)` has underflowed. Pinned by
                // `weibull::tests::cdf_is_nan_when_the_precomputed_power_overflows`.
                let precomputed = scale.powf(-shape);
                Drawn::new(Weibull::new(shape, scale).unwrap()).bad_x(move |x| {
                    !precomputed.is_normal()
                        || (x >= 0.0
                            && x.is_finite()
                            && (shape
                                * (x / scale).powf(shape - 1.0)
                                * (-(x.powf(shape)) * precomputed).exp()
                                / scale)
                                .is_nan())
                })
            }
        }
    }

    fn draw_continuous(tc: &hegel::TestCase, lo: f64, hi: f64) -> Drawn {
        let family = tc.draw(generators::sampled_from(continuous_families()));
        build_continuous(tc, family, lo, hi)
    }

    fn draw_discrete(tc: &hegel::TestCase) -> Box<dyn UnivariateDiscrete> {
        // Counts are held to a few hundred so the properties below can walk a
        // whole support; that bounds the tests' runtime, not any contract.
        let p = || tc.draw(generators::floats::<f64>().min_value(0.0).max_value(1.0));
        let n = || tc.draw(generators::integers::<u64>().max_value(300));
        match tc.draw(generators::sampled_from(discrete_families())) {
            "Bernoulli" => Box::new(Bernoulli::new(p()).unwrap()),
            "Binomial" => Box::new(Binomial::new(p(), n()).unwrap()),
            "Categorical" => {
                let mass: Vec<f64> = tc.draw(
                    generators::vecs(generators::floats::<f64>().min_value(0.0).max_value(1e6))
                        .min_size(1)
                        .max_size(20),
                );
                tc.assume(mass.iter().sum::<f64>() > 0.0);
                Box::new(Categorical::new(&mass).unwrap())
            }
            "Geometric" => {
                let p = p();
                tc.assume(p > 0.0);
                Box::new(Geometric::new(p).unwrap())
            }
            "Hypergeometric" => {
                let population = n();
                let successes = tc.draw(generators::integers::<u64>().max_value(population));
                let draws = tc.draw(generators::integers::<u64>().max_value(population));
                Box::new(Hypergeometric::new(population, successes, draws).unwrap())
            }
            "NegativeBinomial" => {
                let p = p();
                // A certain success gives a NaN mass rather than 1; pinned by
                // `negative_binomial::tests::pmf_is_nan_for_a_certain_success`.
                tc.assume(p < 1.0);
                Box::new(NegativeBinomial::new(log_uniform(tc, -2.0, 2.0), p).unwrap())
            }
            _ => Box::new(Poisson::new(log_uniform(tc, -3.0, 2.0)).unwrap()),
        }
    }

    // ----------------------------------------------------------------------
    // Contracts shared by every continuous distribution.
    // ----------------------------------------------------------------------

    /// A cdf is a probability. Exact: a value outside `[0, 1]`, or a NaN, is
    /// not a rounding error.
    #[hegel::test]
    fn cdf_lies_in_the_unit_interval(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -8.0, 8.0);
        let x = any_real(&tc);
        tc.assume(!(d.bad_x)(x));
        let c = d.dist.cdf(x);
        assert!(
            (0.0..=1.0).contains(&c),
            "{:?}: cdf({x:e}) = {c:.17e}",
            d.dist
        );
    }

    /// A cdf is non-decreasing, for any two reals inside or outside the
    /// support.
    ///
    /// The allowance is sixteen ulps of a value in `[0, 1]`: `StudentsT::cdf`
    /// rounds non-monotonically near its location, by one ulp at ordinary
    /// degrees of freedom and up to eight at 1e-8, pinned by
    /// `students_t::tests::cdf_decreases_near_the_location`. Anything larger
    /// is a real ordering failure.
    #[hegel::test]
    fn cdf_is_nondecreasing(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -8.0, 8.0);
        let x1 = any_real(&tc);
        let x2 = any_real(&tc);
        tc.assume(!(d.bad_x)(x1) && !(d.bad_x)(x2));
        let (lo, hi) = if x1 <= x2 { (x1, x2) } else { (x2, x1) };
        let (c_lo, c_hi) = (d.dist.cdf(lo), d.dist.cdf(hi));
        assert!(
            c_lo <= c_hi + 16.0 * f64::EPSILON,
            "{:?}: cdf({lo:e}) = {c_lo:.17e} > cdf({hi:e}) = {c_hi:.17e}",
            d.dist
        );
    }

    /// All the mass lies inside the support: the cdf is 0 at negative infinity
    /// and at a finite lower bound, and 1 at positive infinity and at a finite
    /// upper bound. A continuous distribution puts no atom on an endpoint, so
    /// these are exact.
    #[hegel::test]
    fn cdf_reaches_zero_and_one_at_the_support_bounds(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -8.0, 8.0);
        let (lo, hi) = (d.dist.min(), d.dist.max());
        tc.assume(!(d.bad_x)(lo) && !(d.bad_x)(hi));
        assert_eq!(
            d.dist.cdf(f64::NEG_INFINITY),
            0.0,
            "{:?}: cdf(-inf)",
            d.dist
        );
        assert_eq!(d.dist.cdf(f64::INFINITY), 1.0, "{:?}: cdf(inf)", d.dist);
        if lo.is_finite() {
            assert_eq!(d.dist.cdf(lo), 0.0, "{:?}: cdf(min = {lo:e})", d.dist);
        }
        if hi.is_finite() {
            assert_eq!(d.dist.cdf(hi), 1.0, "{:?}: cdf(max = {hi:e})", d.dist);
        }
    }

    /// A density is non-negative and never NaN. Infinity is allowed: several
    /// families have a genuine pole at a support boundary, such as `Beta` with
    /// a shape below 1 at zero.
    #[hegel::test]
    fn pdf_is_nonnegative(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -8.0, 8.0);
        let x = any_real(&tc);
        tc.assume(!(d.bad_x)(x));
        let density = d.dist.pdf(x);
        assert!(density >= 0.0, "{:?}: pdf({x:e}) = {density:e}", d.dist);
    }

    /// `ln_pdf` is the log of `pdf`. Most families compute the two by
    /// different routes — the log form sums logarithms where the direct form
    /// multiplies — so this relates two independent computations.
    ///
    /// The density is required to be normal: once `pdf` underflows into the
    /// subnormal range its logarithm reports the subnormal grid rather than
    /// the density, which limits the comparison, not `ln_pdf`.
    ///
    /// Parameter magnitudes are held to `[1e-3, 1e3]` and the tolerance to
    /// 1e-10 absolute. Both are set by the `ln_gamma` cancellation in the
    /// `StudentsT` and `FisherSnedecor` densities, which grows like
    /// `eps * nu * ln(nu)` and would need a 1e-6 allowance at the top of the
    /// range the exact properties above use.
    ///
    /// `x` is kept within three decades of that parameter range so that the
    /// density is usually representable; over the full float range nearly
    /// every draw lands where `pdf` has underflowed to zero, which this
    /// property has nothing to say about.
    #[hegel::test]
    fn ln_pdf_is_the_logarithm_of_pdf(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -3.0, 3.0);
        let magnitude = log_uniform(&tc, -6.0, 6.0);
        let x = if tc.draw(generators::booleans()) {
            magnitude
        } else {
            -magnitude
        };
        tc.assume(!(d.bad_x)(x));
        let density = d.dist.pdf(x);
        tc.assume(density.is_finite() && density >= f64::MIN_POSITIVE);
        let expected = density.ln();
        prec::assert_abs_diff_eq!(
            d.dist.ln_pdf(x),
            expected,
            epsilon = 1e-10 + 1e-12 * expected.abs()
        );
    }

    /// `sf(x) + cdf(x) = 1`. The trait supplies `sf` as `1 - cdf`, but many
    /// families override it with an independently derived formula, and those
    /// are what this checks.
    ///
    /// Tolerance: both terms lie in `[0, 1]` and the crate targets 1e-14
    /// relative accuracy (`prec::DEFAULT_RELATIVE_ACC`), so 1e-11 absolute is
    /// three decades of headroom over two such terms.
    #[hegel::test]
    fn sf_is_the_complement_of_cdf(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, -3.0, 3.0);
        let x = any_real(&tc);
        tc.assume(!(d.bad_x)(x));
        prec::assert_abs_diff_eq!(d.dist.cdf(x) + d.dist.sf(x), 1.0, epsilon = 1e-11);
    }

    /// A quantile lies in the support. Exact: `inverse_cdf` returning a point
    /// the distribution assigns no mass to is a contract violation, not
    /// rounding.
    ///
    /// Parameter magnitudes are held in `[2.5, 1000]` so that every shape
    /// clears the floors in `beta_inverse_is_broken` and
    /// `gamma_inverse_is_broken` by construction rather than by rejection.
    #[hegel::test]
    fn inverse_cdf_lies_within_the_support(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, 0.4, 3.0);
        let p = well_conditioned_probability(&tc);
        tc.assume(!(d.bad_p)(p));
        let q = d.dist.inverse_cdf(p);
        assert!(
            q >= d.dist.min() && q <= d.dist.max(),
            "{:?}: inverse_cdf({p:e}) = {q:e}, support is [{:e}, {:e}]",
            d.dist,
            d.dist.min(),
            d.dist.max()
        );
    }

    /// `inverse_cdf` is non-decreasing in `p`, since it inverts a
    /// non-decreasing cdf. No oracle and no tolerance: two calls on the same
    /// instance must come back in order.
    ///
    /// A quantile below 1e-8 is skipped for the gamma family: its Newton loop
    /// stops on a step of 1e-9 *absolute*, so below that the answer carries no
    /// information about `p` at all. See
    /// `gamma::tests::inverse_cdf_is_far_from_a_tiny_quantile`.
    #[hegel::test]
    fn inverse_cdf_is_nondecreasing_in_p(tc: hegel::TestCase) {
        let d = draw_continuous(&tc, 0.4, 3.0);
        let p1 = well_conditioned_probability(&tc);
        let p2 = well_conditioned_probability(&tc);
        tc.assume(!(d.bad_p)(p1) && !(d.bad_p)(p2));
        let (lo, hi) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        let (q_lo, q_hi) = (d.dist.inverse_cdf(lo), d.dist.inverse_cdf(hi));
        tc.assume(q_lo.abs() >= 1e-8 && q_hi.abs() >= 1e-8);
        assert!(
            q_lo <= q_hi,
            "{:?}: inverse_cdf({lo:e}) = {q_lo:e} > inverse_cdf({hi:e}) = {q_hi:e}",
            d.dist
        );
    }

    // ----------------------------------------------------------------------
    // Contracts shared by every discrete distribution.
    // ----------------------------------------------------------------------

    /// The pmf summed up to `k` equals `cdf(k)` at every `k`, and the whole
    /// support sums to 1.
    ///
    /// The 1e-10 tolerance follows `density_util::check_sum_pmf_is_cdf`, the
    /// crate's own version of this check. `Poisson` and `NegativeBinomial` have
    /// unbounded support, so the walk stops once the running total is within
    /// 1e-12 of 1.
    #[hegel::test]
    fn discrete_pmf_sums_to_cdf(tc: hegel::TestCase) {
        let d = draw_discrete(&tc);
        let mut sum = 0.0;
        let limit = d.max().min(2_000);
        for k in d.min()..=limit {
            let mass = d.pmf(k);
            // Four ulps of a value in [0, 1]: `NegativeBinomial::pmf` rounds
            // just above 1 for a near-certain success, pinned by
            // `negative_binomial::tests::pmf_exceeds_one_for_a_near_certain_success`.
            assert!(
                (0.0..=1.0 + 4.0 * f64::EPSILON).contains(&mass),
                "{d:?}: pmf({k}) = {mass:.17e}"
            );
            sum += mass;
            prec::assert_abs_diff_eq!(sum, d.cdf(k), epsilon = 1e-10);
            if sum > 1.0 - 1e-12 {
                return;
            }
        }
        assert!(
            d.max() > limit || prec::abs_diff_eq!(sum, 1.0, epsilon = 1e-10),
            "{d:?}: the pmf over the whole support sums to {sum:.17e}"
        );
    }

    /// A discrete cdf is a probability.
    ///
    /// `Hypergeometric::cdf` sums its pmf and lands above 1 by up to 8.2e-13
    /// within the population sizes drawn here, pinned by
    /// `hypergeometric::tests::cdf_exceeds_one`; the 1e-11 allowance clears
    /// that while still catching a value that is negative, NaN, or grossly
    /// above 1.
    #[hegel::test]
    fn discrete_cdf_lies_in_the_unit_interval(tc: hegel::TestCase) {
        let d = draw_discrete(&tc);
        let k = tc.draw(generators::integers::<u64>().max_value(2_000));
        let c = d.cdf(k);
        assert!(
            (0.0..=1.0 + 1e-11).contains(&c),
            "{d:?}: cdf({k}) = {c:.17e}"
        );
    }

    /// A discrete cdf is non-decreasing. Same 1e-11 allowance and same cause:
    /// once `Hypergeometric::cdf` has overshot 1 it comes back down at the
    /// next point.
    #[hegel::test]
    fn discrete_cdf_is_nondecreasing(tc: hegel::TestCase) {
        let d = draw_discrete(&tc);
        let k1 = tc.draw(generators::integers::<u64>().max_value(2_000));
        let k2 = tc.draw(generators::integers::<u64>().max_value(2_000));
        let (lo, hi) = if k1 <= k2 { (k1, k2) } else { (k2, k1) };
        let (c_lo, c_hi) = (d.cdf(lo), d.cdf(hi));
        assert!(
            c_lo <= c_hi + 1e-11,
            "{d:?}: cdf({lo}) = {c_lo:.17e} > cdf({hi}) = {c_hi:.17e}"
        );
    }

    /// `ln_pmf` is the log of `pmf` wherever the mass is a normal float. The
    /// two are computed by different routes — sums of `ln_factorial` against
    /// products of `binomial` — so this relates them.
    #[hegel::test]
    fn discrete_ln_pmf_is_the_logarithm_of_pmf(tc: hegel::TestCase) {
        let d = draw_discrete(&tc);
        let k = tc.draw(generators::integers::<u64>().max_value(2_000));
        let mass = d.pmf(k);
        tc.assume(mass.is_finite() && mass >= f64::MIN_POSITIVE);
        let expected = mass.ln();
        prec::assert_abs_diff_eq!(
            d.ln_pmf(k),
            expected,
            epsilon = 1e-8 + 1e-12 * expected.abs()
        );
    }

    // ----------------------------------------------------------------------
    // Relationships between families. These are model-free oracles: the crate
    // ships both sides, standard theory says they are the same distribution,
    // so they must agree to their own accuracy.
    // ----------------------------------------------------------------------

    /// A Bernoulli trial is a binomial with one trial.
    #[hegel::test]
    fn bernoulli_agrees_with_a_one_trial_binomial(tc: hegel::TestCase) {
        let p = tc.draw(generators::floats::<f64>().min_value(0.0).max_value(1.0));
        let k = tc.draw(generators::integers::<u64>().max_value(4));
        assert_eq!(
            Bernoulli::new(p).unwrap().pmf(k),
            Binomial::new(p, 1).unwrap().pmf(k),
            "pmf at {k}, p = {p}"
        );
    }

    /// The exponential distribution is a gamma with unit shape. `Exp` uses the
    /// closed form `1 - exp(-rate*x)` while `Gamma` goes through the
    /// regularized incomplete gamma function, so the two share no code.
    ///
    /// Tolerance: 1e-11 absolute on a cdf value, three decades above the
    /// crate's 1e-14 relative target for two such terms.
    #[hegel::test]
    fn exponential_agrees_with_a_unit_shape_gamma(tc: hegel::TestCase) {
        let rate = log_uniform(&tc, -3.0, 3.0);
        let x = log_uniform(&tc, -3.0, 3.0);
        prec::assert_abs_diff_eq!(
            Exp::new(rate).unwrap().cdf(x),
            Gamma::new(1.0, rate).unwrap().cdf(x),
            epsilon = 1e-11
        );
    }

    /// Chi-squared with `k` degrees of freedom is a gamma with shape `k/2` and
    /// rate `1/2`. `ChiSquared` delegates to `Gamma` for the cdf but writes out
    /// its own density.
    #[hegel::test]
    fn chi_squared_agrees_with_the_corresponding_gamma(tc: hegel::TestCase) {
        let k = log_uniform(&tc, -3.0, 3.0);
        let x = log_uniform(&tc, -3.0, 3.0);
        prec::assert_relative_eq!(
            ChiSquared::new(k).unwrap().pdf(x),
            Gamma::new(k / 2.0, 0.5).unwrap().pdf(x),
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    /// Chi with `k` degrees of freedom is the square root of a chi-squared
    /// with `k`, so `Chi(k).cdf(x) = ChiSquared(k).cdf(x*x)`.
    ///
    /// `x` is kept where `x*x` neither underflows nor overflows; that bounds
    /// the oracle's argument, not `Chi`'s domain — see
    /// `chi::tests::cdf_panics_for_a_tiny_argument`.
    #[hegel::test]
    fn chi_agrees_with_the_square_root_of_a_chi_squared(tc: hegel::TestCase) {
        let k = tc.draw(generators::integers::<u64>().min_value(1).max_value(1_000));
        let x = log_uniform(&tc, -100.0, 100.0);
        prec::assert_abs_diff_eq!(
            Chi::new(k).unwrap().cdf(x),
            ChiSquared::new(k as f64).unwrap().cdf(x * x),
            epsilon = 1e-11
        );
    }

    /// An Erlang distribution is a gamma with integer shape, and `Erlang`
    /// holds a `Gamma` internally, so the two must agree exactly.
    #[hegel::test]
    fn erlang_agrees_with_an_integer_shape_gamma(tc: hegel::TestCase) {
        let shape = tc.draw(generators::integers::<u64>().min_value(1).max_value(1_000));
        let rate = log_uniform(&tc, -3.0, 3.0);
        let x = log_uniform(&tc, -3.0, 3.0);
        assert_eq!(
            Erlang::new(shape, rate).unwrap().cdf(x),
            Gamma::new(shape as f64, rate).unwrap().cdf(x),
            "shape = {shape}, rate = {rate:e}, x = {x:e}"
        );
    }

    /// `Beta(1, 1)` is the standard uniform distribution. The two cdfs share
    /// no code: `Beta` goes through the regularized incomplete beta function
    /// and `Uniform` interpolates linearly.
    #[hegel::test]
    fn beta_with_unit_shapes_agrees_with_the_standard_uniform(tc: hegel::TestCase) {
        let x = tc.draw(generators::floats::<f64>().min_value(0.0).max_value(1.0));
        prec::assert_abs_diff_eq!(
            Beta::new(1.0, 1.0).unwrap().cdf(x),
            Uniform::new(0.0, 1.0).unwrap().cdf(x),
            epsilon = 1e-14
        );
    }

    /// A Weibull with unit shape is an exponential with rate `1/scale`.
    #[hegel::test]
    fn weibull_with_unit_shape_agrees_with_an_exponential(tc: hegel::TestCase) {
        let scale = log_uniform(&tc, -3.0, 3.0);
        let x = log_uniform(&tc, -3.0, 3.0);
        prec::assert_abs_diff_eq!(
            Weibull::new(1.0, scale).unwrap().cdf(x),
            Exp::new(1.0 / scale).unwrap().cdf(x),
            epsilon = 1e-11
        );
    }

    /// A Student's t with many degrees of freedom is a standard normal. At
    /// `nu = 1e6` the two cdfs differ by O(1/nu), so 1e-5 is what the limit
    /// itself allows; the property is that `StudentsT` converges to the normal
    /// and not to something else.
    #[hegel::test]
    fn students_t_approaches_the_normal_for_large_freedom(tc: hegel::TestCase) {
        let x = tc.draw(generators::floats::<f64>().min_value(-8.0).max_value(8.0));
        prec::assert_abs_diff_eq!(
            StudentsT::new(0.0, 1.0, 1e6).unwrap().cdf(x),
            Normal::new(0.0, 1.0).unwrap().cdf(x),
            epsilon = 1e-5
        );
    }

    /// A log-normal variable is the exponential of a normal one, so
    /// `LogNormal(mu, sigma).cdf(x) = Normal(mu, sigma).cdf(ln x)`.
    #[hegel::test]
    fn log_normal_agrees_with_the_normal_of_the_logarithm(tc: hegel::TestCase) {
        let mu = location(&tc, -3.0, 3.0);
        let sigma = log_uniform(&tc, -3.0, 3.0);
        let x = log_uniform(&tc, -300.0, 300.0);
        prec::assert_abs_diff_eq!(
            LogNormal::new(mu, sigma).unwrap().cdf(x),
            Normal::new(mu, sigma).unwrap().cdf(x.ln()),
            epsilon = 1e-11
        );
    }

    /// A geometric distribution is a negative binomial waiting for one
    /// success, shifted by one: `Geometric` counts the trial the success occurs
    /// on where `NegativeBinomial` counts the failures before it.
    ///
    /// `p = 1` is excluded, see
    /// `negative_binomial::tests::pmf_is_nan_for_a_certain_success`.
    #[hegel::test]
    fn geometric_agrees_with_a_shifted_negative_binomial(tc: hegel::TestCase) {
        let p = tc.draw(
            generators::floats::<f64>()
                .min_value(0.0)
                .max_value(1.0)
                .exclude_min(true)
                .exclude_max(true),
        );
        // `k` stops at 300 so the mass stays a normal float for most of the
        // `p` range; a relative comparison inside the subnormals measures the
        // subnormal grid rather than either pmf.
        let k = tc.draw(generators::integers::<u64>().min_value(1).max_value(300));
        let mass = Geometric::new(p).unwrap().pmf(k);
        tc.assume(mass >= f64::MIN_POSITIVE);
        prec::assert_relative_eq!(
            mass,
            NegativeBinomial::new(1.0, p).unwrap().pmf(k - 1),
            epsilon = 0.0,
            max_relative = 1e-12
        );
    }

    // ----------------------------------------------------------------------
    // Sampling.
    // ----------------------------------------------------------------------

    /// A drawn sample lies inside the support. The RNG comes from hegel rather
    /// than a fixed seed, so a failing draw shrinks to a minimal sequence of
    /// random decisions instead of an opaque seed.
    ///
    /// True randomness is used because several samplers reject and retry
    /// (`Binomial`'s BTPE algorithm, the ziggurat normal), and artificial
    /// randomness makes those loops run for a very long time.
    #[cfg(feature = "rand")]
    #[hegel::test]
    fn a_sample_lies_within_the_support(tc: hegel::TestCase) {
        use hegel::extras::rand as rand_gs;

        let mut families = continuous_families();
        // `Gumbel`'s sampler computes `ln(-x)` where it means `ln(-ln(x))`, so
        // every draw is NaN; pinned by
        // `gumbel::tests::every_sample_is_nan`.
        families.retain(|family| *family != "Gumbel");
        let family = tc.draw(generators::sampled_from(families));
        // Magnitudes are kept to [0.1, 10]: `FisherSnedecor`'s sampler is a
        // ratio of two gamma samples, both of which underflow to zero for tiny
        // degrees of freedom, giving a NaN pinned by
        // `fisher_snedecor::tests::sample_is_nan_for_a_tiny_freedom`.
        let d = build_continuous(&tc, family, -1.0, 1.0);
        let mut rng = tc.draw(rand_gs::randoms().use_true_random(true));
        let sample = d.dist.sample_one(&mut rng);
        assert!(
            sample >= d.dist.min() && sample <= d.dist.max(),
            "{:?}: sampled {sample:e}, support is [{:e}, {:e}]",
            d.dist,
            d.dist.min(),
            d.dist.max()
        );
    }

    // ----------------------------------------------------------------------
    // Known failures, pinned deterministically. Each is ignored so the suite
    // stays green; the properties above name them where they narrow a
    // generator to stay clear.
    // ----------------------------------------------------------------------

    /// KNOWN BUG (open PR #458 addresses this): `try_inverse_cdf` panics instead of reporting
    /// failure. Nothing in the crate overrides the trait's default
    /// implementation, which is `Ok(self.inverse_cdf(p))`, and the specialized
    /// `inverse_cdf` implementations panic on a `p` outside `[0, 1]`. So
    /// `InverseCdfError::ArgumentOutOfRange` is never returned by any
    /// distribution and the fallible variant offers nothing over the panicking
    /// one.
    #[test]
    #[ignore = "known bug: try_inverse_cdf panics instead of returning Err"]
    fn try_inverse_cdf_panics_instead_of_reporting_an_error() {
        let d = Chi::new(1).unwrap();
        assert!(d.try_inverse_cdf(f64::INFINITY).is_err());
    }
}

#[cfg(test)]
mod test {

    #[cfg(feature = "std")]
    #[test]
    fn test_integer_bisection() {
        use super::integral_bisection_search;
        fn search(z: usize, data: &[usize]) -> Option<usize> {
            integral_bisection_search(|idx: &usize| data[*idx], z, 0, data.len() - 1)
        }

        let needle = 3;
        let data = (0..5)
            .map(|n| if n >= needle { n + 1 } else { n })
            .collect::<Vec<_>>();

        for i in 0..(data.len()) {
            assert_eq!(search(data[i], &data), Some(i),)
        }
        {
            let infimum = search(needle, &data);
            let found_element = search(needle + 1, &data); // 4 > needle && member of range
            assert_eq!(found_element, Some(needle));
            assert_eq!(infimum, found_element)
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_newton_raphson_quantile() {
        use super::newton_raphson_quantile;
        // Exponential(λ) has the closed-form quantile -ln(1 - p)/λ, so it is an
        // exact oracle for the shared solver across both tails.
        for lambda in [0.5f64, 1.0, 4.0] {
            let cdf = |x: f64| -(-lambda * x).exp_m1();
            let sf = |x: f64| (-lambda * x).exp();
            let pdf = |x: f64| lambda * (-lambda * x).exp();
            for p in [
                1e-12,
                1e-8,
                1e-4,
                1e-2,
                0.25,
                0.5,
                0.75,
                0.99,
                1.0 - 1e-6,
                1.0 - 1e-12,
            ] {
                let got = newton_raphson_quantile(p, cdf, sf, pdf);
                let want = -(-p).ln_1p() / lambda; // -ln(1 - p)/λ, accurate in both tails
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "Exp({lambda}).inverse_cdf({p}) = {got}, want {want}"
                );
            }
        }
    }

    pub mod boiler_tests {
        use crate::distribution::{Beta, BetaError};
        use crate::statistics::*;

        testing_boiler!(shape_a: f64, shape_b: f64; Beta; BetaError);

        #[test]
        fn create_ok_success() {
            let b = create_ok(0.8, 1.2);
            assert_eq!(b.shape_a(), 0.8);
            assert_eq!(b.shape_b(), 1.2);
        }

        #[test]
        #[should_panic]
        fn create_err_failure() {
            create_err(0.8, 1.2);
        }

        #[test]
        fn create_err_success() {
            let err = create_err(-0.5, 1.2);
            assert_eq!(err, BetaError::ShapeAInvalid);
        }

        #[test]
        #[should_panic]
        fn create_ok_failure() {
            create_ok(-0.5, 1.2);
        }

        #[test]
        fn test_exact_success() {
            test_exact(1.5, 1.5, 0.5, |dist| dist.mode().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_exact_failure() {
            test_exact(1.2, 1.4, 0.333333333333, |dist| dist.mode().unwrap());
        }

        #[test]
        fn test_relative_success() {
            test_relative(1.2, 1.4, 0.333333333333, |dist| dist.mode().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_relative_failure() {
            test_relative(1.2, 1.4, 0.333, |dist| dist.mode().unwrap());
        }

        #[test]
        fn test_absolute_success() {
            test_absolute(1.2, 1.4, 0.333333333333, 1e-12, |dist| dist.mode().unwrap());
        }

        #[test]
        #[should_panic]
        fn test_absolute_failure() {
            test_absolute(1.2, 1.4, 0.333333333333, 1e-15, |dist| dist.mode().unwrap());
        }

        #[test]
        fn test_create_err_success() {
            test_create_err(0.0, 0.5, BetaError::ShapeAInvalid);
        }

        #[test]
        #[should_panic]
        fn test_create_err_failure() {
            test_create_err(0.0, 0.5, BetaError::ShapeBInvalid);
        }

        #[test]
        fn test_is_nan_success() {
            // Not sure that any Beta API can return a NaN, so we force the issue
            test_is_nan(0.8, 1.2, |_| f64::NAN);
        }

        #[test]
        #[should_panic]
        fn test_is_nan_failure() {
            test_is_nan(0.8, 1.2, |dist| dist.mean().unwrap());
        }

        #[test]
        fn test_is_none_success() {
            test_none(0.5, 1.2, |dist| dist.mode());
        }

        #[test]
        #[should_panic]
        fn test_is_none_failure() {
            test_none(0.8, 1.2, |dist| dist.mean());
        }
    }
}
