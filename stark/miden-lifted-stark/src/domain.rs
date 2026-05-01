//! Domain hierarchy: subgroup → coset → lifted domain.
//!
//! This module hosts three concrete types representing a STARK's evaluation
//! domains, each adding one level of structure:
//!
//! 1. [`TwoAdicSubgroup<F>`] — the bare multiplicative subgroup `H = ⟨ω⟩ ⊂ F^*` of order
//!    `2^log_size`. **Single home of `F::two_adic_generator(...)` in this crate.**
//!
//! 2. [`TwoAdicCoset<F>`] — a coset `s · H` of a [`TwoAdicSubgroup`], carrying a multiplicative
//!    shift `s ∈ F`.
//!
//! 3. [`LiftedDomain<F>`] — the STARK-protocol object: an LDE coset `g^r · K` together with a
//!    smaller trace subgroup `H ⊆ K` and a lift ratio `r` relative to the max coset `g · K_max`.
//!    **Single home of `F::GENERATOR` in this crate** (used in [`LiftedDomain::canonical`] and
//!    [`LiftedDomain::sub_domain`] to compute the canonical LDE coset shift).
//!
//! [`TwoAdicSubgroup`] and [`TwoAdicCoset`] both implement the [`Coset`]
//! trait, which captures the shared `(log_size, shift, generator)` interface
//! and provides default bodies for `size`, `point_at`, `points`,
//! `bit_reversed_points`, `vanishing_at`, `contains`, `generator_inverse`.
//! [`LiftedDomain`] does *not* implement [`Coset`] — it composes a trace
//! subgroup and an LDE coset, and exposing a single coset interface would
//! silently pick one of two distinct vanishing polynomials. Callers say
//! `domain.trace_subgroup()` or `domain.lde_coset()` to disambiguate.

use alloc::vec::Vec;
use core::marker::PhantomData;

use miden_stark_transcript::Channel;
use p3_field::{ExtensionField, TwoAdicField, batch_multiplicative_inverse};
use p3_maybe_rayon::prelude::*;
use p3_util::reverse_slice_index_bits;

use crate::selectors::Selectors;

// ============================================================================
// Canonical shift
// ============================================================================

/// The canonical multiplicative coset shift for an LDE coset of order
/// `2^log_lde_order`, defined as
///
/// ```text
/// canonical_shift = F::GENERATOR^(2^(F::TWO_ADICITY − log_lde_order))
/// ```
///
/// This is the **single source of `F::GENERATOR`** in this crate. Its value
/// depends only on the field and the LDE order — it is invariant of any
/// surrounding batch context. Two cosets of the same order, in different
/// batches, have the same canonical shift.
#[inline]
fn canonical_shift<F: TwoAdicField>(log_lde_order: u8) -> F {
    F::GENERATOR.exp_power_of_2(F::TWO_ADICITY - log_lde_order as usize)
}

// ============================================================================
// Coset trait
// ============================================================================

/// Shared interface for two-adic coset-like multiplicative domains.
///
/// A coset `s · H` is parameterised by:
/// - `log_size`: log₂ of the order `|H|`,
/// - `shift`: the multiplicative offset `s ∈ F` (`F::ONE` for a plain subgroup),
/// - `generator`: a primitive `2^log_size`-th root of unity.
///
/// Implemented by [`TwoAdicSubgroup`] (with `shift = F::ONE`) and
/// [`TwoAdicCoset`]. The default bodies for `point_at`, `points`,
/// `bit_reversed_points`, `vanishing_at`, `contains`, `size`, and
/// `generator_inverse` are written once here in terms of the three required
/// methods — for a subgroup, the shift collapses to `F::ONE` and the formulas
/// reduce to the unshifted case.
///
/// [`LiftedDomain`] deliberately does **not** implement this trait: it composes
/// a trace subgroup and an LDE coset, each with its own vanishing polynomial,
/// and exposing a single `Coset` interface would force one to silently win.
/// Callers reach into the parts: `domain.trace_subgroup()` or
/// `domain.lde_coset()`.
pub trait Coset<F: TwoAdicField>: Sized {
    /// Log₂ of the domain order.
    fn log_size(&self) -> u8;

    /// Multiplicative shift `s` (`F::ONE` for a subgroup).
    fn shift(&self) -> F;

    /// Primitive `2^log_size`-th root of unity.
    fn generator(&self) -> F;

    /// Cached inverse of [`Self::shift`]. The default body recomputes it on
    /// every call; implementations should override either with a free
    /// constant (`F::ONE` for a subgroup) or by returning a value computed
    /// once at construction time. `vanishing_at` and `contains` route through
    /// this method so multiple invocations on the same coset don't redo the
    /// inversion.
    #[inline]
    fn shift_inverse(&self) -> F {
        self.shift().inverse()
    }

    /// Domain order: `2^log_size`.
    #[inline]
    fn size(&self) -> usize {
        1 << self.log_size() as usize
    }

    /// Inverse of the generator. Convenient for FRI twiddle factors and
    /// last-row Lagrange denominators.
    #[inline]
    fn generator_inverse(&self) -> F {
        self.generator().inverse()
    }

    /// The `i`-th point in natural order: `s · ωⁱ`.
    #[inline]
    fn point_at(&self, i: u64) -> F {
        self.shift() * self.generator().exp_u64(i)
    }

    /// All points in natural order, length `2^log_size`.
    ///
    /// Single-pass iteration via `shifted_powers(self.shift())`: produces
    /// `s, s·ω, s·ω², …` directly without an intermediate "unshifted then map"
    /// step.
    fn points(&self) -> Vec<F> {
        self.generator().shifted_powers(self.shift()).take(self.size()).collect()
    }

    /// All points in bit-reversed order.
    fn bit_reversed_points(&self) -> Vec<F> {
        let mut pts = self.points();
        reverse_slice_index_bits(&mut pts);
        pts
    }

    /// Vanishing polynomial of the domain at `z`: `(z/s)^|H| − 1`.
    ///
    /// For a subgroup (`s = 1`) this reduces to `z^|H| − 1`.
    #[inline]
    fn vanishing_at<EF: ExtensionField<F>>(&self, z: EF) -> EF {
        (z * self.shift_inverse()).exp_power_of_2(self.log_size() as usize) - EF::ONE
    }

    /// Membership test: returns `true` iff `(z/s)^|H| == 1`.
    #[inline]
    fn contains<EF: ExtensionField<F>>(&self, z: EF) -> bool {
        (z * self.shift_inverse()).exp_power_of_2(self.log_size() as usize) == EF::ONE
    }
}

// ============================================================================
// TwoAdicSubgroup
// ============================================================================

/// Multiplicative subgroup of `F^*` of order `2^log_size`.
///
/// Implements [`Coset`] with `shift = F::ONE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TwoAdicSubgroup<F: TwoAdicField> {
    log_size: u8,
    _phantom: PhantomData<F>,
}

impl<F: TwoAdicField> TwoAdicSubgroup<F> {
    /// Create a subgroup of order `2^log_size`.
    ///
    /// # Panics
    ///
    /// Panics if `log_size > F::TWO_ADICITY` — a primitive `2^log_size`-th root
    /// of unity does not exist in `F` for that size.
    #[inline]
    pub fn new(log_size: u8) -> Self {
        assert!(
            (log_size as usize) <= F::TWO_ADICITY,
            "subgroup log size {log_size} exceeds field two-adicity {}",
            F::TWO_ADICITY,
        );
        Self { log_size, _phantom: PhantomData }
    }

    /// Smaller subgroup of order `2^(log_size − log_factor)`.
    ///
    /// Used by FRI to derive the next round's domain.
    ///
    /// # Panics
    ///
    /// Panics if `log_factor > self.log_size`.
    #[inline]
    pub fn shrink(&self, log_factor: u8) -> Self {
        assert!(
            log_factor <= self.log_size,
            "cannot shrink subgroup of log size {} by {log_factor}",
            self.log_size,
        );
        Self::new(self.log_size - log_factor)
    }
}

impl<F: TwoAdicField> Coset<F> for TwoAdicSubgroup<F> {
    #[inline]
    fn log_size(&self) -> u8 {
        self.log_size
    }

    #[inline]
    fn shift(&self) -> F {
        F::ONE
    }

    /// `F::ONE` is its own inverse — skip the inversion entirely.
    #[inline]
    fn shift_inverse(&self) -> F {
        F::ONE
    }

    /// **The only place in the crate that calls `F::two_adic_generator(...)`
    /// directly.** All other domain-indexed two-adic roots flow from this
    /// method via the [`Coset`] trait.
    #[inline]
    fn generator(&self) -> F {
        F::two_adic_generator(self.log_size as usize)
    }
}

// ============================================================================
// TwoAdicCoset
// ============================================================================

/// A coset `s · H` of a [`TwoAdicSubgroup`] `H`, carrying a multiplicative shift
/// `s ∈ F`.
///
/// `TwoAdicSubgroup` is the special case `s = 1`. The two are kept distinct so
/// the type system can express "no shift" without runtime checks. Both
/// implement [`Coset`].
///
/// The shift's multiplicative inverse is computed once at construction and
/// stored, so repeated `vanishing_at` / `contains` calls don't re-invert.
#[derive(Copy, Clone, Debug)]
pub struct TwoAdicCoset<F: TwoAdicField> {
    subgroup: TwoAdicSubgroup<F>,
    shift: F,
    shift_inverse: F,
}

impl<F: TwoAdicField> TwoAdicCoset<F> {
    /// Create a coset `shift · subgroup`. Computes and caches `shift⁻¹`.
    #[inline]
    pub fn new(subgroup: TwoAdicSubgroup<F>, shift: F) -> Self {
        Self {
            subgroup,
            shift,
            shift_inverse: shift.inverse(),
        }
    }

    /// Create the unshifted coset `subgroup` (i.e. `shift = 1`).
    #[inline]
    pub fn unshifted(subgroup: TwoAdicSubgroup<F>) -> Self {
        Self {
            subgroup,
            shift: F::ONE,
            shift_inverse: F::ONE,
        }
    }

    /// The underlying subgroup `H`.
    #[inline]
    pub fn subgroup(&self) -> &TwoAdicSubgroup<F> {
        &self.subgroup
    }
}

impl<F: TwoAdicField> PartialEq for TwoAdicCoset<F> {
    /// Equality ignores the cached `shift_inverse` field — it is fully
    /// determined by `shift`.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.subgroup == other.subgroup && self.shift == other.shift
    }
}

impl<F: TwoAdicField> Eq for TwoAdicCoset<F> {}

impl<F: TwoAdicField> Coset<F> for TwoAdicCoset<F> {
    #[inline]
    fn log_size(&self) -> u8 {
        self.subgroup.log_size()
    }

    #[inline]
    fn shift(&self) -> F {
        self.shift
    }

    /// Returns the cached inverse computed at construction time.
    #[inline]
    fn shift_inverse(&self) -> F {
        self.shift_inverse
    }

    #[inline]
    fn generator(&self) -> F {
        self.subgroup.generator()
    }
}

// ============================================================================
// LiftedDomain
// ============================================================================

/// STARK lifted-domain object: an LDE coset `g^r·K` together with a smaller
/// trace subgroup `H ⊆ K` and a lift ratio `r` relative to the max coset
/// `g·K_max`.
///
/// **Single home of `F::GENERATOR` in this crate** (via the
/// [`canonical_shift`] helper used in [`LiftedDomain::canonical`] and
/// [`LiftedDomain::sub_domain`]).
///
/// # Invariants
///
/// - `lde_coset.log_size() = trace_subgroup.log_size() + log_blowup` (where `log_blowup = lde −
///   trace`)
/// - `lde_coset.shift() = F::GENERATOR.exp_power_of_2(F::TWO_ADICITY − lde_coset.log_size())` — the
///   canonical, batch-independent shift for this LDE order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiftedDomain<F: TwoAdicField> {
    /// Trace domain `H` — size `2^log_trace_height`.
    trace_subgroup: TwoAdicSubgroup<F>,
    /// LDE evaluation coset `g^r·K` — size `2^log_lde_height`, shift `g^r`.
    lde_coset: TwoAdicCoset<F>,
    /// Log₂ of the lift ratio: `log_max_lde_height − log_lde_height`.
    log_lift_ratio: u8,
}

impl<F: TwoAdicField> LiftedDomain<F> {
    /// Create the canonical domain for `(trace, blowup)`: trace height
    /// `2^log_trace_height`, LDE blowup `2^log_blowup`, `log_lift_ratio = 0`.
    ///
    /// The LDE coset shift is the **canonical generator power**
    /// `g^(2^(F::TWO_ADICITY − log_lde_order))`, where `log_lde_order =
    /// log_trace_height + log_blowup`. This shift depends only on the LDE
    /// order — it is invariant of which other matrices appear in the same
    /// batch — making `canonical(t, b).lde_shift()` a function of the field
    /// and `(t, b)` alone.
    ///
    /// This is the **primary constructor**. Per-batch sub-domains derive
    /// from this one via [`sub_domain`](Self::sub_domain), which shrinks
    /// the trace subgroup; the shift is recomputed from the new LDE order
    /// (still canonical for that order).
    ///
    /// # Panics
    ///
    /// Panics if `log_trace_height + log_blowup > F::TWO_ADICITY` (enforced
    /// by [`TwoAdicSubgroup::new`]).
    #[inline]
    pub fn canonical(log_trace_height: u8, log_blowup: u8) -> Self {
        let log_lde_height = log_trace_height + log_blowup;
        let shift = canonical_shift::<F>(log_lde_height);
        Self {
            trace_subgroup: TwoAdicSubgroup::new(log_trace_height),
            lde_coset: TwoAdicCoset::new(TwoAdicSubgroup::new(log_lde_height), shift),
            log_lift_ratio: 0,
        }
    }

    /// Derive a sub-domain with a smaller trace subgroup, sharing this
    /// domain's blowup. The new domain's shift is the canonical shift for
    /// its own (smaller) LDE order — independent of the parent's lift ratio.
    /// The new `log_lift_ratio` grows by the trace shrink amount, recording
    /// the batch context for OOD lifting.
    ///
    /// # Panics
    ///
    /// Panics if `smaller_log_trace_height > self.log_trace_height()`.
    #[inline]
    pub fn sub_domain(&self, smaller_log_trace_height: u8) -> Self {
        let log_trace = self.log_trace_height();
        assert!(
            smaller_log_trace_height <= log_trace,
            "sub_domain trace height {smaller_log_trace_height} exceeds parent {log_trace}"
        );
        let log_blowup = self.log_blowup();
        let log_lift_ratio_inc = log_trace - smaller_log_trace_height;
        let new_log_lift_ratio = self.log_lift_ratio + log_lift_ratio_inc;
        let new_log_lde = smaller_log_trace_height + log_blowup;
        let shift = canonical_shift::<F>(new_log_lde);
        Self {
            trace_subgroup: TwoAdicSubgroup::new(smaller_log_trace_height),
            lde_coset: TwoAdicCoset::new(TwoAdicSubgroup::new(new_log_lde), shift),
            log_lift_ratio: new_log_lift_ratio,
        }
    }

    // ============ Subgroup / coset accessors ============

    /// The trace domain `H` as a [`TwoAdicSubgroup`].
    #[inline]
    pub fn trace_subgroup(&self) -> &TwoAdicSubgroup<F> {
        &self.trace_subgroup
    }

    /// The LDE evaluation coset `g^r·K` as a [`TwoAdicCoset`].
    #[inline]
    pub fn lde_coset(&self) -> &TwoAdicCoset<F> {
        &self.lde_coset
    }

    // ============ Protocol-named height / shift sugar ============

    /// Log₂ of the original trace height.
    #[inline]
    pub fn log_trace_height(&self) -> u8 {
        self.trace_subgroup.log_size()
    }

    /// Log₂ of this matrix's LDE height.
    #[inline]
    pub fn log_lde_height(&self) -> u8 {
        self.lde_coset.log_size()
    }

    /// Log₂ of the blowup factor for this matrix: `log_lde_height − log_trace_height`.
    #[inline]
    pub fn log_blowup(&self) -> u8 {
        self.log_lde_height() - self.log_trace_height()
    }

    /// The trace height (number of constraint rows).
    #[inline]
    pub fn trace_height(&self) -> usize {
        self.trace_subgroup.size()
    }

    /// The LDE height for this matrix.
    #[inline]
    pub fn lde_height(&self) -> usize {
        self.lde_coset.size()
    }

    /// The coset shift `g^r` for this matrix's LDE domain.
    #[inline]
    pub fn lde_shift(&self) -> F {
        self.lde_coset.shift()
    }

    // ============ Quotient evaluation coset ============

    /// The coset where the quotient polynomial is evaluated for per-row
    /// constraint evaluation: a strict sub-coset of [`lde_coset`](Self::lde_coset)
    /// of order `2^(log_trace_height + log_constraint_degree)`, **sharing this
    /// domain's LDE shift**.
    ///
    /// This is *not* a [`LiftedDomain`] — its shift is borrowed from the
    /// parent LDE coset rather than being the canonical shift for its own
    /// (smaller) order. Q is evaluated on this coset, then decomposed into D
    /// chunks each LDE-extended onto the canonical LDE coset for commitment.
    ///
    /// # Panics
    /// Panics if `log_constraint_degree > log_blowup`.
    pub fn evaluation_coset(&self, log_constraint_degree: u8) -> TwoAdicCoset<F> {
        let log_blowup = self.log_blowup();
        assert!(log_constraint_degree <= log_blowup, "constraint degree cannot exceed blowup");
        let log_eval_height = self.log_trace_height() + log_constraint_degree;
        TwoAdicCoset::new(TwoAdicSubgroup::new(log_eval_height), self.lde_coset.shift())
    }

    // ============ Selector computation ============

    /// Unnormalized prover-side Lagrange row selectors, evaluated at every point
    /// of the quotient evaluation coset (natural order).
    ///
    /// For each coset point `xᵢ = s · ωⁱ` (where `s` is the LDE shift and `ω`
    /// generates the size-`2^(log_trace_height + log_constraint_degree)` coset):
    /// - `is_first_row[i]  = Z_H(xᵢ) / (xᵢ − 1)`
    /// - `is_last_row[i]   = Z_H(xᵢ) / (xᵢ − ω_H⁻¹)`
    /// - `is_transition[i] = xᵢ − ω_H⁻¹`
    ///
    /// where `Z_H(x) = x^N_H − 1` is the trace subgroup's vanishing and `ω_H`
    /// is its generator. `Z_H(xᵢ)` is periodic with `2^log_constraint_degree`
    /// distinct values across the coset; we batch-invert the unique
    /// denominators.
    ///
    /// # Panics
    /// Panics if `log_constraint_degree > log_blowup` (via [`Self::evaluation_coset`]).
    pub fn selectors(&self, log_constraint_degree: u8) -> Selectors<Vec<F>> {
        let log_trace_height = self.log_trace_height();
        let eval_coset = self.evaluation_coset(log_constraint_degree);
        let coset_size = eval_coset.size();
        let shift = eval_coset.shift();

        // Z_H(x) = x^N_H − 1 over the eval coset is periodic with
        // 2^log_constraint_degree distinct values.
        let s_pow_n = shift.exp_power_of_2(log_trace_height as usize);
        let blowup_subgroup = eval_coset.subgroup().shrink(log_trace_height);
        let z_h_periodic: Vec<F> = blowup_subgroup
            .generator()
            .shifted_powers(s_pow_n)
            .take(1 << log_constraint_degree)
            .map(|x| x - F::ONE)
            .collect();
        let period = z_h_periodic.len();

        // Coset points in natural order.
        let xs: Vec<F> = eval_coset.generator().shifted_powers(shift).collect_n(coset_size);
        let omega_h_inv = self.trace_subgroup.generator_inverse();

        // Unnormalized Lagrange selector: selᵢ = Z_H(xᵢ) / (xᵢ − basis_point).
        let single_point_selector = |basis_point: F| -> Vec<F> {
            let denoms: Vec<F> = xs.par_iter().map(|&x| x - basis_point).collect();
            let invs = batch_multiplicative_inverse(&denoms);
            (0..coset_size)
                .into_par_iter()
                .map(|i| z_h_periodic[i % period] * invs[i])
                .collect()
        };

        Selectors {
            is_first_row: single_point_selector(F::ONE),
            is_last_row: single_point_selector(omega_h_inv),
            is_transition: xs.into_par_iter().map(|x| x - omega_h_inv).collect(),
        }
    }

    /// Unnormalized Lagrange row selectors at an OOD extension-field point `z`,
    /// evaluated as if `z` lives on the (lifted) trace subgroup.
    ///
    /// Internally the OOD point is first lifted via `z' = z^(2^log_lift_ratio)`
    /// so the selectors line up with the trace domain regardless of the
    /// per-instance lift ratio. For unlifted domains (`log_lift_ratio = 0`)
    /// this reduces to plain trace-subgroup selectors at `z`.
    ///
    /// - `is_first_row  = Z_H(z') / (z' − 1)`
    /// - `is_last_row   = Z_H(z') / (z' − ω_H⁻¹)`
    /// - `is_transition = z' − ω_H⁻¹`
    ///
    /// where `Z_H(z') = z'^N_H − 1` and `ω_H` is the trace subgroup generator.
    pub fn selectors_at<EF>(&self, z: EF) -> Selectors<EF>
    where
        EF: ExtensionField<F>,
    {
        let z_lift = z.exp_power_of_2(self.log_lift_ratio as usize);
        let vanishing = self.trace_subgroup.vanishing_at(z_lift);
        let omega_h_inv = self.trace_subgroup.generator_inverse();
        Selectors {
            is_first_row: vanishing / (z_lift - F::ONE),
            is_last_row: vanishing / (z_lift - omega_h_inv),
            is_transition: z_lift - omega_h_inv,
        }
    }

    // ============ OOD point sampling ============

    /// Sample an OOD evaluation point outside both `H` and the LDE coset.
    pub fn sample_ood_point<EF>(&self, channel: &mut impl Channel<F = F>) -> EF
    where
        EF: ExtensionField<F>,
    {
        loop {
            let candidate: EF = channel.sample_algebra_element();
            if !self.trace_subgroup.contains(candidate) && !self.lde_coset.contains(candidate) {
                break candidate;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use p3_field::{Field, PrimeCharacteristicRing};

    use super::*;
    use crate::testing::configs::goldilocks_poseidon2::{Felt, QuadFelt};

    // ========== TwoAdicSubgroup ==========

    #[test]
    fn subgroup_basic_dimensions() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(5);
        assert_eq!(h.log_size(), 5);
        assert_eq!(h.size(), 32);
    }

    #[test]
    fn subgroup_generator_matches_two_adic_generator() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(7);
        assert_eq!(h.generator(), Felt::two_adic_generator(7));
    }

    #[test]
    fn subgroup_generator_inverse_is_inverse() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(6);
        assert_eq!(h.generator() * h.generator_inverse(), Felt::ONE);
    }

    #[test]
    fn subgroup_point_at_matches_powers() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let g = h.generator();
        for i in 0..h.size() as u64 {
            assert_eq!(h.point_at(i), g.exp_u64(i));
        }
    }

    #[test]
    fn subgroup_points_length_and_first_two() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(3);
        let pts = h.points();
        assert_eq!(pts.len(), h.size());
        assert_eq!(pts[0], Felt::ONE);
        assert_eq!(pts[1], h.generator());
    }

    #[test]
    fn subgroup_bit_reversed_points() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let natural = h.points();
        let br = h.bit_reversed_points();
        assert_eq!(natural[0], br[0]);
        // Adjacent-negation: br[1] = ω^{n/2} = -1
        assert_eq!(br[1], -Felt::ONE);
    }

    #[test]
    fn subgroup_shrink_halves() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(8);
        let h2 = h.shrink(1);
        assert_eq!(h2.log_size(), 7);
        assert_eq!(h2.generator(), h.generator() * h.generator());
    }

    #[test]
    fn subgroup_vanishing_zero_in_subgroup() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        for k in 0..h.size() as u64 {
            assert_eq!(h.vanishing_at(h.point_at(k)), Felt::ZERO);
        }
    }

    #[test]
    fn subgroup_vanishing_outside() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let z = Felt::from_u32(7);
        let expected = z.exp_u64(h.size() as u64) - Felt::ONE;
        assert_eq!(h.vanishing_at(z), expected);
    }

    #[test]
    fn subgroup_contains() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(5);
        for k in 0..h.size() as u64 {
            assert!(h.contains(h.point_at(k)));
        }
        assert!(!h.contains(QuadFelt::from(Felt::from_u32(12345))));
    }

    // ========== TwoAdicCoset ==========

    #[test]
    fn coset_unshifted_matches_subgroup() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let coset = TwoAdicCoset::unshifted(h);
        assert_eq!(coset.shift(), Felt::ONE);
        assert_eq!(coset.points(), h.points());
        assert_eq!(coset.bit_reversed_points(), h.bit_reversed_points());
    }

    #[test]
    fn coset_points_match_shift_times_subgroup() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(5);
        let shift = Felt::from_u32(7);
        let coset = TwoAdicCoset::new(h, shift);

        let expected: Vec<Felt> = h.points().into_iter().map(|p| shift * p).collect();
        assert_eq!(coset.points(), expected);

        for i in 0..coset.size() as u64 {
            assert_eq!(coset.point_at(i), shift * h.point_at(i));
        }
    }

    #[test]
    fn coset_bit_reversed_points_explicit() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let shift = Felt::from_u32(11);
        let coset = TwoAdicCoset::new(h, shift);
        let expected: Vec<Felt> = h.bit_reversed_points().into_iter().map(|p| shift * p).collect();
        assert_eq!(coset.bit_reversed_points(), expected);
    }

    #[test]
    fn coset_vanishing_zero_at_coset_points() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let coset = TwoAdicCoset::new(h, Felt::from_u32(13));
        for k in 0..coset.size() as u64 {
            assert_eq!(coset.vanishing_at(coset.point_at(k)), Felt::ZERO);
        }
    }

    #[test]
    fn coset_vanishing_outside() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(4);
        let shift = Felt::from_u32(13);
        let coset = TwoAdicCoset::new(h, shift);
        let z = Felt::from_u32(999);
        let expected = (z * shift.inverse()).exp_u64(coset.size() as u64) - Felt::ONE;
        assert_eq!(coset.vanishing_at(z), expected);
    }

    #[test]
    fn coset_contains_its_own_points() {
        let h: TwoAdicSubgroup<Felt> = TwoAdicSubgroup::new(5);
        let coset = TwoAdicCoset::new(h, Felt::from_u32(31));
        for k in 0..coset.size() as u64 {
            assert!(coset.contains(coset.point_at(k)));
        }
        assert!(!coset.contains(QuadFelt::from(Felt::from_u32(54321))));
    }

    // ========== LiftedDomain ==========

    #[test]
    fn domain_canonical_is_unlifted() {
        // Canonical of trace 2^10, blowup 2^3 — no lifting.
        let info: LiftedDomain<Felt> = LiftedDomain::canonical(10, 3);
        assert_eq!(info.log_trace_height(), 10);
        assert_eq!(info.log_lde_height(), 13);
        assert_eq!(info.log_blowup(), 3);
        // Field-relative shift: g^(2^(TWO_ADICITY − log_lde_order)).
        let expected = Felt::GENERATOR.exp_power_of_2(Felt::TWO_ADICITY - 13);
        assert_eq!(info.lde_shift(), expected);
    }

    #[test]
    fn sub_domain_lifts_relative_to_canonical() {
        // Canonical: max trace 2^12, blowup 2^3 — sub-domain at trace 2^10 → lift_ratio 2.
        let parent: LiftedDomain<Felt> = LiftedDomain::canonical(12, 3);
        let sub = parent.sub_domain(10);

        assert_eq!(sub.log_trace_height(), 10);
        assert_eq!(sub.log_lde_height(), 13);
        assert_eq!(sub.log_blowup(), 3);
        assert_eq!(sub.trace_height(), 1024);
        assert_eq!(sub.lde_height(), 8192);

        // Shift is canonical for the sub-domain's own LDE order (13), not derived
        // from the parent's lift ratio. Crucially, equal to canonical(10,3).lde_shift().
        let expected_shift = Felt::GENERATOR.exp_power_of_2(Felt::TWO_ADICITY - 13);
        assert_eq!(sub.lde_shift(), expected_shift);
        assert_eq!(sub.lde_shift(), LiftedDomain::<Felt>::canonical(10, 3).lde_shift());
    }

    #[test]
    fn sub_domain_at_same_trace_is_identity() {
        let tallest: LiftedDomain<Felt> = LiftedDomain::canonical(10, 3);
        let same = tallest.sub_domain(10);
        assert_eq!(same.lde_shift(), tallest.lde_shift());
        assert_eq!(same.log_trace_height(), tallest.log_trace_height());
        assert_eq!(same.log_lde_height(), tallest.log_lde_height());
    }

    #[test]
    fn evaluation_coset_inherits_parent_shift() {
        // Sub-domain with lift_ratio = 2.
        let lde: LiftedDomain<Felt> = LiftedDomain::canonical(12, 3).sub_domain(10);
        let eval = lde.evaluation_coset(2);
        // Order N · D = 2^(10 + 2) = 2^12.
        assert_eq!(eval.log_size(), 12);
        // Shift is borrowed from the parent LDE coset (literally a sub-coset).
        // Note: this differs from `LiftedDomain::canonical(10, 2).lde_shift()` —
        // the evaluation coset is *not* a canonical-shift LiftedDomain.
        assert_eq!(eval.shift(), lde.lde_shift());
    }

    #[test]
    fn lde_coset_point_at_matches_shift_times_omega() {
        let info: LiftedDomain<Felt> = LiftedDomain::canonical(5, 2).sub_domain(4);
        let shift = info.lde_shift();
        let omega = info.lde_coset().generator();
        for i in 0..4 {
            assert_eq!(info.lde_coset().point_at(i as u64), shift * omega.exp_u64(i as u64));
        }
    }
}
