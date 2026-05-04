//! Barycentric interpolation for DEEP openings (with lifting).
//!
//! The DEEP technique needs values `f(z)` for many committed polynomials `f` at a
//! small number of out-of-domain (OOD) points `z`. Interpolating a degree-`< d`
//! polynomial from `d` samples naively is `O(d²)`; barycentric interpolation makes
//! it `O(d)` once we precompute the expensive inverses.
//!
//! # Notation
//! - Domain points: `xᵢ = g·ωⁱ` for `i = 0..d−1` (a coset `gH` of size `d`).
//! - OOD points: `zⱼ`, chosen so `zⱼ ≠ xᵢ` for all `i, j`.
//! - Vanishing on `gH`: `V_{gH}(X) = (X/g)ᵈ − 1`.
//!
//! # Barycentric form
//! For `deg(f) < d`:
//!
//! ```text
//! f(z) = s(z) · Σᵢ wᵢ(z) · f(xᵢ)
//! s(z) = V_{gH}(z) / d = ((z/g)ᵈ − 1) / d
//! wᵢ(z) = xᵢ / (z − xᵢ)
//! ```
//!
//! # Point quotients
//! We precompute `qᵢ(zⱼ) = 1/(zⱼ − xᵢ)` for all domain points `xᵢ` and all
//! opening points `zⱼ` using batch inversion (Montgomery's trick). This single
//! table is reused for:
//! - barycentric weights: `wᵢ(zⱼ) = xᵢ · qᵢ(zⱼ)`
//! - DEEP quotients: `(f(zⱼ) − f(X)) / (zⱼ − X)`
//!
//! # Lifting and weight folding
//! In lifted STARKs, different matrices correspond to polynomials on different
//! (power-of-two) domain sizes. A polynomial on a smaller domain is embedded into
//! the max domain by composing with an r-th power map: `f_lift(X) = f(Xʳ)`.
//! The verifier always queries at `z`, so the prover reports `f(zʳ)` for that
//! matrix; equivalently, it is evaluating `f_lift(z)`.
//!
//! To avoid recomputing barycentric weights for every height, we exploit the
//! bit-reversed ordering used by commitments: on a two-adic coset, points come in
//! adjacent `(+x, −x)` pairs. When lifting by a factor of 2, the lifted polynomial
//! `f(X²)` takes the same value on each adjacent pair, so we can fold the barycentric
//! sum by *summing the corresponding weights*. Repeating this `k` times handles lift
//! factors `r = 2ᵏ`.
//!
//! **Why weight summing is correct.** In bit-reversed order, `x_{2i+1} = −x_{2i}`.
//! Adding the two barycentric weights gives
//! `w_{2i} + w_{2i+1} = x/(z−x) + (−x)/(z+x) = 2x²/(z²−x²) = 2·w'ᵢ(z²)`,
//! where `w'ᵢ` is the weight on the squared domain. The factor of 2 cancels with the
//! halved scaling `s'(z²) = 2·s(z)`, so the interpolation identity is preserved.

use alloc::{collections::BTreeSet, vec::Vec};
use core::marker::PhantomData;

use p3_field::{ExtensionField, FieldArray, TwoAdicField, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{linear_map::LinearMap, log2_strict_usize, reverse_bits_len};
use tracing::{debug_span, info_span};

use crate::lmcs::row_list::RowList;

/// Precomputed `1/(z − xᵢ)` and `1/(h·z − xᵢ)` for the lifted-STARK opening pair.
///
/// The lifted STARK PCS only ever opens at `(z, h·z)` — the local and next-row
/// out-of-domain points needed for transition constraints. `PointQuotients`
/// stores both denominators in a single `FieldArray<EF, 2>` per coset point,
/// enabling batched `O(d)` barycentric evaluation and DEEP quotient
/// construction without repeating inversions.
pub struct PointQuotients<F: TwoAdicField, EF: ExtensionField<F>> {
    /// The evaluation points `[z, h·z]`.
    points: FieldArray<EF, 2>,
    /// `point_quotient[i] = [1/(z − xᵢ), 1/(h·z − xᵢ)]` for domain point xᵢ.
    pub(super) point_quotient: Vec<FieldArray<EF, 2>>,
    _marker: PhantomData<F>,
}

impl<F: TwoAdicField, EF: ExtensionField<F>> PointQuotients<F, EF> {
    /// Build the quotient table for openings at `(z, h·z)`.
    ///
    /// `h` must be a primitive `2^log_max_trace_height`-th root of unity in `F`,
    /// equivalently `h = ω_lde^blowup` where `ω_lde` is the LDE primitive root and
    /// `blowup = 2^log_blowup`. This is the structure used for STARK transition
    /// constraints: the verifier opens trace columns at the OOD point `z` and at
    /// `z` shifted by one trace step.
    ///
    /// # Preconditions
    ///
    /// - `coset_points` must be the LDE coset in **bit-reversed** order, matching the layout
    ///   produced by [`bit_reversed_coset_points`](crate::pcs::utils::bit_reversed_coset_points).
    /// - `coset_points.len()` must be a power of two and `≥ 2^log_blowup`.
    /// - `z` and `h·z` must both lie outside the LDE coset.
    pub fn from_z_and_hz(z: EF, h: F, log_blowup: u8, coset_points: &[F]) -> Self {
        let _span =
            info_span!("PointQuotients::from_z_and_hz", n = coset_points.len()).entered();
        let n = coset_points.len();
        let log_n = log2_strict_usize(n);
        let blowup = 1usize << log_blowup as usize;
        assert!(blowup <= n, "blowup factor exceeds coset size");

        // Step 1: invert (z − x_i) for all coset points (one batch inverse over n elements).
        let diffs: Vec<EF> = coset_points.par_iter().map(|&x| z - x).collect();
        let inv_z = batch_multiplicative_inverse(&diffs);

        // Step 2: derive 1/(h·z − x_i) = h⁻¹ · inv_z[perm(i)] where the permutation
        // implements the `blowup`-step cyclic shift in standard order, conjugated by
        // bit-reversal to operate directly on `inv_z` in bit-reversed order.
        let h_inv = h.inverse();
        let point_quotient: Vec<FieldArray<EF, 2>> = (0..n)
            .into_par_iter()
            .map(|j| {
                let std_idx = reverse_bits_len(j, log_n);
                let std_idx_shifted = (std_idx + n - blowup) % n;
                let perm_j = reverse_bits_len(std_idx_shifted, log_n);
                let inv_hz = inv_z[perm_j] * h_inv;
                FieldArray([inv_z[j], inv_hz])
            })
            .collect();

        Self {
            points: FieldArray([z, z * h]),
            point_quotient,
            _marker: PhantomData,
        }
    }

    /// Evaluate all matrix columns at `[zʳ, (h·z)ʳ]`.
    ///
    /// Here `r = domain_size / matrix_height` is the lift factor for that matrix.
    ///
    /// Returns evaluations grouped by commitment: `groups[group_idx][matrix_idx][col_idx]`
    /// where each element is a `FieldArray<EF, 2>` containing the two-point evaluations.
    /// Both opening points are batched together, leveraging
    /// `columnwise_dot_product_batched<2>` for better cache utilization than two
    /// separate calls.
    ///
    /// Implementation note: we compute barycentric weights for the maximum domain once,
    /// then derive weights for smaller heights by folding (summing blocks). All heights
    /// share the same precomputed point quotients `1/(zⱼ − xᵢ)`.
    pub fn batch_eval_lifted<M: Matrix<F>>(
        &self,
        matrices_groups: &[Vec<&M>],
        coset_points: &[F],
        log_blowup: u8,
    ) -> RowList<FieldArray<EF, 2>> {
        let _span = info_span!("batch_eval_lifted", n_groups = matrices_groups.len()).entered();
        let n = coset_points.len();
        let d = n >> log_blowup as usize;
        let log_d = log2_strict_usize(d);

        let shift = coset_points[0]; // g in bit-reversed order
        let shift_inverse = shift.inverse();

        // Compute barycentric scaling factors for each point:
        // sⱼ(zⱼ) = ((zⱼ/g)ᵈ − 1) / d
        let barycentric_scalings = self.points.map(|point| {
            let z_over_shift = point * shift_inverse;
            let t = z_over_shift.exp_power_of_2(log_d) - EF::ONE;
            t.div_2exp_u64(log_d as u64)
        });

        let used_degrees: BTreeSet<usize> = matrices_groups
            .iter()
            .flat_map(|g| g.iter().map(|m| m.height() >> log_blowup as usize))
            .collect();

        // Compute barycentric weights for each point at each height:
        // wᵢⱼ(zⱼ) = xᵢ / (zⱼ − xᵢ) = xᵢ · point_quotient[i][j]
        // For smaller domains, sum chunks (weight folding).
        let barycentric_weights: LinearMap<usize, Vec<FieldArray<EF, 2>>> =
            debug_span!("barycentric_weights", d).in_scope(|| {
                assert_eq!(*used_degrees.last().unwrap(), d);
                // Initial weights at full domain size
                let top_weights: Vec<FieldArray<EF, 2>> = coset_points[..d]
                    .par_iter()
                    .zip(self.point_quotient[..d].par_iter())
                    .map(|(&x, invs)| (*invs).map(|inv| inv * x))
                    .collect();

                let mut weights = Vec::with_capacity(used_degrees.len());
                weights.push(top_weights);

                // Descending order: progressively sum chunks to shrink weights
                for &next_degree in used_degrees.iter().rev().skip(1) {
                    let prev_weights = weights.last().unwrap();
                    let chunk_size = prev_weights.len() / next_degree;
                    let next_weights = prev_weights
                        .par_chunks_exact(chunk_size)
                        .map(|chunk| chunk.iter().copied().sum())
                        .collect();
                    weights.push(next_weights);
                }

                weights.into_iter().map(|w| (w.len(), w)).collect()
            });

        // f(zⱼʳ) = sⱼ(zⱼ)·Σᵢ wᵢⱼ(zⱼ)·f(xᵢ)
        // For each group, evaluate at both points using columnwise_dot_product_batched.
        // Returns Vec<FieldArray<EF, 2>> where result[col] = [eval at z, eval at h·z].
        let all_evals: Vec<Vec<FieldArray<EF, 2>>> = matrices_groups
            .iter()
            .flat_map(|group| {
                group.iter().map(|m| {
                    let weights = &barycentric_weights[&(m.height() >> log_blowup as usize)];
                    let _guard =
                        debug_span!("evaluate matrix", height = weights.len(), width = m.width())
                            .entered();
                    let mut results = m.columnwise_dot_product_batched(weights);
                    for batch_evals in results.iter_mut() {
                        *batch_evals *= barycentric_scalings;
                    }
                    results
                })
            })
            .collect();

        RowList::from_rows(&all_evals)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_interpolation::{interpolate_coset, interpolate_coset_with_precomputation};
    use p3_matrix::{bitrev::BitReversibleMatrix, dense::RowMajorMatrix};
    use p3_util::reverse_slice_index_bits;
    use rand::{RngExt, SeedableRng, distr::StandardUniform, prelude::SmallRng};

    use super::*;
    use crate::{
        pcs::utils::bit_reversed_coset_points,
        testing::configs::goldilocks_poseidon2::{Felt, QuadFelt},
    };

    /// `from_z_and_hz` must place the correct inverse at every coset point for both `z`
    /// and `h·z`. Compared directly against per-element `try_inverse` rather than against
    /// a separate generic constructor.
    #[test]
    fn from_z_and_hz_denominators_are_correct() {
        let rng = &mut SmallRng::seed_from_u64(2024);
        let log_blowup = 2u8;
        // Several LDE sizes and trace heights to exercise the bit-reversed index permutation.
        for log_lde_height in [4u8, 6, 8, 10] {
            let log_max_trace_height = log_lde_height - log_blowup;
            let coset_points_br = bit_reversed_coset_points::<Felt>(log_lde_height);

            let z: QuadFelt = rng.sample(StandardUniform);
            let h = Felt::two_adic_generator(log_max_trace_height as usize);

            let pq = PointQuotients::<Felt, QuadFelt>::from_z_and_hz(
                z,
                h,
                log_blowup,
                &coset_points_br,
            );

            for (i, &x_i) in coset_points_br.iter().enumerate() {
                let expected_inv_z = (z - x_i).try_inverse().expect("z outside coset");
                let expected_inv_hz =
                    (z * h - x_i).try_inverse().expect("h·z outside coset");
                assert_eq!(pq.point_quotient[i][0], expected_inv_z, "z denom at i={i}");
                assert_eq!(pq.point_quotient[i][1], expected_inv_hz, "h·z denom at i={i}");
            }
        }
    }

    /// `batch_eval_lifted` must match `interpolate_coset` for both opening points `z` and
    /// `h·z`, across mixed-height matrices that exercise weight folding.
    #[test]
    fn batch_eval_matches_interpolate_coset() {
        let rng = &mut SmallRng::seed_from_u64(999);
        let log_blowup = 2u8;
        let log_n = 8;
        let n = 1usize << log_n;
        let shift = Felt::GENERATOR;

        let coset_points_br = bit_reversed_coset_points::<Felt>(log_n);

        let log_max_trace_height = log_n - log_blowup;
        let z: QuadFelt = rng.sample(StandardUniform);
        let h = Felt::two_adic_generator(log_max_trace_height as usize);
        let hz = z * h;

        // Matrix 1: full height (no lifting).
        let poly_degree_1 = n >> log_blowup as usize;
        let width = 3;
        let lifted_shift_1 = shift;
        let mut coeffs1 = RowMajorMatrix::<Felt>::rand(rng, poly_degree_1, width).values;
        coeffs1.resize(n * width, Felt::ZERO);
        let evals1_std =
            NaiveDft.coset_dft_batch(RowMajorMatrix::new(coeffs1, width), lifted_shift_1);
        let evals1_br: RowMajorMatrix<Felt> =
            evals1_std.clone().bit_reverse_rows().to_row_major_matrix();

        // Matrix 2: half height (lift factor 2).
        let poly_degree_2 = poly_degree_1 >> 1;
        let lifted_shift_2 = shift.square();
        let mut coeffs2 = RowMajorMatrix::<Felt>::rand(rng, poly_degree_2, width).values;
        coeffs2.resize((n >> 1) * width, Felt::ZERO);
        let evals2_std =
            NaiveDft.coset_dft_batch(RowMajorMatrix::new(coeffs2, width), lifted_shift_2);
        let evals2_br: RowMajorMatrix<Felt> =
            evals2_std.clone().bit_reverse_rows().to_row_major_matrix();

        let matrices_groups: Vec<Vec<&RowMajorMatrix<Felt>>> = vec![vec![&evals1_br, &evals2_br]];

        let pq = PointQuotients::<Felt, QuadFelt>::from_z_and_hz(
            z,
            h,
            log_blowup,
            &coset_points_br,
        );
        let result = pq.batch_eval_lifted(&matrices_groups, &coset_points_br, log_blowup);
        let rows: Vec<&[FieldArray<QuadFelt, 2>]> = result.iter_rows().collect();
        assert_eq!(rows.len(), 2, "expected 2 matrix rows");

        for (point_idx, label, point) in [(0usize, "z", z), (1, "h·z", hz)] {
            // Matrix 1 (no lifting): evaluate at the point directly.
            let expected1: Vec<QuadFelt> = interpolate_coset(&evals1_std, lifted_shift_1, point);
            for (col, (&our, &exp)) in rows[0].iter().zip(expected1.iter()).enumerate() {
                assert_eq!(our[point_idx], exp, "{label}, mat1, col={col}: mismatch");
            }

            // Matrix 2 (lift factor 2): evaluate at point².
            let point_lifted = point.square();
            let expected2: Vec<QuadFelt> =
                interpolate_coset(&evals2_std, lifted_shift_2, point_lifted);
            for (col, (&our, &exp)) in rows[1].iter().zip(expected2.iter()).enumerate() {
                assert_eq!(our[point_idx], exp, "{label}, mat2, col={col}: mismatch");
            }
        }
    }

    /// `batch_eval_lifted` must agree with `interpolate_coset_with_precomputation` when fed
    /// the same precomputed inverses. Exercises the no-lifting full-domain case for both
    /// the `z` and `h·z` columns of `point_quotient`.
    #[test]
    fn batch_eval_matches_interpolate_with_precomputation() {
        let rng = &mut SmallRng::seed_from_u64(123);
        let log_blowup = 2u8;
        let log_n = 8;
        let n = 1usize << log_n;
        let shift = Felt::GENERATOR;

        // Coset points in both orderings.
        let coset_points_br = bit_reversed_coset_points::<Felt>(log_n);
        let mut coset_points_std = coset_points_br.clone();
        reverse_slice_index_bits(&mut coset_points_std);

        let log_max_trace_height = log_n - log_blowup;
        let z: QuadFelt = rng.sample(StandardUniform);
        let h = Felt::two_adic_generator(log_max_trace_height as usize);
        let hz = z * h;

        let pq = PointQuotients::<Felt, QuadFelt>::from_z_and_hz(
            z,
            h,
            log_blowup,
            &coset_points_br,
        );

        // No lifting: poly degree fills the LDE domain after blowup.
        let poly_degree = n >> log_blowup as usize;
        let lde_height = n;
        let width = 4;

        let mut coeffs_values = RowMajorMatrix::<Felt>::rand(rng, poly_degree, width).values;
        coeffs_values.resize(lde_height * width, Felt::ZERO);
        let padded_coeffs = RowMajorMatrix::new(coeffs_values, width);

        let evals_std = NaiveDft.coset_dft_batch(padded_coeffs, shift);
        let evals_br = evals_std.clone().bit_reverse_rows();

        let result = pq.batch_eval_lifted(&[vec![&evals_br]], &coset_points_br, log_blowup);
        let our_rows: Vec<&[FieldArray<QuadFelt, 2>]> = result.iter_rows().collect();
        assert_eq!(our_rows.len(), 1, "expected one matrix row");
        let our_row = our_rows[0];

        for (point_idx, label, point) in [(0usize, "z", z), (1, "h·z", hz)] {
            // Extract the column of inverses for this opening point and convert to standard order.
            let mut diff_invs_std: Vec<QuadFelt> =
                pq.point_quotient[..lde_height].iter().map(|arr| arr[point_idx]).collect();
            reverse_slice_index_bits(&mut diff_invs_std);

            let expected: Vec<QuadFelt> = interpolate_coset_with_precomputation(
                &evals_std,
                shift,
                point,
                &coset_points_std[..lde_height],
                &diff_invs_std,
            );

            assert_eq!(our_row.len(), expected.len(), "{label}: length mismatch");
            for (col, (&our, &exp)) in our_row.iter().zip(expected.iter()).enumerate() {
                assert_eq!(our[point_idx], exp, "{label}, col={col}: evaluation mismatch");
            }
        }
    }
}
