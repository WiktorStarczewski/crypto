use alloc::vec::Vec;
use core::{
    array,
    ops::{Add, Mul},
};

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};

// ============================================================================
// Extension trait for PackedFieldExtension methods not in upstream
// ============================================================================

/// Horner fold with an explicit accumulator.
///
/// Computes `acc·xⁿ + v₀·xⁿ⁻¹ + v₁·xⁿ⁻² + ... + vₙ₋₁·x⁰` where n = len(vals).
/// Equivalently: `((acc·x + v₀)·x + v₁)·x + ... + vₙ₋₁`.
/// The first element gets the highest power of `x`.
///
/// For polynomial evaluation `p(x) = Σᵢ cᵢ·xⁱ`, pass coefficients in
/// descending degree order `[cₙ, ..., c₁, c₀]`.
#[inline]
pub fn horner_acc<Acc, Val, X, I>(acc: Acc, x: X, vals: I) -> Acc
where
    I: IntoIterator<Item = Val>,
    Acc: Mul<X, Output = Acc> + Add<Val, Output = Acc>,
    X: Clone,
{
    vals.into_iter().fold(acc, |acc, val| acc * x.clone() + val)
}

/// Horner fold starting from zero.
///
/// See [`horner_acc`] for the evaluation convention.
#[inline]
pub fn horner<Acc, Val, X, I>(x: X, vals: I) -> Acc
where
    I: IntoIterator<Item = Val>,
    Acc: Default + Mul<X, Output = Acc> + Add<Val, Output = Acc>,
    X: Clone,
{
    horner_acc(Acc::default(), x, vals)
}

/// Extension trait adding `pack_ext_columns` and `to_ext_slice` methods.
///
/// These methods enable efficient SIMD operations on arrays of extension field elements
/// by providing column-wise packing and unpacking utilities.
pub trait PackedFieldExtensionExt<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = Self>,
>: PackedFieldExtension<BaseField, ExtField>
{
    /// Pack N columns from WIDTH rows into N packed extension field elements.
    ///
    /// Input: `rows[lane][col]` - WIDTH rows, each with N extension field elements.
    /// Output: `result[col]` - N packed values, where each packs WIDTH lanes.
    fn pack_ext_columns<const N: usize>(rows: &[[ExtField; N]]) -> [Self; N] {
        let width = BaseField::Packing::WIDTH;
        debug_assert_eq!(rows.len(), width);
        array::from_fn(|col| {
            let col_elems: Vec<ExtField> = (0..width).map(|lane| rows[lane][col]).collect();
            Self::from_ext_slice(&col_elems)
        })
    }

    /// Extract all lanes to an output slice.
    fn to_ext_slice(&self, out: &mut [ExtField]) {
        let width = BaseField::Packing::WIDTH;
        for (lane, slot) in out.iter_mut().enumerate().take(width) {
            *slot = self.extract(lane);
        }
    }
}

impl<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = P>,
    P: PackedFieldExtension<BaseField, ExtField>,
> PackedFieldExtensionExt<BaseField, ExtField> for P
{
}
