//! Auxiliary trace types: builder and external assertion evaluation.
//!
//! # Protocol Overview
//!
//! The auxiliary trace enables cross-AIR interactions (e.g. multiset / logup
//! arguments) in the lifted STARK.
//!
//! ## Prover
//!
//! 1. [`AuxBuilder::build_aux_trace`] constructs the aux trace and returns aux values
//!    (extension-field elements whose meaning is AIR-defined).
//! 2. The aux trace is committed (Merkle commitment).
//! 3. Aux values are sent via the Fiat-Shamir transcript.
//!
//! ## AIR constraints ([`eval`](crate::LiftedAir::eval))
//!
//! 4. The AIR defines how aux values relate to the committed aux trace. A common pattern is to
//!    constrain them to equal the aux trace's last row, but the protocol does not impose this — the
//!    AIR is free to define whatever relationship it needs.
//! 5. Transition constraints enforce the aux trace's internal logic (e.g. running product
//!    accumulation).
//!
//! ## Verifier
//!
//! 6. The verifier receives aux values from the transcript.
//! 7. Constraint evaluation (steps 4–5) is checked at a random point.
//! 8. [`eval_external`](crate::LiftedAir::eval_external) produces each AIR's external assertions
//!    from the aux values, challenges, and public inputs.
//! 9. Per-AIR check: each entry of every AIR's assertion list must equal zero. The verifier checks
//!    them one by one and reports the first non-zero entry along with the offending instance and
//!    assertion index.

mod builder;

pub use builder::AuxBuilder;

/// Boxed error returned by
/// [`LiftedAir::eval_external`](crate::LiftedAir::eval_external).
///
/// Each AIR defines its own concrete error type and boxes it into this alias.
pub type ReductionError = alloc::boxed::Box<dyn core::error::Error + Send + Sync>;
