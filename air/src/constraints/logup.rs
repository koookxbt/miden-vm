//! LogUp rational-fraction algebra for bus constraint packing.
//!
//! Provides three structs mirroring the LogUp accumulator algebra:
//!
//! 1. [`Batch`] — simultaneous interactions normalized to `N / D`
//! 2. [`RationalSet`] — mutually exclusive batches compressed to `V / U` via selector gating
//! 3. [`Column`] — independent sets combined by clearing cross-denominators
//!
//! The final constraint emitted per column is `Δ · U − V = 0` where `Δ = acc_next − acc`.
//!
//! See `docs/src/design/bus_packing_summary.md` §1–7 for the full derivation.

use core::marker::PhantomData;

use miden_core::field::{Algebra, PrimeCharacteristicRing};

use super::logup_msg::LogUpMessage;
use crate::trace::Challenges;

// BATCH OF SIMULTANEOUS INTERACTIONS
// ================================================================================================

/// A batch of simultaneously active interactions, normalized to the fraction `N / D`.
///
/// `E` is the base-field expression type (for multiplicities), `EF` is the extension-field
/// expression type (for denominators and the accumulated pair).
///
/// Given interactions `(m₁, v₁), ..., (mₙ, vₙ)`:
/// - `D = Π vᵢ`
/// - `N = Σ mᵢ · Π_{j≠i} vⱼ`
///
/// Built iteratively: start with `(N, D) = (0, 1)`, then for each interaction `(m, v)`:
/// - `N ← N · v + m · D`
/// - `D ← D · v`
pub struct Batch<'c, E, EF: PrimeCharacteristicRing> {
    challenges: &'c Challenges<EF>,
    pub n: EF,
    pub d: EF,
    _phantom: PhantomData<E>,
}

impl<'c, E, EF> Batch<'c, E, EF>
where
    E: PrimeCharacteristicRing + Clone,
    EF: PrimeCharacteristicRing + Clone + Algebra<E>,
{
    /// Empty batch: `N = 0, D = 1`.
    pub fn new(challenges: &'c Challenges<EF>) -> Self {
        Self {
            challenges,
            n: EF::ZERO,
            d: EF::ONE,
            _phantom: PhantomData,
        }
    }

    /// Absorb an insert interaction (multiplicity = +1).
    pub fn add(&mut self, msg: impl LogUpMessage<E, EF>) {
        self.insert(E::ONE, msg);
    }

    /// Absorb a remove interaction (multiplicity = −1).
    pub fn remove(&mut self, msg: impl LogUpMessage<E, EF>) {
        self.insert(E::ZERO - E::ONE, msg);
    }

    /// Absorb an interaction with arbitrary base-field multiplicity.
    pub fn insert(&mut self, m: E, msg: impl LogUpMessage<E, EF>) {
        let v: EF = msg.encode(self.challenges);
        let d_prev = self.d.clone();
        self.n = self.n.clone() * v.clone() + d_prev * m;
        self.d = self.d.clone() * v;
    }
}

// SET OF MUTUALLY EXCLUSIVE BATCHES
// ================================================================================================

/// A set of mutually exclusive batches, compressed into a single rational pair `V / U`.
///
/// `E` is the base-field expression type, `EF` is the extension-field expression type.
///
/// Given ME batches `(N₁, D₁), ..., (Nₖ, Dₖ)` with boolean selectors `s₁, ..., sₖ`:
/// - `U = 1 + Σ sᵣ · (Dᵣ − 1)`
/// - `V = Σ sᵣ · Nᵣ`
///
/// When no selector is active: `U = 1, V = 0` (contributes zero).
/// When selector `sᵣ = 1`: `U = Dᵣ, V = Nᵣ` (contributes `Nᵣ / Dᵣ`).
pub struct RationalSet<'c, E, EF: PrimeCharacteristicRing> {
    challenges: &'c Challenges<EF>,
    pub u: EF,
    pub v: EF,
    _phantom: PhantomData<E>,
}

impl<'c, E, EF> RationalSet<'c, E, EF>
where
    E: PrimeCharacteristicRing + Clone,
    EF: PrimeCharacteristicRing + Clone + Algebra<E>,
{
    /// Empty set: `U = 1, V = 0` (identity — contributes nothing).
    pub fn new(challenges: &'c Challenges<EF>) -> Self {
        Self {
            challenges,
            u: EF::ONE,
            v: EF::ZERO,
            _phantom: PhantomData,
        }
    }

    /// Add a selector-gated single insert interaction: `+1 / v`.
    ///
    /// The message is constructed lazily via a closure, allowing future implementations
    /// to skip construction when the selector is zero.
    pub fn add_single<M: LogUpMessage<E, EF>>(&mut self, selector: E, msg_fn: impl FnOnce() -> M) {
        let mut b = Batch::new(self.challenges);
        b.add(msg_fn());
        self.fold_batch(selector, b);
    }

    /// Add a selector-gated single remove interaction: `−1 / v`.
    ///
    /// The message is constructed lazily via a closure.
    pub fn remove_single<M: LogUpMessage<E, EF>>(
        &mut self,
        selector: E,
        msg_fn: impl FnOnce() -> M,
    ) {
        let mut b = Batch::new(self.challenges);
        b.remove(msg_fn());
        self.fold_batch(selector, b);
    }

    /// Add a selector-gated single interaction with arbitrary multiplicity: `m / v`.
    ///
    /// The message is constructed lazily via a closure.
    pub fn insert_single<M: LogUpMessage<E, EF>>(
        &mut self,
        selector: E,
        m: E,
        msg_fn: impl FnOnce() -> M,
    ) {
        let mut b = Batch::new(self.challenges);
        b.insert(m, msg_fn());
        self.fold_batch(selector, b);
    }

    /// Add a selector-gated batch of simultaneous interactions.
    pub fn add_batch(&mut self, selector: E, build: impl FnOnce(&mut Batch<'c, E, EF>)) {
        let mut b = Batch::new(self.challenges);
        build(&mut b);
        self.fold_batch(selector, b);
    }

    /// Create a set for an always-active interaction (no selector gating).
    pub fn always(
        challenges: &'c Challenges<EF>,
        build: impl FnOnce(&mut Batch<'c, E, EF>),
    ) -> Self {
        let mut b = Batch::new(challenges);
        build(&mut b);
        Self {
            challenges,
            u: b.d,
            v: b.n,
            _phantom: PhantomData,
        }
    }

    /// Fold a completed batch with its selector into the running `(U, V)` pair.
    fn fold_batch(&mut self, selector: E, batch: Batch<'c, E, EF>) {
        self.u += (batch.d - EF::ONE) * selector.clone();
        self.v += batch.n * selector;
    }
}

// COLUMN ACCUMULATOR
// ================================================================================================

/// A column accumulator that combines independent sets by clearing cross-denominators.
///
/// Given sets with pairs `(U₁, V₁), ..., (Uₜ, Vₜ)`:
/// - `U = Π Uᵢ`
/// - `V = Σ Vᵢ · Π_{j≠i} Uⱼ`
///
/// Built iteratively: start with `(U, V) = (1, 0)`, then for each set `(Ũ, Ṽ)`:
/// - `V ← V · Ũ + Ṽ · U`
/// - `U ← U · Ũ`
///
/// The final constraint for this column is `Δ · U − V = 0` where `Δ = acc_next − acc`.
pub struct Column<E, EF> {
    pub u: EF,
    pub v: EF,
    _phantom: PhantomData<E>,
}

impl<E, EF> Column<E, EF>
where
    E: PrimeCharacteristicRing + Clone,
    EF: PrimeCharacteristicRing + Clone + Algebra<E>,
{
    /// Empty column: `U = 1, V = 0`.
    pub fn new() -> Self {
        Self {
            u: EF::ONE,
            v: EF::ZERO,
            _phantom: PhantomData,
        }
    }

    /// Create a column containing exactly one set.
    pub fn from_set(set: RationalSet<'_, E, EF>) -> Self {
        Self {
            u: set.u,
            v: set.v,
            _phantom: PhantomData,
        }
    }

    /// Add an independent set to the column.
    pub fn add_set(&mut self, set: RationalSet<'_, E, EF>) {
        self.v = self.v.clone() * set.u.clone() + set.v * self.u.clone();
        self.u = self.u.clone() * set.u;
    }

    /// Return the constraint expression `Δ · U − V`.
    pub fn constraint(&self, delta: EF) -> EF {
        delta * self.u.clone() - self.v.clone()
    }
}

// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    extern crate std;

    use miden_core::{
        Felt,
        field::{Field, QuadFelt},
    };

    use super::*;
    use crate::trace::Challenges;

    type E = Felt;
    type EF = QuadFelt;

    fn ef(a: u64) -> EF {
        EF::from(Felt::new(a))
    }

    fn challenges() -> Challenges<EF> {
        Challenges::new(ef(100), ef(7))
    }

    /// A trivial test message: encodes a single field element.
    struct TestMsg<E> {
        val: E,
    }

    impl<E, EF2> LogUpMessage<E, EF2> for TestMsg<E>
    where
        E: PrimeCharacteristicRing + Clone,
        EF2: PrimeCharacteristicRing + Algebra<E>,
    {
        fn encode(&self, challenges: &Challenges<EF2>) -> EF2 {
            challenges.encode([self.val.clone()])
        }
    }

    type B<'c> = Batch<'c, E, EF>;
    type S<'c> = RationalSet<'c, E, EF>;

    #[test]
    fn empty_batch() {
        let ch = challenges();
        let batch = B::new(&ch);
        assert_eq!(batch.n, EF::ZERO);
        assert_eq!(batch.d, EF::ONE);
    }

    #[test]
    fn single_add() {
        let ch = challenges();
        let v = ef(42);
        let mut batch = B::new(&ch);
        batch.add(TestMsg { val: Felt::new(42) });
        // D should equal the encoded value of 42
        let expected_d = ch.encode([Felt::new(42)]);
        assert_eq!(batch.d, expected_d);
        assert_eq!(batch.n, EF::ONE);
    }

    #[test]
    fn add_then_remove() {
        let ch = challenges();
        let mut batch = B::new(&ch);
        batch.add(TestMsg { val: Felt::new(3) });
        batch.remove(TestMsg { val: Felt::new(5) });
        let v1 = ch.encode([Felt::new(3)]);
        let v2 = ch.encode([Felt::new(5)]);
        assert_eq!(batch.d, v1 * v2);
        assert_eq!(batch.n, v2 - v1);
    }

    #[test]
    fn set_add_single() {
        let ch = challenges();
        let mut set = S::new(&ch);
        set.add_single(Felt::ONE, || TestMsg { val: Felt::new(10) });
        let v = ch.encode([Felt::new(10)]);
        assert_eq!(set.u, v);
        assert_eq!(set.v, EF::ONE);
    }

    #[test]
    fn set_inactive() {
        let ch = challenges();
        let mut set = S::new(&ch);
        set.add_single(Felt::ZERO, || TestMsg { val: Felt::new(10) });
        assert_eq!(set.u, EF::ONE);
        assert_eq!(set.v, EF::ZERO);
    }

    #[test]
    fn set_add_batch_closure() {
        let ch = challenges();
        let mut set = S::new(&ch);
        set.add_batch(Felt::ONE, |b| {
            b.add(TestMsg { val: Felt::new(3) });
            b.remove(TestMsg { val: Felt::new(5) });
        });
        let v1 = ch.encode([Felt::new(3)]);
        let v2 = ch.encode([Felt::new(5)]);
        assert_eq!(set.u, v1 * v2);
        assert_eq!(set.v, v2 - v1);
    }

    #[test]
    fn column_two_sets() {
        let ch = challenges();
        let v1 = ch.encode([Felt::new(3)]);
        let v2 = ch.encode([Felt::new(7)]);

        let mut set1 = S::new(&ch);
        set1.add_single(Felt::ONE, || TestMsg { val: Felt::new(3) });
        let mut set2 = S::new(&ch);
        set2.add_single(Felt::ONE, || TestMsg { val: Felt::new(7) });

        let mut col = Column::from_set(set1);
        col.add_set(set2);

        assert_eq!(col.u, v1 * v2);
        assert_eq!(col.v, v1 + v2);
    }

    #[test]
    fn column_constraint_zero() {
        let ch = challenges();
        let v = ch.encode([Felt::new(5)]);
        let mut set = S::new(&ch);
        set.add_single(Felt::ONE, || TestMsg { val: Felt::new(5) });
        let col = Column::from_set(set);
        assert_eq!(col.constraint(v.inverse()), EF::ZERO);
    }

    #[test]
    fn three_interactions_rational_identity() {
        let ch = challenges();
        let v1 = ch.encode([Felt::new(2)]);
        let v2 = ch.encode([Felt::new(3)]);
        let v3 = ch.encode([Felt::new(5)]);

        let mut batch = B::new(&ch);
        batch.add(TestMsg { val: Felt::new(2) });
        batch.add(TestMsg { val: Felt::new(3) });
        batch.remove(TestMsg { val: Felt::new(5) });

        assert_eq!(batch.d, v1 * v2 * v3);
        let lhs = batch.n * batch.d.inverse();
        let rhs = v1.inverse() + v2.inverse() - v3.inverse();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn end_to_end() {
        let ch = challenges();
        let v1 = ch.encode([Felt::new(3)]);
        let v2 = ch.encode([Felt::new(7)]);

        let set1 = S::always(&ch, |b| b.add(TestMsg { val: Felt::new(3) }));
        let mut set2 = S::new(&ch);
        set2.add_single(Felt::ONE, || TestMsg { val: Felt::new(7) });

        let mut col = Column::from_set(set1);
        col.add_set(set2);

        let delta = v1.inverse() + v2.inverse();
        assert_eq!(col.constraint(delta), EF::ZERO);
    }
}
