//! Message structs for LogUp bus interactions.
//!
//! Each struct represents a reduced denominator encoding: `α + Σ βⁱ · field_i`.
//! Fields are named for readability; the [`LogUpMessage`] trait provides the
//! `encode` method that produces the extension-field value.
//!
//! Chiplet messages (hasher, memory, bitwise) use a **header-as-builder** pattern:
//! construct a header with the shared context, then call a named method that bakes in
//! the operation label and produces the final message. The label enums are internal.
//!
//! All structs are generic over `E` (base-field expression type, typically `AB::Expr`).

use miden_core::field::{Algebra, PrimeCharacteristicRing};

use crate::trace::Challenges;

// TRAIT
// ================================================================================================

/// A bus message that can be encoded into a reduced denominator `α + Σ βⁱ · elemᵢ`.
///
/// `E` is the base-field expression type, `EF` is the extension-field expression type.
pub trait LogUpMessage<E: PrimeCharacteristicRing + Clone, EF: PrimeCharacteristicRing + Algebra<E>>
{
    /// Encode this message using the given challenges, producing an extension-field value.
    fn encode(&self, challenges: &Challenges<EF>) -> EF;
}

// HASHER MESSAGES
// ================================================================================================

/// Hasher chiplet message. Variants differ by payload size.
///
/// Constructed via associated functions — the label is baked in by each constructor.
/// Encodes as `[label, addr, node_index, ...payload]`.
#[derive(Clone)]
pub enum HasherMsg<E> {
    /// 15-element message: addr + node_index + 12-lane sponge state.
    State {
        label_value: u16,
        addr: E,
        node_index: E,
        state: [E; 12],
    },
    /// 11-element message: addr + node_index + 8-lane rate.
    Rate {
        label_value: u16,
        addr: E,
        node_index: E,
        rate: [E; 8],
    },
    /// 7-element message: addr + node_index + 4-element word/digest.
    Word {
        label_value: u16,
        addr: E,
        node_index: E,
        word: [E; 4],
    },
}

impl<E: PrimeCharacteristicRing + Clone> HasherMsg<E> {
    // --- State messages (15 elements) ---

    /// Linear hash / control block init: full 12-lane sponge state.
    ///
    /// Used by: HPERM input, LOGPRECOMPILE input.
    pub fn linear_hash_init(addr: E, state: [E; 12]) -> Self {
        use crate::trace::chiplets::hasher::LINEAR_HASH_LABEL;
        Self::State {
            label_value: LINEAR_HASH_LABEL as u16 + 16,
            addr,
            node_index: E::ZERO,
            state,
        }
    }

    /// Control block init: 8 rate lanes + opcode at capacity[1], zeros elsewhere.
    ///
    /// Used by: JOIN, SPLIT, LOOP, SPAN, CALL, SYSCALL, DYN, DYNCALL.
    pub fn control_block(addr: E, rate: &[E; 8], opcode: u8) -> Self {
        use crate::trace::chiplets::hasher::LINEAR_HASH_LABEL;
        Self::State {
            label_value: LINEAR_HASH_LABEL as u16 + 16,
            addr,
            node_index: E::ZERO,
            state: [
                rate[0].clone(),
                rate[1].clone(),
                rate[2].clone(),
                rate[3].clone(),
                rate[4].clone(),
                rate[5].clone(),
                rate[6].clone(),
                rate[7].clone(),
                E::ZERO,
                E::from_u16(opcode as u16),
                E::ZERO,
                E::ZERO,
            ],
        }
    }

    /// Return full sponge state after permutation.
    ///
    /// Used by: HPERM output, LOGPRECOMPILE output.
    pub fn return_state(addr: E, state: [E; 12]) -> Self {
        use crate::trace::chiplets::hasher::RETURN_STATE_LABEL;
        Self::State {
            label_value: RETURN_STATE_LABEL as u16 + 32,
            addr,
            node_index: E::ZERO,
            state,
        }
    }

    // --- Rate messages (11 elements) ---

    /// Absorb new rate into running hash.
    ///
    /// Used by: RESPAN.
    pub fn absorption(addr: E, rate: [E; 8]) -> Self {
        use crate::trace::chiplets::hasher::LINEAR_HASH_LABEL;
        Self::Rate {
            label_value: LINEAR_HASH_LABEL as u16 + 32,
            addr,
            node_index: E::ZERO,
            rate,
        }
    }

    // --- Word messages (7 elements) ---

    /// Return digest only (node_index = 0).
    ///
    /// Used by: END, MPVERIFY output, MRUPDATE output.
    pub fn return_hash(addr: E, word: [E; 4]) -> Self {
        use crate::trace::chiplets::hasher::RETURN_HASH_LABEL;
        Self::Word {
            label_value: RETURN_HASH_LABEL as u16 + 32,
            addr,
            node_index: E::ZERO,
            word,
        }
    }

    /// Start Merkle path verification (with explicit node_index).
    ///
    /// Used by: MPVERIFY input.
    pub fn merkle_verify_init(addr: E, node_index: E, word: [E; 4]) -> Self {
        use crate::trace::chiplets::hasher::MP_VERIFY_LABEL;
        Self::Word {
            label_value: MP_VERIFY_LABEL as u16 + 16,
            addr,
            node_index,
            word,
        }
    }

    /// Start Merkle update, old path (with explicit node_index).
    ///
    /// Used by: MRUPDATE old input.
    pub fn merkle_old_init(addr: E, node_index: E, word: [E; 4]) -> Self {
        use crate::trace::chiplets::hasher::MR_UPDATE_OLD_LABEL;
        Self::Word {
            label_value: MR_UPDATE_OLD_LABEL as u16 + 16,
            addr,
            node_index,
            word,
        }
    }

    /// Start Merkle update, new path (with explicit node_index).
    ///
    /// Used by: MRUPDATE new input.
    pub fn merkle_new_init(addr: E, node_index: E, word: [E; 4]) -> Self {
        use crate::trace::chiplets::hasher::MR_UPDATE_NEW_LABEL;
        Self::Word {
            label_value: MR_UPDATE_NEW_LABEL as u16 + 16,
            addr,
            node_index,
            word,
        }
    }

    // --- Encoding ---

    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        match self {
            Self::State { label_value, addr, node_index, state } => challenges.encode([
                E::from_u16(*label_value),
                addr.clone(),
                node_index.clone(),
                state[0].clone(),
                state[1].clone(),
                state[2].clone(),
                state[3].clone(),
                state[4].clone(),
                state[5].clone(),
                state[6].clone(),
                state[7].clone(),
                state[8].clone(),
                state[9].clone(),
                state[10].clone(),
                state[11].clone(),
            ]),
            Self::Rate { label_value, addr, node_index, rate } => challenges.encode([
                E::from_u16(*label_value),
                addr.clone(),
                node_index.clone(),
                rate[0].clone(),
                rate[1].clone(),
                rate[2].clone(),
                rate[3].clone(),
                rate[4].clone(),
                rate[5].clone(),
                rate[6].clone(),
                rate[7].clone(),
            ]),
            Self::Word { label_value, addr, node_index, word } => challenges.encode([
                E::from_u16(*label_value),
                addr.clone(),
                node_index.clone(),
                word[0].clone(),
                word[1].clone(),
                word[2].clone(),
                word[3].clone(),
            ]),
        }
    }
}

// MEMORY MESSAGES
// ================================================================================================

/// Common header for all memory messages: `[ctx, addr, clk]`.
///
/// Call a named method to produce a [`MemoryMsg`] with the correct operation label baked in.
#[derive(Clone)]
pub struct MemoryHeader<E> {
    pub ctx: E,
    pub addr: E,
    pub clk: E,
}

impl<E: PrimeCharacteristicRing + Clone> MemoryHeader<E> {
    /// Read a single element from memory.
    pub fn read_element(&self, element: E) -> MemoryMsg<E> {
        use crate::trace::chiplets::memory::MEMORY_READ_ELEMENT_LABEL;
        MemoryMsg::Element {
            op_value: MEMORY_READ_ELEMENT_LABEL as u16,
            header: self.clone(),
            element,
        }
    }

    /// Write a single element to memory.
    pub fn write_element(&self, element: E) -> MemoryMsg<E> {
        use crate::trace::chiplets::memory::MEMORY_WRITE_ELEMENT_LABEL;
        MemoryMsg::Element {
            op_value: MEMORY_WRITE_ELEMENT_LABEL as u16,
            header: self.clone(),
            element,
        }
    }

    /// Read a 4-element word from memory.
    pub fn read_word(&self, word: [E; 4]) -> MemoryMsg<E> {
        use crate::trace::chiplets::memory::MEMORY_READ_WORD_LABEL;
        MemoryMsg::Word {
            op_value: MEMORY_READ_WORD_LABEL as u16,
            header: self.clone(),
            word,
        }
    }

    /// Write a 4-element word to memory.
    pub fn write_word(&self, word: [E; 4]) -> MemoryMsg<E> {
        use crate::trace::chiplets::memory::MEMORY_WRITE_WORD_LABEL;
        MemoryMsg::Word {
            op_value: MEMORY_WRITE_WORD_LABEL as u16,
            header: self.clone(),
            word,
        }
    }
}

/// Memory chiplet message. Variants differ by payload size.
///
/// Constructed via methods on [`MemoryHeader`] — the operation label is baked in.
/// Encodes as `[op_label, ctx, addr, clk, ...payload]`.
#[derive(Clone)]
pub enum MemoryMsg<E> {
    /// 5-element message: header + one field element.
    Element {
        op_value: u16,
        header: MemoryHeader<E>,
        element: E,
    },
    /// 8-element message: header + 4-element word.
    Word {
        op_value: u16,
        header: MemoryHeader<E>,
        word: [E; 4],
    },
}

impl<E: PrimeCharacteristicRing + Clone> MemoryMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        match self {
            Self::Element { op_value, header, element } => challenges.encode([
                E::from_u16(*op_value),
                header.ctx.clone(),
                header.addr.clone(),
                header.clk.clone(),
                element.clone(),
            ]),
            Self::Word { op_value, header, word } => challenges.encode([
                E::from_u16(*op_value),
                header.ctx.clone(),
                header.addr.clone(),
                header.clk.clone(),
                word[0].clone(),
                word[1].clone(),
                word[2].clone(),
                word[3].clone(),
            ]),
        }
    }
}

// BITWISE MESSAGE
// ================================================================================================

/// Bitwise chiplet message (4 elements): `[label, a, b, result]`.
///
/// Constructed via [`BitwiseMsg::and`] or [`BitwiseMsg::xor`].
#[derive(Clone)]
pub struct BitwiseMsg<E> {
    op_value: u16,
    pub a: E,
    pub b: E,
    pub result: E,
}

impl<E: PrimeCharacteristicRing> BitwiseMsg<E> {
    /// Bitwise AND message (label = 2).
    pub fn and(a: E, b: E, result: E) -> Self {
        Self { op_value: 2, a, b, result }
    }

    /// Bitwise XOR message (label = 6).
    pub fn xor(a: E, b: E, result: E) -> Self {
        Self { op_value: 6, a, b, result }
    }
}

impl<E: PrimeCharacteristicRing + Clone> BitwiseMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([
            E::from_u16(self.op_value),
            self.a.clone(),
            self.b.clone(),
            self.result.clone(),
        ])
    }
}

// DECODER MESSAGES
// ================================================================================================

/// Block stack message: `[block_id, parent_id, is_loop, ctx, fmp, depth, fn_hash[4]]`.
///
/// `Simple` — for blocks that don't save context (JOIN/SPLIT/SPAN/DYN/LOOP/RESPAN/END-simple).
/// Context fields are encoded as zeros.
///
/// `Full` — for blocks that save/restore the caller's execution context
/// (CALL/SYSCALL/DYNCALL/END-call).
#[derive(Clone)]
pub enum BlockStackMsg<E> {
    Simple {
        block_id: E,
        parent_id: E,
        is_loop: E,
    },
    Full {
        block_id: E,
        parent_id: E,
        is_loop: E,
        ctx: E,
        fmp: E,
        depth: E,
        fn_hash: [E; 4],
    },
}

impl<E: PrimeCharacteristicRing + Clone> BlockStackMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        match self {
            Self::Simple { block_id, parent_id, is_loop } => challenges.encode([
                block_id.clone(),
                parent_id.clone(),
                is_loop.clone(),
                E::ZERO,
                E::ZERO,
                E::ZERO,
                E::ZERO,
                E::ZERO,
                E::ZERO,
                E::ZERO,
            ]),
            Self::Full {
                block_id,
                parent_id,
                is_loop,
                ctx,
                fmp,
                depth,
                fn_hash,
            } => challenges.encode([
                block_id.clone(),
                parent_id.clone(),
                is_loop.clone(),
                ctx.clone(),
                fmp.clone(),
                depth.clone(),
                fn_hash[0].clone(),
                fn_hash[1].clone(),
                fn_hash[2].clone(),
                fn_hash[3].clone(),
            ]),
        }
    }
}

/// Block hash queue message (7 elements):
/// `[parent, child_hash[4], is_first_child, is_loop_body]`.
///
/// `FirstChild` — first child of a JOIN (is_first_child = 1, is_loop_body = 0).
/// `Child` — non-first, non-loop child (is_first_child = 0, is_loop_body = 0).
/// `LoopBody` — loop body entry (is_first_child = 0, is_loop_body = 1).
/// `End` — removal at END; both flags are computed expressions.
#[derive(Clone)]
pub enum BlockHashMsg<E> {
    FirstChild {
        parent: E,
        child_hash: [E; 4],
    },
    Child {
        parent: E,
        child_hash: [E; 4],
    },
    LoopBody {
        parent: E,
        child_hash: [E; 4],
    },
    End {
        parent: E,
        child_hash: [E; 4],
        is_first_child: E,
        is_loop_body: E,
    },
}

impl<E: PrimeCharacteristicRing + Clone> BlockHashMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        let (parent, child_hash, is_first_child, is_loop_body) = match self {
            Self::FirstChild { parent, child_hash } => {
                (parent, child_hash, E::ONE, E::ZERO)
            },
            Self::Child { parent, child_hash } => (parent, child_hash, E::ZERO, E::ZERO),
            Self::LoopBody { parent, child_hash } => (parent, child_hash, E::ZERO, E::ONE),
            Self::End { parent, child_hash, is_first_child, is_loop_body } => {
                (parent, child_hash, is_first_child.clone(), is_loop_body.clone())
            },
        };
        challenges.encode([
            parent.clone(),
            child_hash[0].clone(),
            child_hash[1].clone(),
            child_hash[2].clone(),
            child_hash[3].clone(),
            is_first_child,
            is_loop_body,
        ])
    }
}

/// Op group table message (3 elements): `[batch_id, group_pos, group_value]`.
#[derive(Clone)]
pub struct OpGroupMsg<E> {
    pub batch_id: E,
    pub group_pos: E,
    pub group_value: E,
}

impl<E: PrimeCharacteristicRing + Clone> OpGroupMsg<E> {
    /// Create an op group message. Computes `group_pos = group_count - offset`.
    pub fn new<V>(batch_id: &E, group_count: V, offset: u16, group_value: E) -> Self
    where
        V: core::ops::Sub<E, Output = E> + Clone,
    {
        Self {
            batch_id: batch_id.clone(),
            group_pos: group_count - E::from_u16(offset),
            group_value,
        }
    }

    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([self.batch_id.clone(), self.group_pos.clone(), self.group_value.clone()])
    }
}

// STACK MESSAGE
// ================================================================================================

/// Stack overflow table message (3 elements): `[clk, val, prev]`.
///
/// `clk` is the clock cycle (unique address), `val` is the overflowed element,
/// `prev` is the pointer to the previous overflow entry (linked list).
#[derive(Clone)]
pub struct OverflowMsg<E> {
    pub clk: E,
    pub val: E,
    pub prev: E,
}

impl<E: PrimeCharacteristicRing + Clone> OverflowMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([self.clk.clone(), self.val.clone(), self.prev.clone()])
    }
}

// KERNEL ROM MESSAGE
// ================================================================================================

/// Kernel ROM message (5 elements): `[label, digest[4]]`.
///
/// The label is `KERNEL_PROC_CALL_LABEL = 16`, baked into encode.
#[derive(Clone)]
pub struct KernelRomMsg<E> {
    pub digest: [E; 4],
}

impl<E: PrimeCharacteristicRing + Clone> KernelRomMsg<E> {
    // KERNEL_PROC_CALL_LABEL = Felt::new(0b001111 + 1) = 16.
    const LABEL: u16 = 16;

    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([
            E::from_u16(Self::LABEL),
            self.digest[0].clone(),
            self.digest[1].clone(),
            self.digest[2].clone(),
            self.digest[3].clone(),
        ])
    }
}

// ACE MESSAGE
// ================================================================================================

/// ACE circuit evaluation init message (6 elements): `[label, clk, ctx, ptr, num_read, num_eval]`.
#[derive(Clone)]
pub struct AceInitMsg<E> {
    pub clk: E,
    pub ctx: E,
    pub ptr: E,
    pub num_read: E,
    pub num_eval: E,
}

impl<E: PrimeCharacteristicRing + Clone> AceInitMsg<E> {
    /// ACE_INIT_LABEL = Felt(0b0111 + 1) = 8.
    const LABEL: u16 = 8;

    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([
            E::from_u16(Self::LABEL),
            self.clk.clone(),
            self.ctx.clone(),
            self.ptr.clone(),
            self.num_read.clone(),
            self.num_eval.clone(),
        ])
    }
}

// RANGE CHECK MESSAGE
// ================================================================================================

/// Range check message (1 element): `[value]`.
///
/// The denominator is `α + β⁰ · value`.
#[derive(Clone)]
pub struct RangeMsg<E> {
    pub value: E,
}

impl<E: PrimeCharacteristicRing + Clone> RangeMsg<E> {
    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([self.value.clone()])
    }
}

// LOG-PRECOMPILE CAPACITY MESSAGE
// ================================================================================================

/// Log-precompile capacity state message (5 elements): `[label, cap[4]]`.
#[derive(Clone)]
pub struct LogCapacityMsg<E> {
    pub capacity: [E; 4],
}

impl<E: PrimeCharacteristicRing + Clone> LogCapacityMsg<E> {
    /// LOG_PRECOMPILE_LABEL = 14.
    const LABEL: u16 = crate::trace::LOG_PRECOMPILE_LABEL as u16;

    pub fn encode<EF>(&self, challenges: &Challenges<EF>) -> EF
    where
        EF: PrimeCharacteristicRing + Algebra<E>,
    {
        challenges.encode([
            E::from_u16(Self::LABEL),
            self.capacity[0].clone(),
            self.capacity[1].clone(),
            self.capacity[2].clone(),
            self.capacity[3].clone(),
        ])
    }
}

// TRAIT IMPLEMENTATIONS
// ================================================================================================

macro_rules! impl_logup_message {
    ($ty:ident) => {
        impl<E, EF> LogUpMessage<E, EF> for $ty<E>
        where
            E: PrimeCharacteristicRing + Clone,
            EF: PrimeCharacteristicRing + Algebra<E>,
        {
            fn encode(&self, challenges: &Challenges<EF>) -> EF {
                self.encode(challenges)
            }
        }
    };
}

impl_logup_message!(HasherMsg);
impl_logup_message!(MemoryMsg);
impl_logup_message!(BitwiseMsg);
impl_logup_message!(BlockStackMsg);
impl_logup_message!(BlockHashMsg);
impl_logup_message!(OpGroupMsg);
impl_logup_message!(OverflowMsg);
impl_logup_message!(KernelRomMsg);
impl_logup_message!(AceInitMsg);
impl_logup_message!(RangeMsg);
impl_logup_message!(LogCapacityMsg);
