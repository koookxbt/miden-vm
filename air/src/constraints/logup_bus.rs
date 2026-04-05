//! All-LogUp bus constraints for the Miden VM.
//!
//! Replaces the existing running-product + LogUp bus constraints with a unified
//! LogUp formulation using the rational-fraction algebra from [`super::logup`].
//!
//! Two entry points:
//! - [`enforce_main`] — 5 columns over system/decoder/stack/range columns
//! - [`enforce_chiplet`] — 3 columns over chiplet columns
//!
//! See `docs/src/design/bus_packing_summary.md` §8–9 for the full interaction inventory.

use core::array;

use miden_core::{FMP_ADDR, FMP_INIT_VALUE, field::PrimeCharacteristicRing, operations::opcodes};
use miden_crypto::stark::air::{ExtensionBuilder, LiftedAirBuilder, WindowAccess};

use super::{
    chiplets::{bitwise::P_BITWISE_K_TRANSITION, hasher},
    logup::{Column, RationalSet},
    logup_msg::*,
    op_flags::{ExprDecoderAccess, OpFlags},
};
use crate::{
    Felt, MainTraceRow,
    trace::{
        CHIPLETS_OFFSET, Challenges,
        chiplets::{
            HASHER_NODE_INDEX_COL_IDX, HASHER_SELECTOR_COL_RANGE, HASHER_STATE_COL_RANGE,
            NUM_ACE_SELECTORS,
            NUM_BITWISE_SELECTORS, NUM_KERNEL_ROM_SELECTORS, NUM_MEMORY_SELECTORS,
            ace::{
                ACE_INSTRUCTION_ID1_OFFSET, ACE_INSTRUCTION_ID2_OFFSET, CLK_IDX,
                CTX_IDX, EVAL_OP_IDX, ID_0_IDX, ID_1_IDX, ID_2_IDX, M_0_IDX, M_1_IDX, PTR_IDX,
                READ_NUM_EVAL_IDX, SELECTOR_BLOCK_IDX, SELECTOR_START_IDX, V_0_0_IDX, V_0_1_IDX,
                V_1_0_IDX, V_1_1_IDX, V_2_0_IDX, V_2_1_IDX,
            },
            bitwise::{self, BITWISE_AND_LABEL, BITWISE_XOR_LABEL},
            hasher::{
                HASH_CYCLE_LEN, LINEAR_HASH_LABEL, MP_VERIFY_LABEL, MR_UPDATE_NEW_LABEL,
                MR_UPDATE_OLD_LABEL, RETURN_HASH_LABEL, RETURN_STATE_LABEL,
            },
            kernel_rom::{KERNEL_PROC_CALL_LABEL, KERNEL_PROC_INIT_LABEL},
            memory::{
                self, MEMORY_READ_ELEMENT_LABEL, MEMORY_READ_WORD_LABEL,
                MEMORY_WRITE_ELEMENT_LABEL, MEMORY_WRITE_WORD_LABEL,
            },
        },
        decoder::{
            ADDR_COL_IDX, GROUP_COUNT_COL_IDX, HASHER_STATE_RANGE, IN_SPAN_COL_IDX,
            IS_CALL_FLAG_COL_IDX, IS_LOOP_BODY_FLAG_COL_IDX, IS_LOOP_FLAG_COL_IDX,
            IS_SYSCALL_FLAG_COL_IDX, OP_BATCH_FLAGS_OFFSET, OP_BITS_RANGE, USER_OP_HELPERS_OFFSET,
        },
        log_precompile::{
            HELPER_ADDR_IDX, HELPER_CAP_PREV_RANGE, STACK_CAP_NEXT_RANGE, STACK_COMM_RANGE,
            STACK_R0_RANGE, STACK_R1_RANGE, STACK_TAG_RANGE,
        },
    },
};

// COLUMN INDICES
// ================================================================================================

/// Main trace LogUp column indices (in the auxiliary/permutation trace).
pub mod main_cols {
    pub const M1: usize = 0;
    pub const M2: usize = 1;
    pub const M3: usize = 2;
    pub const M4: usize = 3;
    pub const M5: usize = 4;
}

/// Chiplet trace LogUp column indices.
pub mod chip_cols {
    pub const C1: usize = 5;
    pub const C2: usize = 6;
    pub const C3: usize = 7;
}

// MAIN TRACE LOGUP
// ================================================================================================

/// Enforces all main-trace LogUp bus constraints (5 columns: M1–M5).
pub fn enforce_main<AB>(
    builder: &mut AB,
    local: &MainTraceRow<AB::Var>,
    next: &MainTraceRow<AB::Var>,
) where
    AB: LiftedAirBuilder<F = Felt>,
{
    let r = builder.permutation_randomness();
    let challenges = Challenges::<AB::ExprEF>::new(r[0].into(), r[1].into());
    let op_flags = OpFlags::new(ExprDecoderAccess::<_, AB::Expr>::new(local));
    let op_flags_next = OpFlags::new(ExprDecoderAccess::<_, AB::Expr>::new(next));

    // Trace section aliases.
    let dec = &local.decoder;
    let dec_next = &next.decoder;
    let stk = &local.stack;
    let stk_next = &next.stack;

    // Decoder Var bindings (Copy — no clones needed).
    let addr = dec[ADDR_COL_IDX];
    let addr_next = dec_next[ADDR_COL_IDX];
    let h: [AB::Var; 8] = array::from_fn(|i| dec[HASHER_STATE_RANGE.start + i]);
    let h1_next = dec_next[HASHER_STATE_RANGE.start + 1];
    let is_loop_flag = dec[IS_LOOP_FLAG_COL_IDX];
    let is_loop_body_flag = dec[IS_LOOP_BODY_FLAG_COL_IDX];
    let is_call_flag = dec[IS_CALL_FLAG_COL_IDX];
    let is_syscall_flag = dec[IS_SYSCALL_FLAG_COL_IDX];
    let helper0 = dec[USER_OP_HELPERS_OFFSET];

    // Stack Var bindings.
    let s0 = stk[0];
    let s1 = stk[1];
    let b0 = stk[16];
    let b1 = stk[17];
    let b0_next = stk_next[16];
    let b1_next = stk_next[17];

    // System Var bindings.
    let clk = local.clk;
    let ctx = local.ctx;
    let ctx_next = next.ctx;

    // Range Var bindings.
    let range_m = local.range[0];
    let range_v = local.range[1];

    // Pre-lifted Expr arrays (used across multiple groups).
    let fn_hash: [AB::Expr; 4] = local.fn_hash.map(Into::into);
    let fn_hash_next: [AB::Expr; 4] = next.fn_hash.map(Into::into);
    let he: [AB::Expr; 8] = h.map(Into::into);
    let h_first: [AB::Expr; 4] = array::from_fn(|i| h[i].into());
    let h_second: [AB::Expr; 4] = array::from_fn(|i| h[4 + i].into());

    // Stack as 4-element words (Expr). The stack is logically [word0, word1, word2, word3, b0, b1,
    // h0].
    let stk_words: [[AB::Expr; 4]; 4] =
        array::from_fn(|w| array::from_fn(|i| stk[w * 4 + i].into()));
    let stk_next_words: [[AB::Expr; 4]; 4] =
        array::from_fn(|w| array::from_fn(|i| stk_next[w * 4 + i].into()));

    // =====================================================================
    // G_bstack: block-stack table
    // =====================================================================
    let g_bstack = {
        let mut set = RationalSet::new(&challenges);

        // JOIN/SPLIT/SPAN/DYN: simple push
        let f = op_flags.join() + op_flags.split() + op_flags.span() + op_flags.dyn_op();
        set.add_single(f, || BlockStackMsg::Simple {
            block_id: addr_next.into(),
            parent_id: addr.into(),
            is_loop: AB::Expr::ZERO,
        });

        // LOOP: push with is_loop = s0
        set.add_single(op_flags.loop_op(), || BlockStackMsg::Simple {
            block_id: addr_next.into(),
            parent_id: addr.into(),
            is_loop: s0.into(),
        });

        // DYNCALL: full push with h4/h5 as fmp/depth
        set.add_single(op_flags.dyncall(), || BlockStackMsg::Full {
            block_id: addr_next.into(),
            parent_id: addr.into(),
            is_loop: AB::Expr::ZERO,
            ctx: ctx.into(),
            fmp: h[4].into(),
            depth: h[5].into(),
            fn_hash: fn_hash.clone(),
        });

        // CALL/SYSCALL: full push with context
        let f = op_flags.call() + op_flags.syscall();
        set.add_single(f, || BlockStackMsg::Full {
            block_id: addr_next.into(),
            parent_id: addr.into(),
            is_loop: AB::Expr::ZERO,
            ctx: ctx.into(),
            fmp: b0.into(),
            depth: b1.into(),
            fn_hash: fn_hash.clone(),
        });

        // END (simple blocks): pop
        let f = op_flags.end() * (AB::Expr::ONE - is_call_flag - is_syscall_flag);
        set.remove_single(f, || BlockStackMsg::Simple {
            block_id: addr.into(),
            parent_id: addr_next.into(),
            is_loop: is_loop_flag.into(),
        });

        // END (after CALL/SYSCALL): pop with restored context
        let f = op_flags.end() * (is_call_flag + is_syscall_flag);
        set.remove_single(f, || BlockStackMsg::Full {
            block_id: addr.into(),
            parent_id: addr_next.into(),
            is_loop: is_loop_flag.into(),
            ctx: ctx_next.into(),
            fmp: b0_next.into(),
            depth: b1_next.into(),
            fn_hash: fn_hash_next.clone(),
        });

        // RESPAN: simultaneous push + pop
        set.add_batch(op_flags.respan(), |b| {
            b.add(BlockStackMsg::Simple {
                block_id: addr_next.into(),
                parent_id: h1_next.into(),
                is_loop: AB::Expr::ZERO,
            });
            b.remove(BlockStackMsg::Simple {
                block_id: addr.into(),
                parent_id: h1_next.into(),
                is_loop: AB::Expr::ZERO,
            });
        });

        set
    };

    // =====================================================================
    // G_rtable: range table response (always active, m = M)
    // =====================================================================
    let g_rtable = RationalSet::always(&challenges, |b| {
        b.insert(range_m.into(), RangeMsg { value: range_v.into() });
    });

    // =====================================================================
    // G_bqueue: block-hash queue
    // =====================================================================
    let g_bqueue = {
        let mut set = RationalSet::new(&challenges);
        let parent: AB::Expr = addr_next.into();

        // JOIN: two children (left h[0..3], right h[4..7])
        set.add_batch(op_flags.join(), |b| {
            b.add(BlockHashMsg::FirstChild {
                parent: parent.clone(),
                child_hash: h_first.clone(),
            });
            b.add(BlockHashMsg::Child {
                parent: parent.clone(),
                child_hash: h_second.clone(),
            });
        });

        // SPLIT: conditional select child
        set.add_single(op_flags.split(), || {
            let split_h: [AB::Expr; 4] =
                array::from_fn(|i| s0 * h[i] + (AB::Expr::ONE - s0) * h[i + 4]);
            BlockHashMsg::Child {
                parent: parent.clone(),
                child_hash: split_h,
            }
        });

        // LOOP body + REPEAT (merged: same message)
        let f = op_flags.loop_op() * s0 + op_flags.repeat();
        set.add_single(f, || BlockHashMsg::LoopBody {
            parent: parent.clone(),
            child_hash: h_first.clone(),
        });

        // DYN/DYNCALL/CALL/SYSCALL: single child (merged: same message)
        let f = op_flags.dyn_op() + op_flags.dyncall() + op_flags.call() + op_flags.syscall();
        set.add_single(f, || BlockHashMsg::Child {
            parent: parent.clone(),
            child_hash: h_first.clone(),
        });

        // END: remove
        set.remove_single(op_flags.end(), || {
            let is_first_child: AB::Expr =
                AB::Expr::ONE - op_flags_next.end() - op_flags_next.repeat() - op_flags_next.halt();
            BlockHashMsg::End {
                parent: parent.clone(),
                child_hash: h_first.clone(),
                is_first_child,
                is_loop_body: is_loop_body_flag.into(),
            }
        });

        set
    };

    // =====================================================================
    // G_creq: chiplet requests
    // =====================================================================
    let g_creq = {
        let mut set = RationalSet::new(&challenges);

        let addr_e: AB::Expr = addr.into();
        let addr_next_e: AB::Expr = addr_next.into();
        let helper0_e: AB::Expr = helper0.into();
        let last_off: AB::Expr = AB::Expr::from_u16((HASH_CYCLE_LEN - 1) as u16);
        let cycle_len: AB::Expr = AB::Expr::from_u16(HASH_CYCLE_LEN as u16);
        let zeros8: [AB::Expr; 8] = array::from_fn(|_| AB::Expr::ZERO);

        // Non-word-aligned stack slices and 12-element states (only used by G_creq).
        let old_root: [AB::Expr; 4] = array::from_fn(|i| stk[6 + i].into());
        let new_node: [AB::Expr; 4] = array::from_fn(|i| stk[10 + i].into());
        let stk_state: [AB::Expr; 12] = array::from_fn(|i| stk[i].into());
        let stk_next_state: [AB::Expr; 12] = array::from_fn(|i| stk_next[i].into());

        // Reusable memory header: MLOAD/MSTORE/MLOADW/MSTOREW share ctx, addr=s0, clk.
        let mem = MemoryHeader {
            ctx: ctx.into(),
            addr: s0.into(),
            clk: clk.into(),
        };

        // JOIN
        set.remove_single(op_flags.join(), || {
            HasherMsg::control_block(addr_next_e.clone(), &he, opcodes::JOIN)
        });

        // SPLIT
        set.remove_single(op_flags.split(), || {
            HasherMsg::control_block(addr_next_e.clone(), &he, opcodes::SPLIT)
        });

        // LOOP
        set.remove_single(op_flags.loop_op(), || {
            HasherMsg::control_block(addr_next_e.clone(), &he, opcodes::LOOP)
        });

        // SPAN
        set.remove_single(op_flags.span(), || {
            HasherMsg::control_block(addr_next_e.clone(), &he, 0)
        });

        // RESPAN
        set.remove_single(op_flags.respan(), || {
            HasherMsg::absorption(
                addr_next_e.clone() - AB::Expr::ONE,
                he.clone().try_into().ok().unwrap(),
            )
        });

        // END
        set.remove_single(op_flags.end(), || {
            HasherMsg::return_hash(addr_e + last_off.clone(), h_first.clone())
        });

        // CALL: control block + FMP write
        set.add_batch(op_flags.call(), |b| {
            b.remove(HasherMsg::control_block(addr_next_e.clone(), &he, opcodes::CALL));
            let fmp_hdr = MemoryHeader {
                ctx: ctx_next.into(),
                addr: FMP_ADDR.into(),
                clk: clk.into(),
            };
            b.remove(fmp_hdr.write_element(FMP_INIT_VALUE.into()));
        });

        // SYSCALL: control block + kernel ROM lookup
        set.add_batch(op_flags.syscall(), |b| {
            b.remove(HasherMsg::control_block(addr_next_e.clone(), &he, opcodes::SYSCALL));
            b.remove(KernelRomMsg { digest: h_first.clone() });
        });

        // DYN: zeros-hasher + callee word read
        set.add_batch(op_flags.dyn_op(), |b| {
            b.remove(HasherMsg::control_block(addr_next_e.clone(), &zeros8, opcodes::DYN));
            b.remove(mem.read_word(h_first.clone()));
        });

        // DYNCALL: zeros-hasher + callee word read + FMP write
        set.add_batch(op_flags.dyncall(), |b| {
            b.remove(HasherMsg::control_block(addr_next_e.clone(), &zeros8, opcodes::DYNCALL));
            b.remove(mem.read_word(h_first.clone()));
            let fmp_hdr = MemoryHeader {
                ctx: ctx_next.into(),
                addr: FMP_ADDR.into(),
                clk: clk.into(),
            };
            b.remove(fmp_hdr.write_element(FMP_INIT_VALUE.into()));
        });

        // HPERM: full state in + out
        set.add_batch(op_flags.hperm(), |b| {
            b.remove(HasherMsg::linear_hash_init(helper0_e.clone(), stk_state));
            b.remove(HasherMsg::return_state(helper0_e.clone() + last_off.clone(), stk_next_state));
        });

        // MPVERIFY: leaf word in + root word out
        let mp_index = stk[5];
        let mp_depth = stk[4];
        set.add_batch(op_flags.mpverify(), |b| {
            b.remove(HasherMsg::merkle_verify_init(
                helper0_e.clone(),
                mp_index.into(),
                stk_words[0].clone(),
            ));
            b.remove(HasherMsg::return_hash(
                helper0_e.clone() + mp_depth * cycle_len.clone() - AB::Expr::ONE,
                old_root.clone(),
            ));
        });

        // MRUPDATE: 4 word messages
        let mr_depth = stk[4];
        let mr_index = stk[5];
        set.add_batch(op_flags.mrupdate(), |b| {
            b.remove(HasherMsg::merkle_old_init(
                helper0_e.clone(),
                mr_index.into(),
                stk_words[0].clone(),
            ));
            b.remove(HasherMsg::return_hash(
                helper0_e.clone() + mr_depth * cycle_len.clone() - AB::Expr::ONE,
                old_root,
            ));
            b.remove(HasherMsg::merkle_new_init(
                helper0_e.clone() + mr_depth * cycle_len.clone(),
                mr_index.into(),
                new_node,
            ));
            b.remove(HasherMsg::return_hash(
                helper0_e + mr_depth * (cycle_len.clone() + cycle_len) - AB::Expr::ONE,
                stk_next_words[0].clone(),
            ));
        });

        // MLOAD
        set.remove_single(op_flags.mload(), || mem.read_element(stk_next[0].into()));

        // MSTORE
        set.remove_single(op_flags.mstore(), || mem.write_element(s1.into()));

        // MLOADW
        set.remove_single(op_flags.mloadw(), || mem.read_word(stk_next_words[0].clone()));

        // MSTOREW
        set.remove_single(op_flags.mstorew(), || {
            mem.write_word([s1.into(), stk[2].into(), stk[3].into(), stk[4].into()])
        });

        // U32AND
        set.remove_single(op_flags.u32and(), || {
            BitwiseMsg::and(s0.into(), s1.into(), stk_next[0].into())
        });

        // U32XOR
        set.remove_single(op_flags.u32xor(), || {
            BitwiseMsg::xor(s0.into(), s1.into(), stk_next[0].into())
        });

        // EVALCIRCUIT
        set.remove_single(op_flags.evalcircuit(), || AceInitMsg {
            clk: clk.into(),
            ctx: ctx.into(),
            ptr: s0.into(),
            num_read: s1.into(),
            num_eval: stk[2].into(),
        });

        // LOGPRECOMPILE: hasher in + out
        let log_addr = dec[USER_OP_HELPERS_OFFSET + HELPER_ADDR_IDX];
        let cap_prev: [AB::Var; 4] =
            array::from_fn(|i| dec[USER_OP_HELPERS_OFFSET + HELPER_CAP_PREV_RANGE.start + i]);
        let cap_next: [AB::Var; 4] = array::from_fn(|i| stk_next[STACK_CAP_NEXT_RANGE.start + i]);
        let comm: [AB::Expr; 4] = array::from_fn(|i| stk[STACK_COMM_RANGE.start + i].into());
        let tag: [AB::Expr; 4] = array::from_fn(|i| stk[STACK_TAG_RANGE.start + i].into());
        let r0: [AB::Expr; 4] = array::from_fn(|i| stk_next[STACK_R0_RANGE.start + i].into());
        let r1: [AB::Expr; 4] = array::from_fn(|i| stk_next[STACK_R1_RANGE.start + i].into());
        let logpre_in: [AB::Expr; 12] = [
            comm[0].clone(),
            comm[1].clone(),
            comm[2].clone(),
            comm[3].clone(),
            tag[0].clone(),
            tag[1].clone(),
            tag[2].clone(),
            tag[3].clone(),
            cap_prev[0].into(),
            cap_prev[1].into(),
            cap_prev[2].into(),
            cap_prev[3].into(),
        ];
        let logpre_out: [AB::Expr; 12] = [
            r0[0].clone(),
            r0[1].clone(),
            r0[2].clone(),
            r0[3].clone(),
            r1[0].clone(),
            r1[1].clone(),
            r1[2].clone(),
            r1[3].clone(),
            cap_next[0].into(),
            cap_next[1].into(),
            cap_next[2].into(),
            cap_next[3].into(),
        ];
        set.add_batch(op_flags.log_precompile(), |b| {
            b.remove(HasherMsg::linear_hash_init(log_addr.into(), logpre_in));
            b.remove(HasherMsg::return_state(log_addr + last_off, logpre_out));
        });

        // TODO: MSTREAM, PIPE, CRYPTOSTREAM, HORNERBASE, HORNEREXT
        set
    };

    // =====================================================================
    // G_opgrp: op group table
    // =====================================================================
    let g_opgrp = {
        let mut set = RationalSet::new(&challenges);
        let gc = dec[GROUP_COUNT_COL_IDX];
        let gc_next = dec_next[GROUP_COUNT_COL_IDX];
        let batch_id: AB::Expr = addr_next.into();
        let c0 = dec[OP_BATCH_FLAGS_OFFSET];
        let c1 = dec[OP_BATCH_FLAGS_OFFSET + 1];
        let c2 = dec[OP_BATCH_FLAGS_OFFSET + 2];

        // g8
        set.add_batch(c0.into(), |b| {
            for i in 1u16..=7 {
                b.add(OpGroupMsg::new(&batch_id, gc, i, h[i as usize].into()));
            }
        });
        // g4
        set.add_batch((AB::Expr::ONE - c0) * c1 * (AB::Expr::ONE - c2), |b| {
            for i in 1u16..=3 {
                b.add(OpGroupMsg::new(&batch_id, gc, i, h[i as usize].into()));
            }
        });
        // g2
        let f = (AB::Expr::ONE - c0) * (AB::Expr::ONE - c1) * c2;
        set.add_single(f, || OpGroupMsg::new(&batch_id, gc, 1, h[1].into()));

        // Removal
        let f = dec[IN_SPAN_COL_IDX] * (gc.clone() - gc_next);
        set.remove_single(f, || {
            let is_push: AB::Expr = op_flags.push();
            let h0_next = dec_next[HASHER_STATE_RANGE.start];
            let opcode_next: AB::Expr = (0..7).fold(AB::Expr::ZERO, |acc, i| {
                acc + dec_next[1 + i] * AB::Expr::from_u16(1u16 << i)
            });
            let group_value = is_push.clone() * stk_next[0]
                + (AB::Expr::ONE - is_push) * (h0_next * AB::Expr::from_u16(128) + opcode_next);
            OpGroupMsg {
                batch_id: addr.into(),
                group_pos: gc.into(),
                group_value,
            }
        });

        set
    };

    // =====================================================================
    // G_rstack + G_logcap: range stack lookups + log capacity (ME → one set)
    // =====================================================================
    let g_rstack_logcap = {
        let mut set = RationalSet::new(&challenges);

        // Range: 4 simultaneous lookups
        let op_bit4 = dec[OP_BITS_RANGE.start + 4];
        let op_bit5 = dec[OP_BITS_RANGE.start + 5];
        let op_bit6 = dec[OP_BITS_RANGE.start + 6];
        let f_u32rc: AB::Expr =
            op_bit6.into() * (AB::Expr::ONE - op_bit5) * (AB::Expr::ONE - op_bit4);
        set.add_batch(f_u32rc, |b| {
            let helpers: [AB::Var; 4] = array::from_fn(|i| dec[USER_OP_HELPERS_OFFSET + i]);
            for i in 0..4 {
                b.remove(RangeMsg { value: helpers[i].into() });
            }
        });

        // Log-precompile capacity
        set.add_batch(op_flags.log_precompile(), |b| {
            let cap_prev: [AB::Var; 4] =
                array::from_fn(|i| dec[USER_OP_HELPERS_OFFSET + HELPER_CAP_PREV_RANGE.start + i]);
            let cap_next: [AB::Var; 4] =
                array::from_fn(|i| stk_next[STACK_CAP_NEXT_RANGE.start + i]);
            b.remove(LogCapacityMsg { capacity: cap_prev.map(Into::into) });
            b.add(LogCapacityMsg { capacity: cap_next.map(Into::into) });
        });

        set
    };

    // =====================================================================
    // Combine groups into columns and emit constraints
    // =====================================================================

    let aux = builder.permutation();
    let aux_local = aux.current_slice();
    let aux_next = aux.next_slice();

    let mut m1 = Column::from_set(aux_local[main_cols::M1].into(), aux_next[main_cols::M1].into(), g_bstack);
    m1.add_set(g_rtable);
    m1.constrain(builder);

    Column::from_set(aux_local[main_cols::M2].into(), aux_next[main_cols::M2].into(), g_bqueue)
        .constrain(builder);
    Column::from_set(aux_local[main_cols::M3].into(), aux_next[main_cols::M3].into(), g_creq)
        .constrain(builder);
    Column::from_set(aux_local[main_cols::M4].into(), aux_next[main_cols::M4].into(), g_rstack_logcap)
        .constrain(builder);
    Column::from_set(aux_local[main_cols::M5].into(), aux_next[main_cols::M5].into(), g_opgrp)
        .constrain(builder);
}

// CHIPLET TRACE LOGUP
// ================================================================================================

// Chiplet-local column offsets (relative to `local.chiplets[]`).
const S_START: usize = HASHER_SELECTOR_COL_RANGE.start - CHIPLETS_OFFSET;
const H_START: usize = HASHER_STATE_COL_RANGE.start - CHIPLETS_OFFSET;
const IDX_COL: usize = HASHER_NODE_INDEX_COL_IDX - CHIPLETS_OFFSET;

/// ACE chiplet column offset (after s0, s1, s2, s3 chiplet selectors).
const ACE_OFFSET: usize = 4;

/// Enforces all chiplet-trace LogUp bus constraints (3 columns: C1–C3).
///
/// - **C1** — chiplet bus responses (hasher, bitwise, memory, ACE init, kernel ROM)
/// - **C2** — hash-kernel virtual table (sibling table + ACE memory reads)
/// - **C3** — ACE wiring bus (READ/EVAL wire interactions)
pub fn enforce_chiplet<AB>(
    builder: &mut AB,
    local: &MainTraceRow<AB::Var>,
    next: &MainTraceRow<AB::Var>,
) where
    AB: LiftedAirBuilder<F = Felt>,
{
    let r = builder.permutation_randomness();
    let challenges = Challenges::<AB::ExprEF>::new(r[0].into(), r[1].into());

    // =====================================================================
    // Periodic values
    // =====================================================================
    let (cycle_row_0, cycle_row_31, k_transition) = {
        let p = builder.periodic_values();
        let cycle_row_0: AB::Expr = p[hasher::periodic::P_CYCLE_ROW_0].into();
        let cycle_row_31: AB::Expr = p[hasher::periodic::P_CYCLE_ROW_31].into();
        let k_transition: AB::Expr = p[P_BITWISE_K_TRANSITION].into();
        (cycle_row_0, cycle_row_31, k_transition)
    };

    // =====================================================================
    // Chiplet selector flags
    // =====================================================================
    let s0: AB::Expr = local.chiplets[0].clone().into();
    let s1: AB::Expr = local.chiplets[1].clone().into();
    let s2: AB::Expr = local.chiplets[2].clone().into();
    let s3: AB::Expr = local.chiplets[3].clone().into();
    let s4: AB::Expr = local.chiplets[4].clone().into();

    let is_hasher: AB::Expr = AB::Expr::ONE - s0.clone();

    // Hasher internal selectors (meaningful when is_hasher = 1).
    let hs0: AB::Expr = local.chiplets[S_START].clone().into();
    let hs1: AB::Expr = local.chiplets[S_START + 1].clone().into();
    let hs2: AB::Expr = local.chiplets[S_START + 2].clone().into();

    // Hasher state (12 elements) and node index.
    let h: [AB::Expr; 12] = array::from_fn(|i| local.chiplets[H_START + i].clone().into());
    let h_next: [AB::Expr; 12] = array::from_fn(|i| next.chiplets[H_START + i].clone().into());
    let node_index: AB::Expr = local.chiplets[IDX_COL].clone().into();
    let node_index_next: AB::Expr = next.chiplets[IDX_COL].clone().into();

    // Hasher addr proxy: clk + 1 (hasher row address is clk-based).
    let hasher_addr: AB::Expr = local.clk.clone().into() + AB::Expr::ONE;

    // Bit for conditional leaf selection: bit = node_index - 2 * node_index_next.
    let bit: AB::Expr = node_index.clone() - node_index_next.clone().double();

    // =====================================================================
    // C1: Chiplet bus responses
    // =====================================================================
    let g_chiplet_resp = {
        let mut set = RationalSet::new(&challenges);

        // --- Hasher responses (7 ME flags, each degree 5 within is_hasher) ---

        // f_bp: linear hash / 2-to-1 hash init — full 15-element state.
        let f = is_hasher.clone()
            * hasher::flags::f_bp(
                cycle_row_0.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::State {
                label_value: LINEAR_HASH_LABEL as u16 + 16,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                state: h.clone(),
            }
        });

        // f_mp: Merkle path verify init — conditional leaf word.
        let f = is_hasher.clone()
            * hasher::flags::f_mp(
                cycle_row_0.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::Word {
                label_value: MP_VERIFY_LABEL as u16 + 16,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                word: leaf_word(&h, &bit),
            }
        });

        // f_mv: Merkle update old path init — conditional leaf word.
        let f = is_hasher.clone()
            * hasher::flags::f_mv(
                cycle_row_0.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::Word {
                label_value: MR_UPDATE_OLD_LABEL as u16 + 16,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                word: leaf_word(&h, &bit),
            }
        });

        // f_mu: Merkle update new path init — conditional leaf word.
        let f = is_hasher.clone()
            * hasher::flags::f_mu(
                cycle_row_0.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::Word {
                label_value: MR_UPDATE_NEW_LABEL as u16 + 16,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                word: leaf_word(&h, &bit),
            }
        });

        // f_hout: return hash — 4-element digest from RATE0.
        let f = is_hasher.clone()
            * hasher::flags::f_hout(
                cycle_row_31.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::Word {
                label_value: RETURN_HASH_LABEL as u16 + 32,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                word: [h[0].clone(), h[1].clone(), h[2].clone(), h[3].clone()],
            }
        });

        // f_sout: return full state — 15-element state message.
        let f = is_hasher.clone()
            * hasher::flags::f_sout(
                cycle_row_31.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::State {
                label_value: RETURN_STATE_LABEL as u16 + 32,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                state: h.clone(),
            }
        });

        // f_abp: absorption — 8-element rate from the NEXT row.
        let f = is_hasher.clone()
            * hasher::flags::f_abp(
                cycle_row_31.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        set.add_single(f, || {
            HasherMsg::Rate {
                label_value: LINEAR_HASH_LABEL as u16 + 32,
                addr: hasher_addr.clone(),
                node_index: node_index.clone(),
                rate: array::from_fn(|i| h_next[i].clone()),
            }
        });

        // --- Bitwise response (flag deg 4) ---
        // Active on last row of 8-row cycle: s0*(1-s1)*(1-k_transition).
        let is_bitwise_responding: AB::Expr =
            s0.clone() * (AB::Expr::ONE - s1.clone()) * (AB::Expr::ONE - k_transition);
        set.add_single(is_bitwise_responding, || {
            let bw_offset = NUM_BITWISE_SELECTORS;
            let sel: AB::Expr = local.chiplets[bw_offset].clone().into();
            let label: AB::Expr = (AB::Expr::ONE - sel.clone()) * AB::Expr::from(BITWISE_AND_LABEL)
                + sel * AB::Expr::from(BITWISE_XOR_LABEL);
            let a: AB::Expr = local.chiplets[bw_offset + bitwise::A_COL_IDX].clone().into();
            let b: AB::Expr = local.chiplets[bw_offset + bitwise::B_COL_IDX].clone().into();
            let z: AB::Expr = local.chiplets[bw_offset + bitwise::OUTPUT_COL_IDX].clone().into();
            BitwiseResponseMsg { label, a, b, z }
        });

        // --- Memory response (flag deg 3) ---
        // Active on all memory rows: s0*s1*(1-s2).
        let is_memory: AB::Expr =
            s0.clone() * s1.clone() * (AB::Expr::ONE - s2.clone());
        set.add_single(is_memory, || {
            compute_memory_response_msg::<AB>(local)
        });

        // --- ACE init response (flag deg 5) ---
        // Active on ACE start rows: s0*s1*s2*(1-s3)*start_sel.
        let ace_start: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + SELECTOR_START_IDX]
            .clone()
            .into();
        let is_ace: AB::Expr =
            s0.clone() * s1.clone() * s2.clone() * (AB::Expr::ONE - s3.clone()) * ace_start;
        set.add_single(is_ace, || {
            let clk: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CLK_IDX].clone().into();
            let ctx: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CTX_IDX].clone().into();
            let ptr: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + PTR_IDX].clone().into();
            let read_num_eval: AB::Expr =
                local.chiplets[NUM_ACE_SELECTORS + READ_NUM_EVAL_IDX].clone().into();
            let num_eval_rows: AB::Expr = read_num_eval + AB::Expr::ONE;
            let id_0: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + ID_0_IDX].clone().into();
            let num_read_rows: AB::Expr = id_0 + AB::Expr::ONE - num_eval_rows.clone();
            AceInitMsg { clk, ctx, ptr, num_read: num_read_rows, num_eval: num_eval_rows }
        });

        // --- Kernel ROM response (flag deg 5) ---
        // Active on all kernel ROM rows: s0*s1*s2*s3*(1-s4).
        let is_kernel_rom: AB::Expr =
            s0.clone() * s1.clone() * s2.clone() * s3.clone() * (AB::Expr::ONE - s4.clone());
        set.add_single(is_kernel_rom, || {
            let s_first: AB::Expr = local.chiplets[NUM_KERNEL_ROM_SELECTORS].clone().into();
            let init_label: AB::Expr = AB::Expr::from(KERNEL_PROC_INIT_LABEL);
            let call_label: AB::Expr = AB::Expr::from(KERNEL_PROC_CALL_LABEL);
            let label: AB::Expr =
                s_first.clone() * init_label + (AB::Expr::ONE - s_first) * call_label;
            let root0: AB::Expr =
                local.chiplets[NUM_KERNEL_ROM_SELECTORS + 1].clone().into();
            let root1: AB::Expr =
                local.chiplets[NUM_KERNEL_ROM_SELECTORS + 2].clone().into();
            let root2: AB::Expr =
                local.chiplets[NUM_KERNEL_ROM_SELECTORS + 3].clone().into();
            let root3: AB::Expr =
                local.chiplets[NUM_KERNEL_ROM_SELECTORS + 4].clone().into();
            KernelRomResponseMsg { label, digest: [root0, root1, root2, root3] }
        });

        set
    };

    // Shared sibling msg constructors.
    let sibling_curr = || SiblingMsg {
        node_index: node_index.clone(),
        bit: bit.clone(),
        h_lo: array::from_fn(|i| h[i].clone()),
        h_hi: array::from_fn(|i| h[4 + i].clone()),
    };
    let sibling_next = || SiblingMsg {
        node_index: node_index.clone(),
        bit: bit.clone(),
        h_lo: array::from_fn(|i| h_next[i].clone()),
        h_hi: array::from_fn(|i| h_next[4 + i].clone()),
    };

    // =====================================================================
    // C2: Hash-kernel virtual table (sibling table + ACE memory reads)
    // =====================================================================
    let g_hash_kernel = {
        let mut set = RationalSet::new(&challenges);

        // --- Sibling table ---
        // MV/MVA: add (response — store sibling during old Merkle path).
        // MU/MUA: remove (request — retrieve sibling during new Merkle path).

        let f_mv: AB::Expr = is_hasher.clone()
            * hasher::flags::f_mv(cycle_row_0.clone(), hs0.clone(), hs1.clone(), hs2.clone());
        let f_mu: AB::Expr = is_hasher.clone()
            * hasher::flags::f_mu(cycle_row_0.clone(), hs0.clone(), hs1.clone(), hs2.clone());
        let f_mva: AB::Expr = is_hasher.clone()
            * hasher::flags::f_mva(
                cycle_row_31.clone(),
                hs0.clone(),
                hs1.clone(),
                hs2.clone(),
            );
        let f_mua: AB::Expr = is_hasher.clone()
            * hasher::flags::f_mua(cycle_row_31, hs0, hs1, hs2);

        // MV (+1) and MU (-1) share sibling_curr.
        set.replace(f_mv, f_mu, sibling_curr);
        // MVA (+1) and MUA (-1) share sibling_next.
        set.replace(f_mva, f_mua, sibling_next);

        // --- ACE memory reads ---
        let is_ace_row: AB::Expr =
            s0.clone() * s1.clone() * s2.clone() * (AB::Expr::ONE - s3.clone());
        let block_sel: AB::Expr =
            local.chiplets[NUM_ACE_SELECTORS + SELECTOR_BLOCK_IDX].clone().into();

        // f_ace_read: word read on READ rows.
        let f_ace_read: AB::Expr = is_ace_row.clone() * (AB::Expr::ONE - block_sel.clone());
        set.remove_single(f_ace_read, || {
            let ace_clk: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CLK_IDX].clone().into();
            let ace_ctx: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CTX_IDX].clone().into();
            let ace_ptr: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + PTR_IDX].clone().into();
            let v0_0: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + V_0_0_IDX].clone().into();
            let v0_1: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + V_0_1_IDX].clone().into();
            let v1_0: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + V_1_0_IDX].clone().into();
            let v1_1: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + V_1_1_IDX].clone().into();
            MemoryMsg::Word {
                op_value: MEMORY_READ_WORD_LABEL as u16,
                header: MemoryHeader {
                    ctx: ace_ctx,
                    addr: ace_ptr,
                    clk: ace_clk,
                },
                word: [v0_0, v0_1, v1_0, v1_1],
            }
        });

        // f_ace_eval: element read on EVAL rows.
        let f_ace_eval: AB::Expr = is_ace_row * block_sel;
        set.remove_single(f_ace_eval, || {
            let ace_clk: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CLK_IDX].clone().into();
            let ace_ctx: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + CTX_IDX].clone().into();
            let ace_ptr: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + PTR_IDX].clone().into();
            let id_1: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + ID_1_IDX].clone().into();
            let id_2: AB::Expr = local.chiplets[NUM_ACE_SELECTORS + ID_2_IDX].clone().into();
            let eval_op: AB::Expr =
                local.chiplets[NUM_ACE_SELECTORS + EVAL_OP_IDX].clone().into();
            let element: AB::Expr = id_1
                + id_2 * AB::Expr::from(ACE_INSTRUCTION_ID1_OFFSET)
                + (eval_op + AB::Expr::ONE) * AB::Expr::from(ACE_INSTRUCTION_ID2_OFFSET);
            MemoryMsg::Element {
                op_value: MEMORY_READ_ELEMENT_LABEL as u16,
                header: MemoryHeader {
                    ctx: ace_ctx,
                    addr: ace_ptr,
                    clk: ace_clk,
                },
                element,
            }
        });

        set
    };

    // =====================================================================
    // C3: ACE wiring bus
    // =====================================================================
    let g_wiring = {
        let mut set = RationalSet::new(&challenges);

        let ace_flag: AB::Expr =
            s0.clone() * s1.clone() * s2.clone() * (AB::Expr::ONE - s3.clone());
        let sblock: AB::Expr =
            local.chiplets[ACE_OFFSET + SELECTOR_BLOCK_IDX].clone().into();
        let is_read: AB::Expr = ace_flag.clone() * (AB::Expr::ONE - sblock.clone());
        let is_eval: AB::Expr = ace_flag * sblock;

        let clk: AB::Expr = local.chiplets[ACE_OFFSET + CLK_IDX].clone().into();
        let ctx: AB::Expr = local.chiplets[ACE_OFFSET + CTX_IDX].clone().into();
        let m0: AB::Expr = local.chiplets[ACE_OFFSET + M_0_IDX].clone().into();
        let m1: AB::Expr = local.chiplets[ACE_OFFSET + M_1_IDX].clone().into();

        let wire_0 = AceWireMsg {
            clk: clk.clone(),
            ctx: ctx.clone(),
            id: local.chiplets[ACE_OFFSET + ID_0_IDX].clone().into(),
            v0: local.chiplets[ACE_OFFSET + V_0_0_IDX].clone().into(),
            v1: local.chiplets[ACE_OFFSET + V_0_1_IDX].clone().into(),
        };
        let wire_1 = AceWireMsg {
            clk: clk.clone(),
            ctx: ctx.clone(),
            id: local.chiplets[ACE_OFFSET + ID_1_IDX].clone().into(),
            v0: local.chiplets[ACE_OFFSET + V_1_0_IDX].clone().into(),
            v1: local.chiplets[ACE_OFFSET + V_1_1_IDX].clone().into(),
        };
        let wire_2 = AceWireMsg {
            clk: clk.clone(),
            ctx: ctx.clone(),
            id: local.chiplets[ACE_OFFSET + ID_2_IDX].clone().into(),
            v0: local.chiplets[ACE_OFFSET + V_2_0_IDX].clone().into(),
            v1: local.chiplets[ACE_OFFSET + V_2_1_IDX].clone().into(),
        };

        // READ batch: insert wire_0 (m0 times) + insert wire_1 (m1 times).
        set.add_batch(is_read, |b| {
            b.insert(m0.clone(), wire_0.clone());
            b.insert(m1, wire_1.clone());
        });

        // EVAL batch: insert wire_0 (m0 times) + remove wire_1 + remove wire_2.
        set.add_batch(is_eval, |b| {
            b.insert(m0, wire_0);
            b.remove(wire_1);
            b.remove(wire_2);
        });

        set
    };

    // =====================================================================
    // Combine groups into columns and emit constraints
    // =====================================================================

    let aux = builder.permutation();
    let aux_local = aux.current_slice();
    let aux_next = aux.next_slice();

    Column::from_set(aux_local[chip_cols::C1].into(), aux_next[chip_cols::C1].into(), g_chiplet_resp)
        .constrain(builder);
    Column::from_set(aux_local[chip_cols::C2].into(), aux_next[chip_cols::C2].into(), g_hash_kernel)
        .constrain(builder);
    Column::from_set(aux_local[chip_cols::C3].into(), aux_next[chip_cols::C3].into(), g_wiring)
        .constrain(builder);
}

// CHIPLET RESPONSE HELPERS
// ================================================================================================

/// Compute the conditional leaf word for Merkle path operations.
///
/// When `bit = 0`, selects `h[0..4]` (RATE0); when `bit = 1`, selects `h[4..8]` (RATE1).
fn leaf_word<E: PrimeCharacteristicRing + Clone>(h: &[E; 12], bit: &E) -> [E; 4] {
    array::from_fn(|i| {
        (E::ONE - bit.clone()) * h[i].clone() + bit.clone() * h[i + 4].clone()
    })
}

/// Compute the memory chiplet response message value.
///
/// Encodes `[label, ctx, addr, clk, data...]` where label depends on is_read/is_word flags,
/// addr = word + 2*idx1 + idx0, and data is either a single element (muxed by idx0/idx1)
/// or a full 4-element word.
fn compute_memory_response_msg<AB: LiftedAirBuilder<F = Felt>>(
    local: &MainTraceRow<AB::Var>,
) -> MemoryResponseMsg<AB::Expr> {
    let mem_offset = NUM_MEMORY_SELECTORS;
    let is_read: AB::Expr = local.chiplets[mem_offset + memory::IS_READ_COL_IDX].clone().into();
    let is_word: AB::Expr =
        local.chiplets[mem_offset + memory::IS_WORD_ACCESS_COL_IDX].clone().into();
    let ctx: AB::Expr = local.chiplets[mem_offset + memory::CTX_COL_IDX].clone().into();
    let word: AB::Expr = local.chiplets[mem_offset + memory::WORD_COL_IDX].clone().into();
    let idx0: AB::Expr = local.chiplets[mem_offset + memory::IDX0_COL_IDX].clone().into();
    let idx1: AB::Expr = local.chiplets[mem_offset + memory::IDX1_COL_IDX].clone().into();
    let clk: AB::Expr = local.chiplets[mem_offset + memory::CLK_COL_IDX].clone().into();

    // Compute address: addr = word + 2*idx1 + idx0.
    let addr: AB::Expr = word + idx1.clone() * AB::Expr::from_u16(2) + idx0.clone();

    // Compute label from flags.
    let write_element_label = AB::Expr::from_u16(MEMORY_WRITE_ELEMENT_LABEL as u16);
    let write_word_label = AB::Expr::from_u16(MEMORY_WRITE_WORD_LABEL as u16);
    let read_element_label = AB::Expr::from_u16(MEMORY_READ_ELEMENT_LABEL as u16);
    let read_word_label = AB::Expr::from_u16(MEMORY_READ_WORD_LABEL as u16);
    let write_label = (AB::Expr::ONE - is_word.clone()) * write_element_label
        + is_word.clone() * write_word_label;
    let read_label = (AB::Expr::ONE - is_word.clone()) * read_element_label
        + is_word.clone() * read_word_label;
    let label =
        (AB::Expr::ONE - is_read.clone()) * write_label + is_read * read_label;

    // Value columns.
    let v0: AB::Expr = local.chiplets[mem_offset + memory::V_COL_RANGE.start].clone().into();
    let v1: AB::Expr = local.chiplets[mem_offset + memory::V_COL_RANGE.start + 1].clone().into();
    let v2: AB::Expr = local.chiplets[mem_offset + memory::V_COL_RANGE.start + 2].clone().into();
    let v3: AB::Expr = local.chiplets[mem_offset + memory::V_COL_RANGE.start + 3].clone().into();

    // Element selection: v0*(1-idx0)*(1-idx1) + v1*idx0*(1-idx1) + v2*(1-idx0)*idx1 + v3*idx0*idx1.
    let element: AB::Expr =
        v0.clone() * (AB::Expr::ONE - idx0.clone()) * (AB::Expr::ONE - idx1.clone())
            + v1.clone() * idx0.clone() * (AB::Expr::ONE - idx1.clone())
            + v2.clone() * (AB::Expr::ONE - idx0.clone()) * idx1.clone()
            + v3.clone() * idx0 * idx1;

    MemoryResponseMsg { label, ctx, addr, clk, is_word, element, word: [v0, v1, v2, v3] }
}

