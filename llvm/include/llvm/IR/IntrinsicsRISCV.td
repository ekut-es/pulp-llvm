//===- IntrinsicsRISCV.td - Defines RISCV intrinsics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the RISCV-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Atomics

// Atomic Intrinsics have multiple versions for different access widths, which
// all follow one of the following signatures (depending on how many arguments
// they require). We carefully instantiate only specific versions of these for
// specific integer widths, rather than using `llvm_anyint_ty`.
//
// In fact, as these intrinsics take `llvm_anyptr_ty`, the given names are the
// canonical names, and the intrinsics used in the code will have a name
// suffixed with the pointer type they are specialised for (denoted `<p>` in the
// names below), in order to avoid type conflicts.

let TargetPrefix = "riscv" in {

  // T @llvm.<name>.T.<p>(any*, T, T, T imm);
  class MaskedAtomicRMWFourArg<LLVMType itype>
      : Intrinsic<[itype], [llvm_anyptr_ty, itype, itype, itype],
                  [IntrArgMemOnly, NoCapture<0>, ImmArg<3>]>;
  // T @llvm.<name>.T.<p>(any*, T, T, T, T imm);
  class MaskedAtomicRMWFiveArg<LLVMType itype>
      : Intrinsic<[itype], [llvm_anyptr_ty, itype, itype, itype, itype],
                  [IntrArgMemOnly, NoCapture<0>, ImmArg<4>]>;

  // We define 32-bit and 64-bit variants of the above, where T stands for i32
  // or i64 respectively:
  multiclass MaskedAtomicRMWFourArgIntrinsics {
    // i32 @llvm.<name>.i32.<p>(any*, i32, i32, i32 imm);
    def _i32 : MaskedAtomicRMWFourArg<llvm_i32_ty>;
    // i64 @llvm.<name>.i32.<p>(any*, i64, i64, i64 imm);
    def _i64 : MaskedAtomicRMWFourArg<llvm_i64_ty>;
  }

  multiclass MaskedAtomicRMWFiveArgIntrinsics {
    // i32 @llvm.<name>.i32.<p>(any*, i32, i32, i32, i32 imm);
    def _i32 : MaskedAtomicRMWFiveArg<llvm_i32_ty>;
    // i64 @llvm.<name>.i64.<p>(any*, i64, i64, i64, i64 imm);
    def _i64 : MaskedAtomicRMWFiveArg<llvm_i64_ty>;
  }

  // @llvm.riscv.masked.atomicrmw.*.{i32,i64}.<p>(...)
  defm int_riscv_masked_atomicrmw_xchg : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_add : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_sub : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_nand : MaskedAtomicRMWFourArgIntrinsics;
  // Signed min and max need an extra operand to do sign extension with.
  defm int_riscv_masked_atomicrmw_max : MaskedAtomicRMWFiveArgIntrinsics;
  defm int_riscv_masked_atomicrmw_min : MaskedAtomicRMWFiveArgIntrinsics;
  // Unsigned min and max don't need the extra operand.
  defm int_riscv_masked_atomicrmw_umax : MaskedAtomicRMWFourArgIntrinsics;
  defm int_riscv_masked_atomicrmw_umin : MaskedAtomicRMWFourArgIntrinsics;

  // @llvm.riscv.masked.cmpxchg.{i32,i64}.<p>(...)
  defm int_riscv_masked_cmpxchg : MaskedAtomicRMWFiveArgIntrinsics;





  // Pulp specific intrinsics
  multiclass PulpIntrinsic<list<LLVMType> ret_types,
                           list<LLVMType> param_types,
                           list<IntrinsicProperty> intr_properties> {

    def int_riscv_pulp_ # NAME
        : GCCBuiltin<"__builtin_pulp_" # NAME>,
          Intrinsic<ret_types, param_types, intr_properties>;
  }

  multiclass PulpUnary32 : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
  multiclass PulpBinary32 : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  multiclass PulpTernary32 : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  defm fl1    : PulpUnary32;
  defm ff1    : PulpUnary32;
  defm clb    : PulpUnary32;
  defm cnt    : PulpUnary32;
  defm parity : PulpUnary32;
  defm abs    : PulpUnary32;

  defm mac : PulpTernary32;
  defm msu : PulpTernary32;

  defm macs : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm macu : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  defm machhs : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm machhu : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

  defm rotr : PulpBinary32;

  defm maxsi : PulpBinary32;
  defm minsi : PulpBinary32;
  defm maxusi : PulpBinary32;
  defm minusi : PulpBinary32;

  multiclass PulpBinaryV2 : PulpIntrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem]>;
  multiclass PulpBinaryV4 : PulpIntrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty], [IntrNoMem]>;

  multiclass PulpBinaryV {
    defm "2" : PulpBinaryV2;
    defm "4" : PulpBinaryV4;
  }

  multiclass PulpUnaryV2 : PulpIntrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty], [IntrNoMem]>;
  multiclass PulpUnaryV4 : PulpIntrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty], [IntrNoMem]>;

  multiclass PulpUnaryV {
    defm "2" : PulpUnaryV2;
    defm "4" : PulpUnaryV4;
  }

  defm add  : PulpBinaryV;
  defm sub  : PulpBinaryV;
  defm avg  : PulpBinaryV;
  defm avgu : PulpBinaryV;
  defm min  : PulpBinaryV;
  defm minu : PulpBinaryV;
  defm max  : PulpBinaryV;
  defm maxu : PulpBinaryV;
  defm and  : PulpBinaryV;
  defm or   : PulpBinaryV;
  defm exor : PulpBinaryV;
  defm srl  : PulpBinaryV;
  defm sra  : PulpBinaryV;
  defm sll  : PulpBinaryV;
  defm abs  : PulpUnaryV;
  defm neg  : PulpUnaryV;

  multiclass PulpDotp {
    defm p2   : PulpIntrinsic<[llvm_i32_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem]>;
    defm psc2 : PulpIntrinsic<[llvm_i32_ty], [llvm_v2i16_ty, llvm_i32_ty],   [IntrNoMem]>;
    defm p4   : PulpIntrinsic<[llvm_i32_ty], [llvm_v4i8_ty,  llvm_v4i8_ty],  [IntrNoMem]>;
    defm psc4 : PulpIntrinsic<[llvm_i32_ty], [llvm_v4i8_ty,  llvm_i32_ty],   [IntrNoMem]>;
    defm s # NAME # p2   : PulpIntrinsic<[llvm_i32_ty], [llvm_v2i16_ty, llvm_v2i16_ty, llvm_i32_ty], [IntrNoMem]>;
    defm s # NAME # psc2 : PulpIntrinsic<[llvm_i32_ty], [llvm_v2i16_ty, llvm_i32_ty,   llvm_i32_ty], [IntrNoMem]>;
    defm s # NAME # p4   : PulpIntrinsic<[llvm_i32_ty], [llvm_v4i8_ty,  llvm_v4i8_ty,  llvm_i32_ty], [IntrNoMem]>;
    defm s # NAME # psc4 : PulpIntrinsic<[llvm_i32_ty], [llvm_v4i8_ty,  llvm_i32_ty,   llvm_i32_ty], [IntrNoMem]>;
  }

  defm dots  : PulpDotp;
  defm dotu  : PulpDotp;
  defm dotus : PulpDotp;


  defm muls   : PulpBinary32;
  defm mulu   : PulpBinary32;
  defm mulhhs : PulpBinary32;
  defm mulhhu : PulpBinary32;

  multiclass mul_mac_N_RN {
    defm mul # NAME # N   : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
    defm mul # NAME # RN  : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
    defm mac # NAME # N   : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
    defm mac # NAME # RN  : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  }

  defm s   : mul_mac_N_RN;
  defm u   : mul_mac_N_RN;
  defm hhs : mul_mac_N_RN;
  defm hhu : mul_mac_N_RN;



  multiclass N_RN_r {
    defm N    : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
    defm N_r  : PulpTernary32;
    defm RN   : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
    defm RN_r : PulpTernary32;
  }

  defm add  : N_RN_r;
  defm addu : N_RN_r;
  defm sub  : N_RN_r;
  defm subu : N_RN_r;

  defm pack2 : PulpIntrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm pack4 : PulpIntrinsic<[llvm_v4i8_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm shuffleh : PulpBinaryV2;
  defm shuffleb : PulpBinaryV4;
  defm pack4_lo : PulpIntrinsic<[llvm_v4i8_ty], [llvm_i32_ty, llvm_i32_ty, llvm_v4i8_ty], [IntrNoMem]>;
  defm pack4_hi : PulpIntrinsic<[llvm_v4i8_ty], [llvm_i32_ty, llvm_i32_ty, llvm_v4i8_ty], [IntrNoMem]>;
  defm shuffle4b : PulpIntrinsic<[llvm_v4i8_ty], [llvm_v4i8_ty, llvm_v4i8_ty, llvm_v4i8_ty], [IntrNoMem]>;
  defm shuffle2h : PulpIntrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem]>;

  defm bclr : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm bclr_r : PulpBinary32;
  defm bset : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm bset_r : PulpBinary32;
  defm bextract : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm bextract_r : PulpBinary32;
  defm bextractu : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm bextractu_r : PulpBinary32;
  defm binsert : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm binsert_r : PulpTernary32;
  defm clip : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm clip_r : PulpBinary32;
  defm clipu : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
  defm clipu_r : PulpBinary32;

  defm OffsetedRead : PulpIntrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i32_ty>, llvm_i32_ty], [IntrReadMem]>;
  defm OffsetedReadHalf : PulpIntrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i16_ty>, llvm_i32_ty], [IntrReadMem]>;
  defm OffsetedReadByte : PulpIntrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i8_ty>, llvm_i32_ty], [IntrReadMem]>;

  defm OffsetedWrite : PulpIntrinsic<[], [llvm_i32_ty, LLVMPointerType<llvm_i32_ty>, llvm_i32_ty], [IntrWriteMem]>;
  defm OffsetedWriteHalf : PulpIntrinsic<[], [llvm_i32_ty, LLVMPointerType<llvm_i16_ty>, llvm_i32_ty], [IntrWriteMem]>;
  defm OffsetedWriteByte : PulpIntrinsic<[], [llvm_i32_ty, LLVMPointerType<llvm_i8_ty>, llvm_i32_ty], [IntrWriteMem]>;

  defm CoreId : PulpIntrinsic<[llvm_i32_ty], [], []>;
  defm ClusterId : PulpIntrinsic<[llvm_i32_ty], [], []>;

  defm read_base_off : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrReadMem]>;
  defm write_base_off : PulpIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrWriteMem]>;

  defm spr_read_vol : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  defm spr_read : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;

  defm spr_write : PulpIntrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  defm spr_bit_set : PulpIntrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  defm spr_bit_clr : PulpIntrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;

  defm read_then_spr_write : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  defm read_then_spr_bit_set : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;
  defm read_then_spr_bit_clr : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrHasSideEffects]>;


  defm read_base_off_v : PulpIntrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrReadMem]>;
  defm write_base_off_v : PulpIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrWriteMem]>;


} // TargetPrefix = "riscv"
