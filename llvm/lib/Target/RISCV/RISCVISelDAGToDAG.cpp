//===-- RISCVISelDAGToDAG.cpp - A dag to dag inst selector for RISCV ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the RISCV target.
//
//===----------------------------------------------------------------------===//

#include "RISCVISelDAGToDAG.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "Utils/RISCVMatInt.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-isel"

void RISCVDAGToDAGISel::PostprocessISelDAG() {
  doPeepholeLoadStoreADDI();
}

static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, int64_t Imm,
                         MVT XLenVT) {
  RISCVMatInt::InstSeq Seq;
  RISCVMatInt::generateInstSeq(Imm, XLenVT == MVT::i64, Seq);

  SDNode *Result = nullptr;
  SDValue SrcReg = CurDAG->getRegister(RISCV::X0, XLenVT);
  for (RISCVMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.Imm, DL, XLenVT);
    if (Inst.Opc == RISCV::LUI)
      Result = CurDAG->getMachineNode(RISCV::LUI, DL, XLenVT, SDImm);
    else
      Result = CurDAG->getMachineNode(Inst.Opc, DL, XLenVT, SrcReg, SDImm);

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}

// Returns true if the Node is an ISD::AND with a constant argument. If so,
// set Mask to that constant value.
static bool isConstantMask(SDNode *Node, uint64_t &Mask) {
  if (Node->getOpcode() == ISD::AND &&
      Node->getOperand(1).getOpcode() == ISD::Constant) {
    Mask = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    return true;
  }
  return false;
}

void RISCVDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();
  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  EVT VT = Node->getValueType(0);

  switch (Opcode) {
  case ISD::Constant: {
    auto ConstNode = cast<ConstantSDNode>(Node);
    if (VT == XLenVT && ConstNode->isNullValue()) {
      SDValue New = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                           RISCV::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    if (XLenVT == MVT::i64) {
      ReplaceNode(Node, selectImm(CurDAG, SDLoc(Node), Imm, XLenVT));
      return;
    }
    break;
  }
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, DL, XLenVT);
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    ReplaceNode(Node, CurDAG->getMachineNode(RISCV::ADDI, DL, VT, TFI, Imm));
    return;
  }
  case ISD::SRL: {
    if (!Subtarget->is64Bit())
      break;
    SDValue Op0 = Node->getOperand(0);
    SDValue Op1 = Node->getOperand(1);
    uint64_t Mask;
    // Match (srl (and val, mask), imm) where the result would be a
    // zero-extended 32-bit integer. i.e. the mask is 0xffffffff or the result
    // is equivalent to this (SimplifyDemandedBits may have removed lower bits
    // from the mask that aren't necessary due to the right-shifting).
    if (Op1.getOpcode() == ISD::Constant &&
        isConstantMask(Op0.getNode(), Mask)) {
      uint64_t ShAmt = cast<ConstantSDNode>(Op1.getNode())->getZExtValue();

      if ((Mask | maskTrailingOnes<uint64_t>(ShAmt)) == 0xffffffff) {
        SDValue ShAmtVal =
            CurDAG->getTargetConstant(ShAmt, SDLoc(Node), XLenVT);
        CurDAG->SelectNodeTo(Node, RISCV::SRLIW, XLenVT, Op0.getOperand(0),
                             ShAmtVal);
        return;
      }
    }
    break;
  }
  case RISCVISD::READ_CYCLE_WIDE:
    assert(!Subtarget->is64Bit() && "READ_CYCLE_WIDE is only used on riscv32");

    ReplaceNode(Node, CurDAG->getMachineNode(RISCV::ReadCycleWide, DL, MVT::i32,
                                             MVT::i32, MVT::Other,
                                             Node->getOperand(0)));
    return;
  case ISD::LOAD: {
    LoadSDNode *Load = cast<LoadSDNode>(Node);
    if (Load->getAddressingMode() == ISD::UNINDEXED)
      break;

    SDValue Chain = Node->getOperand(0);
    SDValue Base = Node->getOperand(1);
    SDValue Offset = Node->getOperand(2);

    bool simm12 = false;
    bool signExtend = Load->getExtensionType() == ISD::SEXTLOAD;

    if(auto ConstantOffset = dyn_cast<ConstantSDNode>(Offset)) {
      int ConstantVal = ConstantOffset->getSExtValue();
      simm12 = isInt<12>(ConstantVal);
      if (simm12)
        Offset = CurDAG->getTargetConstant(ConstantVal, DL,
                                           Offset.getValueType());
    }

    unsigned Opcode = 0;


    switch (Load->getMemoryVT().getSimpleVT().SimpleTy) {
      case MVT::i8:
        if      ( simm12 &&  signExtend) Opcode = RISCV::P_LB_ri_PostIncrement;
        else if ( simm12 && !signExtend) Opcode = RISCV::P_LBU_ri_PostIncrement;
        else if (!simm12 &&  signExtend) Opcode = RISCV::P_LB_rr_PostIncrement;
        else                             Opcode = RISCV::P_LBU_rr_PostIncrement;
        break;
      case MVT::i16:
        if      ( simm12 &&  signExtend) Opcode = RISCV::P_LH_ri_PostIncrement;
        else if ( simm12 && !signExtend) Opcode = RISCV::P_LHU_ri_PostIncrement;
        else if (!simm12 &&  signExtend) Opcode = RISCV::P_LH_rr_PostIncrement;
        else                             Opcode = RISCV::P_LHU_rr_PostIncrement;
        break;
      case MVT::i32:
        if (simm12) Opcode = RISCV::P_LW_ri_PostIncrement;
        else        Opcode = RISCV::P_LW_rr_PostIncrement;
      default: break;
    }

    if (!Opcode) break;

    ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, MVT::i32, MVT::i32,
                                             Chain.getSimpleValueType(),
                                             Base, Offset, Chain));
    return;
  }
  case ISD::VECTOR_SHUFFLE: {
    ShuffleVectorSDNode *Shuffle = cast<ShuffleVectorSDNode>(Node);
    SDValue Vec0 = Shuffle->getOperand(0);
    SDValue Vec1 = Shuffle->getOperand(1);
    unsigned Opcode;
    int imm;
    if (Vec1->getOpcode() == ISD::UNDEF) {
      if (VT == MVT::v2i16) {
        Opcode = RISCV::PV_SHUFFLE_SCI_H;
        imm = (Shuffle->getMaskElt(1) & 1) << 1;
        imm |= Shuffle->getMaskElt(0) & 1;
      } else {
        switch (Shuffle->getMaskElt(3)) {
          default:
          case 0: Opcode = RISCV::PV_SHUFFLEI0_SCI_B; break;
          case 1: Opcode = RISCV::PV_SHUFFLEI1_SCI_B; break;
          case 2: Opcode = RISCV::PV_SHUFFLEI2_SCI_B; break;
          case 3: Opcode = RISCV::PV_SHUFFLEI3_SCI_B; break;
        }
        imm = (Shuffle->getMaskElt(2) & 3) << 4;
        imm |= (Shuffle->getMaskElt(1) & 3) << 2;
        imm |= Shuffle->getMaskElt(0) & 3;
      }
      SDValue Imm = CurDAG->getTargetConstant(imm, SDLoc(Node), MVT::i32);
      ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, VT, Vec0, Imm));
      return;
    }
    if (VT == MVT::v2i16) {
      Opcode = RISCV::PV_SHUFFLE2_H;
      imm = (Shuffle->getMaskElt(1) & 1) << 16;
      imm |= Shuffle->getMaskElt(0) & 1;
    } else {
      Opcode = RISCV::PV_SHUFFLE2_B;
      imm  = (Shuffle->getMaskElt(3) & 3) << 24;
      imm |= (Shuffle->getMaskElt(2) & 3) << 16;
      imm |= (Shuffle->getMaskElt(1) & 3) << 8;
      imm |=  Shuffle->getMaskElt(0) & 3;
    }
    SDValue Imm = SDValue(selectImm(CurDAG, DL, imm, MVT::i32), 0);
    ReplaceNode(Node, CurDAG->getMachineNode(Opcode, DL, VT, Vec0, Vec1, Imm));
    return;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    if (Node->getConstantOperandVal(1) == Intrinsic::loop_decrement) {
      ReplaceUses(SDValue(Node, 1),
                  Node->getOperand(0));
      return;
    }
  }
  }

  // Select the default instruction.
  SelectCode(Node);
}

bool RISCVDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
  switch (ConstraintID) {
  case InlineAsm::Constraint_m:
    // We just support simple memory operands that have a single address
    // operand and need no special handling.
    OutOps.push_back(Op);
    return false;
  case InlineAsm::Constraint_A:
    OutOps.push_back(Op);
    return false;
  default:
    break;
  }

  return true;
}

bool RISCVDAGToDAGISel::SelectAddrFI(SDValue Addr, SDValue &Base) {
  if (auto FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}

bool RISCVDAGToDAGISel::SelectLoopDecrement(SDValue LoopDecrement) {
  return (LoopDecrement->getOpcode() == ISD::INTRINSIC_W_CHAIN &&
         LoopDecrement->getConstantOperandVal(1) == Intrinsic::loop_decrement);
}

// Merge an ADDI into the offset of a load/store instruction where possible.
// (load (add base, off), 0) -> (load base, off)
// (store val, (add base, off)) -> (store val, base, off)
void RISCVDAGToDAGISel::doPeepholeLoadStoreADDI() {
  SelectionDAG::allnodes_iterator Position(CurDAG->getRoot().getNode());
  ++Position;

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    int OffsetOpIdx;
    int BaseOpIdx;

    // Only attempt this optimisation for I-type loads and S-type stores.
    switch (N->getMachineOpcode()) {
    default:
      continue;
    case RISCV::LB:
    case RISCV::LH:
    case RISCV::LW:
    case RISCV::LBU:
    case RISCV::LHU:
    case RISCV::LWU:
    case RISCV::LD:
    case RISCV::FLW:
    case RISCV::FLD:
      BaseOpIdx = 0;
      OffsetOpIdx = 1;
      break;
    case RISCV::SB:
    case RISCV::SH:
    case RISCV::SW:
    case RISCV::SD:
    case RISCV::FSW:
    case RISCV::FSD:
      BaseOpIdx = 1;
      OffsetOpIdx = 2;
      break;
    }

    // Currently, the load/store offset must be 0 to be considered for this
    // peephole optimisation.
    if (!isa<ConstantSDNode>(N->getOperand(OffsetOpIdx)) ||
        N->getConstantOperandVal(OffsetOpIdx) != 0)
      continue;

    SDValue Base = N->getOperand(BaseOpIdx);

    // If the base is an ADDI, we can merge it in to the load/store.
    if (!Base.isMachineOpcode() || Base.getMachineOpcode() != RISCV::ADDI)
      continue;

    SDValue ImmOperand = Base.getOperand(1);

    if (auto Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
      ImmOperand = CurDAG->getTargetConstant(
          Const->getSExtValue(), SDLoc(ImmOperand), ImmOperand.getValueType());
    } else if (auto GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
      ImmOperand = CurDAG->getTargetGlobalAddress(
          GA->getGlobal(), SDLoc(ImmOperand), ImmOperand.getValueType(),
          GA->getOffset(), GA->getTargetFlags());
    } else if (auto CP = dyn_cast<ConstantPoolSDNode>(ImmOperand)) {
      ImmOperand = CurDAG->getTargetConstantPool(
          CP->getConstVal(), ImmOperand.getValueType(), CP->getAlign(),
          CP->getOffset(), CP->getTargetFlags());
    } else {
      continue;
    }

    LLVM_DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
    LLVM_DEBUG(Base->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\nN: ");
    LLVM_DEBUG(N->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\n");

    // Modify the offset operand of the load/store.
    if (BaseOpIdx == 0) // Load
      CurDAG->UpdateNodeOperands(N, Base.getOperand(0), ImmOperand,
                                 N->getOperand(2));
    else // Store
      CurDAG->UpdateNodeOperands(N, N->getOperand(0), Base.getOperand(0),
                                 ImmOperand, N->getOperand(3));

    // The add-immediate may now be dead, in which case remove it.
    if (Base.getNode()->use_empty())
      CurDAG->RemoveDeadNode(Base.getNode());
  }
}

// This pass converts a legalized DAG into a RISCV-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createRISCVISelDag(RISCVTargetMachine &TM) {
  return new RISCVDAGToDAGISel(TM);
}
