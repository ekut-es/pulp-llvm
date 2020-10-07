
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCContext.h"
using namespace llvm;


#define PULP_HWLOOPS_NAME "RISCV Pulp hardware loops pass"

#define DEBUG_TYPE "pulp-hardware-loops"

namespace {

  class RISCVPulpHWLoops : public MachineFunctionPass {

    public:
      static char ID;
      RISCVPulpHWLoops() : MachineFunctionPass(ID) { }

      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesCFG();
        AU.addRequired<MachineLoopInfo>();
        MachineFunctionPass::getAnalysisUsage(AU);
      }

      bool runOnMachineFunction(MachineFunction &MF) override;

      bool ProcessLoop(MachineLoop *ML, MachineFunction &MF);

      StringRef getPassName() const override {
        return PULP_HWLOOPS_NAME;
      }
  };
}

char RISCVPulpHWLoops::ID = 0;

bool RISCVPulpHWLoops::runOnMachineFunction(MachineFunction &MF) {
    MachineLoopInfo *MLI = &getAnalysis<MachineLoopInfo>();


    bool Changed = false;
    for (auto ML : *MLI) {
      if (!ML->getParentLoop())
        Changed |= ProcessLoop(ML, MF);
    }
    return Changed;
}

bool RISCVPulpHWLoops::ProcessLoop(MachineLoop *ML, MachineFunction &MF) {

  const TargetSubtargetInfo &ST = MF.getSubtarget();
  const TargetInstrInfo *TII = ST.getInstrInfo();

  bool Changed = false;

  for (auto I = ML->begin(), E = ML->end(); I != E; ++I)
    Changed |= ProcessLoop(*I, MF);

  MachineBasicBlock *Preheader = ML->getLoopPreheader(), *Latch;
  MachineInstr *Init = nullptr, *Branch = nullptr;

  SmallVector<MachineBasicBlock*, 4> Latches;
  ML->getLoopLatches(Latches);
  for (auto *BB : Latches) {
    for (auto &MI : *BB) {
      if (MI.getOpcode() == RISCV::LoopBranch) {
        Branch = &MI;
        Latch = BB;
      }
    }
  }

  if (!Latch || !Preheader)
    return Changed;


  for (auto &MI : *Preheader) {
    if (MI.getOpcode() == RISCV::LoopIterations ||
        MI.getOpcode() == RISCV::LoopIterationsImm) {
      Init = &MI;
      break;
    }
  }

  if (!Init || !Branch) {
    return Changed;
  }

  MachineBasicBlock *LoopHeader = Branch->getOperand(2).getMBB();

  // find last executed instruction in the loop
  MachineInstr *LastInstr = Branch->getPrevNode();
  MachineBasicBlock *LastBB = Latch;

  while (LastInstr && LastInstr->isDebugInstr()) {
    LastInstr = LastInstr->getPrevNode();

    // continue search in single predecessor
    if (!LastInstr && LastBB->pred_size() == 1) {
      LastInstr = &*(*LastBB->pred_begin())->rbegin();

      // skip branch to LastBB
      if (LastInstr && LastInstr->isBranch() &&
          TII->getBranchDestBlock(*LastInstr) == LastBB) {
        LastInstr = LastInstr->getPrevNode();
      }

      LastBB = *LastBB->pred_begin();
    }
  }

  MachineInstr *FirstInstr;
  for (auto &MI : *LoopHeader) {
    if (!MI.isDebugInstr()) {
      FirstInstr = &MI;
      break;
    }
  }

  // If there are multiple possible last instructions or the last instruction is
  // a branch or the first instruction, create a nop as the last instruction
  if (!LastInstr || LastInstr->isBranch() || LastInstr == FirstInstr) {
    LastInstr = BuildMI(*Latch, Branch, Branch->getDebugLoc(),
                         TII->get(RISCV::ADDI)).addReg(RISCV::X0)
                         .addReg(RISCV::X0).addImm(0);
  }



  MCSymbol *LastInstrSymbol = LastInstr->getPreInstrSymbol();
  if (!LastInstrSymbol) {
    LastInstrSymbol = MF.getContext().createLinkerPrivateTempSymbol();
    LastInstr->setPreInstrSymbol(MF, LastInstrSymbol);
  }

  int Offset = 0, SetupOffset = 0, HeaderOffset = 0, LastInstrOffset = 0;
  for (auto &BB : MF) {
    if (&BB == LoopHeader) {
      HeaderOffset = Offset;
    }
    for (auto &MI : BB) {
      if (&MI == Init) {
        SetupOffset = Offset;
      } else if (&MI == LastInstr) {
        LastInstrOffset = Offset;
      }
      Offset += TII->getInstSizeInBytes(MI);
    }
  }
  // FIXME: rearrange basic blocks
  assert(isUInt<13>(HeaderOffset - SetupOffset) &&
         isUInt<13>(LastInstrOffset - SetupOffset) &&
         "Loop Blocks badly positioned");

  int outerLoop = Changed ? 1 : 0;

  if(Init->getOpcode() == RISCV::LoopIterations) {
    Register count = Init->getOperand(0).getReg();
    if (HeaderOffset - SetupOffset == 12) {
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_SETUP))
              .addImm(outerLoop).addReg(count).addSym(LastInstrSymbol);
    }
    else {
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_COUNT))
              .addImm(outerLoop).addReg(count);
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_STARTI))
              .addImm(outerLoop).addMBB(LoopHeader);
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_ENDI))
              .addImm(outerLoop).addSym(LastInstrSymbol);
    }
  }
  else { // LoopIterationsImm
    int64_t count = Init->getOperand(0).getImm();
    if (HeaderOffset - SetupOffset == 12 && LastInstrOffset - SetupOffset < 64) {
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_SETUPI))
              .addImm(outerLoop).addImm(count).addSym(LastInstrSymbol);
    }
    else {
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_COUNTI))
              .addImm(outerLoop).addImm(count);
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_STARTI))
              .addImm(outerLoop).addMBB(LoopHeader);
      BuildMI(*Preheader, Init, Init->getDebugLoc(), TII->get(RISCV::LP_ENDI))
              .addImm(outerLoop).addSym(LastInstrSymbol);
    }
  }
  Preheader->erase(Init);
  Latch->erase(Branch);

  return true;
}


INITIALIZE_PASS(RISCVPulpHWLoops, DEBUG_TYPE, PULP_HWLOOPS_NAME, false, false)


FunctionPass *llvm::createRISCVPulpHWLoopsPass() { return new RISCVPulpHWLoops(); }
