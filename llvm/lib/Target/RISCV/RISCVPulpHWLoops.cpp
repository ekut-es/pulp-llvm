
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineDominators.h"
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

    private:
      const TargetInstrInfo *TII;
      SmallVector<MachineInstr*, 4> Setups;
      SmallVector<MachineInstr*, 4> Branches;
      SmallVector<int, 4> LoopNums;

      void FindInstrPairs(MachineDomTreeNode *Node, MachineInstr *LoopSetup,
                          MachineInstr *OuterLoopSetup);

      void ProcessHardwareLoop(MachineInstr *Setup, MachineInstr *Branch,
                               int LoopNum, MachineFunction &MF);

    public:
      static char ID;
      RISCVPulpHWLoops() : MachineFunctionPass(ID) { }

      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesCFG();
        AU.addRequired<MachineDominatorTree>();
        MachineFunctionPass::getAnalysisUsage(AU);
      }

      bool runOnMachineFunction(MachineFunction &MF) override;

      StringRef getPassName() const override {
        return PULP_HWLOOPS_NAME;
      }
  };
}

char RISCVPulpHWLoops::ID = 0;

bool RISCVPulpHWLoops::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getSubtarget<RISCVSubtarget>().hasNonStdExtPulp()) {
    return false;
  }
  TII = MF.getSubtarget().getInstrInfo();

  MachineDominatorTree *MDT = &getAnalysis<MachineDominatorTree>();
  FindInstrPairs(MDT->getRootNode(), nullptr, nullptr);

  assert(Setups.size() == Branches.size() && Setups.size() == LoopNums.size());
  if (Setups.empty())
    return false;

  while (!Setups.empty()) {
    ProcessHardwareLoop(Setups.pop_back_val(), Branches.pop_back_val(),
                        LoopNums.pop_back_val(), MF);
  }
  return true;
}

// The hardware loop setup and branch instruction might not be in loops anymore,
// but it should still be the case, that the setup dominates the branch and the
// branch post-dominates the setup. Therefore, a nested loop is dominated by the
// setup of the outer loop, but not by it's loop branch.
// We use a DFS on the dominator tree and pair loop branches with the last found
// loop setup.
void RISCVPulpHWLoops::FindInstrPairs(MachineDomTreeNode *Node,
                                      MachineInstr *LoopSetup,
                                      MachineInstr *OuterLoopSetup) {
  for (auto &MI : *Node->getBlock()) {
    if (MI.getOpcode() == RISCV::LoopIterations ||
        MI.getOpcode() == RISCV::LoopIterationsImm) {
      assert(!OuterLoopSetup && "hardware loop nesting is too deep");
      if (LoopSetup) {
        OuterLoopSetup = LoopSetup;
      }
      LoopSetup = &MI;
    }
    if (MI.getOpcode() == RISCV::LoopBranch) {
      assert(LoopSetup && "Unpaired hardware loop branch");
      Setups.push_back(LoopSetup);
      Branches.push_back(&MI);
      LoopNums.push_back(OuterLoopSetup ? 0 : 1);
      LoopSetup = OuterLoopSetup;
      OuterLoopSetup = nullptr;
    }
  }
  for (auto &NextNode : *Node) {
    FindInstrPairs(NextNode, LoopSetup, OuterLoopSetup);
  }
}

void RISCVPulpHWLoops::ProcessHardwareLoop(MachineInstr *Setup,
                                           MachineInstr *Branch,
                                           int LoopNum, MachineFunction &MF) {


  MachineBasicBlock *LoopHeader = Branch->getOperand(2).getMBB();
  LoopHeader->setLabelMustBeEmitted();

  // find last executed instruction in the loop
  MachineInstr *LastInstr = Branch->getPrevNode();
  MachineBasicBlock *LastBB = Branch->getParent();

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
  if (!LastInstr || LastInstr->isBranch() || LastInstr->isCall() ||
      LastInstr->isReturn() || LastInstr == FirstInstr) {
    LastInstr = BuildMI(*Branch->getParent(), Branch, Branch->getDebugLoc(),
                         TII->get(RISCV::ADDI)).addReg(RISCV::X0)
                         .addReg(RISCV::X0).addImm(0);
  }



  MCSymbol *LastInstrSymbol = LastInstr->getPreInstrSymbol();
  if (!LastInstrSymbol) {
    LastInstrSymbol = MF.getContext().createLinkerPrivateTempSymbol();
    LastInstr->setPreInstrSymbol(MF, LastInstrSymbol);
  }

  int Offset = 0, SetupOffset = 0, FirstInstrOffset = 0, LastInstrOffset = 0;
  for (auto &BB : MF) {
    for (auto &MI : BB) {
      if (&MI == Setup) {
        SetupOffset = Offset;
      } else if (&MI == FirstInstr) {
        FirstInstrOffset = Offset;
      } else if (&MI == LastInstr) {
        LastInstrOffset = Offset;
      }
      Offset += TII->getInstSizeInBytes(MI);
    }
  }
  // FIXME: rearrange basic blocks
  assert(isUInt<13>(FirstInstrOffset - SetupOffset) &&
         isUInt<13>(LastInstrOffset - SetupOffset) &&
         "Loop Blocks badly positioned");

  MachineBasicBlock *Preheader = Setup->getParent();
  if(Setup->getOpcode() == RISCV::LoopIterations) {
    Register count = Setup->getOperand(0).getReg();
    if (FirstInstrOffset - SetupOffset == 12) {
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_SETUP))
              .addImm(LoopNum).addReg(count).addSym(LastInstrSymbol);
    }
    else {
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_COUNT))
              .addImm(LoopNum).addReg(count);
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_STARTI))
              .addImm(LoopNum).addMBB(LoopHeader);
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_ENDI))
              .addImm(LoopNum).addSym(LastInstrSymbol);
    }
  }
  else { // LoopIterationsImm
    int64_t count = Setup->getOperand(0).getImm();
    if (FirstInstrOffset - SetupOffset == 12 && LastInstrOffset - SetupOffset < 64) {
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_SETUPI))
              .addImm(LoopNum).addImm(count).addSym(LastInstrSymbol);
    }
    else {
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_COUNTI))
              .addImm(LoopNum).addImm(count);
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_STARTI))
              .addImm(LoopNum).addMBB(LoopHeader);
      BuildMI(*Preheader, Setup, Setup->getDebugLoc(), TII->get(RISCV::LP_ENDI))
              .addImm(LoopNum).addSym(LastInstrSymbol);
    }
  }
  Setup->removeFromParent();
  Branch->removeFromParent();
}


INITIALIZE_PASS(RISCVPulpHWLoops, DEBUG_TYPE, PULP_HWLOOPS_NAME, false, false)


FunctionPass *llvm::createRISCVPulpHWLoopsPass() { return new RISCVPulpHWLoops(); }
