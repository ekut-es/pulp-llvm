


let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PulpUnary<bits<7> funct7, bits<5> funct5, bits<3> funct3,
               RISCVOpcode opcode, string opcodestr>
    : RVInstR<funct7, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1"> {
  let Inst{24-20} = funct5;
}


let Predicates = [HasNonStdExtPulp] in {
def P_FF1 : PulpUnary<0b0001000, 0b00000, 0b000, OPC_OP, "p.ff1">,
            Sched<[WriteIALU, ReadIALU]>;
def P_FL1 : PulpUnary<0b0001000, 0b00000, 0b001, OPC_OP, "p.fl1">,
            Sched<[WriteIALU, ReadIALU]>;
def P_CLB : PulpUnary<0b0001000, 0b00000, 0b010, OPC_OP, "p.clb">,
            Sched<[WriteIALU, ReadIALU]>;
def P_CNT : PulpUnary<0b0001000, 0b00000, 0b011, OPC_OP, "p.cnt">,
            Sched<[WriteIALU, ReadIALU]>;
}
