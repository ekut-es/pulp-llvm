


let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PulpUnary<bits<7> funct7, bits<5> funct5, bits<3> funct3,
               string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1"> {
  let Inst{24-20} = funct5;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class PulpInstRii<bits<2> funct2, bits<3> funct3,
                  string opcodestr>
    : RVInstI<funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1, uimm5:$ls3, uimm5:$ls2),
              opcodestr, "$rd, $rs1, $ls3, $ls2">,
      Sched<[WriteIALU, ReadIALU]> {
  bits<5> ls3;
  bits<5> ls2;
  
  let imm12{11-10} = funct2;
  let imm12{9-5} = ls3;
  let imm12{4-0} = ls2;
}


let Predicates = [HasNonStdExtPulp] in {
def P_EXTRACT :   PulpInstRii<0b11, 0b000, "p.extract">;
def P_EXTRACTU :  PulpInstRii<0b11, 0b001, "p.extractu">;
def P_INSERT :    PulpInstRii<0b11, 0b010, "p.insert">;
def P_BCLR :      PulpInstRii<0b11, 0b011, "p.bclr">;
def P_BSET :      PulpInstRii<0b11, 0b100, "p.bset">;

def P_EXTRACTR :  ALU_rr<0b1000000, 0b000, "p.extractr">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def P_EXTRACTUR : ALU_rr<0b1000000, 0b001, "p.extractur">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def P_INSERTR :   ALU_rr<0b1000000, 0b010, "p.insertr">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def P_BCLRR :     ALU_rr<0b1000000, 0b011, "p.bclrr">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def P_BSETR :     ALU_rr<0b1000000, 0b100, "p.bsetr">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;

def P_ROR :       ALU_rr<0b0000100, 0b101, "p.ror">,
                  Sched<[WriteIALU, ReadIALU, ReadIALU]>;
def P_FF1 :       PulpUnary<0b0001000, 0b00000, 0b000, "p.ff1">,
                  Sched<[WriteIALU, ReadIALU]>;
def P_FL1 :       PulpUnary<0b0001000, 0b00000, 0b001, "p.fl1">,
                  Sched<[WriteIALU, ReadIALU]>;
def P_CLB :       PulpUnary<0b0001000, 0b00000, 0b010, "p.clb">,
                  Sched<[WriteIALU, ReadIALU]>;
def P_CNT :       PulpUnary<0b0001000, 0b00000, 0b011, "p.cnt">,
                  Sched<[WriteIALU, ReadIALU]>;
}
