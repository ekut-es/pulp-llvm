


let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_r<bits<7> funct7, bits<5> funct5, bits<3> funct3,
               string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1">,
      Sched<[WriteIALU, ReadIALU]> {
  let Inst{24-20} = funct5;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_rr<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode, string opcodestr>
    : RVInstR<funct7, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2),
              opcodestr, "$rd, $rs1, $rs2">,
      Sched<[WriteIALU, ReadIALU, ReadIALU]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_ri<bits<7> funct7, bits<3> funct3, string opcodestr>
    : RVInstR<funct7, funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1, uimm5:$Is2),
              opcodestr, "$rd, $rs1, $Is2">,
      Sched<[WriteIALU, ReadIALU]> {
  bits<5> Is2;
  let Inst{24-20} = Is2;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_rii<bits<2> funct2, bits<3> funct3, string opcodestr>
    : RVInstI<funct3, OPC_OP, (outs GPR:$rd), (ins GPR:$rs1, uimm5:$Is3, uimm5:$Is2),
              opcodestr, "$rd, $rs1, $Is3, $Is2">,
      Sched<[WriteIALU, ReadIALU]> {
  bits<5> Is3;
  bits<5> Is2;

  let Inst{31-30} = funct2;
  let Inst{29-25} = Is3;
  let Inst{24-20} = Is2;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_rri<bits<2> funct2, bits<3> funct3, RISCVOpcode opcode, string opcodestr>
    : RVInstI<funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2, uimm5:$Is3),
              opcodestr, "$rd, $rs1, $rs2, $Is3">,
      Sched<[WriteIALU, ReadIALU, ReadIALU]> {
  bits<5> rs2;
  bits<5> Is3;

  let Inst{31-30} = funct2;
  let Inst{29-25} = Is3;
  let Inst{24-20} = rs2;
}


let Predicates = [HasNonStdExtPulp] in {

// Bit manipulation operations
def P_EXTRACT :   Pulp_ALU_rii<0b11, 0b000, "p.extract">;
def P_EXTRACTU :  Pulp_ALU_rii<0b11, 0b001, "p.extractu">;
def P_INSERT :    Pulp_ALU_rii<0b11, 0b010, "p.insert">;
def P_BCLR :      Pulp_ALU_rii<0b11, 0b011, "p.bclr">;
def P_BSET :      Pulp_ALU_rii<0b11, 0b100, "p.bset">;

def P_EXTRACTR :  Pulp_ALU_rr<0b1000000, 0b000, OPC_OP, "p.extractr">;
def P_EXTRACTUR : Pulp_ALU_rr<0b1000000, 0b001, OPC_OP, "p.extractur">;
def P_INSERTR :   Pulp_ALU_rr<0b1000000, 0b010, OPC_OP, "p.insertr">;
def P_BCLRR :     Pulp_ALU_rr<0b1000000, 0b011, OPC_OP, "p.bclrr">;
def P_BSETR :     Pulp_ALU_rr<0b1000000, 0b100, OPC_OP, "p.bsetr">;

def P_ROR :       Pulp_ALU_rr<0b0000100, 0b101, OPC_OP, "p.ror">;
def P_FF1 :       Pulp_ALU_r<0b0001000, 0b00000, 0b000, "p.ff1">;
def P_FL1 :       Pulp_ALU_r<0b0001000, 0b00000, 0b001, "p.fl1">;
def P_CLB :       Pulp_ALU_r<0b0001000, 0b00000, 0b010, "p.clb">;
def P_CNT :       Pulp_ALU_r<0b0001000, 0b00000, 0b011, "p.cnt">;


// General ALU operations
def P_ABS :    Pulp_ALU_r<0b0000010, 0b00000, 0b000, "p.abs">;
def P_SLET :   Pulp_ALU_rr<0b0000010, 0b010, OPC_OP, "p.slet">;
def P_SLETU :  Pulp_ALU_rr<0b0000010, 0b011, OPC_OP, "p.sletu">;
def P_MIN :    Pulp_ALU_rr<0b0000010, 0b100, OPC_OP, "p.min">;
def P_MINU :   Pulp_ALU_rr<0b0000010, 0b101, OPC_OP, "p.minu">;
def P_MAX :    Pulp_ALU_rr<0b0000010, 0b110, OPC_OP, "p.max">;
def P_MAXU :   Pulp_ALU_rr<0b0000010, 0b111, OPC_OP, "p.maxu">;
def P_EXTHS :  Pulp_ALU_r<0b0001000, 0b00000, 0b100, "p.exths">;
def P_EXTHZ :  Pulp_ALU_r<0b0001000, 0b00000, 0b101, "p.exthz">;
def P_EXTBS :  Pulp_ALU_r<0b0001000, 0b00000, 0b110, "p.extbs">;
def P_EXTBZ :  Pulp_ALU_r<0b0001000, 0b00000, 0b111, "p.extbz">;

def P_CLIP :   Pulp_ALU_ri<0b0001010, 0b001, "p.clip">;
def P_CLIPU :  Pulp_ALU_ri<0b0001010, 0b010, "p.clipu">;
def P_CLIPR :  Pulp_ALU_rr<0b0001010, 0b101, OPC_OP, "p.clipr">;
def P_CLIPUR : Pulp_ALU_rr<0b0001010, 0b110, OPC_OP, "p.clipur">;

def P_ADDN :    Pulp_ALU_rri<0b00, 0b010, RISCVOpcode<0b1011011>, "p.addn">;
def P_ADDUN :   Pulp_ALU_rri<0b10, 0b010, RISCVOpcode<0b1011011>, "p.addun">;
def P_ADDRN :   Pulp_ALU_rri<0b00, 0b110, RISCVOpcode<0b1011011>, "p.addrn">;
def P_ADDURN :  Pulp_ALU_rri<0b10, 0b110, RISCVOpcode<0b1011011>, "p.addurn">;
def P_SUBN :    Pulp_ALU_rri<0b00, 0b011, RISCVOpcode<0b1011011>, "p.subn">;
def P_SUBUN :   Pulp_ALU_rri<0b10, 0b011, RISCVOpcode<0b1011011>, "p.subun">;
def P_SUBRN :   Pulp_ALU_rri<0b00, 0b111, RISCVOpcode<0b1011011>, "p.subrn">;
def P_SUBURN :  Pulp_ALU_rri<0b10, 0b111, RISCVOpcode<0b1011011>, "p.suburn">;
def P_ADDNR :   Pulp_ALU_rr<0b0100000, 0b010, RISCVOpcode<0b1011011>, "p.addnr">;
def P_ADDUNR :  Pulp_ALU_rr<0b1100000, 0b010, RISCVOpcode<0b1011011>, "p.addunr">;
def P_ADDRNR :  Pulp_ALU_rr<0b0100000, 0b110, RISCVOpcode<0b1011011>, "p.addrnr">;
def P_ADDURNR : Pulp_ALU_rr<0b1100000, 0b110, RISCVOpcode<0b1011011>, "p.addurnr">;
def P_SUBNR :   Pulp_ALU_rr<0b0100000, 0b011, RISCVOpcode<0b1011011>, "p.subnr">;
def P_SUBUNR :  Pulp_ALU_rr<0b1100000, 0b011, RISCVOpcode<0b1011011>, "p.subunr">;
def P_SUBRNR :  Pulp_ALU_rr<0b0100000, 0b111, RISCVOpcode<0b1011011>, "p.subrnr">;
def P_SUBURNR : Pulp_ALU_rr<0b1100000, 0b111, RISCVOpcode<0b1011011>, "p.suburnr">;
}


// MAC Operations
let Predicates = [HasNonStdExtPulp] in {

def P_MAC :   Pulp_ALU_rr<0b0100001, 0b000, OPC_OP, "p.mac">;
def P_MSU :   Pulp_ALU_rr<0b0100001, 0b001, OPC_OP, "p.msu">;

def P_MULS :   Pulp_ALU_rr<0b1000000, 0b000, RISCVOpcode<0b1011011>, "p.muls">;
def P_MULHHS : Pulp_ALU_rr<0b1100000, 0b000, RISCVOpcode<0b1011011>, "p.mulhhs">;
def P_MULSN :   Pulp_ALU_rri<0b10, 0b000, RISCVOpcode<0b1011011>, "p.mulsn">;
def P_MULHHSN :   Pulp_ALU_rri<0b11, 0b000, RISCVOpcode<0b1011011>, "p.mulhhsn">;
def P_MULSRN :   Pulp_ALU_rri<0b10, 0b100, RISCVOpcode<0b1011011>, "p.mulsrn">;
def P_MULHHSRN :   Pulp_ALU_rri<0b11, 0b100, RISCVOpcode<0b1011011>, "p.mulhhsrn">;
def P_MULU :   Pulp_ALU_rr<0b0000000, 0b000, RISCVOpcode<0b1011011>, "p.mulu">;
def P_MULHHU : Pulp_ALU_rr<0b0100000, 0b000, RISCVOpcode<0b1011011>, "p.mulhhu">;
def P_MULUN :   Pulp_ALU_rri<0b00, 0b000, RISCVOpcode<0b1011011>, "p.mulun">;
def P_MULHHUN :   Pulp_ALU_rri<0b01, 0b000, RISCVOpcode<0b1011011>, "p.mulhhun">;
def P_MULURN :   Pulp_ALU_rri<0b00, 0b100, RISCVOpcode<0b1011011>, "p.mulurn">;
def P_MULHHURN :   Pulp_ALU_rri<0b01, 0b100, RISCVOpcode<0b1011011>, "p.mulhhurn">;
def P_MACSN :   Pulp_ALU_rri<0b10, 0b001, RISCVOpcode<0b1011011>, "p.macsn">;
def P_MACHHSN :   Pulp_ALU_rri<0b11, 0b001, RISCVOpcode<0b1011011>, "p.machhsn">;
def P_MACSRN :   Pulp_ALU_rri<0b10, 0b101, RISCVOpcode<0b1011011>, "p.macsrn">;
def P_MACHHSRN :   Pulp_ALU_rri<0b11, 0b101, RISCVOpcode<0b1011011>, "p.machhsrn">;
def P_MACUN :   Pulp_ALU_rri<0b00, 0b001, RISCVOpcode<0b1011011>, "p.macun">;
def P_MACHHUN :   Pulp_ALU_rri<0b01, 0b001, RISCVOpcode<0b1011011>, "p.machhun">;
def P_MACURN :   Pulp_ALU_rri<0b00, 0b101, RISCVOpcode<0b1011011>, "p.macurn">;
def P_MACHHURN :   Pulp_ALU_rri<0b01, 0b101, RISCVOpcode<0b1011011>, "p.machhurn">;
}
