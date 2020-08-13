


let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_r<bits<7> funct7, bits<3> funct3,
                 RISCVOpcode opcode, string opcodestr>
    : RVInstR<funct7, funct3, opcode, (outs GPR:$rd), (ins GPR:$rs1),
              opcodestr, "$rd, $rs1">,
      Sched<[WriteIALU, ReadIALU]> {
  let rs2 = 0;
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
def P_FF1 :       Pulp_ALU_r<0b0001000, 0b000, OPC_OP, "p.ff1">;
def P_FL1 :       Pulp_ALU_r<0b0001000, 0b001, OPC_OP, "p.fl1">;
def P_CLB :       Pulp_ALU_r<0b0001000, 0b010, OPC_OP, "p.clb">;
def P_CNT :       Pulp_ALU_r<0b0001000, 0b011, OPC_OP, "p.cnt">;


// General ALU operations
def P_ABS :    Pulp_ALU_r<0b0000010, 0b000, OPC_OP, "p.abs">;
def P_SLET :   Pulp_ALU_rr<0b0000010, 0b010, OPC_OP, "p.slet">;
def P_SLETU :  Pulp_ALU_rr<0b0000010, 0b011, OPC_OP, "p.sletu">;
def P_MIN :    Pulp_ALU_rr<0b0000010, 0b100, OPC_OP, "p.min">;
def P_MINU :   Pulp_ALU_rr<0b0000010, 0b101, OPC_OP, "p.minu">;
def P_MAX :    Pulp_ALU_rr<0b0000010, 0b110, OPC_OP, "p.max">;
def P_MAXU :   Pulp_ALU_rr<0b0000010, 0b111, OPC_OP, "p.maxu">;
def P_EXTHS :  Pulp_ALU_r<0b0001000, 0b100, OPC_OP, "p.exths">;
def P_EXTHZ :  Pulp_ALU_r<0b0001000, 0b101, OPC_OP, "p.exthz">;
def P_EXTBS :  Pulp_ALU_r<0b0001000, 0b110, OPC_OP, "p.extbs">;
def P_EXTBZ :  Pulp_ALU_r<0b0001000, 0b111, OPC_OP, "p.extbz">;

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

def uimm6 : Operand<XLenVT>, ImmLeaf<XLenVT, [{return isUInt<6>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<6>;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<6>";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return isUInt<6>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
class Pulp_ALU_sci<bits<5> funct5, bit F, bits<3> funct3, RISCVOpcode opcode,
                   bit signedImm, string opcodestr>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1, !if(signedImm, simm6, uimm6):$Imm6), opcodestr, "$rd, $rs1, $Imm6", [], InstFormatR>,
      Sched<[WriteIALU, ReadIALU]>  {
  bits<6> Imm6;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-27} = funct5;
  let Inst{26} = F;
  let Inst{25} = Imm6{0};
  let Inst{24-20} = Imm6{5-1};
  let Inst{19-15} = rs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = rd;
  let Opcode = opcode.Value;
}

multiclass SC_SCI_H_B<bits<5> funct5, bit F, bit signedImm, string opcodestr> {
  def _H     : Pulp_ALU_rr<{funct5, F, 0}, 0b000, RISCVOpcode<0b1010111>,            "pv." # opcodestr # ".h">;
  def _SC_H  : Pulp_ALU_rr<{funct5, F, 0}, 0b100, RISCVOpcode<0b1010111>,            "pv." # opcodestr # ".sc.h">;
  def _SCI_H : Pulp_ALU_sci<funct5, F,     0b110, RISCVOpcode<0b1010111>, signedImm, "pv." # opcodestr # ".sci.h">;
  def _B     : Pulp_ALU_rr<{funct5, F, 0}, 0b001, RISCVOpcode<0b1010111>,            "pv." # opcodestr # ".b">;
  def _SC_B  : Pulp_ALU_rr<{funct5, F, 0}, 0b101, RISCVOpcode<0b1010111>,            "pv." # opcodestr # ".sc.b">;
  def _SCI_B : Pulp_ALU_sci<funct5, F,     0b111, RISCVOpcode<0b1010111>, signedImm, "pv." # opcodestr # ".sci.b">;
}

// Vectorial ALU Operations
let Predicates = [HasNonStdExtPulp] in {

defm PV_ADD  : SC_SCI_H_B<0b00000, 0, 1, "add">;
defm PV_SUB  : SC_SCI_H_B<0b00001, 0, 1, "sub">;
defm PV_AVG  : SC_SCI_H_B<0b00010, 0, 1, "avg">;
defm PV_AVGU : SC_SCI_H_B<0b00011, 0, 0, "avgu">;
defm PV_MIN  : SC_SCI_H_B<0b00100, 0, 1, "min">;
defm PV_MINU : SC_SCI_H_B<0b00101, 0, 0, "minu">;
defm PV_MAX  : SC_SCI_H_B<0b00110, 0, 1, "max">;
defm PV_MAXU : SC_SCI_H_B<0b00111, 0, 0, "maxu">;
defm PV_SRL  : SC_SCI_H_B<0b01000, 0, 0, "srl">;
defm PV_SRA  : SC_SCI_H_B<0b01001, 0, 0, "sra">;
defm PV_SLL  : SC_SCI_H_B<0b01010, 0, 0, "sll">;
defm PV_OR   : SC_SCI_H_B<0b01011, 0, 1, "or">;
defm PV_XOR  : SC_SCI_H_B<0b01100, 0, 1, "xor">;
defm PV_AND  : SC_SCI_H_B<0b01101, 0, 1, "and">;

def PV_ABS_H      : Pulp_ALU_r  <0b0111000,  0b000, RISCVOpcode<0b1010111>,    "pv.abs.h">;
def PV_ABS_B      : Pulp_ALU_r  <0b0111000,  0b001, RISCVOpcode<0b1010111>,    "pv.abs.b">;
def PV_EXTRACT_H  : Pulp_ALU_sci<0b01111, 0, 0b110, RISCVOpcode<0b1010111>, 1, "pv.extract.h">;
def PV_EXTRACT_B  : Pulp_ALU_sci<0b01111, 0, 0b111, RISCVOpcode<0b1010111>, 1, "pv.extract.b">;
def PV_EXTRACTU_H : Pulp_ALU_sci<0b10010, 0, 0b110, RISCVOpcode<0b1010111>, 1, "pv.extractu.h">;
def PV_EXTRACTU_B : Pulp_ALU_sci<0b10010, 0, 0b111, RISCVOpcode<0b1010111>, 1, "pv.extractu.b">;
def PV_INSERT_H   : Pulp_ALU_sci<0b10110, 0, 0b110, RISCVOpcode<0b1010111>, 1, "pv.insert.h">;
def PV_INSERT_B   : Pulp_ALU_sci<0b10110, 0, 0b111, RISCVOpcode<0b1010111>, 1, "pv.insert.b">;

defm PV_DOTUP   : SC_SCI_H_B<0b10000, 0, 0, "dotup">;
defm PV_DOTUSP  : SC_SCI_H_B<0b10001, 0, 1, "dotusp">;
defm PV_DOTSP   : SC_SCI_H_B<0b10011, 0, 1, "dotsp">;
defm PV_SDOTUP  : SC_SCI_H_B<0b10100, 0, 0, "sdotup">;
defm PV_SDOTUSP : SC_SCI_H_B<0b10101, 0, 1, "sdotusp">;
defm PV_SDOTSP  : SC_SCI_H_B<0b10111, 0, 1, "sdotsp">;

def PV_SHUFFLE_H       : Pulp_ALU_rr <0b1100000,  0b000, RISCVOpcode<0b1010111>,    "pv.shuffle.h">;
def PV_SHUFFLE_SCI_H   : Pulp_ALU_sci<0b11000, 0, 0b110, RISCVOpcode<0b1010111>, 0, "pv.shuffle.sci.h">;
def PV_SHUFFLE_B       : Pulp_ALU_rr <0b1100000,  0b001, RISCVOpcode<0b1010111>,    "pv.shuffle.b">;
def PV_SHUFFLEI0_SCI_B : Pulp_ALU_sci<0b11000, 0, 0b111, RISCVOpcode<0b1010111>, 0, "pv.shufflei0.sci.b">;
def PV_SHUFFLEI1_SCI_B : Pulp_ALU_sci<0b11101, 0, 0b111, RISCVOpcode<0b1010111>, 0, "pv.shufflei1.sci.b">;
def PV_SHUFFLEI2_SCI_B : Pulp_ALU_sci<0b11110, 0, 0b111, RISCVOpcode<0b1010111>, 0, "pv.shufflei2.sci.b">;
def PV_SHUFFLEI3_SCI_B : Pulp_ALU_sci<0b11111, 0, 0b111, RISCVOpcode<0b1010111>, 0, "pv.shufflei3.sci.b">;
def PV_SHUFFLE2_H      : Pulp_ALU_rr <0b1100100,  0b000, RISCVOpcode<0b1010111>,    "pv.shuffle2.h">;
def PV_SHUFFLE2_B      : Pulp_ALU_rr <0b1100100,  0b001, RISCVOpcode<0b1010111>,    "pv.shuffle2.b">;
def PV_PACK_H          : Pulp_ALU_rr <0b1101000,  0b000, RISCVOpcode<0b1010111>,    "pv.pack.h">;
def PV_PACKHI_B        : Pulp_ALU_rr <0b1101100,  0b001, RISCVOpcode<0b1010111>,    "pv.packhi.b">;
def PV_PACKLO_B        : Pulp_ALU_rr <0b1110000,  0b001, RISCVOpcode<0b1010111>,    "pv.packlo.b">;


// Vectorial comparison operations
defm PV_CMPEQ : SC_SCI_H_B<0b00000, 1, 1, "cmpeq">;
defm PV_CMPNE : SC_SCI_H_B<0b00001, 1, 1, "cmpne">;
defm PV_CMPGT : SC_SCI_H_B<0b00010, 1, 1, "cmpgt">;
defm PV_CMPGE : SC_SCI_H_B<0b00011, 1, 1, "cmpge">;
defm PV_CMPLT : SC_SCI_H_B<0b00100, 1, 1, "cmplt">;
defm PV_CMPLE : SC_SCI_H_B<0b00101, 1, 1, "cmple">;
defm PV_CMPGTU : SC_SCI_H_B<0b00110, 1, 0, "cmpgtu">;
defm PV_CMPGEU : SC_SCI_H_B<0b00111, 1, 0, "cmpgeu">;
defm PV_CMPLTU : SC_SCI_H_B<0b01000, 1, 0, "cmpltu">;
defm PV_CMPLEU : SC_SCI_H_B<0b01001, 1, 0, "cmpleu">;
}
