//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qinterface.hpp"

#include <algorithm>

namespace Qrack {

#define REG_GATE_1(gate)                                                                                               \
    void QInterface::gate(bitLenInt start, bitLenInt length)                                                           \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(start + bit);                                                                                         \
        }                                                                                                              \
    }

#define REG_GATE_2(gate)                                                                                               \
    void QInterface::gate(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)                                        \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(qubit1 + bit, qubit2 + bit);                                                                          \
        }                                                                                                              \
    }

#define REG_GATE_3(gate)                                                                                               \
    void QInterface::gate(bitLenInt qubit1, bitLenInt qubit2, bitLenInt qubit3, bitLenInt length)                      \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(qubit1 + bit, qubit2 + bit, qubit3 + bit);                                                            \
        }                                                                                                              \
    }

#define REG_GATE_3B(gate)                                                                                              \
    void QInterface::gate(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)    \
    {                                                                                                                  \
        for (bitLenInt i = 0; i < length; i++) {                                                                       \
            gate(qInputStart + i, (bitCapIntOcl)bitSlice(i, classicalInput), outputStart + i);                         \
        }                                                                                                              \
    }

#define REG_GATE_1R(gate)                                                                                              \
    void QInterface::gate(real1_f radians, bitLenInt start, bitLenInt length)                                          \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(radians, start + bit);                                                                                \
        }                                                                                                              \
    }

#define REG_GATE_1D(gate)                                                                                              \
    void QInterface::gate(int numerator, int denominator, bitLenInt start, bitLenInt length)                           \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(numerator, denominator, start + bit);                                                                 \
        }                                                                                                              \
    }

#define REG_GATE_C1_1(gate)                                                                                            \
    void QInterface::gate(bitLenInt control, bitLenInt target, bitLenInt length)                                       \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(control + bit, target + bit);                                                                         \
        }                                                                                                              \
    }

#define REG_GATE_C2_1(gate)                                                                                            \
    void QInterface::gate(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)                  \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(control1 + bit, control2 + bit, target + bit);                                                        \
        }                                                                                                              \
    }

#define REG_GATE_C1_1R(gate)                                                                                           \
    void QInterface::gate(real1_f radians, bitLenInt control, bitLenInt target, bitLenInt length)                      \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(radians, control + bit, target + bit);                                                                \
        }                                                                                                              \
    }

#define REG_GATE_C1_1D(gate)                                                                                           \
    void QInterface::gate(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)       \
    {                                                                                                                  \
        for (bitLenInt bit = 0; bit < length; bit++) {                                                                 \
            gate(numerator, denominator, control + bit, target + bit);                                                 \
        }                                                                                                              \
    }

inline real1_f dyadAngle(int numerator, int denomPower) { return (-M_PI * numerator * 2) / pow(2, denomPower); };

/// Bit-wise apply swap to two registers
REG_GATE_2(Swap);

/// Bit-wise apply iswap to two registers
REG_GATE_2(ISwap);

/// Bit-wise apply square root of swap to two registers
REG_GATE_2(SqrtSwap);

/// Bit-wise apply inverse square root of swap to two registers
REG_GATE_2(ISqrtSwap);

void QInterface::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        FSim(theta, phi, qubit1 + bit, qubit2 + bit);
    }
}

/// Bit-wise apply "anti-"controlled-z to two control registers and one target register
REG_GATE_C2_1(AntiCCZ);

/// Bit-wise apply doubly-controlled-z to two control registers and one target register
REG_GATE_C2_1(CCZ);

/// Apply "Anti-"CZ gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCZ);

/// Apply controlled Pauli Z matrix to each bit
REG_GATE_C1_1(CZ);

/// Bit-wise apply "anti-"controlled-not to two control registers and one target register
REG_GATE_C2_1(AntiCCNOT);

/// Bit-wise apply controlled-not to two control registers and one target register
REG_GATE_C2_1(CCNOT);

/// Apply "Anti-"CNOT gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCNOT);

/// Apply CNOT gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(CNOT);

/// Bit-wise apply "anti-"controlled-y to two control registers and one target register
REG_GATE_C2_1(AntiCCY);

/// Bit-wise apply doubly-controlled-y to two control registers and one target register
REG_GATE_C2_1(CCY);

// Apply "Anti-"CY gate for "length" starting from "control" and "target," respectively
REG_GATE_C1_1(AntiCY);

/// Apply controlled Pauli Y matrix to each bit
REG_GATE_C1_1(CY);

/// Apply S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
REG_GATE_1(S);

/// Apply inverse S gate (1/4 phase rotation) to each bit in "length," starting from bit index "start"
REG_GATE_1(IS);

/// Apply T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
REG_GATE_1(T);

/// Apply inverse T gate (1/8 phase rotation)  to each bit in "length," starting from bit index "start"
REG_GATE_1(IT);

/// Apply X ("not") gate to each bit in "length," starting from bit index "start"
REG_GATE_1(X);

/// Apply square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SqrtX);

/// Apply inverse square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(ISqrtX);

/// Apply phased square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SqrtXConjT);

/// Apply inverse phased square root of X gate to each bit in "length," starting from bit index "start"
REG_GATE_1(ISqrtXConjT);

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
REG_GATE_1(H);

/// Apply Y-basis transformation gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SH);

/// Apply inverse Y-basis transformation gate to each bit in "length," starting from bit index "start"
REG_GATE_1(HIS);

/// Apply square root of Hadamard gate to each bit in "length," starting from bit index "start"
REG_GATE_1(SqrtH);

/// Apply Pauli Y matrix to each bit
REG_GATE_1(Y);

/// Apply square root of Pauli Y matrix to each bit
REG_GATE_1(SqrtY);

/// Apply square root of Pauli Y matrix to each bit
REG_GATE_1(ISqrtY);

/// Apply Pauli Z matrix to each bit
REG_GATE_1(Z);

/// Apply controlled H gate to each bit
REG_GATE_C1_1(CH);

/// Apply controlled S gate to each bit
REG_GATE_C1_1(CS);

/// Apply controlled IS gate to each bit
REG_GATE_C1_1(CIS);

/// Apply controlled T gate to each bit
REG_GATE_C1_1(CT);

/// Apply controlled IT gate to each bit
REG_GATE_C1_1(CIT);

/// "AND" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(AND);

/// "OR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(OR);

/// "XOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(XOR);

/// "NAND" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(NAND);

/// "NOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(NOR);

/// "XNOR" compare a 2 bit ranges in QInterface and store result in range starting at output
REG_GATE_3(XNOR);

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLAND);

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLOR);

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLXOR);

/// "NAND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLNAND);

/// "NOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLNOR);

/// "XNOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
REG_GATE_3B(CLXNOR);

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
REG_GATE_1R(RT);

/// Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / 2^denomPower) around |1> state.
void QInterface::RTDyad(int numerator, int denomPower, bitLenInt qubit) { RT(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
REG_GATE_1D(RTDyad);

/// Bitwise (identity) exponentiation gate - Applies exponentiation of the identity operator
REG_GATE_1R(Exp);

/// Dyadic fraction (identity) exponentiation gate - Applies exponentiation of the identity operator
void QInterface::ExpDyad(int numerator, int denomPower, bitLenInt qubit)
{
    Exp(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction (identity) exponentiation gate - Applies \f$ e^{-i * \pi * numerator * I / 2^denomPower} \f$,
REG_GATE_1D(ExpDyad);

/// Bitwise Pauli X exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
REG_GATE_1R(ExpX);

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
void QInterface::ExpXDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpX(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction Pauli X exponentiation gate - Applies exponentiation of the Pauli X operator
REG_GATE_1D(ExpXDyad);

/// Bitwise Pauli Y exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
REG_GATE_1R(ExpY);

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
void QInterface::ExpYDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpY(dyadAngle(numerator, denomPower), qubit);
}

/// Dyadic fraction Pauli Y exponentiation gate - Applies exponentiation of the Pauli Y operator
REG_GATE_1D(ExpYDyad);

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
void QInterface::ExpZDyad(int numerator, int denomPower, bitLenInt qubit)
{
    ExpZ(dyadAngle(numerator, denomPower), qubit);
}

/**
 * Bitwise Pauli Z exponentiation gate - Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
 */
REG_GATE_1R(ExpZ);

/// Dyadic fraction Pauli Z exponentiation gate - Applies exponentiation of the Pauli Z operator
REG_GATE_1D(ExpZDyad);

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
REG_GATE_1R(RX);

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::RXDyad(int numerator, int denomPower, bitLenInt qubit) { RX(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction x axis rotation gate - Rotates around Pauli x
REG_GATE_1D(RXDyad);

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
REG_GATE_1R(RY);

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RYDyad(int numerator, int denomPower, bitLenInt qubit) { RY(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction y axis rotation gate - Rotates each bit around Pauli y axis.
REG_GATE_1D(RYDyad);

/// z axis rotation gate - Rotates each bit around Pauli z axis
REG_GATE_1R(RZ);

/// Dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::RZDyad(int numerator, int denomPower, bitLenInt qubit) { RZ(dyadAngle(numerator, denomPower), qubit); }

/// Dyadic fraction z axis rotation gate - Rotates each bit around Pauli y axis.
REG_GATE_1D(RZDyad)

/// Controlled "phase shift gate"
REG_GATE_C1_1R(CRT);

/// Controlled dyadic "phase shift gate" - if control bit is true, rotates target bit as e^(i*(M_PI * numerator) /
/// 2^denomPower) around |1> state
void QInterface::CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRT(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction "phase shift gate"
REG_GATE_C1_1D(CRTDyad);

/// Controlled x axis rotation
REG_GATE_C1_1R(CRX);

/// Controlled dyadic fraction x axis rotation gate - Rotates around Pauli x axis.
void QInterface::CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRX(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
REG_GATE_C1_1D(CRXDyad);

/// Controlled y axis rotation
REG_GATE_C1_1R(CRY);

/// Controlled dyadic fraction y axis rotation gate - Rotates around Pauli y axis.
void QInterface::CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRY(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
REG_GATE_C1_1D(CRYDyad);

/// Controlled z axis rotation
REG_GATE_C1_1R(CRZ);

/// Controlled dyadic fraction z axis rotation gate - Rotates around Pauli z axis.
void QInterface::CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target)
{
    CRZ(dyadAngle(numerator, denomPower), control, target);
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
REG_GATE_C1_1D(CRZDyad);

/// Apply "PhaseRootN" gate (1/(2^N) phase rotation) to each bit in "length", starting from bit index "start"
void QInterface::PhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        PhaseRootN(n, start + bit);
    }
}

/// Apply inverse "PhaseRootN" gate (1/(2^N) phase rotation) to each bit in "length", starting from bit index "start"
void QInterface::IPhaseRootN(bitLenInt n, bitLenInt start, bitLenInt length)
{
    for (bitLenInt bit = 0; bit < length; bit++) {
        IPhaseRootN(n, start + bit);
    }
}

/// Apply controlled "PhaseRootN" gate to each bit
void QInterface::CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        CZ(control, target, length);
        return;
    }

    for (bitLenInt bit = 0; bit < length; bit++) {
        CPhaseRootN(n, control + bit, target + bit);
    }
}

/// Apply controlled IT gate to each bit
void QInterface::CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target, bitLenInt length)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        CZ(control, target, length);
        return;
    }

    for (bitLenInt bit = 0; bit < length; bit++) {
        CIPhaseRootN(n, control + bit, target + bit);
    }
}

/// Arithmetic shift left, with last 2 bits as sign and carry
void QInterface::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROL(shift, start, length);
            SetReg(start, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void QInterface::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROR(shift, start, length);
            SetReg(end - shift - 1, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Logical shift left, filling the extra bits with |0>
void QInterface::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROL(shift, start, length);
            SetReg(start, shift, 0);
        }
    }
}

/// Logical shift right, filling the extra bits with |0>
void QInterface::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            SetReg(start, shift, 0);
            ROR(shift, start, length);
        }
    }
}

/// Quantum Fourier Transform - Optimized for going from |0>/|1> to |+>/|-> basis
void QInterface::QFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    bitLenInt end = start + (length - 1U);
    for (bitLenInt i = 0; i < length; i++) {
        bitLenInt hBit = end - i;
        for (bitLenInt j = 0; j < i; j++) {
            bitLenInt c = hBit;
            bitLenInt t = hBit + 1U + j;
            CPhaseRootN(j + 2U, c, t);
            if (trySeparate) {
                TrySeparate(c, t);
            }
        }
        H(hBit);
    }
}

/// Inverse Quantum Fourier Transform - Quantum Fourier transform optimized for going from |+>/|-> to |0>/|1> basis
void QInterface::IQFT(bitLenInt start, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    for (bitLenInt i = 0; i < length; i++) {
        for (bitLenInt j = 0; j < i; j++) {
            bitLenInt c = (start + i) - (j + 1U);
            bitLenInt t = start + i;
            CIPhaseRootN(j + 2U, c, t);
            if (trySeparate) {
                TrySeparate(c, t);
            }
        }
        H(start + i);
    }
}

/// Quantum Fourier Transform - Optimized for going from |0>/|1> to |+>/|-> basis
void QInterface::QFTR(bitLenInt* qubits, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    bitLenInt end = (length - 1U);
    for (bitLenInt i = 0; i < length; i++) {
        H(qubits[end - i]);
        for (bitLenInt j = 0; j < (bitLenInt)((length - 1U) - i); j++) {
            CPhaseRootN(j + 2U, qubits[(end - i) - (j + 1U)], qubits[end - i]);
        }

        if (trySeparate) {
            TrySeparate(qubits[end - i]);
        }
    }
}

/// Inverse Quantum Fourier Transform - Quantum Fourier transform optimized for going from |+>/|-> to |0>/|1> basis
void QInterface::IQFTR(bitLenInt* qubits, bitLenInt length, bool trySeparate)
{
    if (length == 0) {
        return;
    }

    for (bitLenInt i = 0; i < length; i++) {
        for (bitLenInt j = 0; j < i; j++) {
            CIPhaseRootN(j + 2U, qubits[i - (j + 1U)], qubits[i]);
        }
        H(qubits[i]);

        if (trySeparate) {
            TrySeparate(qubits[i]);
        }
    }
}

/// Set register bits to given permutation
void QInterface::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (value == 1));
    } else if ((start == 0) && (length == qubitCount)) {
        SetPermutation(value);
    } else {
        bitCapInt regVal = MReg(start, length);
        for (bitLenInt i = 0; i < length; i++) {
            bool bitVal = (bitCapIntOcl)bitSlice(i, regVal);
            if ((bitVal && !bitSlice(i, value)) || (!bitVal && bitSlice(i, value)))
                X(start + i);
        }
    }
}

/// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    bitCapInt res = 0;
    for (bitLenInt bit = 0; bit < length; bit++) {
        bitCapInt power = pow2(bit);
        res |= ForceM(start + bit, (bool)(power & result), doForce, doApply) ? power : 0;
    }
    return res;
}

/// Bit-wise apply measurement gate to a register
bitCapInt QInterface::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values, bool doApply)
{
    bitCapInt result = 0;
    if (values == NULL) {
        if (doApply) {
            for (bitLenInt bit = 0; bit < length; bit++) {
                result |= M(bits[bit]) ? pow2(bits[bit]) : 0;
            }
        } else {
            std::vector<bitCapInt> qPowers(length);
            for (bitLenInt bit = 0; bit < length; bit++) {
                qPowers[bit] = pow2(bits[bit]);
            }
            result = MultiShotMeasureMask(&(qPowers[0]), qPowers.size(), 1).begin()->first;
        }
    } else {
        for (bitLenInt bit = 0; bit < length; bit++) {
            result |= ForceM(bits[bit], values[bit], true, doApply) ? pow2(bits[bit]) : 0;
        }
    }
    return result;
}

/// Returns probability of permutation of the register
real1_f QInterface::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    real1 prob = ONE_R1;
    for (bitLenInt i = 0; i < length; i++) {
        if ((permutation >> i) & ONE_BCI) {
            prob *= Prob(start + i);
        } else {
            prob *= (ONE_R1 - Prob(start + i));
        }
    }
    return prob;
}

/// Returns probability of permutation of the mask
real1_f QInterface::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    real1 prob = ZERO_R1;
    for (bitCapInt lcv = 0; lcv < maxQPower; lcv++) {
        if ((lcv & mask) == permutation) {
            prob += ProbAll(lcv);
        }
    }

    return prob;
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QInterface::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        Reverse(start, end);
        Reverse(start, start + shift);
        Reverse(start + shift, end);
    }
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QInterface::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) { ROL(length - shift, start, length); }

std::map<QInterfacePtr, bitLenInt> QInterface::Compose(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto&& q : toCopy) {
        ret[q] = Compose(q);
    }

    return ret;
}

void QInterface::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    bitLenInt length;
    std::vector<bitCapInt> bitPowers;
    for (length = 0; v; length++) {
        bitCapInt oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        bitPowers.push_back((v ^ oldV) & oldV);
    }

    std::fill(probsArray, probsArray + pow2Ocl(length), ZERO_R1);

    for (bitCapInt lcv = 0; lcv < maxQPower; lcv++) {
        bitCapIntOcl retIndex = 0;
        for (bitLenInt p = 0; p < length; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2Ocl(p);
            }
        }
        probsArray[retIndex] += ProbAll(lcv);
    }
}

void QInterface::ProbBitsAll(const bitLenInt* bits, const bitLenInt& length, real1* probsArray)
{
    std::fill(probsArray, probsArray + pow2Ocl(length), ZERO_R1);

    std::vector<bitCapInt> bitPowers(length);
    for (bitLenInt p = 0; p < length; p++) {
        bitPowers[p] = pow2(bits[p]);
    }

    for (bitCapInt lcv = 0; lcv < maxQPower; lcv++) {
        bitCapIntOcl retIndex = 0;
        for (bitLenInt p = 0; p < length; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2Ocl(p);
            }
        }
        probsArray[retIndex] += ProbAll(lcv);
    }
}

real1_f QInterface::ExpectationBitsAll(const bitLenInt* bits, const bitLenInt& length, const bitCapInt& offset)
{
    if (length == 1U) {
        return Prob(bits[0]);
    }

    std::vector<bitCapInt> bitPowers(length);
    for (bitLenInt p = 0; p < length; p++) {
        bitPowers[p] = pow2(bits[p]);
    }

    real1_f expectation = 0;
    for (bitCapInt lcv = 0; lcv < maxQPower; lcv++) {
        bitCapInt retIndex = 0;
        for (bitLenInt p = 0; p < length; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= pow2(p);
            }
        }
        expectation += (bitCapIntOcl)(offset + retIndex) * ProbAll(lcv);
    }

    return expectation;
}

std::map<bitCapInt, int> QInterface::MultiShotMeasureMask(
    const bitCapInt* qPowers, const bitLenInt qPowerCount, const unsigned int shots)
{
    if (!shots) {
        return std::map<bitCapInt, int>();
    }

    std::vector<bitCapInt> maskMap(qPowerCount);
    for (bitLenInt i = 0; i < qPowerCount; i++) {
        maskMap[i] = qPowers[i];
    }

    std::unique_ptr<real1[]> allProbsArray(new real1[(bitCapIntOcl)maxQPower]);
    GetProbs(allProbsArray.get());

    bitCapInt maskMaxQPower = pow2(qPowerCount);
    std::unique_ptr<real1[]> maskProbsArray(new real1[(bitCapIntOcl)maskMaxQPower]());
    for (bitCapIntOcl j = 0; j < maxQPower; j++) {
        bitCapIntOcl maskPerm = 0;
        for (bitLenInt i = 0; i < qPowerCount; i++) {
            if (j & maskMap[i]) {
                maskPerm |= pow2Ocl(i);
            }
        }
        maskProbsArray[maskPerm] += allProbsArray[j];
    }
    allProbsArray.reset();

    if (shots == 1U) {
        real1 maskProb = (real1)Rand();
        real1 cumulativeProb = ZERO_R1;
        std::map<bitCapInt, int> results;
        for (bitCapIntOcl j = 0U; j < maskMaxQPower; j++) {
            cumulativeProb += maskProbsArray[j];
            if (cumulativeProb >= maskProb) {
                results[j] = 1U;
                break;
            }
        }
        return results;
    }

    bitCapIntOcl singlePerm = (maskProbsArray[0] > FP_NORM_EPSILON) ? 0U : maskMaxQPower;
    bitCapIntOcl j;
    for (j = 1U; j < maskMaxQPower; j++) {
        if (maskProbsArray[j] > REAL1_EPSILON) {
            if (singlePerm == maskMaxQPower) {
                singlePerm = j;
            } else {
                break;
            }
        }

        maskProbsArray[j] = maskProbsArray[j - 1U] + maskProbsArray[j];
    }

    if ((j == maskMaxQPower) && (singlePerm < maskMaxQPower)) {
        std::map<bitCapInt, int> results;
        results[singlePerm] = shots;
        return results;
    }

    for (; j < maskMaxQPower; j++) {
        maskProbsArray[j] = maskProbsArray[j - 1U] + maskProbsArray[j];
    }

    std::map<bitCapInt, int> results;
    for (unsigned int shot = 0; shot < shots; shot++) {
        real1 maskProb = (real1)Rand();
        real1* bound = std::upper_bound(maskProbsArray.get(), maskProbsArray.get() + maskMaxQPower, maskProb);
        size_t dist = bound - maskProbsArray.get();
        if (dist >= maskMaxQPower) {
            bound--;
        }
        if (maskProb > 0) {
            while (dist && maskProbsArray[dist - 1U] == maskProb) {
                dist--;
            }
        }

        auto result = results.find(dist);
        if (result == results.end()) {
            results[dist] = 1;
        } else {
            result->second++;
        }
    }

    return results;
}


bool QInterface::TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol)
{
    Finish();

    bool tempDoNorm = doNormalize;
    doNormalize = false;
    QInterfacePtr unitCopy = Clone();
    doNormalize = tempDoNorm;

    unitCopy->Decompose(start, dest);
    unitCopy->Compose(dest, start);

    bool didSeparate = ApproxCompare(unitCopy, error_tol);

    if (didSeparate) {
        // The subsystem is separable.
        Dispose(start, dest->GetQubitCount());
    }

    return didSeparate;
}

} // namespace Qrack
