//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>
#include <atomic>
#include <ctime>
#include <future>
#include <math.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <thread>

#include "complex16simd.hpp"

#define Complex16 Complex16Simd
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {

class CoherentUnit;

/*
 * Enumerated list of supported engines.
 *
 * Not currently published since selection isn't supported by the API.
 */
enum CoherentUnitEngine {
    COHERENT_UNIT_ENGINE_SOFTWARE = 0,
    COHERENT_UNIT_ENGINE_OPENCL,

    COHERENT_UNIT_ENGINE_MAX
};

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState);

/// The "Qrack::CoherentUnit" class represents one or more coherent quantum processor registers, including primitive bit
/// logic gates and (abstract) opcodes-like methods.
/**
 * A "Qrack::CoherentUnit" is a qubit permutation state vector with methods to operate on it as by gates and
register-like instructions. In brief: All directly interacting qubits must be contained in a single CoherentUnit object,
by requirement of quantum mechanics, unless a certain collection of bits represents a "separable quantum subsystem." All
registers of a virtual chip will usually be contained in a single CoherentUnit, and they are accesible similar to a
one-dimensional array of qubits.

Introduction: Like classical bits, a set of qubits has a maximal respresentation as the permutation of bits. (An 8 bit
byte has 256 permutations, commonly numbered 0 to 255, as does an 8 bit qubyte.) Additionally, the state of a qubyte is
fully specified in terms of probability and phase of each permutation of qubits. This is the "|0>/|1>" "permutation
basis." There are other fully descriptive bases, such as the |+>/|-> permutation basis, which is characteristic of
Hadamard gates. The notation "|x>" represents a "ket" of the "x" state in the quantum "bra-ket" notation of Dirac. It is
a quantum state vector as described by Schrödinger's equation. When we say |01>, we mean the qubit equivalent of the
binary bit pemutation "01."

The state of a two bit permutation can be described as follows: where one in the set of variables "x_0, x_1, x_2, and
x_3" is equal to 1 and the rest are equal to zero, the state of the bit permutation can always be described by

|psi> = x_0 * |00> + x_1 * |01> + x_2 * |10> + x_3 * |11>

One of the leading variables is always 1 and the rest are always 0. That is, the state of the classical bit combination
is always exactly one of |00>, |01>, |10>, or |11>, and never a mix of them at once, however we would mix them. One way
to mix them is probabilistically, in which the sum of probabilities of states should be 100% or 1. This suggests
splitting for example x_0 and x_1 into 1/2 and 1/2 to represent a potential |psi>, but Schrödinger's equation actually
requires us to split into 1/sqrt(2) and 1/sqrt(2) to get 100% probability, like so,

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |10>,

where the leading coefficients are ultimately squared to give probabilities. This is a valid description of a 2 qubit
permutation. The first equation given before it above encompasses all possible states of a 2 qubit combination, when the
x_n variables are constrained so that the total probability of all states adds up to one. However, the domain of the x_n
variables must also be the complex numbers. This is also a valid state, for example:

|psi> = (1+i) / sqrt(4) * |00> + (1-i) / sqrt(4) * |10>


where "i" is defined as the sqrt(-1). This imparts "phase" to each permutation state vector component like |00> or |10>,
(which are "eigenstates"). Phase and probability of permutation state fully (but not uniquely) specify the state of a
coherent set of qubits.

For N bits, there are 2^N permutation basis "eigenstates" that with probability normalization and phase fully describe
every possible quantum state of the N qubits. A CoherentUnit tracks the 2^N dimensional state vector of eigenstate
components. It optimizes certain register-like methods by operating in parallel over the "entanglements" of these
permutation basis states. For example, the state "|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11>" has a probablity of
both bits being 1 or neither bit being 1, but it has no independent probability for the bits being different, when
measured. If this state is acted on by an X or NOT gate on the left qubit, for example, we need only act on the states
entangled into the original state:

|psi> = 1 / sqrt(2) * |00> + 1 / sqrt(2) * |11>
(When acted on by an X gate on the left bit, goes to:)
|psi> = 1 / sqrt(2) * |10> + 1 / sqrt(2) * |01>

In the permutation basis, "entanglement" is as simple as the ability to restrain bit combinations in specificying an
arbitrary "|psi>" state, as we have just described at length.

 */
class CoherentUnit {
public:
    /// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
    CoherentUnit(bitLenInt qBitCount);

    /// Initialize a coherent unit with qBitCount number pf bits, to initState unsigned integer permutation state
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState);

    /// PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
    CoherentUnit(const CoherentUnit& pqs);

    virtual ~CoherentUnit() {}

    /// Set the random seed (primarily used for testing)
    void SetRandomSeed(uint32_t seed);

    /// Get the count of bits in this register
    int GetQubitCount() { return qubitCount; }

    /// Get the 1 << GetQubitCount()
    int GetMaxQPower() { return maxQPower; }

    /// PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
    void CloneRawState(Complex16* output);

    /// Generate a random double from 0 to 1
    double Rand();

    /// Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
    void SetPermutation(bitCapInt perm);

    /// Set arbitrary pure quantum state, in unsigned int permutation basis
    void SetQuantumState(Complex16* inputState);

    /**
     * Combine (a copy of) another CoherentUnit with this one, after the last bit index of this one. (If the programmer
     * doesn't want to "cheat," it is left up to them to delete the old coherent unit that was added.
     */
    void Cohere(CoherentUnit& toCopy);

    /**
     * Minimally decohere a set of contigious bits from the full coherent unit. The length of this coherent unit is
     * reduced by the length of bits decohered, and the bits removed are output in the destination CoherentUnit pointer.
     * The destination object must be initialized to the correct number of bits, in 0 permutation state.
     */
    void Decohere(bitLenInt start, bitLenInt length, CoherentUnit& destination);

    void Dispose(bitLenInt start, bitLenInt length);

    // Logic Gates
    //
    // Each bit is paired with a CL* variant that utilizes a classical bit as
    // an input.

    void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);

    void CNOT(bitLenInt control, bitLenInt target);
    void AntiCNOT(bitLenInt control, bitLenInt target);

    void H(bitLenInt qubitIndex);
    bool M(bitLenInt qubitIndex);

    void X(bitLenInt qubitIndex);
    void Y(bitLenInt qubitIndex);
    void Z(bitLenInt qubitIndex);

    // Controlled variants
    void CY(bitLenInt control, bitLenInt target);
    void CZ(bitLenInt control, bitLenInt target);

    /// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
    double Prob(bitLenInt qubitIndex);

    /// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
    double ProbAll(bitCapInt fullRegister);

    /// PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
    void ProbArray(double* probArray);

    /*
     * Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     */

    /// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
    void R1(double radians, bitLenInt qubitIndex);

    /// Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state.
    void R1Dyad(int numerator, int denominator, bitLenInt qubitIndex);

    /// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
    /// Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis.
    void RX(double radians, bitLenInt qubitIndex);
    void RXDyad(int numerator, int denominator, bitLenInt qubitIndex);
    void CRX(double radians, bitLenInt control, bitLenInt target);
    void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
    /// Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis.
    void RY(double radians, bitLenInt qubitIndex);
    void RYDyad(int numerator, int denominator, bitLenInt qubitIndex);
    void CRY(double radians, bitLenInt control, bitLenInt target);
    void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
    /// Dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis.
    void RZ(double radians, bitLenInt qubitIndex);
    void RZDyad(int numerator, int denominator, bitLenInt qubitIndex);
    void CRZ(double radians, bitLenInt control, bitLenInt target);
    void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /// Set individual bit to pure |0> (false) or |1> (true) state
    void SetBit(bitLenInt qubitIndex1, bool value);

    /// Swap values of two bits in register
    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    /// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
    void CRT(double radians, bitLenInt control, bitLenInt target);
    void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    // Register-spanning gates
    //
    // Convienence functions implementing gates are applied from the bit
    // 'start' for 'length' bits for the register.

    void H(bitLenInt start, bitLenInt length);
    void X(bitLenInt start, bitLenInt length);
    void Y(bitLenInt start, bitLenInt length);
    void Z(bitLenInt start, bitLenInt length);

    void CNOT(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt length);
    void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

    void R1(double radians, bitLenInt start, bitLenInt length);
    void R1Dyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    void RX(double radians, bitLenInt start, bitLenInt length);
    void RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    void CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    void RY(double radians, bitLenInt start, bitLenInt length);
    void RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    void CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    void RZ(double radians, bitLenInt start, bitLenInt length);
    void RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    void CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    void CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    void CY(bitLenInt control, bitLenInt target, bitLenInt length);
    void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /// Arithmetic shift left, with last 2 bits as sign and carry
    void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Arithmetic shift right, with last 2 bits as sign and carry
    void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Logical shift left, filling the extra bits with |0>
    void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Logical shift right, filling the extra bits with |0>
    void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// "Circular shift left" - shift bits left, and carry last bits.
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// "Circular shift right" - shift bits right, and carry first bits.
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    /// Add integer (without sign)
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Add integer (without sign, with carry)
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Add an integer to the register, with sign and without carry. Because the register length is an arbitrary number
     * of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as cast
     * to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.s
     */
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /**
     * Add an integer to the register, with sign and with carry. If oveflow is set, flip phase on overflow. Because the
     * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence,
     * the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
     * appropriate position before the cast.
     */
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    /**
     * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is
     * an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add
     * is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position
     * before the cast.
     */
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Add BCD integer (without sign)
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Add BCD integer (without sign, with carry)
    void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract integer (without sign)
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);

    /// Subtract integer (without sign, with carry)
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
     * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified
     * as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
     */
    void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /**
     * Subtract an integer from the register, with sign and with carry. If oveflow is set, flip phase on overflow.
     * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is
     * variable. Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be
     * set at the appropriate position before the cast.
     */
    void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    /**
     * Subtract an integer from the register, with sign and with carry. Flip phase on overflow. Because the register
     * length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
     * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
     * position before the cast.
     */
    void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract BCD integer (without sign)
    void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Subtract BCD integer (without sign, with carry)
    void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // virtual void ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    // void ADDC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt
    // carryIndex);

    /**
     * Add signed integer of "length" bits in "inStart" to signed integer of "length" bits in "inOutStart," and store
     * result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    // void ADDS(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    // void ADDSC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length,
    //    const bitLenInt overflowIndex, const bitLenInt carryIndex);

    /**
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // void ADDBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart," with carry in/out.
     */
    // void ADDBCDC(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    // virtual void SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length);

    /**
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart."
     */
    // void SUBBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart," with carry in/out.
     */
    // void SUBBCDC(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    // void SUBC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract signed integer of "length" bits in "inStart" from signed integer of "length" bits in "inOutStart," and
     * store result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    // void SUBS(
    //    const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /**
     *Subtract integer of "length" bits in "inStart" from integer of "length" bits in "inOutStart," and store result
     * in "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    // void SUBSC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt
    // overflowIndex,
    //    const bitLenInt carryIndex);

    /// Quantum Fourier Transform - Apply the quantum Fourier transform to the register.
    void QFT(bitLenInt start, bitLenInt length);

    /// "Entangled Hadamard" - perform an operation on two entangled registers like a bitwise Hadamard on a single
    /// unentangled register.
    void EntangledH(bitLenInt targetStart, bitLenInt entangledStart, bitLenInt length);

    /// For chips with a zero flag, apply a Z to the zero flag, entangled with the state where the register equals zero.
    void SetZeroFlag(bitLenInt start, bitLenInt length, bitLenInt zeroFlag);

    /// For chips with a zero flag, flip the phase of the state where the register equals zero.
    void SetZeroFlag(bitLenInt start, bitLenInt length);

    /// For chips with a sign flag, apply a Z to the sign flag, entangled with the states where the register is
    /// negative.
    void SetSignFlag(bitLenInt toTest, bitLenInt toSet);

    /// For chips with a sign flag, flip the phase of states where the register is negative.
    void SetSignFlag(bitLenInt toTest);

    /// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
    void SetLessThanFlag(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);

    /// Phase flip always - equivalent to Z X Z X on any bit in the CoherentUnit
    void PhaseFlip();

    /// Set register bits to given permutation
    void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /// Measure permutation state of a register
    bitCapInt MReg(bitLenInt start, bitLenInt length);

    /// Measure permutation state of an 8 bit register
    unsigned char MReg8(bitLenInt start);

    /// Set 8 bit register bits based on read from classical memory
    unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);

    /// Add based on an indexed load from classical memory
    unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

    /// Subtract based on an indexed load from classical memory
    unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

protected:
    double runningNorm;
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    std::unique_ptr<Complex16[]> stateVec;

    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double> rand_distribution;

    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);
    void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm);
    void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    void Carry(bitLenInt integerStart, bitLenInt integerLength, bitLenInt carryBit);
    void NormalizeState();
    void Reverse(bitLenInt first, bitLenInt last);
    void UpdateRunningNorm();
};

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

} // namespace Qrack
