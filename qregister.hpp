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

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

/**
 * The "Qrack::CoherentUnit" class represents one or more coherent quantum
 * processor registers, including primitive bit logic gates and (abstract)
 * opcodes-like methods.
 */
class CoherentUnit {
public:
    /// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
    CoherentUnit(bitLenInt qBitCount);

    /// Initialize a coherent unit with qBitCount number pf bits, to initState unsigned integer permutation state
    CoherentUnit(bitLenInt qBitCount, bitCapInt initState);

    /// PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
    CoherentUnit(const CoherentUnit& pqs);

    /// Set the random seed (primarily used for testing)
    void SetRandomSeed(uint32_t seed);

    /// Get the count of bits in this register
    int GetQubitCount();

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
    void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Add an integer to the register, with sign and without carry. Because the register length is an arbitrary number
     * of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as cast
     * to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.s
     */
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /**
     * Add an integer to the register, with sign and with carry. Because the register length is an arbitrary number of
     * bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as cast to
     * an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast. */
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);

    /// Add BCD integer (without sign)
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Add BCD integer (without sign, with carry)
    void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /// Subtract integer (without sign)
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);

    /// Subtract integer (without sign, with carry)
    void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
     * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified
     * as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
     */
    void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);

    /**
     * Subtract an integer from the register, with sign and with carry. Because the register length is an arbitrary
     * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified
     * as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
     */
    void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);

    /// Subtract BCD integer (without sign)
    void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);

    /// Subtract BCD integer (without sign, with carry)
    void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    virtual void ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    void ADDC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Add signed integer of "length" bits in "inStart" to signed integer of "length" bits in "inOutStart," and store
     * result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    void ADDS(
        const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /**
     * Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    void ADDSC(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length,
        const bitLenInt overflowIndex, const bitLenInt carryIndex);

    /**
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    void ADDBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Add BCD number of "length" bits in "inStart" to BCD number of "length" bits in "inOutStart," and store result in
     * "inOutStart," with carry in/out.
     */
    void ADDBCDC(
        const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart."
     */
    virtual void SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length);

    /**
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart."
     */
    void SUBBCD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);

    /**
     * Subtract BCD number of "length" bits in "inStart" from BCD number of "length" bits in "inOutStart," and store
     * result in "inOutStart," with carry in/out.
     */
    void SUBBCDC(
        const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
     * "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit.
     */
    void SUBC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt carryIndex);

    /**
     * Subtract signed integer of "length" bits in "inStart" from signed integer of "length" bits in "inOutStart," and
     * store result in "inOutStart." Set overflow bit when input to output wraps past minimum or maximum integer.
     */
    void SUBS(
        const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length, const bitLenInt overflowIndex);

    /**
     *Subtract integer of "length" bits in "inStart" from integer of "length" bits in "inOutStart," and store result
     * in "inOutStart." Get carry value from bit at "carryIndex" and place end result into this bit. Set overflow for
     * signed addition if result wraps past the minimum or maximum signed integer.
     */
    void SUBSC(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length, const bitLenInt overflowIndex,
        const bitLenInt carryIndex);

    /// Quantum Fourier Transform - Apply the quantum Fourier transform to the register.
    void QFT(bitLenInt start, bitLenInt length);

    /// For chips with a zero flag, set the zero flag after a register operation.
    void SetZeroFlag(bitLenInt start, bitLenInt length, bitLenInt zeroFlag);

    /// For chips with a sign flag, set the sign flag after a register operation.
    void SetSignFlag(bitLenInt toTest, bitLenInt toSet);

    /// Set register bits to given permutation
    void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);

    /// Measure permutation state of a register
    bitCapInt MReg(bitLenInt start, bitLenInt length);

    /// Measure permutation state of an 8 bit register
    unsigned char MReg8(bitLenInt start);

    /// Set 8 bit register bits based on read from classical memory
    unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);

protected:
#if ENABLE_OPENCL
    virtual void InitOCL();
    virtual void ReInitOCL();
#endif

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
} // namespace Qrack
