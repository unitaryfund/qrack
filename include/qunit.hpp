//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

#include "common/parallel_for.hpp"

namespace Qrack {

/**
 * General purpose QUnit implementation
 */
class QUnit : public QInterface, public ParallelFor
{
protected:
    uint32_t randomSeed;
    double runningNorm;
    bitLenInt qubitCount;
    bitCapInt maxQPower;
    std::unique_ptr<Complex16[]> stateVec;

    std::shared_ptr<std::default_random_engine> rand_generator_ptr;
    std::uniform_real_distribution<double> rand_distribution;

public:
    QUnit(
        bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);

    virtual void SetQuantumState(Complex16* inputState);
    virtual void Cohere(QUnitPtr toCopy);
    virtual void Cohere(std::vector<QUnitPtr> toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QUnitPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void CNOT(bitLenInt control, bitLenInt target);
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);
    virtual void H(bitLenInt qubitIndex);
    virtual bool M(bitLenInt qubitIndex);
    virtual void X(bitLenInt qubitIndex);
    virtual void Y(bitLenInt qubitIndex);
    virtual void Z(bitLenInt qubitIndex);
    virtual void CY(bitLenInt control, bitLenInt target);
    virtual void CZ(bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup LogicGates Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     *
     * @{
     */

    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /** @} */

    /**
     * \defgroup RotGates Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     *
     * @{
     */

    virtual void RT(double radians, bitLenInt qubitIndex);
    virtual void RTDyad(int numerator, int denominator, bitLenInt qubitIndex);
    virtual void RX(double radians, bitLenInt qubitIndex);
    virtual void RXDyad(int numerator, int denominator, bitLenInt qubitIndex);
    virtual void CRX(double radians, bitLenInt control, bitLenInt target);
    virtual void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);
    virtual void RY(double radians, bitLenInt qubitIndex);
    virtual void RYDyad(int numerator, int denominator, bitLenInt qubitIndex);
    virtual void CRY(double radians, bitLenInt control, bitLenInt target);
    virtual void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);
    virtual void RZ(double radians, bitLenInt qubitIndex);
    virtual void RZDyad(int numerator, int denominator, bitLenInt qubitIndex);
    virtual void CRZ(double radians, bitLenInt control, bitLenInt target);
    virtual void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);
    virtual void CRT(double radians, bitLenInt control, bitLenInt target);
    virtual void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup RegGates Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     *
     * @{
     */

    virtual void H(bitLenInt start, bitLenInt length);
    virtual void X(bitLenInt start, bitLenInt length);
    virtual void Y(bitLenInt start, bitLenInt length);
    virtual void Z(bitLenInt start, bitLenInt length);
    virtual void CNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length);
    virtual void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    virtual void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    virtual void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length);
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    virtual void RT(double radians, bitLenInt start, bitLenInt length);
    virtual void RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    virtual void RX(double radians, bitLenInt start, bitLenInt length);
    virtual void RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    virtual void CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void RY(double radians, bitLenInt start, bitLenInt length);
    virtual void RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    virtual void CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void RZ(double radians, bitLenInt start, bitLenInt length);
    virtual void RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    virtual void CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length);
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ASL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ASR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void LSL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void LSR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void QFT(bitLenInt start, bitLenInt length);
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlip();
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);
    virtual unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);
    virtual unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);
    virtual unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);
    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual double Prob(bitLenInt qubitIndex);
    virtual double ProbAll(bitCapInt fullRegister);
    virtual void ProbArray(double* probArray);
    virtual void SetBit(bitLenInt qubitIndex1, bool value);

    /** @} */

protected:
    /** Generate a random double from 0 to 1 */
    double Rand();

    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);
    virtual void ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm);
    virtual void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    virtual void ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm);
    virtual void Carry(bitLenInt integerStart, bitLenInt integerLength, bitLenInt carryBit);
    virtual void NormalizeState();
    virtual void Reverse(bitLenInt first, bitLenInt last);
    virtual void UpdateRunningNorm();
};
} // namespace Qrack
