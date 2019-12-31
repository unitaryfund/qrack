//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>

#include "qinterface.hpp"

namespace Qrack {

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

/**
 * Abstract QEngine implementation, for all "Schroedinger method" engines
 */
class QEngine : public QInterface {
protected:
    bool useHostRam;
    /// The value stored in runningNorm should always be the total probability implied by the norm of all amplitudes,
    /// summed, at each update. To normalize, we should always multiply by 1/sqrt(runningNorm).
    real1 runningNorm;

    complex GetNonunitaryPhase()
    {
        if (randGlobalPhase) {
            real1 angle = Rand() * 2 * M_PI;
            return complex(cos(angle), sin(angle));
        } else {
            return ONE_CMPLX;
        }
    }

public:
    QEngine(bitLenInt qBitCount, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, bool useHardwareRNG = true, real1 norm_thresh = -999.0)
        : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
        , useHostRam(useHostMem)
        , runningNorm(ONE_R1)
    {
        if (qBitCount > (sizeof(bitCapInt) * bitsInByte)) {
            throw std::invalid_argument(
                "Cannot instantiate a register with greater capacity than native types on emulating system.");
        }
    };

    QEngine()
    {
        // Intentionally left blank
    }

    virtual ~QEngine() { Finish(); }

    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true);
    virtual bitCapInt ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values);
    virtual bitCapInt ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true);

    virtual void ApplyM(bitCapInt qPower, bool result, complex nrm)
    {
        bitCapInt powerTest = result ? qPower : 0;
        ApplyM(qPower, powerTest, nrm);
    }
    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) = 0;

    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);

    using QInterface::Swap;
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISwap;
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::SqrtSwap;
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);

    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation) = 0;
    virtual void ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation) = 0;

    virtual void INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    virtual void NormalizeState(real1 nrm = -999.0, real1 norm_thresh = -999.0) = 0;

protected:
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1 norm_thresh = -999.0) = 0;
    virtual void ApplyControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
        const complex* mtrx, bool doCalcNorm);
    virtual void ApplyAntiControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
        const complex* mtrx, bool doCalcNorm);

    /**
     * Common driver method behind INCC and DECC
     */
    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex) = 0;
    /**
     * Common driver method behind INCSC and DECSC (without overflow flag)
     */
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex) = 0;
    /**
     * Common driver method behind INCSC and DECSC (with overflow flag)
     */
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex) = 0;
    /**
     * Common driver method behind INCSC and DECSC (without overflow flag)
     */
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex) = 0;
};
} // namespace Qrack
