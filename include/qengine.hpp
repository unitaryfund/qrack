//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"
#include "qparity.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

#include <algorithm>

namespace Qrack {

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

/**
 * Abstract QEngine implementation, for all "Schroedinger method" engines
 */
#if ENABLE_ALU
class QEngine : public QAlu, public QParity, public QInterface {
#else
class QEngine : public QParity, public QInterface {
#endif
protected:
    bool useHostRam;
    /// The value stored in runningNorm should always be the total probability implied by the norm of all amplitudes,
    /// summed, at each update. To normalize, we should always multiply by 1/sqrt(runningNorm).
    bitCapIntOcl maxQPowerOcl;
    real1 runningNorm;

    bool IsPhase(const complex* mtrx) { return IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2]); }

    bool IsInvert(const complex* mtrx) { return IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3]); }

    bool IsIdentity(const complex* mtrx, bool isControlled);

public:
    QEngine(bitLenInt qBitCount, qrack_rand_gen_ptr rgp = nullptr, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, bool useHardwareRNG = true, real1_f norm_thresh = REAL1_EPSILON)
        : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
        , useHostRam(useHostMem)
        , maxQPowerOcl(pow2Ocl(qBitCount))
        , runningNorm(ONE_R1)
    {
        if (qBitCount > (sizeof(bitCapIntOcl) * bitsInByte)) {
            throw std::invalid_argument(
                "Cannot instantiate a register with greater capacity than native types on emulating system.");
        }
    };

    /** Default constructor, primarily for protected internal use */
    QEngine()
        : useHostRam(false)
        , maxQPowerOcl(0)
        , runningNorm(ONE_R1)
    {
        // Intentionally left blank
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        maxQPowerOcl = (bitCapIntOcl)maxQPower;
    }

    /** Get in-flight renormalization factor */
    virtual real1_f GetRunningNorm()
    {
        Finish();
        return runningNorm;
    }

    /** Set all amplitudes to 0, and optionally temporarily deallocate state vector RAM */
    virtual void ZeroAmplitudes() = 0;
    /** Exactly copy the state vector of a different QEngine instance */
    virtual void CopyStateVec(QEnginePtr src) = 0;
    /** Returns "true" only if amplitudes are all totally 0 */
    virtual bool IsZeroAmplitude() = 0;
    /** Copy a "page" of amplitudes from this QEngine's internal state, into `pagePtr`. */
    virtual void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length) = 0;
    /** Copy a "page" of amplitudes from `pagePtr` into this QEngine's internal state. */
    virtual void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length) = 0;
    /** Copy a "page" of amplitudes from another QEngine, pointed to by `pageEnginePtr`, into this QEngine's internal
     * state. */
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length) = 0;
    /** Swap the high half of this engine with the low half of another. This is necessary for gates which cross
     * sub-engine  boundaries. */
    virtual void ShuffleBuffers(QEnginePtr engine) = 0;
    /** Clone this QEngine's settings, with a zeroed state vector */
    virtual QEnginePtr CloneEmpty() = 0;

    /** Add an operation to the (OpenCL) queue, to set the value of `doNormalize`, which controls whether to
     * automatically normalize the state. */
    virtual void QueueSetDoNormalize(bool doNorm) = 0;
    /** Add an operation to the (OpenCL) queue, to set the value of `runningNorm`, which is the normalization constant
     * for the next normalization operation. */
    virtual void QueueSetRunningNorm(real1_f runningNrm) = 0;

    virtual void ZMask(bitCapInt mask) { PhaseParity(PI_R1, mask); }

    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, bool doApply = true);
    virtual bitCapInt ForceM(const bitLenInt* bits, bitLenInt length, const bool* values, bool doApply = true);
    virtual bitCapInt ForceMReg(
        bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true, bool doApply = true);

    virtual void ApplyM(bitCapInt qPower, bool result, complex nrm)
    {
        bitCapInt powerTest = result ? qPower : 0;
        ApplyM(qPower, powerTest, nrm);
    }
    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) = 0;

    virtual void Mtrx(const complex* mtrx, bitLenInt qubit);
    virtual void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    virtual void CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);
    virtual void AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit1, bitLenInt qubit2);

#if ENABLE_ALU
    virtual bool M(bitLenInt q) { return QInterface::M(q); }
    virtual void X(bitLenInt q) { QInterface::X(q); }
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { QInterface::INC(toAdd, start, length); }
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCC(toAdd, start, length, carryIndex);
    }
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::DECC(toSub, start, length, carryIndex);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::INCS(toAdd, start, length, overflowIndex);
    }
    virtual void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls, controlLen);
    }
    virtual void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
#endif

    using QInterface::Swap;
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISwap;
    virtual void ISwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::SqrtSwap;
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2);
    using QInterface::FSim;
    virtual void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual real1_f ProbAll(bitCapInt fullRegister);
    virtual real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation) = 0;
    virtual void ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray);
    virtual real1_f ProbMask(bitCapInt mask, bitCapInt permutation) = 0;

    virtual real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength) = 0;

    virtual void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG) = 0;
    virtual void ApplyControlled2x2(
        const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx);
    virtual void ApplyAntiControlled2x2(
        const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, const complex* mtrx);

    using QInterface::Decompose;
    virtual QInterfacePtr Decompose(bitLenInt start, bitLenInt length);

    virtual void FreeStateVec(complex* sv = NULL) = 0;
};
} // namespace Qrack
