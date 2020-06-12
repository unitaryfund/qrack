//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qfactory.hpp"

namespace Qrack {

class QEngineHybrid;
typedef std::shared_ptr<QEngineHybrid> QEngineHybridPtr;

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);
template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

/**
 * General purpose QEngineHybrid implementation
 */
class QEngineHybrid : virtual public QEngine {
protected:
    const bitLenInt MIN_OCL_QUBIT_COUNT = 4U;
    QEnginePtr qEngine;
    QInterfaceEngine qEngineType;
    int deviceID;
    bool useRDRAND;
    bool isSparse;

    QEnginePtr ConvertEngineType(QInterfaceEngine oQEngineType, QInterfaceEngine nQEngineType, QEnginePtr oQEngine);

public:
    /**
     * \defgroup HybridInterface Special implementations for QEngineHybrid.
     *
     * @{
     */

    QEngineHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = true, int devID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored = {});

    virtual ~QEngineHybrid()
    {
        // Intentionally left blank
    }

    bitLenInt GetQubitCount() { return qEngine->GetQubitCount(); }

    bitCapInt GetMaxQPower() { return qEngine->GetMaxQPower(); }

    virtual bitLenInt Compose(QEngineHybridPtr toCopy)
    {
        QInterfaceEngine composeType = QINTERFACE_CPU;
        if (qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if (toCopy->qEngineType == QINTERFACE_OPENCL) {
            composeType = QINTERFACE_OPENCL;
        }
        if ((qEngine->GetQubitCount() + toCopy->GetQubitCount()) >= MIN_OCL_QUBIT_COUNT) {
            composeType = QINTERFACE_OPENCL;
        }

        qEngine = ConvertEngineType(qEngineType, composeType, qEngine);
        QEnginePtr nCopyQEngine = ConvertEngineType(toCopy->qEngineType, composeType, toCopy->qEngine);

        bitLenInt toRet = qEngine->Compose(nCopyQEngine);

        qEngineType = composeType;

        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy)
    {
        return Compose(std::dynamic_pointer_cast<QEngineHybrid>(toCopy));
    }
    virtual bitLenInt Compose(QEngineHybridPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineHybrid>(toCopy), start);
    }

    virtual void Decompose(bitLenInt start, bitLenInt length, QEngineHybridPtr dest)
    {
        QInterfaceEngine decomposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            decomposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            decomposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, dest->qEngineType, qEngine);
        qEngine->Decompose(start, length, dest->qEngine);

        if (decomposeType != dest->qEngineType) {
            qEngine = ConvertEngineType(qEngineType, decomposeType, qEngine);
        }

        qEngineType = decomposeType;
    }

    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decompose(start, length, std::dynamic_pointer_cast<QEngineHybrid>(dest));
    }

    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        QInterfaceEngine disposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            disposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            disposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, disposeType, qEngine);
        qEngine->Dispose(start, length);
    }
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        QInterfaceEngine disposeType = QINTERFACE_OPENCL;
        if (qEngineType == QINTERFACE_CPU) {
            disposeType = QINTERFACE_CPU;
        }
        if ((qEngine->GetQubitCount() - length) < MIN_OCL_QUBIT_COUNT) {
            disposeType = QINTERFACE_CPU;
        }

        qEngine = ConvertEngineType(qEngineType, disposeType, qEngine);
        qEngine->Dispose(start, length, disposedPerm);
    }

    virtual void FreeStateVec() { qEngine = NULL; }

    virtual void SetQuantumState(const complex* inputState) { qEngine->SetQuantumState(inputState); }

    virtual void GetQuantumState(complex* outputState) { qEngine->GetQuantumState(outputState); }
    virtual void GetProbs(real1* outputProbs) { qEngine->GetProbs(outputProbs); }
    virtual complex GetAmplitude(bitCapInt perm) { return qEngine->GetAmplitude(perm); }
    virtual void SetAmplitude(bitCapInt perm, complex amp) { qEngine->SetAmplitude(perm, amp); }

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QEngineHybrid>(toCompare));
    }
    virtual bool ApproxCompare(QEngineHybridPtr toCompare) { return qEngine->ApproxCompare(toCompare->qEngine); }
    virtual QInterfacePtr Clone() { return qEngine->Clone(); }

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { qEngine->ROL(shift, start, length); }
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { qEngine->INC(toAdd, start, length); }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        qEngine->INCS(toAdd, start, length, overflowIndex);
    }
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { qEngine->INCBCD(toAdd, start, length); }
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qEngine->MUL(toMul, inOutStart, carryStart, length);
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        qEngine->DIV(toDiv, inOutStart, carryStart, length);
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        qEngine->POWModNOut(base, modN, inStart, outStart, length);
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        qEngine->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
    }
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        qEngine->FullAdd(inputBit1, inputBit2, carryInSumOut, carryOut);
    }
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        qEngine->IFullAdd(inputBit1, inputBit2, carryInSumOut, carryOut);
    }

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) { qEngine->ZeroPhaseFlip(start, length); }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        qEngine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        qEngine->PhaseFlipIfLess(greaterPerm, start, length);
    }
    virtual void PhaseFlip() { qEngine->PhaseFlip(); }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        qEngine->SetPermutation(perm, phaseFac);
    }
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true)
    {
        return qEngine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return qEngine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        return qEngine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values)
    {
        qEngine->Hash(start, length, values);
    }
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        qEngine->UniformlyControlledSingleBit(
            controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask);
    }

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1 Prob(bitLenInt qubitIndex) { return qEngine->Prob(qubitIndex); }
    virtual real1 ProbAll(bitCapInt fullRegister) { return qEngine->ProbAll(fullRegister); }
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
    {
        return qEngine->ProbReg(start, length, permutation);
    }
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        return qEngine->ProbMask(mask, permutation);
    }
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        qEngine->NormalizeState(nrm, norm_thresh);
    }

    /** @} */

protected:
    /**
     * \defgroup HybridInterfaceProtected QEngineHybrid implementation of pure virtual QEngine methods.
     *
     * @{
     */

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        qEngine->Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh);
    }

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        qEngine->INCDECC(toMod, inOutStart, length, carryIndex);
    }
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        qEngine->INCDECSC(toMod, inOutStart, length, carryIndex);
    }
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
    {
        qEngine->INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex);
    }
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        qEngine->INCDECBCDC(toMod, inOutStart, length, carryIndex);
    }

    /** @} */
};
} // namespace Qrack
