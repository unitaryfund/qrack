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

#include <memory>

#include "common/parallel_for.hpp"
#include "qengine.hpp"
#include "statevector.hpp"

namespace Qrack {

class QEngineCPU;
typedef std::shared_ptr<QEngineCPU> QEngineCPUPtr;

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride);
template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride);

/**
 * General purpose QEngineCPU implementation
 */
class QEngineCPU : virtual public QEngine, public ParallelFor {
protected:
    StateVectorPtr stateVec;
    bool isSparse;

    StateVectorSparsePtr CastStateVecSparse() { return std::dynamic_pointer_cast<StateVectorSparse>(stateVec); }

public:
    QEngineCPU(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true, bool ignored = false,
        int ignored2 = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored3 = {});

    virtual ~QEngineCPU()
    {
        // Intentionally left blank
    }

    virtual void FreeStateVec(complex* sv = NULL) { stateVec = NULL; }

    virtual void GetAmplitudePage(complex* pagePtr, const bitCapInt offset, const bitCapInt length)
    {
        if (stateVec) {
            stateVec->copy_out(pagePtr, offset, length);
        } else {
            std::fill(pagePtr, pagePtr + length, ZERO_CMPLX);
        }
    }
    virtual void SetAmplitudePage(const complex* pagePtr, const bitCapInt offset, const bitCapInt length)
    {
        if (!stateVec) {
            ResetStateVec(AllocStateVec(maxQPower));
            stateVec->clear();
        }

        stateVec->copy_in(pagePtr, offset, length);
    }
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, const bitCapInt srcOffset, const bitCapInt dstOffset, const bitCapInt length)
    {
        if (!stateVec) {
            ResetStateVec(AllocStateVec(maxQPower));
            stateVec->clear();
        }

        QEngineCPUPtr pageEngineCpuPtr = std::dynamic_pointer_cast<QEngineCPU>(pageEnginePtr);
        stateVec->copy_in(pageEngineCpuPtr->stateVec, srcOffset, dstOffset, length);
    }
    virtual void ShuffleBuffers(QEnginePtr engine)
    {
        QEngineCPUPtr engineCpu = std::dynamic_pointer_cast<QEngineCPU>(engine);
        stateVec->shuffle(engineCpu->stateVec);
    }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);

    virtual bitLenInt Compose(QEngineCPUPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy)); }
    virtual std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy);
    virtual bitLenInt Compose(QEngineCPUPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy), start);
    }

    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest);

    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values);
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QEngineCPU>(toCompare));
    }
    virtual bool ApproxCompare(QEngineCPUPtr toCompare);
    virtual QInterfacePtr Clone();

    /** @} */

protected:
    virtual StateVectorPtr AllocStateVec(bitCapInt elemCount);
    virtual void ResetStateVec(StateVectorPtr sv);

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr dest);
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void ApplyM(bitCapInt mask, bitCapInt result, complex nrm);

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex);
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);

    typedef std::function<bitCapInt(const bitCapInt&, const bitCapInt&)> IOFn;
    void MULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length);
    void CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt controlLen);

    typedef std::function<bitCapInt(const bitCapInt&)> MFn;
    void ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const bool& inverse = false);
    void CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen, const bool& inverse = false);
};
} // namespace Qrack
