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

#pragma once

#include "qengine.hpp"
#include "statevector.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif

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
class QEngineCPU : virtual public QEngine {
protected:
    StateVectorPtr stateVec;
    bool isSparse;
    std::mutex asyncSharedMutex;
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
    DispatchQueue dispatchQueue;
    bitLenInt dispatchThreshold;
#endif

    StateVectorSparsePtr CastStateVecSparse() { return std::dynamic_pointer_cast<StateVectorSparse>(stateVec); }

public:
    QEngineCPU(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true, bool ignored = false,
        int ignored2 = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> ignored3 = {}, bitLenInt ignored4 = 0,
        real1_f ignored5 = FP_NORM_EPSILON_F);

    virtual ~QEngineCPU() { Dump(); }

    virtual void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.finish();
#endif
    };

    virtual bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        return dispatchQueue.isFinished();
#else
        return true;
#endif
    }

    virtual void Dump()
    {
        // Make sure that async copy is finished, before we free the state vector.
        std::lock_guard<std::mutex> lock(asyncSharedMutex);
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dump();
#endif
    }

    virtual real1_f FirstNonzeroPhase()
    {
        if (!stateVec) {
            return ZERO_R1_F;
        }

        return QInterface::FirstNonzeroPhase();
    }

    virtual void ZeroAmplitudes()
    {
        Dump();
        FreeStateVec();
        runningNorm = ZERO_R1;
    }

    virtual void FreeStateVec(complex* sv = NULL) { stateVec = NULL; }

    virtual bool IsZeroAmplitude() { return !stateVec; }
    virtual void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    virtual void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length);
    virtual void ShuffleBuffers(QEnginePtr engine);
    virtual void CopyStateVec(QEnginePtr src);

    virtual QEnginePtr CloneEmpty();

    virtual void QueueSetDoNormalize(bool doNorm)
    {
        Dispatch(1U, [this, doNorm] { doNormalize = doNorm; });
    }
    virtual void QueueSetRunningNorm(real1_f runningNrm)
    {
        Dispatch(1U, [this, runningNrm] { runningNorm = runningNrm; });
    }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);

    using QEngine::Compose;
    virtual bitLenInt Compose(QEngineCPUPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy)); }
    virtual std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy);
    virtual bitLenInt Compose(QEngineCPUPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineCPU>(toCopy), start);
    }

    using QEngine::Decompose;
    virtual void Decompose(bitLenInt start, QInterfacePtr dest);

    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    /** @} */

    virtual void XMask(bitCapInt mask);
    virtual void PhaseParity(real1_f radians, bitCapInt mask);

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
#if ENABLE_ALU
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
#if ENABLE_BCD
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
#endif
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, const unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);

    /** @} */
#endif

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
        const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask);
    virtual void UniformParityRZ(bitCapInt mask, real1_f angle);
    virtual void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual real1_f Prob(bitLenInt qubitIndex);
    virtual real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation);
    virtual real1_f ProbMask(bitCapInt mask, bitCapInt permutation);
    virtual real1_f ProbParity(bitCapInt mask);
    virtual bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    virtual void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);
    virtual real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QEngineCPU>(toCompare));
    }
    virtual real1_f SumSqrDiff(QEngineCPUPtr toCompare);
    virtual QInterfacePtr Clone();

    /** @} */

protected:
    virtual real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength);

    virtual StateVectorPtr AllocStateVec(bitCapIntOcl elemCount);
    virtual void ResetStateVec(StateVectorPtr sv) { stateVec = sv; }

    typedef std::function<void(void)> DispatchFn;
    virtual void Dispatch(bitCapInt workItemCount, DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        if ((workItemCount >= (bitCapIntOcl)(ONE_BCI << dispatchThreshold)) && (workItemCount < GetStride())) {
            dispatchQueue.dispatch(fn);
        } else {
            Finish();
            fn();
        }
#else
        fn();
#endif
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr dest);
    virtual void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    virtual void ApplyM(bitCapInt mask, bitCapInt result, complex nrm);

#if ENABLE_ALU
    virtual void INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    virtual void INCDECSC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    virtual void INCDECSC(
        bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
#if ENABLE_BCD
    virtual void INCDECBCDC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif

    typedef std::function<bitCapIntOcl(const bitCapIntOcl&, const bitCapIntOcl&)> IOFn;
    void MULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length);
    void CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
        const bitLenInt& carryStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt controlLen);

    typedef std::function<bitCapIntOcl(const bitCapIntOcl&)> MFn;
    void ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const bool& inverse = false);
    void CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart, const bitLenInt& outStart,
        const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen, const bool& inverse = false);
#endif
};
} // namespace Qrack
