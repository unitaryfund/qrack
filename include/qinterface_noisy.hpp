//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "qparity.hpp"

#if ENABLE_ALU
#include "qalu.hpp"
#endif

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error OpenCL or CUDA has not been enabled
#endif

#if ENABLE_OPENCL
#define QRACK_GPU_ENGINE QINTERFACE_OPENCL
#else
#define QRACK_GPU_ENGINE QINTERFACE_CUDA
#endif

namespace Qrack {

class QInterfaceNoisy;
typedef std::shared_ptr<QInterfaceNoisy> QInterfaceNoisyPtr;

/**
 * A "Qrack::QInterfaceNoisy" that wraps any other QInterface with a simple noise model
 */
#if ENABLE_ALU
class QInterfaceNoisy : public QAlu, public QParity, public QInterface {
#else
class QInterfaceNoisy : public QParity, public QInterface {
#endif
protected:
    real1_f noiseParam;
    QInterfacePtr engine;
    std::vector<QInterfaceEngine> engines;

public:
    QInterfaceNoisy(bitLenInt qBitCount, bitCapInt initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QInterfaceNoisy({ QINTERFACE_TENSOR_NETWORK }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
        // Intentionally left blank;
    }

    QInterfaceNoisy(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QInterfaceNoisy(QInterfaceNoisy* o)
        : noiseParam(o->noiseParam)
        , engine(o->engine)
        , engines(o->engines)
    {
        // Intentionally left blank
    }

    void SetNoiseLevel(real1_f lambda) { noiseParam = lambda; }

    void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        engine->SetQubitCount(qb);
    }

    bool isOpenCL() { return engine->isOpenCL(); }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        engine->SetConcurrency(GetConcurrencyLevel());
    }

    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
    {
        return engine->ProbReg(start, length, permutation);
    }

    using QInterface::Compose;
    bitLenInt Compose(QInterfaceNoisyPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->Compose(toCopy->engine);
    }
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy)); }
    bitLenInt Compose(QInterfaceNoisyPtr toCopy, bitLenInt start)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->Compose(toCopy->engine, start);
    }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QInterfaceNoisyPtr toCopy)
    {
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return engine->ComposeNoClone(toCopy->engine);
    }
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QInterfaceNoisy>(toCopy));
    }
    using QInterface::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QInterfaceNoisy>(dest));
    }
    bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return TryDecompose(start, std::dynamic_pointer_cast<QInterfaceNoisy>(dest), error_tol);
    }
    void Decompose(bitLenInt start, QInterfaceNoisyPtr dest)
    {
        engine->Decompose(start, dest->engine);
        SetQubitCount(qubitCount - dest->GetQubitCount());
    }
    void Dispose(bitLenInt start, bitLenInt length)
    {
        engine->Dispose(start, length);
        SetQubitCount(qubitCount - length);
    }
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        engine->Dispose(start, length, disposedPerm);
        SetQubitCount(qubitCount - length);
    }

    bool TryDecompose(bitLenInt start, QInterfaceNoisyPtr dest, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        const bitLenInt nQubitCount = qubitCount - dest->GetQubitCount();
        const bool result = engine->TryDecompose(start, dest->engine, error_tol);
        if (result) {
            SetQubitCount(nQubitCount);
        }
        return result;
    }

    void SetQuantumState(const complex* inputState) { engine->SetQuantumState(inputState); }
    void GetQuantumState(complex* outputState) { engine->GetQuantumState(outputState); }
    void GetProbs(real1* outputProbs) { engine->GetProbs(outputProbs); }
    complex GetAmplitude(bitCapInt perm) { return engine->GetAmplitude(perm); }
    void SetAmplitude(bitCapInt perm, complex amp) { engine->SetAmplitude(perm, amp); }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        engine->SetPermutation(perm, phaseFac);
    }

    void Mtrx(const complex* mtrx, bitLenInt qubitIndex) { engine->Mtrx(mtrx, qubitIndex); }
    void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
    {
        engine->Phase(topLeft, bottomRight, qubitIndex);
    }
    void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
    {
        engine->Invert(topRight, bottomLeft, qubitIndex);
    }
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MCMtrx(controls, mtrx, target);
    }
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target)
    {
        engine->MACMtrx(controls, mtrx, target);
    }

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
        const complex* mtrxs, const std::vector<bitCapInt> mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
    {
        engine->UniformlyControlledSingleBit(controls, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipValueMask);
    }

    void XMask(bitCapInt mask) { engine->XMask(mask); }
    void PhaseParity(real1_f radians, bitCapInt mask) { engine->PhaseParity(radians, mask); }

    real1_f CProb(bitLenInt control, bitLenInt target) { return engine->CProb(control, target); }
    real1_f ACProb(bitLenInt control, bitLenInt target) { return engine->ACProb(control, target); }

    void CSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSwap(controls, qubit1, qubit2);
    }
    void AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSwap(controls, qubit1, qubit2);
    }
    void CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CSqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCSqrtSwap(controls, qubit1, qubit2);
    }
    void CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->CISqrtSwap(controls, qubit1, qubit2);
    }
    void AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt qubit1, bitLenInt qubit2)
    {
        engine->AntiCISqrtSwap(controls, qubit1, qubit2);
    }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        return engine->ForceM(qubit, result, doForce, doApply);
    }

#if ENABLE_ALU
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { engine->INC(toAdd, start, length); }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        engine->CINC(toAdd, inOutStart, length, controls);
    }
    void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCC(toAdd, start, length, carryIndex);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCSC(toAdd, start, length, carryIndex);
    }
    void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECC(toSub, start, length, carryIndex);
    }
    void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, overflowIndex, carryIndex);
    }
    void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECSC(toSub, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) { engine->INCBCD(toAdd, start, length); }
    void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->INCBCDC(toAdd, start, length, carryIndex);
    }
    void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        engine->DECBCDC(toSub, start, length, carryIndex);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        engine->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        engine->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        engine->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        return engine->IndexedLDA(indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return engine->IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return engine->IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values) { engine->Hash(start, length, values); }

    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        engine->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        engine->PhaseFlipIfLess(greaterPerm, start, length);
    }
#endif

    void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->Swap(qubitIndex1, qubitIndex2); }
    void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->ISwap(qubitIndex1, qubitIndex2); }
    void IISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->IISwap(qubitIndex1, qubitIndex2); }
    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->SqrtSwap(qubitIndex1, qubitIndex2); }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) { engine->ISqrtSwap(qubitIndex1, qubitIndex2); }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        engine->FSim(theta, phi, qubitIndex1, qubitIndex2);
    }

    real1_f Prob(bitLenInt qubitIndex) { return engine->Prob(qubitIndex); }
    real1_f ProbAll(bitCapInt fullRegister) { return engine->ProbAll(fullRegister); }
    real1_f ProbMask(bitCapInt mask, bitCapInt permutation) { return engine->ProbMask(mask, permutation); }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QInterfaceNoisy>(toCompare));
    }
    real1_f SumSqrDiff(QInterfaceNoisyPtr toCompare) { return engine->SumSqrDiff(toCompare->engine); }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG) { engine->UpdateRunningNorm(norm_thresh); }
    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        engine->NormalizeState(nrm, norm_thresh, phaseArg);
    }

    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, const bitCapInt& offset = ZERO_BCI)
    {
        return engine->ExpectationBitsAll(bits, offset);
    }

    void Finish() { engine->Finish(); }

    bool isFinished() { return engine->isFinished(); }

    void Dump() { engine->Dump(); }

    QInterfacePtr Clone()
    {
        QInterfaceNoisyPtr c = std::make_shared<QInterfaceNoisy>(this);
        c->engine = engine->Clone();
        return c;
    }

    void SetDevice(int64_t dID)
    {
        devID = dID;
        engine->SetDevice(dID);
    }

    int64_t GetDevice() { return engine->GetDevice(); }

    bitCapIntOcl GetMaxSize() { return engine->GetMaxSize(); };
};
} // namespace Qrack
