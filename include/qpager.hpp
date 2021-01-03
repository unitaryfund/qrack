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

#include "qinterface.hpp"

namespace Qrack {

class QPager;
typedef std::shared_ptr<QPager> QPagerPtr;

/**
 * A "Qrack::QPager" splits a "Qrack::QEngine" implementation into equal-length "pages." This helps both optimization
 * and distribution of a single coherent quantum register across multiple devices.
 */
class QPager : public QInterface {
protected:
    QInterfaceEngine engine;
    int devID;
    complex phaseFactor;
    bool useHostRam;
    bool useRDRAND;
    bool isSparse;
    real1 runningNorm;
    std::vector<QEnginePtr> qPages;
    std::vector<int> deviceIDs;

    bitLenInt thresholdQubitsPerPage;
    bitLenInt baseQubitsPerPage;
    bitCapInt basePageCount;
    bitCapIntOcl basePageMaxQPower;

    QEnginePtr MakeEngine(bitLenInt length, bitCapInt perm, int deviceId);

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);

        baseQubitsPerPage = (qubitCount < thresholdQubitsPerPage) ? qubitCount : thresholdQubitsPerPage;
        basePageCount = pow2Ocl(qubitCount - baseQubitsPerPage);
        basePageMaxQPower = pow2Ocl(baseQubitsPerPage);
    }

    bitCapInt pageMaxQPower() { return maxQPower / qPages.size(); }
    bitLenInt pagedQubitCount() { return log2(qPages.size()); }
    bitLenInt qubitsPerPage() { return log2(pageMaxQPower()); }

    void CombineEngines(bitLenInt thresholdBits);
    void CombineEngines() { CombineEngines(qubitCount); }
    void SeparateEngines(bitLenInt thresholdBits);
    void SeparateEngines() { SeparateEngines(baseQubitsPerPage); }

    template <typename Qubit1Fn> void SingleBitGate(bitLenInt target, Qubit1Fn fn);
    template <typename Qubit1Fn>
    void MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn, const complex* mtrx);
    template <typename Qubit1Fn>
    void SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, Qubit1Fn fn);
    void MetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac);
    void SemiMetaSwap(bitLenInt qubit1, bitLenInt qubit2, bool isIPhaseFac);

    template <typename F> void CombineAndOp(F fn, std::vector<bitLenInt> bits);
    template <typename F>
    void CombineAndOpControlled(
        F fn, std::vector<bitLenInt> bits, const bitLenInt* controls, const bitLenInt controlLen);

    void ApplySingleEither(const bool& isInvert, complex top, complex bottom, bitLenInt target);
    void ApplyEitherControlledSingleBit(const bool& anti, const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex* mtrx);

public:
    QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool ignored = false, bool useHostMem = false,
        int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0);

    QPager(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool ignored = false, bool useHostMem = false,
        int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0)
        : QPager(QINTERFACE_HYBRID, qBitCount, initState, rgp, phaseFac, doNorm, ignored, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold)
    {
    }

    virtual void SetConcurrency(uint32_t threadsPerEngine)
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->SetConcurrency(threadsPerEngine);
        }
    }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm)
    {
        bitCapIntOcl subIndex = (bitCapIntOcl)(perm / pageMaxQPower());
        return qPages[subIndex]->GetAmplitude(perm - (subIndex * pageMaxQPower()));
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        bitCapIntOcl subIndex = (bitCapIntOcl)(perm / pageMaxQPower());
        return qPages[subIndex]->SetAmplitude(perm - (subIndex * pageMaxQPower()), amp);
    }

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    using QInterface::Compose;
    virtual bitLenInt Compose(QPagerPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QPager>(toCopy)); }
    virtual bitLenInt Compose(QPagerPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QPager>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QPager>(dest));
    }
    virtual void Decompose(bitLenInt start, QPagerPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex);
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex)
    {
        ApplySingleEither(false, topLeft, bottomRight, qubitIndex);
    }
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex)
    {
        ApplySingleEither(true, topRight, bottomLeft, qubitIndex);
    }
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        ApplyEitherControlledSingleBit(false, controls, controlLen, target, mtrx);
    }
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
    {
        ApplyEitherControlledSingleBit(true, controls, controlLen, target, mtrx);
    }
    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);
    virtual void UniformParityRZ(const bitCapInt& mask, const real1& angle);
    virtual void CUniformParityRZ(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitCapInt& mask, const real1& angle);

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

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
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

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values);

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    // TODO: QPager not yet used in Q#, but this would need a real implementation:
    virtual real1 ProbParity(const bitCapInt& mask)
    {
        if (!mask) {
            return ZERO_R1;
        }

        CombineEngines();
        return qPages[0]->ProbParity(mask);
    }
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true)
    {
        if (!mask) {
            return ZERO_R1;
        }

        CombineEngines();
        return qPages[0]->ForceMParity(mask, result, doForce);
    }

    virtual bool ApproxCompare(QInterfacePtr toCompare);
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG) {
    } // TODO: skip implementation for now

    virtual void Finish()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->Finish();
        }
    };

    virtual bool isFinished()
    {
        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            if (!qPages[i]->isFinished()) {
                return false;
            }
        }

        return true;
    };

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1) { return false; }

    virtual QInterfacePtr Clone();

    virtual void SetDevice(const int& dID, const bool& forceReInit = false)
    {
        deviceIDs.clear();
        deviceIDs.push_back(dID);

        for (bitCapIntOcl i = 0; i < qPages.size(); i++) {
            qPages[i]->SetDevice(dID, forceReInit);
        }
    }

    virtual int GetDeviceID() { return qPages[0]->GetDeviceID(); }

    bitCapIntOcl GetMaxSize() { return qPages[0]->GetMaxSize(); };

    virtual real1 SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QPager>(toCompare));
    }

    virtual real1 SumSqrDiff(QPagerPtr toCompare)
    {
        CombineEngines();
        toCompare->CombineEngines();
        return qPages[0]->SumSqrDiff(toCompare->qPages[0]);
    }
};
} // namespace Qrack
