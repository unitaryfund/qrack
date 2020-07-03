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

#include "qfactory.hpp"

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
    bitLenInt qPageQubitCount;
    bitCapIntOcl qPageMaxQPower;
    std::vector<QEnginePtr> qPages;

    // TODO: Make this a constructor argument:
    const bitCapInt qubitsPerPage = 4U;
    bitLenInt qPagePow;
    bitCapInt qPageCount;

    QEnginePtr MakeEngine(bitLenInt length, bitCapInt perm)
    {
        return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engine, length, perm, rand_generator,
            phaseFactor, false, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse));
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);

        qPagePow = qubitCount - qubitsPerPage;
        qPageCount = pow2(qPagePow);
    }

    void CombineEngines();
    void SeparateEngines();

    template <typename F, typename... Args>
    void MetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt target, F fn, Args... gfnArgs);
    template <typename F, typename... Args>
    void SemiMetaControlled(bool anti, std::vector<bitLenInt> controls, bitLenInt targetBit, F fn, Args... gfnArgs);

public:
    QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool ignored = false, bool ignored2 = false, bool useHostMem = false,
        int deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> devList = {});

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm)
    {
        return qPages[perm / qPageMaxQPower]->GetAmplitude(perm & (qPageMaxQPower - ONE_BCI));
    }
    virtual void SetAmplitude(bitCapInt perm, complex amp)
    {
        return qPages[perm / qPageMaxQPower]->SetAmplitude(perm & (qPageMaxQPower - ONE_BCI), amp);
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
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decompose(start, length, std::dynamic_pointer_cast<QPager>(dest));
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QPagerPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void ApplySingleBit(const complex* mtrx, bitLenInt qubitIndex);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx) = 0;
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx) = 0;
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex);
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex);
    virtual void ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);
    virtual void ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topLeft, const complex bottomRight);
    virtual void ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
        const bitLenInt& target, const complex topRight, const complex bottomLeft);

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2) = 0;

    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true) = 0;

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex) = 0;
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length) = 0;
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;

    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void DECSC(
        bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex) = 0;
    virtual void DECSC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex) = 0;
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length) = 0;
    virtual void MULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    virtual void IMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    virtual void POWModNOut(
        bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length) = 0;
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen) = 0;

    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void CFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void CIFullAdd(bitLenInt* controls, bitLenInt controlLen, bitLenInt inputBit1, bitLenInt inputBit2,
        bitLenInt carryInSumOut, bitLenInt carryOut);

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) = 0;
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex) = 0;
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length) = 0;
    virtual void PhaseFlip() = 0;

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true) = 0;
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values) = 0;
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values) = 0;

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void ISwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;
    virtual void FSim(real1 theta, real1 phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2) = 0;

    virtual real1 Prob(bitLenInt qubitIndex) = 0;
    virtual real1 ProbAll(bitCapInt fullRegister) = 0;

    virtual bool ApproxCompare(QInterfacePtr toCompare) = 0;
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG) = 0;
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG) = 0;

    virtual void Finish()
    {
        for (bitLenInt i = 0; i < qPageCount; i++) {
            qPages[i]->Finish();
        }
    };

    virtual bool isFinished()
    {
        for (bitLenInt i = 0; i < qPageCount; i++) {
            if (!qPages[i]->isFinished()) {
                return false;
            }
        }

        return true;
    };

    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1) { return false; }

    virtual QInterfacePtr Clone() = 0;
};
} // namespace Qrack
