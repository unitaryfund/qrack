//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qbdt_qengine_node.hpp"
#include "qengine.hpp"

#define NODE_TO_QENGINE(leaf) (std::dynamic_pointer_cast<QBdtQEngineNode>(leaf)->qReg)
#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

class QBdt;
typedef std::shared_ptr<QBdt> QBdtPtr;

#if ENABLE_ALU
class QBdt : public QAlu, public QParity, public QInterface {
#else
class QBdt : public QParity, public QInterface {
#endif
protected:
    bitLenInt attachedQubitCount;
    bitLenInt bdtQubitCount;
    bitLenInt maxPageQubits;
    int64_t devID;
    QBdtNodeInterfacePtr root;
    bitCapInt bdtMaxQPower;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;

    void SetQubitCount(bitLenInt qb, bitLenInt aqb)
    {
        attachedQubitCount = aqb;
        SetQubitCount(qb);
    }

    void SetQubitCount(bitLenInt qb)
    {
        QInterface::SetQubitCount(qb);
        bdtQubitCount = qubitCount - attachedQubitCount;
        bdtMaxQPower = pow2(bdtQubitCount);
    }

    QBdtQEngineNodePtr MakeQEngineNode(complex scale, bitLenInt qbCount, bitCapInt perm = 0U);

    QInterfacePtr MakeTempStateVector()
    {
        QInterfacePtr copyPtr = NODE_TO_QENGINE(MakeQEngineNode(ONE_R1, qubitCount));
        Finish();
        GetQuantumState(copyPtr);

        // If the calling function fully deferences our return, it's automatically freed.
        return copyPtr;
    }
    void SetStateVector()
    {
        if (!bdtQubitCount) {
            return;
        }

        if (attachedQubitCount) {
            throw std::domain_error("QBdt::SetStateVector() not yet implemented, after Attach() call!");
        }

        QBdtQEngineNodePtr nRoot = MakeQEngineNode(ONE_R1, qubitCount);
        GetQuantumState(NODE_TO_QENGINE(nRoot));
        root = nRoot;
        SetQubitCount(qubitCount, qubitCount);
    }

    template <typename Fn> void GetTraversal(Fn getLambda);
    template <typename Fn> void SetTraversal(Fn setLambda);
    template <typename Fn> void ExecuteAsStateVector(Fn operation)
    {
        SetStateVector();
        operation(NODE_TO_QENGINE(root));
    }

    template <typename Fn> bitCapInt BitCapIntAsStateVector(Fn operation)
    {
        SetStateVector();
        return operation(NODE_TO_QENGINE(root));
    }

    void DecomposeDispose(bitLenInt start, bitLenInt length, QBdtPtr dest);

    void ApplyControlledSingle(
        const complex* mtrx, const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, bool isAnti);

    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }

    static bitCapInt RemovePower(bitCapInt perm, bitCapInt power)
    {
        bitCapInt mask = power - ONE_BCI;
        return (perm & mask) | ((perm >> ONE_BCI) & ~mask);
    }

    void ApplySingle(const complex* mtrx, bitLenInt target);

    void Init();

    void ResetStateVector(bitLenInt aqb = 0U)
    {
        if (attachedQubitCount <= aqb) {
            return;
        }

        if (!bdtQubitCount) {
            QBdtQEngineNodePtr oRoot = std::dynamic_pointer_cast<QBdtQEngineNode>(root);
            SetQubitCount(qubitCount, aqb);
            SetQuantumState(NODE_TO_QENGINE(oRoot));
        }

        const bitLenInt length = attachedQubitCount - aqb;
        const bitLenInt oBdtQubitCount = bdtQubitCount;
        QBdtPtr nQubits = std::make_shared<QBdt>(length, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
            false, -1, (hardware_rand_generator == NULL) ? false : true, false, (real1_f)amplitudeFloor);
        nQubits->ResetStateVector();
        root->InsertAtDepth(nQubits->root, oBdtQubitCount, length);
        SetQubitCount(qubitCount + length, aqb);
        ROR(length, oBdtQubitCount, qubitCount);
        Dispose(qubitCount - length, length);
    }

public:
    QBdt(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QBdt(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
#if ENABLE_OPENCL
        : QBdt({ OCLEngine::Instance().GetDeviceCount() ? QINTERFACE_OPENCL : QINTERFACE_CPU }, qBitCount, initState,
              rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec,
              norm_thresh, devList, qubitThreshold, separation_thresh)
#else
        : QBdt({ QINTERFACE_CPU }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
#endif
    {
    }

    QBdt(QEnginePtr enginePtr, std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt ignored = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QEnginePtr ReleaseEngine()
    {
        if (bdtQubitCount) {
            throw std::domain_error("Cannot release QEngine from QBdt with BDT qubits!");
        }

        return NODE_TO_QENGINE(root);
    }

    void LockEngine(QEnginePtr eng) { root = std::make_shared<QBdtQEngineNode>(ONE_CMPLX, eng); }

    bool isBinaryDecisionTree() { return true; };

    void SetDevice(int64_t dID);

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        root->Normalize(bdtQubitCount);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QBdt>(toCompare)); }
    real1_f SumSqrDiff(QBdtPtr toCompare);

    void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG);

    QInterfacePtr Clone();

    void GetQuantumState(complex* state);
    void GetQuantumState(QInterfacePtr eng);
    void SetQuantumState(const complex* state);
    void SetQuantumState(QInterfacePtr eng);
    void GetProbs(real1* outputProbs);

    complex GetAmplitude(bitCapInt perm);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->SetAmplitude(perm, amp); });
    }

    using QInterface::Compose;
    bitLenInt Compose(QBdtPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QBdt>(toCopy), start);
    }
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        QBdtPtr d = std::dynamic_pointer_cast<QBdt>(dest);
        if (!bdtQubitCount) {
            d->root = d->MakeQEngineNode(ONE_CMPLX, d->qubitCount, 0U);
            NODE_TO_QENGINE(root)->Decompose(start, NODE_TO_QENGINE(d->root));
            d->SetQubitCount(d->qubitCount, d->qubitCount);
            SetQubitCount(qubitCount - d->qubitCount, qubitCount - d->qubitCount);

            return;
        }

        DecomposeDispose(start, dest->GetQubitCount(), d);
    }
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length)
    {
        if (!bdtQubitCount) {
            NODE_TO_QENGINE(root)->Dispose(start, length);
            SetQubitCount(qubitCount - length, qubitCount - length);

            return;
        }

        DecomposeDispose(start, length, NULL);
    }

    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
    {
        if (!bdtQubitCount) {
            NODE_TO_QENGINE(root)->Dispose(start, length, disposedPerm);
            SetQubitCount(qubitCount - length, qubitCount - length);

            return;
        }

        DecomposeDispose(start, length, NULL);
    }

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    real1_f Prob(bitLenInt qubitIndex);
    real1_f ProbAll(bitCapInt fullRegister);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);
    bitCapInt MAll();

    void Mtrx(const complex* mtrx, bitLenInt target);
    void MCMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    void MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target);
    void MCPhase(
        const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(
        const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target);

    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { eng->FSim(theta, phi, qubitIndex1, qubitIndex2); });
    }

    real1_f ProbParity(bitCapInt mask)
    {
        if (!mask) {
            return ZERO_R1_F;
        }

        if (!(mask & (mask - ONE_BCI))) {
            return Prob(log2(mask));
        }

        real1_f toRet;
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { toRet = QINTERFACE_TO_QPARITY(NODE_TO_QENGINE(root))->ProbParity(mask); });
        return toRet;
    }
    void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QPARITY(eng)->CUniformParityRZ(controls, controlLen, mask, angle);
        });
    }
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true)
    {
        // If no bits in mask:
        if (!mask) {
            return false;
        }

        // If only one bit in mask:
        if (!(mask & (mask - ONE_BCI))) {
            return ForceM(log2(mask), result, doForce);
        }

        SetStateVector();
        return QINTERFACE_TO_QPARITY(NODE_TO_QENGINE(root))->ForceMParity(mask, result, doForce);
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length) { QInterface::INC(toAdd, start, length); }
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length) { QInterface::DEC(toSub, start, length); }
    void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCC(toAdd, start, length, carryIndex);
    }
    void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::DECC(toSub, start, length, carryIndex);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::INCS(toAdd, start, length, overflowIndex);
    }
    void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        QInterface::DECS(toSub, start, length, overflowIndex);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CINC(toAdd, inOutStart, length, controls, controlLen);
    }
    void CDEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
    {
        QInterface::CDEC(toSub, inOutStart, length, controls, controlLen);
    }
    void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        QInterface::INCDECC(toAdd, start, length, carryIndex);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QInterface::IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->PhaseFlipIfLess(greaterPerm, start, length); });
    }
    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
        });
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
        });
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCDECSC(toAdd, start, length, carryIndex); });
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCBCD(toAdd, start, length); });
    }
    void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->INCDECBCDC(toAdd, start, length, carryIndex); });
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->MUL(toMul, inOutStart, carryStart, length); });
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->DIV(toDiv, inOutStart, carryStart, length); });
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        ExecuteAsStateVector(
            [&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->POWModNOut(base, modN, inStart, outStart, length); });
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CMUL(toMul, inOutStart, carryStart, length, controls, controlLen);
        });
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen);
        });
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) {
            QINTERFACE_TO_QALU(eng)->CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen);
        });
    }
    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedLDA(
                indexStart, indexLength, valueStart, valueLength, values, resetValue);
        });
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedADC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        return BitCapIntAsStateVector([&](QInterfacePtr eng) {
            return QINTERFACE_TO_QALU(eng)->IndexedSBC(
                indexStart, indexLength, valueStart, valueLength, carryIndex, values);
        });
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        ExecuteAsStateVector([&](QInterfacePtr eng) { QINTERFACE_TO_QALU(eng)->Hash(start, length, values); });
    }
#endif
};
} // namespace Qrack
