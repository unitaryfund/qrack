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

#include "mpsshard.hpp"
#include "qengine.hpp"
#include "qunitclifford.hpp"

#define QINTERFACE_TO_QALU(qReg) std::dynamic_pointer_cast<QAlu>(qReg)
#define QINTERFACE_TO_QPARITY(qReg) std::dynamic_pointer_cast<QParity>(qReg)

namespace Qrack {

struct QUnitCliffordAmp {
    complex amp;
    QUnitCliffordPtr stabilizer;

    QUnitCliffordAmp(complex a, QUnitCliffordPtr s)
        : amp(a)
        , stabilizer(s)
    {
        // Intentionally left blank.
    }
};

class QStabilizerHybrid;
typedef std::shared_ptr<QStabilizerHybrid> QStabilizerHybridPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QStabilizer and Qrack::QEngine to maximize
 * performance.
 */
#if ENABLE_ALU
class QStabilizerHybrid : public QAlu, public QParity, public QInterface {
#else
class QStabilizerHybrid : public QParity, public QInterface {
#endif
protected:
    bool useHostRam;
    bool doNormalize;
    bool isSparse;
    bool useTGadget;
    bool isRoundingFlushed;
    bitLenInt thresholdQubits;
    bitLenInt ancillaCount;
    bitLenInt deadAncillaCount;
    bitLenInt maxEngineQubitCount;
    bitLenInt maxAncillaCount;
    bitLenInt maxStateMapCacheQubitCount;
    real1_f separabilityThreshold;
    int64_t devID;
    complex phaseFactor;
    QInterfacePtr engine;
    QUnitCliffordPtr stabilizer;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engineTypes;
    std::vector<QInterfaceEngine> cloneEngineTypes;
    std::vector<MpsShardPtr> shards;
    std::map<bitCapInt, complex> stateMapCache;

    QUnitCliffordPtr MakeStabilizer(bitCapInt perm = 0U);
    QInterfacePtr MakeEngine(bitCapInt perm = 0U);
    QInterfacePtr MakeEngine(bitCapInt perm, bitLenInt qbCount);

    void InvertBuffer(bitLenInt qubit);
    void FlushH(bitLenInt qubit);
    void FlushIfBlocked(bitLenInt control, bitLenInt target, bool isPhase = false);
    bool CollapseSeparableShard(bitLenInt qubit);
    bool TrimControls(const std::vector<bitLenInt>& lControls, std::vector<bitLenInt>& output, bool anti = false);
    void CacheEigenstate(bitLenInt target);
    void FlushBuffers();
    void DumpBuffers()
    {
        for (size_t i = 0; i < shards.size(); ++i) {
            shards[i] = NULL;
        }
    }
    bool EitherIsBuffered(bool logical)
    {
        const size_t maxLcv = logical ? (size_t)qubitCount : shards.size();
        for (size_t i = 0U; i < maxLcv; ++i) {
            if (shards[i]) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsBuffered() { return EitherIsBuffered(false); }
    bool IsLogicalBuffered() { return EitherIsBuffered(true); }
    bool EitherIsProbBuffered(bool logical)
    {
        const size_t maxLcv = logical ? (size_t)qubitCount : shards.size();
        for (size_t i = 0U; i < maxLcv; ++i) {
            MpsShardPtr shard = shards[i];
            if (!shard) {
                continue;
            }
            if (shard->IsHPhase() || shard->IsHInvert()) {
                FlushH(i);
            }
            if (shard->IsInvert()) {
                InvertBuffer(i);
            }
            if (!shard->IsPhase()) {
                // We have a cached non-Clifford operation.
                return true;
            }
        }

        return false;
    }
    bool IsProbBuffered() { return EitherIsProbBuffered(false); }
    bool IsLogicalProbBuffered() { return EitherIsProbBuffered(true); }

    std::unique_ptr<complex[]> GetQubitReducedDensityMatrix(bitLenInt qubit)
    {
        // Form the reduced density matrix of the single qubit.
        const real1 z = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->H(qubit);
        const real1 x = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->S(qubit);
        const real1 y = (real1)(ONE_R1_F - 2 * stabilizer->Prob(qubit));
        stabilizer->IS(qubit);
        stabilizer->H(qubit);

        std::unique_ptr<complex[]> dMtrx(new complex[4]);
        dMtrx[0] = (ONE_CMPLX + z) / complex((real1)2, ZERO_R1);
        dMtrx[1] = x / complex((real1)2, ZERO_R1) - I_CMPLX * (y / complex((real1)2, ZERO_R1));
        dMtrx[2] = x / complex((real1)2, ZERO_R1) + I_CMPLX * (y / complex((real1)2, ZERO_R1));
        dMtrx[3] = (ONE_CMPLX + z) / complex((real1)2, ZERO_R1);
        if (shards[qubit]) {
            complex adj[4]{ std::conj(shards[qubit]->gate[0]), std::conj(shards[qubit]->gate[2]),
                std::conj(shards[qubit]->gate[1]), std::conj(shards[qubit]->gate[3]) };
            complex out[4];
            mul2x2(dMtrx.get(), adj, out);
            mul2x2(shards[qubit]->gate, out, dMtrx.get());
        }

        return dMtrx;
    }

    template <typename F>
    void CheckShots(unsigned shots, bitCapInt m, real1_f partProb, const std::vector<bitCapInt>& qPowers,
        std::vector<real1_f>& rng, F fn)
    {
        for (int64_t shot = rng.size() - 1U; shot >= 0; --shot) {
            if (rng[shot] >= partProb) {
                break;
            }

            bitCapInt sample = 0U;
            for (size_t i = 0U; i < qPowers.size(); ++i) {
                if (m & qPowers[i]) {
                    sample |= pow2(i);
                }
            }
            fn(sample, shot);

            rng.erase(rng.begin() + shot);
            if (!rng.size()) {
                break;
            }
        }
    }

    std::vector<real1_f> GenerateShotProbs(unsigned shots)
    {
        std::vector<real1_f> rng;
        rng.reserve(shots);
        for (unsigned shot = 0U; shot < shots; ++shot) {
            rng.push_back(Rand());
        }
        std::sort(rng.begin(), rng.end());
        std::reverse(rng.begin(), rng.end());

        return rng;
    }

    real1_f FractionalRzAngleWithFlush(bitLenInt i, real1_f angle, bool isGateSuppressed = false)
    {
        const real1_f sectorAngle = PI_R1 / 2;
        const real1_f Period = 2 * PI_R1;
        while (angle >= Period) {
            angle -= Period;
        }
        while (angle < 0) {
            angle += Period;
        }

        int sector = std::round(angle / sectorAngle);
        if (!isGateSuppressed) {
            switch (sector) {
            case 1U:
                stabilizer->S(i);
                break;
            case 2U:
                stabilizer->Z(i);
                break;
            case 3U:
                stabilizer->IS(i);
                break;
            case 0U:
            default:
                break;
            }
        }

        real1_f correctionAngle = angle - (sector * sectorAngle);
        if (correctionAngle > PI_R1) {
            correctionAngle -= Period;
        }
        if (correctionAngle <= -PI_R1) {
            correctionAngle += Period;
        }

        return correctionAngle;
    }

    void FlushCliffordFromBuffers()
    {
        for (size_t i = 0U; i < qubitCount; ++i) {
            // Flush all buffers as close as possible to Clifforrd.
            const MpsShardPtr& shard = shards[i];
            if (!shard) {
                continue;
            }
            if (shard->IsHPhase() || shard->IsHInvert()) {
                FlushH(i);
            }
            if (shard->IsInvert()) {
                InvertBuffer(i);
            }
            if (!shard->IsPhase()) {
                // We have a cached non-phase operation.
                continue;
            }
            const real1 angle = (real1)(FractionalRzAngleWithFlush(i, std::arg(shard->gate[3U] / shard->gate[0U])) / 2);
            if ((2 * abs(angle) / PI_R1) <= FP_NORM_EPSILON) {
                shards[i] = NULL;
                continue;
            }
            const real1 angleCos = cos(angle);
            const real1 angleSin = sin(angle);
            shard->gate[0U] = complex(angleCos, -angleSin);
            shard->gate[3U] = complex(angleCos, angleSin);
        }

        RdmCloneFlush();
    }

    void CombineAncillae();

    QStabilizerHybridPtr RdmCloneHelper()
    {
        CombineAncillae();
        QStabilizerHybridPtr clone = std::dynamic_pointer_cast<QStabilizerHybrid>(Clone());
        clone->RdmCloneFlush(ONE_R1 / 2);

        return clone;
    }
    void RdmCloneFlush(real1_f threshold = FP_NORM_EPSILON);

    real1_f ExpectationFactorized(bool isFloat, const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms,
        const std::vector<real1_f>& weights, bitCapInt offset, bool roundRz)
    {
        if (engine) {
            return isFloat ? engine->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                           : engine->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset);
            ;
        }

        CombineAncillae();

        if (!roundRz) {
            return isFloat ? stabilizer->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                           : stabilizer->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset);
        }

        return isFloat ? RdmCloneHelper()->stabilizer->ExpectationFloatsFactorizedRdm(roundRz, bits, weights)
                       : RdmCloneHelper()->stabilizer->ExpectationBitsFactorizedRdm(roundRz, bits, perms, offset);
    }

    void ClearAncilla(bitLenInt i)
    {
        if (stabilizer->TrySeparate(i)) {
            stabilizer->Dispose(i, 1U);
            shards.erase(shards.begin() + i);
        } else {
            const bitLenInt deadIndex = qubitCount + ancillaCount - 1U;
            stabilizer->SetBit(i, false);
            if (i != deadIndex) {
                stabilizer->Swap(i, deadIndex);
                shards[i].swap(shards[deadIndex]);
            }
            shards.erase(shards.begin() + deadIndex);
            ++deadAncillaCount;
        }
        --ancillaCount;
    }

    real1_f ApproxCompareHelper(
        QStabilizerHybridPtr toCompare, bool isDiscreteBool, real1_f error_tol = TRYDECOMPOSE_EPSILON);

    void ISwapHelper(bitLenInt qubit1, bitLenInt qubit2, bool inverse);

    complex GetAmplitudeOrProb(bitCapInt perm, bool isProb = false);

public:
    QStabilizerHybrid(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QStabilizerHybrid(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QStabilizerHybrid({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    void SetTInjection(bool useGadget) { useTGadget = useGadget; }
    bool GetTInjection() { return useTGadget; }

    void Finish()
    {
        if (stabilizer) {
            stabilizer->Finish();
        } else {
            engine->Finish();
        }
    };

    bool isFinished() { return (!stabilizer || stabilizer->isFinished()) && (!engine || engine->isFinished()); }

    void Dump()
    {
        if (stabilizer) {
            stabilizer->Dump();
        } else {
            engine->Dump();
        }
    }

    void SetConcurrency(uint32_t threadCount)
    {
        QInterface::SetConcurrency(threadCount);
        if (engine) {
            SetConcurrency(GetConcurrencyLevel());
        }
    }

    real1_f ProbRdm(bitLenInt qubit)
    {
        if (!ancillaCount || stabilizer->IsSeparable(qubit)) {
            return Prob(qubit);
        }

        std::unique_ptr<complex[]> dMtrx = GetQubitReducedDensityMatrix(qubit);
        constexpr complex ONE_CMPLX_NEG = complex(-ONE_R1, ZERO_R1);
        constexpr complex pauliZ[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX_NEG };
        complex pMtrx[4];
        mul2x2(dMtrx.get(), pauliZ, pMtrx);
        return (ONE_R1 - std::real(pMtrx[0] + pMtrx[1])) / 2;
    }

    real1_f CProbRdm(bitLenInt control, bitLenInt target)
    {
        AntiCNOT(control, target);
        const real1_f prob = ProbRdm(target);
        AntiCNOT(control, target);

        return prob;
    }

    real1_f ACProbRdm(bitLenInt control, bitLenInt target)
    {
        CNOT(control, target);
        const real1_f prob = ProbRdm(target);
        CNOT(control, target);

        return prob;
    }

    /**
     * Switches between CPU and GPU used modes. (This will not incur a performance penalty, if the chosen mode matches
     * the current mode.) Mode switching happens automatically when qubit counts change, but Compose() and Decompose()
     * might leave their destination QInterface parameters in the opposite mode.
     */
    void SwitchToEngine();

    bool isClifford() { return !engine; }

    bool isClifford(bitLenInt qubit) { return !engine && !shards[qubit]; };

    bool isBinaryDecisionTree() { return engine && engine->isBinaryDecisionTree(); };

    using QInterface::Compose;
    bitLenInt Compose(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, false); };
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy)); }
    bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy), start);
    }
    bitLenInt ComposeNoClone(QStabilizerHybridPtr toCopy) { return ComposeEither(toCopy, true); };
    bitLenInt ComposeNoClone(QInterfacePtr toCopy)
    {
        return ComposeNoClone(std::dynamic_pointer_cast<QStabilizerHybrid>(toCopy));
    }
    bitLenInt ComposeEither(QStabilizerHybridPtr toCopy, bool willDestroy);
    void Decompose(bitLenInt start, QInterfacePtr dest)
    {
        Decompose(start, std::dynamic_pointer_cast<QStabilizerHybrid>(dest));
    }
    void Decompose(bitLenInt start, QStabilizerHybridPtr dest);
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);
    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length);

    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(bitCapInt perm) { return GetAmplitudeOrProb(perm, false); }
    real1_f ProbAll(bitCapInt perm) { return (real1_f)norm(GetAmplitudeOrProb(perm, true)); }
    void SetQuantumState(const complex* inputState);
    void SetAmplitude(bitCapInt perm, complex amp)
    {
        SwitchToEngine();
        engine->SetAmplitude(perm, amp);
    }
    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    void Swap(bitLenInt qubit1, bitLenInt qubit2);
    void ISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, false); }
    void IISwap(bitLenInt qubit1, bitLenInt qubit2) { ISwapHelper(qubit1, qubit2, true); }
    void CSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void CSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void AntiCSqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void CISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);
    void AntiCISqrtSwap(const std::vector<bitLenInt>& lControls, bitLenInt qubit1, bitLenInt qubit2);

    void XMask(bitCapInt mask);
    void YMask(bitCapInt mask);
    void ZMask(bitCapInt mask);

    real1_f Prob(bitLenInt qubit);

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll();

    void Mtrx(const complex* mtrx, bitLenInt target);
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target);
    void MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target);
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target);
    void MACPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target);
    void MACInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target);

    using QInterface::UniformlyControlledSingleBit;
    void UniformlyControlledSingleBit(
        const std::vector<bitLenInt>& controls, bitLenInt qubitIndex, const complex* mtrxs);

    std::map<bitCapInt, int> MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots);
    void MultiShotMeasureMask(const std::vector<bitCapInt>& qPowers, unsigned shots, unsigned long long* shotsArray);

    real1_f ProbParity(bitCapInt mask);
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    void CUniformParityRZ(const std::vector<bitLenInt>& controls, bitCapInt mask, real1_f angle)
    {
        SwitchToEngine();
        QINTERFACE_TO_QPARITY(engine)->CUniformParityRZ(controls, mask, angle);
    }

#if ENABLE_ALU
    using QInterface::M;
    bool M(bitLenInt q) { return QInterface::M(q); }
    using QInterface::X;
    void X(bitLenInt q) { QInterface::X(q); }
    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPhaseFlipIfLess(greaterPerm, start, length, flagIndex);
    }
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->PhaseFlipIfLess(greaterPerm, start, length);
    }

    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::INC(toAdd, start, length);
            return;
        }

        engine->INC(toAdd, start, length);
    }
    void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
    {
        if (stabilizer) {
            QInterface::DEC(toSub, start, length);
            return;
        }

        engine->DEC(toSub, start, length);
    }
    void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::DECS(toSub, start, length, overflowIndex);
            return;
        }

        engine->DECS(toSub, start, length, overflowIndex);
    }
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
    {
        if (stabilizer) {
            QInterface::CINC(toAdd, inOutStart, length, controls);
            return;
        }

        engine->CINC(toAdd, inOutStart, length, controls);
    }
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
    {
        if (stabilizer) {
            QInterface::INCS(toAdd, start, length, overflowIndex);
            return;
        }

        engine->INCS(toAdd, start, length, overflowIndex);
    }
    void INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        if (stabilizer) {
            QInterface::INCDECC(toAdd, start, length, carryIndex);
            return;
        }

        engine->INCDECC(toAdd, start, length, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, overflowIndex, carryIndex);
    }
    void INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECSC(toAdd, start, length, carryIndex);
    }
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCBCD(toAdd, start, length);
    }
    void INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->INCDECBCDC(toAdd, start, length, carryIndex);
    }
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MUL(toMul, inOutStart, carryStart, length);
    }
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->DIV(toDiv, inOutStart, carryStart, length);
    }
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->MULModNOut(toMul, modN, inStart, outStart, length);
    }
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->IMULModNOut(toMul, modN, inStart, outStart, length);
    }
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->POWModNOut(base, modN, inStart, outStart, length);
    }
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMUL(toMul, inOutStart, carryStart, length, controls);
    }
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CDIV(toDiv, inOutStart, carryStart, length, controls);
    }
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CIMULModNOut(toMul, modN, inStart, outStart, length, controls);
    }
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const std::vector<bitLenInt>& controls)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->CPOWModNOut(base, modN, inStart, outStart, length, controls);
    }

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedLDA(
            indexStart, indexLength, valueStart, valueLength, values, resetValue);
    }
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedADC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values)
    {
        SwitchToEngine();
        return QINTERFACE_TO_QALU(engine)->IndexedSBC(
            indexStart, indexLength, valueStart, valueLength, carryIndex, values);
    }
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
    {
        SwitchToEngine();
        QINTERFACE_TO_QALU(engine)->Hash(start, length, values);
    }
#endif

    void PhaseFlip()
    {
        if (stabilizer) {
            stabilizer->PhaseFlip();
        } else {
            engine->PhaseFlip();
        }
    }
    void ZeroPhaseFlip(bitLenInt start, bitLenInt length)
    {
        SwitchToEngine();
        engine->ZeroPhaseFlip(start, length);
    }

    void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::SqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->SqrtSwap(qubitIndex1, qubitIndex2);
    }
    void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
    {
        if (stabilizer) {
            QInterface::ISqrtSwap(qubitIndex1, qubitIndex2);
            return;
        }

        SwitchToEngine();
        engine->ISqrtSwap(qubitIndex1, qubitIndex2);
    }
    void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
    {
        const std::vector<bitLenInt> controls{ qubit1 };
        real1 sinTheta = (real1)sin(theta);

        if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
            MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
            return;
        }

        const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
        if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON) {
            ISwap(qubit1, qubit2);
            MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
            return;
        }

        const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
        if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON) {
            IISwap(qubit1, qubit2);
            MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
            return;
        }

        SwitchToEngine();
        engine->FSim(theta, phi, qubit1, qubit2);
    }

    real1_f ProbMask(bitCapInt mask, bitCapInt permutation)
    {
        SwitchToEngine();
        return engine->ProbMask(mask, permutation);
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), false);
    }
    bool ApproxCompare(QInterfacePtr toCompare, real1_f error_tol = TRYDECOMPOSE_EPSILON)
    {
        return error_tol >=
            ApproxCompareHelper(std::dynamic_pointer_cast<QStabilizerHybrid>(toCompare), true, error_tol);
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        if (engine) {
            engine->UpdateRunningNorm(norm_thresh);
        }
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);

    real1_f ProbAllRdm(bool roundRz, bitCapInt fullRegister);
    real1_f ProbMaskRdm(bool roundRz, bitCapInt mask, bitCapInt permutation);
    real1_f ExpectationBitsAll(const std::vector<bitLenInt>& bits, bitCapInt offset = 0)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsAll(bits, offset);
        }

        return engine->ExpectationBitsAll(bits, offset);
    }
    real1_f ExpectationBitsAllRdm(bool roundRz, const std::vector<bitLenInt>& bits, bitCapInt offset = 0U)
    {
        if (engine) {
            return engine->ExpectationBitsAllRdm(roundRz, bits, offset);
        }

        CombineAncillae();

        if (!roundRz) {
            return stabilizer->ExpectationBitsAll(bits, offset);
        }

        return RdmCloneHelper()->stabilizer->ExpectationBitsAll(bits, offset);
    }
    real1_f ExpectationBitsFactorized(
        const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, bitCapInt offset = 0U)
    {
        if (stabilizer) {
            return QInterface::ExpectationBitsFactorized(bits, perms, offset);
        }

        return engine->ExpectationBitsFactorized(bits, perms, offset);
    }
    real1_f ExpectationBitsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<bitCapInt>& perms, bitCapInt offset = 0U)
    {
        return ExpectationFactorized(false, bits, perms, std::vector<real1_f>(), offset, roundRz);
    }
    real1_f ExpectationFloatsFactorized(const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        if (stabilizer) {
            return QInterface::ExpectationFloatsFactorized(bits, weights);
        }

        return engine->ExpectationFloatsFactorized(bits, weights);
    }
    real1_f ExpectationFloatsFactorizedRdm(
        bool roundRz, const std::vector<bitLenInt>& bits, const std::vector<real1_f>& weights)
    {
        return ExpectationFactorized(true, bits, std::vector<bitCapInt>(), weights, 0U, roundRz);
    }

    bool TrySeparate(bitLenInt qubit);
    bool TrySeparate(bitLenInt qubit1, bitLenInt qubit2);
    bool TrySeparate(const std::vector<bitLenInt>& qubits, real1_f error_tol);

    QInterfacePtr Clone();

    void SetDevice(int64_t dID)
    {
        devID = dID;
        if (engine) {
            engine->SetDevice(dID);
        }
    }

    int64_t GetDeviceID() { return devID; }

    bitCapIntOcl GetMaxSize()
    {
        if (stabilizer) {
            return QInterface::GetMaxSize();
        }

        return engine->GetMaxSize();
    }

    friend std::ostream& operator<<(std::ostream& os, const QStabilizerHybridPtr s);
    friend std::istream& operator>>(std::istream& is, const QStabilizerHybridPtr s);
};
} // namespace Qrack
