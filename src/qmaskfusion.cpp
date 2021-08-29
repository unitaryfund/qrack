//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <thread>

#include "common/oclengine.hpp"

#include "qfactory.hpp"
#include "qmaskfusion.hpp"

namespace Qrack {

QMaskFusion::QMaskFusion(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , engType(eng)
    , subEngType(subEng)
    , devID(deviceId)
    , devices(devList)
    , phaseFactor(phaseFac)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , separabilityThreshold(sep_thresh)
    , zxShards(qBitCount)
{
    if ((engType == QINTERFACE_OPTIMAL_SCHROEDINGER) && (engType == subEngType)) {
        subEngType = QINTERFACE_OPTIMAL_SINGLE_PAGE;
    }

    engine = MakeEngine(initState);
}

QEnginePtr QMaskFusion::MakeEngine(bitCapInt initState)
{
    QEnginePtr toRet = std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engType, subEngType, qubitCount,
        initState, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse,
        (real1_f)amplitudeFloor, devices, thresholdQubits, separabilityThreshold));
    return toRet;
}

QInterfacePtr QMaskFusion::Clone()
{
    QMaskFusionPtr c = std::dynamic_pointer_cast<QMaskFusion>(CreateQuantumInterface(QINTERFACE_MASK_FUSION, engType,
        subEngType, qubitCount, 0, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID,
        useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}, thresholdQubits, separabilityThreshold));
    c->engine->CopyStateVec(engine);
    return c;
}

void QMaskFusion::DumpBuffers()
{
    zxShards = std::vector<QMaskFusionShard>(qubitCount);
    mpsShards = std::vector<MpsShardPtr>(qubitCount);
}

void QMaskFusion::FlushBuffers()
{
    bitCapInt bitPow;
    bitCapInt rZMask = 0U;
    bitCapInt xMask = 0U;
    bitCapInt lZMask = 0U;
    uint8_t phase = 0U;
    for (bitLenInt i = 0U; i < qubitCount; i++) {
        QMaskFusionShard& shard = zxShards[i];
        bitPow = pow2(i);
        if (shard.isX) {
            xMask |= bitPow;
        }
        if (shard.isZ) {
            if (shard.isXZ) {
                lZMask = bitPow;
            } else {
                rZMask = bitPow;
            }
        }
        phase = (phase + shard.phase) & 3U;
    }

    ZMask(rZMask);
    XMask(xMask);
    ZMask(lZMask);

    if (!randGlobalPhase) {
        if (phase == 1U) {
            ApplySinglePhase(I_CMPLX, I_CMPLX, 0);
        } else if (phase == 2U) {
            PhaseFlip();
        } else if (phase == 3U) {
            ApplySinglePhase(-I_CMPLX, -I_CMPLX, 0);
        }
    }

    for (bitLenInt i = 0; i < qubitCount; i++) {
        MpsShardPtr shard = mpsShards[i];
        if (shard) {
            mpsShards[i] = NULL;
            ApplySingleBit(shard->gate, i);
        }
    }
}

void QMaskFusion::X(bitLenInt target)
{
    if (mpsShards[target]) {
        const complex mtrx[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        ApplySingleBit(mtrx, target);
        return;
    }

    QMaskFusionShard& shard = zxShards[target];

    if (shard.isZ && shard.isXZ) {
        shard.isXZ = false;
    }
    shard.isX = !shard.isX;
}

void QMaskFusion::Y(bitLenInt target)
{
    if (mpsShards[target]) {
        const complex mtrx[4] = { ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
        ApplySingleBit(mtrx, target);
        return;
    }

    QMaskFusionShard& shard = zxShards[target];

    if (shard.isZ) {
        if (shard.isX && !shard.isXZ) {
            shard.phase = (shard.phase + 2U) & 3U;
        }
    } else if (shard.isX) {
        shard.isXZ = true;
    }
    shard.isZ = !shard.isZ;

    if (shard.isZ && shard.isXZ) {
        shard.isXZ = false;
    }
    shard.isX = !shard.isX;

    shard.phase += 1U;
}

void QMaskFusion::Z(bitLenInt target)
{
    if (mpsShards[target]) {
        const complex mtrx[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
        ApplySingleBit(mtrx, target);
        return;
    }

    QMaskFusionShard& shard = zxShards[target];

    if (shard.isZ) {
        if (shard.isX && !shard.isXZ) {
            shard.phase = (shard.phase + 2U) & 3U;
        }
    } else if (shard.isX) {
        shard.isXZ = true;
    }
    shard.isZ = !shard.isZ;
}

void QMaskFusion::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4];
    if (mpsShards[target]) {
        mpsShards[target]->Compose(lMtrx);
        std::copy(mpsShards[target]->gate, mpsShards[target]->gate + 4, mtrx);
        mpsShards[target] = NULL;
    } else {
        std::copy(lMtrx, lMtrx + 4, mtrx);
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        if (IS_SAME(mtrx[0], mtrx[3])) {
            return;
        }
        if (IS_SAME(mtrx[0], -mtrx[3])) {
            Z(target);
            return;
        }
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        if (IS_SAME(mtrx[1], mtrx[2])) {
            X(target);
            return;
        }
        if (IS_SAME(mtrx[1], -mtrx[1])) {
            Y(target);
            return;
        }
    }

    engine->ApplySingleBit(mtrx, target);
}

} // namespace Qrack
