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

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

#include "qfactory.hpp"
#include "qmaskfusion.hpp"

namespace Qrack {

QMaskFusion::QMaskFusion(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engType(eng)
    , subEngType(subEng)
    , devID(deviceId)
    , devices(devList)
    , phaseFactor(phaseFac)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , useHostRam(useHostMem)
    , isCacheEmpty(true)
    , separabilityThreshold(sep_thresh)
    , zxShards(qBitCount)
{
    if (engType == subEngType) {
        if (engType == QINTERFACE_MASK_FUSION) {
            engType = QINTERFACE_OPTIMAL_G2_CHILD;
            subEngType = QINTERFACE_OPTIMAL_G2_CHILD;
        }
#if ENABLE_OPENCL
        if (engType == QINTERFACE_OPTIMAL_G2_CHILD) {
            subEngType = OCLEngine::Instance()->GetDeviceCount() ? QINTERFACE_HYBRID : QINTERFACE_CPU;
        }
#else
        if (engType == QINTERFACE_OPTIMAL_G2_CHILD) {
            subEngType = QINTERFACE_OPTIMAL_G3_CHILD;
        }
#endif
    }

    engine = MakeEngine(initState);
}

QInterfacePtr QMaskFusion::MakeEngine(bitCapInt initState)
{
    QInterfacePtr toRet = CreateQuantumInterface(engType, subEngType, qubitCount, initState, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        devices, thresholdQubits, separabilityThreshold);
    return toRet;
}

QInterfacePtr QMaskFusion::Clone()
{
    FlushBuffers();
    QMaskFusionPtr c = std::dynamic_pointer_cast<QMaskFusion>(CreateQuantumInterface(QINTERFACE_MASK_FUSION, engType,
        subEngType, qubitCount, 0, rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID,
        useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}, thresholdQubits, separabilityThreshold));
    c->engine = engine->Clone();
    return c;
}

void QMaskFusion::FlushBuffers()
{
    bitLenInt i;
    bitCapInt bitPow;
    bitCapInt zMask = 0U;
    bitCapInt xMask = 0U;
    for (i = 0U; i < qubitCount; i++) {
        QMaskFusionShard& shard = zxShards[i];
        bitPow = pow2(i);
        if (shard.isZ) {
            zMask |= bitPow;
        }
        if (shard.isX) {
            xMask |= bitPow;
        }
    }

    engine->ZMask(zMask);
    engine->XMask(xMask);

    DumpBuffers();
}

void QMaskFusion::X(bitLenInt target)
{
    QMaskFusionShard& shard = zxShards[target];
    shard.isX = !shard.isX;
    isCacheEmpty = false;
}

void QMaskFusion::Y(bitLenInt target)
{
    QMaskFusionShard& shard = zxShards[target];
    shard.isZ = !shard.isZ;
    shard.isX = !shard.isX;
    isCacheEmpty = false;
}

void QMaskFusion::Z(bitLenInt target)
{
    QMaskFusionShard& shard = zxShards[target];
    shard.isZ = !shard.isZ;
    isCacheEmpty = false;
}

void QMaskFusion::H(bitLenInt target)
{
    QMaskFusionShard& shard = zxShards[target];
    if (shard.isZ != shard.isX) {
        shard.isZ = !shard.isZ;
        shard.isX = !shard.isX;
    }

    engine->H(target);
}

void QMaskFusion::ApplySingleBit(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4] = { lMtrx[0], lMtrx[1], lMtrx[2], lMtrx[3] };
    if (zxShards[target].isZ) {
        zxShards[target].isZ = false;
        mtrx[1] = -mtrx[1];
        mtrx[3] = -mtrx[3];
    }

    if (zxShards[target].isX) {
        zxShards[target].isX = false;
        std::swap(mtrx[0], mtrx[1]);
        std::swap(mtrx[2], mtrx[3]);
    }

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        ApplySinglePhase(mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        ApplySingleInvert(mtrx[1], mtrx[2], target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        H(target);
        return;
    }

    if (IS_SAME(mtrx[0], mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        H(target);
        X(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], mtrx[2]) && IS_SAME(mtrx[0], mtrx[3])) {
        X(target);
        H(target);
        return;
    }

    if (IS_SAME(mtrx[0], -mtrx[1]) && IS_SAME(mtrx[0], -mtrx[2]) && IS_SAME(mtrx[0], -mtrx[3])) {
        Y(target);
        H(target);
        return;
    }

    engine->ApplySingleBit(mtrx, target);
}

} // namespace Qrack
