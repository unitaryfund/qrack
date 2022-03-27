//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

#include <thread>

namespace Qrack {

QMaskFusion::QMaskFusion(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , engTypes(eng)
    , devID(deviceId)
    , devices(devList)
    , phaseFactor(phaseFac)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , isCacheEmpty(true)
    , separabilityThreshold(sep_thresh)
    , zxShards(qBitCount)
{
    if ((engTypes[0] == QINTERFACE_HYBRID) || (engTypes[0] == QINTERFACE_OPENCL)) {
#if ENABLE_OPENCL
        if (!OCLEngine::Instance().GetDeviceCount()) {
            engTypes[0] = QINTERFACE_CPU;
        }
#else
        engTypes[0] = QINTERFACE_CPU;
#endif
    }

    engine = MakeEngine(initState);
}

QEnginePtr QMaskFusion::MakeEngine(bitCapInt initState)
{
    return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(engTypes, qubitCount, initState, rand_generator,
        phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        devices, thresholdQubits, separabilityThreshold));
}

QInterfacePtr QMaskFusion::Clone()
{
    FlushBuffers();

    QMaskFusionPtr c = std::make_shared<QMaskFusion>(engTypes, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, devices, thresholdQubits,
        separabilityThreshold);
    c->engine = std::dynamic_pointer_cast<QEngine>(engine->Clone());
    return c;
}

void QMaskFusion::FlushBuffers()
{
    bitCapInt zMask = 0U;
    bitCapInt xMask = 0U;
    uint8_t phase = 0U;
    for (bitLenInt i = 0U; i < qubitCount; i++) {
        QMaskFusionShard& shard = zxShards[i];
        bitCapInt bitPow = pow2(i);
        if (shard.isZ) {
            zMask |= bitPow;
        }
        if (shard.isX) {
            xMask |= bitPow;
        }
        phase = (phase + shard.phase) & 3U;
    }

    engine->ZMask(zMask);
    engine->XMask(xMask);

    if (!randGlobalPhase) {
        switch (phase) {
        case 1U:
            engine->Phase(I_CMPLX, I_CMPLX, 0U);
            break;
        case 2U:
            engine->Phase(-ONE_CMPLX, -ONE_CMPLX, 0U);
            break;
        case 3U:
            engine->Phase(-I_CMPLX, -I_CMPLX, 0U);
            break;
        default:
            // Identity
            break;
        }
    }

    DumpBuffers();
}

void QMaskFusion::Phase(complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_SAME(topLeft, bottomRight) && (randGlobalPhase || IS_SAME(topLeft, ONE_CMPLX))) {
        return;
    }

    if (IS_SAME(topLeft, -bottomRight) && (randGlobalPhase || IS_SAME(topLeft, ONE_CMPLX))) {
        Z(target);
        return;
    }

    if (zxShards[target].isZ) {
        zxShards[target].isZ = false;
        bottomRight = -bottomRight;
    }

    if (zxShards[target].isX) {
        zxShards[target].isX = false;
        engine->Invert(topLeft, bottomRight, target);
    } else {
        engine->Phase(topLeft, bottomRight, target);
    }
}
void QMaskFusion::Invert(complex topRight, complex bottomLeft, bitLenInt target)
{
    if (IS_SAME(topRight, bottomLeft) && (randGlobalPhase || IS_SAME(topRight, ONE_CMPLX))) {
        X(target);
        return;
    }

    if (IS_SAME(topRight, -bottomLeft) && (randGlobalPhase || IS_SAME(topRight, -I_CMPLX))) {
        Y(target);
        return;
    }

    if (zxShards[target].isZ) {
        zxShards[target].isZ = false;
        topRight = -topRight;
    }

    if (zxShards[target].isX) {
        zxShards[target].isX = false;
        engine->Phase(topRight, bottomLeft, target);
    } else {
        engine->Invert(topRight, bottomLeft, target);
    }
}

void QMaskFusion::Mtrx(const complex* lMtrx, bitLenInt target)
{
    complex mtrx[4] = { lMtrx[0], lMtrx[1], lMtrx[2], lMtrx[3] };

    if (zxShards[target].isX) {
        zxShards[target].isX = false;
        std::swap(mtrx[0], mtrx[1]);
        std::swap(mtrx[2], mtrx[3]);
    }

    if (zxShards[target].isZ) {
        zxShards[target].isZ = false;
        mtrx[1] = -mtrx[1];
        mtrx[3] = -mtrx[3];
    }

    switch (zxShards[target].phase) {
    case 1U:
        mtrx[0] *= I_CMPLX;
        mtrx[1] *= I_CMPLX;
        mtrx[2] *= I_CMPLX;
        mtrx[3] *= I_CMPLX;
        break;
    case 2U:
        mtrx[0] *= -ONE_CMPLX;
        mtrx[1] *= -ONE_CMPLX;
        mtrx[2] *= -ONE_CMPLX;
        mtrx[3] *= -ONE_CMPLX;
        break;
    case 3U:
        mtrx[0] *= -I_CMPLX;
        mtrx[1] *= -I_CMPLX;
        mtrx[2] *= -I_CMPLX;
        mtrx[3] *= -I_CMPLX;
        break;
    default:
        // Identity
        break;
    }
    zxShards[target].phase = 0U;

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        Phase(mtrx[0], mtrx[3], target);
        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        Invert(mtrx[1], mtrx[2], target);
        return;
    }

    engine->Mtrx(mtrx, target);
}

} // namespace Qrack
