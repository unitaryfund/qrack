//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#include "common/oclengine.hpp"

#include "qfactory.hpp"
#include "qhybrid.hpp"

namespace Qrack {

QHybrid::QHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1_f norm_thresh, std::vector<int> ignored, bitLenInt qubitThreshold)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
{
    concurrency = std::thread::hardware_concurrency();

    if (qubitThreshold != 0) {
        thresholdQubits = qubitThreshold;
    } else {
        // Single bit gates act pairwise on amplitudes, so add at least 1 qubit to the log2 of the preferred
        // concurrency.
        bitLenInt gpuQubits = log2(OCLEngine::Instance()->GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;

        bitLenInt pStridePow =
            getenv("QRACK_PSTRIDEPOW") ? (bitLenInt)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW;

        bitLenInt cpuQubits = (concurrency == 1 ? pStridePow : (log2(concurrency - 1) + pStridePow + 1));

        thresholdQubits = gpuQubits < cpuQubits ? gpuQubits : cpuQubits;
    }

    if (qubitCount >= pagingThresholdQubits) {
        engineType = TYPE_PAGER;
    } else if (qubitCount >= thresholdQubits) {
        engineType = TYPE_GPU;
    } else {
        engineType = TYPE_CPU;
    }
    engine = MakeEngine(engineType, initState);
}

QEnginePtr QHybrid::MakeEngine(QHybridEngineType eType, bitCapInt initState)
{
    QInterfaceEngine t;
    if (eType == TYPE_CPU) {
        t = QINTERFACE_CPU;
    } else if (eType == TYPE_GPU) {
        t = QINTERFACE_OPENCL;
    } else {
        t = QINTERFACE_QPAGER;
    }

    QEnginePtr toRet = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(t, qubitCount, initState, rand_generator, phaseFactor, doNormalize, randGlobalPhase,
            useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}));
    toRet->SetConcurrency(concurrency);
    return toRet;
}

QInterfacePtr QHybrid::Clone()
{
    QHybridPtr c = std::dynamic_pointer_cast<QHybrid>(
        CreateQuantumInterface(QINTERFACE_HYBRID, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
            randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int>{}));
    c->SetConcurrency(concurrency);
    c->engine->CopyStateVec(engine);
    return c;
}
} // namespace Qrack
