//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QPager breaks a QEngine instance into pages of contiguous amplitudes.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <thread>

#include "qfactory.hpp"
#include "qhybrid.hpp"

namespace Qrack {

QHybrid::QHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<int> ignored, bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, false, useHardwareRNG, false, norm_thresh)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , isGpu(qubitCount >= qubitThreshold)
{
    concurrency = std::thread::hardware_concurrency();
    thresholdQubits = qubitThreshold ? qubitThreshold : (log2(concurrency) + PSTRIDEPOW - 1);
    engine = MakeEngine(qubitCount >= thresholdQubits, initState);
}

QEnginePtr QHybrid::MakeEngine(bool isOpenCL, bitCapInt initState)
{
    return std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(isOpenCL ? QINTERFACE_OPENCL : QINTERFACE_CPU, qubitCount, initState, rand_generator,
            phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, amplitudeFloor));
}

QInterfacePtr QHybrid::Clone()
{
    QHybridPtr c =
        std::dynamic_pointer_cast<QHybrid>(CreateQuantumInterface(QINTERFACE_HYBRID, qubitCount, 0, rand_generator,
            phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, amplitudeFloor));
    c->engine->CopyStateVec(engine);
    return c;
}
} // namespace Qrack
