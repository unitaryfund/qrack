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
    , thresholdQubits(qubitThreshold)
{
    concurrency = std::thread::hardware_concurrency();
    engine = MakeEngine(qubitCount >= thresholdQubits, initState);
}

QEnginePtr QHybrid::MakeEngine(bool isOpenCL, bitCapInt initState)
{
    if (isOpenCL) {
        return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(QINTERFACE_OPENCL, qubitCount, initState,
            rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse));
    } else {
        return std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(QINTERFACE_CPU, qubitCount, initState,
            rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse));
    }
}

QInterfacePtr QHybrid::Clone()
{
    QHybridPtr c = std::dynamic_pointer_cast<QHybrid>(CreateQuantumInterface(QINTERFACE_HYBRID, qubitCount, 0,
        rand_generator, phaseFactor, doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND, isSparse));
    c->engine->CopyStateVec(engine);
    return c;
}
} // namespace Qrack
