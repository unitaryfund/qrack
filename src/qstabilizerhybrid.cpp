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
#include "qstabilizerhybrid.hpp"

namespace Qrack {

QStabilizerHybrid::QStabilizerHybrid(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1 norm_thresh, std::vector<int> ignored, bitLenInt qubitThreshold)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engineType(eng)
    , engine(NULL)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , doNormalize(doNorm)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , thresholdQubits(qubitThreshold)
{
    concurrency = std::thread::hardware_concurrency();
    stabilizer = MakeStabilizer(initState);
}

QStabilizerPtr QStabilizerHybrid::MakeStabilizer(const bitCapInt& perm)
{
    return std::make_shared<QStabilizer>(qubitCount, perm, useRDRAND, rand_generator);
}

QInterfacePtr QStabilizerHybrid::MakeEngine(const bitCapInt& perm)
{
    QInterfacePtr toRet = CreateQuantumInterface(engineType, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, amplitudeFloor, std::vector<int>{}, thresholdQubits);
    toRet->SetConcurrency(concurrency);
    return toRet;
}

QInterfacePtr QStabilizerHybrid::Clone()
{
    Finish();

    QStabilizerHybridPtr c = std::dynamic_pointer_cast<QStabilizerHybrid>(CreateQuantumInterface(
        QINTERFACE_STABILIZER_HYBRID, engineType, qubitCount, 0, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, devID, useRDRAND, isSparse, amplitudeFloor, std::vector<int>{}, thresholdQubits));

    if (stabilizer) {
        c->stabilizer = std::make_shared<QStabilizer>(*stabilizer);
    } else {
        complex* stateVec = new complex[maxQPower];
        engine->GetQuantumState(stateVec);
        c->SwitchToEngine();
        c->engine->SetQuantumState(stateVec);
        delete[] stateVec;
    }

    return c;
}
} // namespace Qrack
