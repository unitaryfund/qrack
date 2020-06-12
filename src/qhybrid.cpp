//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qhybrid.hpp"
#include "qfactory.hpp"

namespace Qrack {

QEnginePtr QHybrid::ConvertEngineType(QInterfaceEngine oQEngineType, QInterfaceEngine nQEngineType, QEnginePtr oQEngine)
{
    if (oQEngineType == nQEngineType) {
        return oQEngine;
    }

    QEnginePtr nQEngine = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(nQEngineType, oQEngine->GetQubitCount(), 0, rand_generator, CMPLX_DEFAULT_ARG,
            doNormalize, randGlobalPhase, useHostRam, deviceID, useRDRAND, isSparse));

    complex* nStateVec = new complex[nQEngine->GetMaxQPower()];
    oQEngine->GetQuantumState(nStateVec);
    nQEngine->SetQuantumState(nStateVec);
    delete[] nStateVec;

    return nQEngine;
}

QHybrid::QHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int devID, bool useHardwareRNG, bool useSparseStateVec, real1 norm_thresh,
    std::vector<bitLenInt> ignored)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , deviceID(devID)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
    , useHostRam(useHostMem)
{
    if (qBitCount < MIN_OCL_QUBIT_COUNT) {
        qEngineType = QINTERFACE_CPU;
    } else {
        qEngineType = QINTERFACE_OPENCL;
    }

    qEngine = std::dynamic_pointer_cast<QEngine>(CreateQuantumInterface(qEngineType, qBitCount, initState,
        rand_generator, phaseFac, doNormalize, randGlobalPhase, useHostRam, deviceID, useRDRAND, isSparse));
}

} // namespace Qrack
