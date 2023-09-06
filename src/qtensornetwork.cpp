//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QTensorNetwork is a gate-based QInterface descendant wrapping cuQuantum.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

// TODO: qtensornetwork.hpp will be included in qfactory.hpp, then the former include can be removed.
#include "qtensornetwork.hpp"
#include "qfactory.hpp"

#include <cuda_runtime.h>
#include <cutensornet.h>

namespace Qrack {

QTensorNetwork::QTensorNetwork(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceId,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QInterface(qBitCount, rgp, doNorm, useHardwareRNG, randomGlobalPhase, doNorm ? norm_thresh : ZERO_R1_F)
    , devID(deviceId)
    , deviceIDs(devList)
    , engines(eng)
    , circuit({ std::make_shared<QCircuit>() })
{
}

void QTensorNetwork::MakeLayerStack()
{
    if (layerStack) {
        // We have a cached layerStack.
        return;
    }

    // We need to prepare the layer stack (and cache it).
    layerStack = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(engines, qubitCount, 0U, rand_generator, ONE_CMPLX, doNormalize, false, false, devID,
            hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor, deviceIDs));

    const size_t maxLcv = std::max(circuit.size(), measurements.size());
    for (size_t i = 0U; i < maxLcv; ++i) {
        if (circuit.size() <= i) {
            continue;
        }

        circuit[i]->Run(layerStack);

        if (measurements.size() <= i) {
            continue;
        }

        const size_t bitCount = measurements[i].size();
        std::vector<bitLenInt> bits;
        bits.reserve(bitCount);
        std::vector<bool> values;
        values.reserve(bitCount);

        for (const auto& m : measurements[i]) {
            bits.push_back(m.first);
            values.push_back(m.second);
        }

        layerStack->ForceM(bits, values);
    }
}

void QTensorNetwork::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const std::vector<bitLenInt> controls{ qubit1 };
    real1 sinTheta = (real1)sin(theta);

    if ((sinTheta * sinTheta) <= FP_NORM_EPSILON) {
        MCPhase(controls, ONE_CMPLX, exp(complex(ZERO_R1, (real1)phi)), qubit2);
        return;
    }

    const complex expIPhi = exp(complex(ZERO_R1, (real1)phi));

    const real1 sinThetaDiffNeg = ONE_R1 + sinTheta;
    if ((sinThetaDiffNeg * sinThetaDiffNeg) <= FP_NORM_EPSILON) {
        ISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    const real1 sinThetaDiffPos = ONE_R1 - sinTheta;
    if ((sinThetaDiffPos * sinThetaDiffPos) <= FP_NORM_EPSILON) {
        IISwap(qubit1, qubit2);
        MCPhase(controls, ONE_CMPLX, expIPhi, qubit2);
        return;
    }

    throw std::domain_error("QTensorNetwork::FSim() not implemented for irreducible cases!");
}
} // namespace Qrack
