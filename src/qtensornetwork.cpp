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

#if ENABLE_CUDA
#include <cuda_runtime.h>
#include <cutensornet.h>
#endif

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

bool QTensorNetwork::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
#if ENABLE_CUDA
    const bitLenInt maxQb = GetThresholdQb();
    bool toRet;
    if (qubitCount <= maxQb) {
        MakeLayerStack();
        toRet = layerStack->ForceM(qubit, result, doForce, doApply);
    } else {
        TensorNetworkMetaPtr network = MakeTensorNetwork();

        // TODO: Calculate result of measurement with cuTensorNetwork
        throw std::runtime_error("QTensorNetwork doesn't have cuTensorNetwork capabilities yet!");
    }
#else
    MakeLayerStack();
    const bool toRet = layerStack->ForceM(qubit, result, doForce, doApply);
#endif

    size_t layerId = circuit.size() - 1U;
    // Starting from latest circuit layer, if measurement commutes...
    while (layerId && !(circuit[layerId]->IsNonPhaseTarget(qubit))) {
        const QCircuitPtr& c = circuit[layerId];
        c->DeletePhaseTarget(qubit, toRet);
        if (measurements.size() > layerId) {
            // We will insert a terminal measurement on this qubit, again.
            // This other measurement commutes, as it is in the same basis.
            // So, erase any redundant later measurement.
            std::map<bitLenInt, bool>& m = measurements[layerId];
            m.erase(qubit);

            // If the measurement layer is empty, telescope the layers.
            if (!m.size()) {
                measurements.erase(measurements.begin() + layerId);
                if (layerId < (circuit.size() - 1U)) {
                    c->Combine(circuit[layerId + 1U]);
                    circuit.erase(circuit.begin() + layerId + 1U);
                }
            }
        }
        // ...Fill an earlier layer.
        --layerId;
    }

    // Identify whether we need a totally new measurement layer.
    if (layerId > measurements.size()) {
        // Insert the required measurement layer.
        measurements.emplace_back();
    }

    // Insert terminal measurement.
    measurements[layerId][qubit] = toRet;

    // If no qubit in this layer is target of a non-phase gate, it can be completely telescoped into classical state
    // preparation.
    while (layerId) {
        std::vector<bitLenInt> nonMeasuredQubits;
        nonMeasuredQubits.reserve(qubitCount);
        for (size_t i = 0U; i < qubitCount; ++i) {
            nonMeasuredQubits.push_back(i);
        }
        for (const auto& m : measurements[layerId]) {
            nonMeasuredQubits.erase(std::find(nonMeasuredQubits.begin(), nonMeasuredQubits.end(), m.first));
        }
        const QCircuitPtr& c = circuit[layerId];
        for (const bitLenInt& q : nonMeasuredQubits) {
            if (c->IsNonPhaseTarget(q)) {
                // Nothing more to do; tell the user the result.
                return toRet;
            }
        }

        // If we did not return, this circuit layer is fully collapsed.
        circuit.erase(circuit.begin() + layerId);

        const std::map<bitLenInt, bool>& m = measurements[layerId];
        measurements[layerId - 1U].insert(m.begin(), m.end());
        measurements.erase(measurements.begin() + layerId);

        --layerId;
    }

    // Tell the user the result.
    return toRet;
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
