//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QTensorNetwork is a gate-based QInterface descendant wrapping cuQuantum.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

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
    , useHostRam(useHostMem)
    , isSparse(useSparseStateVec)
    , isReactiveSeparate(true)
    , useTGadget(true)
    , devID(deviceId)
    , deviceIDs(devList)
    , engines(eng)
{
    if (!engines.size()) {
#if ENABLE_OPENCL
        engines.push_back((OCLEngine::Instance().GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#elif ENABLE_CUDA
        engines.push_back(
            (CUDAEngine::Instance().GetDeviceCount() > 1) ? QINTERFACE_OPTIMAL_MULTI : QINTERFACE_OPTIMAL);
#else
        engines.push_back(QINTERFACE_OPTIMAL);
#endif
    }

    SetPermutation(initState, phaseFac);
}

void QTensorNetwork::MakeLayerStack(const std::set<bitLenInt>& qubits)
{
    if (layerStack) {
        // We have a cached layerStack.
        return;
    }

    // We need to prepare the layer stack (and cache it).
    layerStack =
        CreateQuantumInterface(engines, qubitCount, 0U, rand_generator, ONE_CMPLX, doNormalize, randGlobalPhase,
            useHostRam, devID, hardware_rand_generator != NULL, isSparse, (real1_f)amplitudeFloor, deviceIDs);
    layerStack->SetReactiveSeparate(isReactiveSeparate);
    layerStack->SetTInjection(useTGadget);

    Finish();

    if (qubits.size() && (circuit.size() == 1U)) {
        circuit[0U]->PastLightCone(qubits)->Run(layerStack);
        if (measurements.size()) {
            RunMeasurmentLayer(0U);
        }

        return;
    }

    for (size_t i = 0U; i < circuit.size(); ++i) {
        circuit[i]->Run(layerStack);

        if (measurements.size() <= i) {
            continue;
        }

        RunMeasurmentLayer(i);
    }
}

QInterfacePtr QTensorNetwork::Clone()
{
    QTensorNetworkPtr clone = std::make_shared<QTensorNetwork>(engines, qubitCount, 0U, rand_generator, ONE_CMPLX,
        doNormalize, randGlobalPhase, useHostRam, devID, hardware_rand_generator != NULL, isSparse,
        (real1_f)amplitudeFloor, deviceIDs);

    Finish();
    clone->circuit.clear();
    for (const QCircuitPtr& c : circuit) {
        clone->circuit.push_back(c->Clone());
    }
    clone->measurements = measurements;
    if (layerStack) {
        clone->layerStack = layerStack->Clone();
    }

    clone->SetReactiveSeparate(isReactiveSeparate);
    clone->SetTInjection(useTGadget);

    return clone;
}

bool QTensorNetwork::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if ((qubit + 1U) > qubitCount) {
        if (doForce && result) {
            throw std::runtime_error("QTensorNetwork::ForceM() forced a measurement with 0 probability!");
        }
        return false;
    }

    bool toRet;
    RunAsAmplitudes([&](QInterfacePtr ls) { toRet = ls->ForceM(qubit, result, doForce, doApply); }, { qubit });

    if (!doApply) {
        return toRet;
    }

    size_t layerId = circuit.size() - 1U;
    // Starting from latest circuit layer, if measurement commutes...
    while (layerId && !(circuit[layerId]->IsNonPhaseTarget(qubit))) {
        const QCircuitPtr& c = circuit[layerId];
        c->DeletePhaseTarget(qubit, toRet);

        if (measurements.size() <= layerId) {
            // ...Fill an earlier layer.
            --layerId;
            continue;
        }

        // We will insert a terminal measurement on this qubit, again.
        // This other measurement commutes, as it is in the same basis.
        // So, erase any redundant later measurement.
        std::map<bitLenInt, bool>& m = measurements[layerId];
        m.erase(qubit);

        // If the measurement layer is empty, telescope the layers.
        if (!m.size()) {
            measurements.erase(measurements.begin() + layerId);
            const size_t prevLayerId = layerId + 1U;
            if (prevLayerId < circuit.size()) {
                c->Combine(circuit[prevLayerId]);
                circuit.erase(circuit.begin() + prevLayerId);
            }
        }

        // ...Fill an earlier layer.
        --layerId;
    }

    // Identify whether we need a totally new measurement layer.
    if ((layerId + 1U) > measurements.size()) {
        // Insert the required measurement layer.
        measurements.emplace_back();
    }

    // Insert terminal measurement.
    measurements[layerId][qubit] = toRet;

    // If no qubit in this layer is target of a non-phase gate, it can be completely telescoped into classical state
    // preparation.
    while (true) {
        std::vector<bitLenInt> nonMeasuredQubits;
        nonMeasuredQubits.reserve(qubitCount);
        for (size_t i = 0U; i < qubitCount; ++i) {
            nonMeasuredQubits.push_back(i);
        }
        std::map<bitLenInt, bool>& m = measurements[layerId];
        for (const auto& b : m) {
            nonMeasuredQubits.erase(std::find(nonMeasuredQubits.begin(), nonMeasuredQubits.end(), b.first));
        }

        const QCircuitPtr& c = circuit[layerId];
        for (const bitLenInt& q : nonMeasuredQubits) {
            if (c->IsNonPhaseTarget(q)) {
                // Nothing more to do; tell the user the result.
                return toRet;
            }
        }

        // If we did not return, this circuit layer is fully collapsed.
        constexpr complex pauliX[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };

        if (!layerId) {
            circuit[0U] = std::make_shared<QCircuit>();
            for (const auto& b : m) {
                if (b.second) {
                    circuit[0U]->AppendGate(std::make_shared<QCircuitGate>(b.first, pauliX));
                }
            }

            return toRet;
        }
        circuit.erase(circuit.begin() + layerId);

        const size_t layerIdMin1 = layerId - 1U;
        const std::map<bitLenInt, bool>& mMin1 = measurements[layerIdMin1];
        for (const auto& b : m) {
            const auto it = mMin1.find(b.first);
            if ((it == mMin1.end()) || (b.second == it->second)) {
                continue;
            }
            circuit[layerIdMin1]->AppendGate(std::make_shared<QCircuitGate>(b.first, pauliX));
        }
        m.insert(mMin1.begin(), mMin1.end());
        measurements.erase(measurements.begin() + (layerId - 1U));

        // ...Repeat until we reach the terminal layer.
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
