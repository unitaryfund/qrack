//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QTensorNetwork is a gate-based QInterface descendant wrapping cuQuantum.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qcircuit.hpp"

#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
#include "common/dispatchqueue.hpp"
#endif

namespace Qrack {

class QTensorNetwork;
typedef std::shared_ptr<QTensorNetwork> QTensorNetworkPtr;

struct TensorMeta {
    std::vector<std::vector<int32_t>> modes;
    std::vector<std::vector<int64_t>> extents;
};

typedef std::vector<TensorMeta> TensorNetworkMeta;
typedef std::shared_ptr<TensorNetworkMeta> TensorNetworkMetaPtr;

class QTensorNetwork : public QInterface {
protected:
    int64_t devID;
    QInterfacePtr layerStack;
    std::vector<int64_t> deviceIDs;
    std::vector<QInterfaceEngine> engines;
    std::vector<QCircuitPtr> circuit;
    std::vector<std::map<bitLenInt, bool>> measurements;
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
    DispatchQueue dispatchQueue;
#endif

    void Dispatch(DispatchFn fn)
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dispatch(fn);
#else
        fn();
#endif
    }

    QCircuitPtr GetCircuit(bitLenInt target, std::vector<bitLenInt> controls = std::vector<bitLenInt>())
    {
        for (size_t i = 0U; i < measurements.size(); ++i) {
            const size_t l = measurements.size() - (i + 1U);
            std::map<bitLenInt, bool>& m = measurements[l];

            if (m.find(target) != m.end()) {
                if (circuit.size() == l) {
                    circuit.emplace_back();
                }

                return circuit[l];
            }

            for (size_t j = 0U; j < controls.size(); ++j) {
                if (m.find(controls[j]) != m.end()) {
                    if (circuit.size() == l) {
                        circuit.emplace_back();
                    }

                    return circuit[l];
                }
            }
        }

        return circuit[0U];
    }

    TensorNetworkMetaPtr GetTensorNetwork() { return NULL; }

    void MakeLayerStack();

public:
    QTensorNetwork(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QTensorNetwork(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QTensorNetwork({ QINTERFACE_OPTIMAL_MULTI }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    ~QTensorNetwork() { Dump(); }

    void Finish()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.finish();
#endif
    };

    bool isFinished()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        return dispatchQueue.isFinished();
#else
        return true;
#endif
    }

    void Dump()
    {
#if ENABLE_QUNIT_CPU_PARALLEL && ENABLE_PTHREAD
        dispatchQueue.dump();
#endif
    }

    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        // Intentionally left blank.
    }

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F)
    {
        // Intentionally left blank
    }

    real1_f SumSqrDiff(QInterfacePtr toCompare)
    {
        return SumSqrDiff(std::dynamic_pointer_cast<QTensorNetwork>(toCompare));
    }
    real1_f SumSqrDiff(QTensorNetworkPtr toCompare) { return ONE_R1_F; }

    void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        Dump();
        circuit.clear();
        measurements.clear();

        circuit.emplace_back();

        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            if (initState & pow2(i)) {
                X(i);
            }
        }

        if (phaseFac == CMPLX_DEFAULT_ARG) {
            if (randGlobalPhase) {
                real1_f angle = Rand() * 2 * (real1_f)PI_R1;
                phaseFac = complex((real1)cos(angle), (real1)sin(angle));
            } else {
                phaseFac = ONE_CMPLX;
            }
        }

        Phase(phaseFac, phaseFac, 0U);
    }

    QInterfacePtr Clone() { return NULL; }

    void GetQuantumState(complex* state) {}
    void GetQuantumState(QInterfacePtr eng) {}
    void SetQuantumState(const complex* state) {}
    void SetQuantumState(QInterfacePtr eng) {}
    void GetProbs(real1* outputProbs) {}

    complex GetAmplitude(bitCapInt perm) { return ZERO_CMPLX; }
    void SetAmplitude(bitCapInt perm, complex amp) {}

    using QInterface::Compose;
    bitLenInt Compose(QTensorNetworkPtr toCopy, bitLenInt start) { return 0U; }
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start) { return 0U; }
    void Decompose(bitLenInt start, QInterfacePtr dest) {}
    QInterfacePtr Decompose(bitLenInt start, bitLenInt length) { return NULL; }
    void Dispose(bitLenInt start, bitLenInt length) {}
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm) {}

    using QInterface::Allocate;
    bitLenInt Allocate(bitLenInt start, bitLenInt length) { return 0U; }

    real1_f Prob(bitLenInt qubitIndex) { return ZERO_R1_F; }
    real1_f ProbAll(bitCapInt fullRegister) { return ZERO_R1_F; }

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true)
    {
        TensorNetworkMetaPtr network = GetTensorNetwork();

#if ENABLE_ENV_VARS
        const bitLenInt maxQb = getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")
            ? (bitLenInt)std::stoi(std::string(getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")))
            : 27U;
#else
        constexpr bitLenInt maxQb = 27U;
#endif

        bool toRet;
        if (qubitCount <= maxQb) {
            MakeLayerStack();
            toRet = layerStack->ForceM(qubit, result, doForce, doApply);
        } else {
            // TODO: Calculate result of measurement with cuTensorNetwork
            throw std::runtime_error("QTensorNetwork doesn't have cuTensorNetwork capabilities yet!");
        }

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
                        c->Append(circuit[layerId + 1U]);
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
    bitCapInt MAll() { return 0U; }

    void Mtrx(const complex* mtrx, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        std::copy(mtrx, mtrx + 4U, lMtrx.get());
        Dispatch([this, target, lMtrx] {
            GetCircuit(target)->AppendGate(std::make_shared<QCircuitGate>(target, lMtrx.get()));
        });
    }
    void MCMtrx(const std::vector<bitLenInt> controls, const complex* mtrx, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        std::copy(mtrx, mtrx + 4U, lMtrx.get());
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(target, lMtrx.get(),
                    std::set<bitLenInt>{ controls.begin(), controls.end() }, pow2(controls.size()) - 1U));
        });
    }
    void MACMtrx(const std::vector<bitLenInt> controls, const complex* mtrx, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        std::copy(mtrx, mtrx + 4U, lMtrx.get());
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(
                    target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, 0U));
        });
    }
    void MCPhase(const std::vector<bitLenInt> controls, complex topLeft, complex bottomRight, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(target, lMtrx.get(),
                    std::set<bitLenInt>{ controls.begin(), controls.end() }, pow2(controls.size()) - 1U));
        });
    }
    void MACPhase(const std::vector<bitLenInt> controls, complex topLeft, complex bottomRight, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = topLeft;
        lMtrx.get()[1U] = ZERO_CMPLX;
        lMtrx.get()[2U] = ZERO_CMPLX;
        lMtrx.get()[3U] = bottomRight;
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(
                    target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, 0U));
        });
    }
    void MCInvert(const std::vector<bitLenInt> controls, complex topRight, complex bottomLeft, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(target, lMtrx.get(),
                    std::set<bitLenInt>{ controls.begin(), controls.end() }, pow2(controls.size()) - 1U));
        });
    }
    void MACInvert(const std::vector<bitLenInt> controls, complex topRight, complex bottomLeft, bitLenInt target)
    {
        layerStack = NULL;
        std::shared_ptr<complex> lMtrx(new complex[4U], std::default_delete<complex[]>());
        lMtrx.get()[0U] = ZERO_CMPLX;
        lMtrx.get()[1U] = topRight;
        lMtrx.get()[2U] = bottomLeft;
        lMtrx.get()[3U] = ZERO_CMPLX;
        Dispatch([this, target, controls, lMtrx] {
            GetCircuit(target, controls)
                ->AppendGate(std::make_shared<QCircuitGate>(
                    target, lMtrx.get(), std::set<bitLenInt>{ controls.begin(), controls.end() }, 0U));
        });
    }

    void FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2);
};
} // namespace Qrack
