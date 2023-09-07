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

#if ENABLE_CUDA
struct TensorMeta {
    std::vector<std::vector<int32_t>> modes;
    std::vector<std::vector<int64_t>> extents;
};
typedef std::vector<TensorMeta> TensorNetworkMeta;
typedef std::shared_ptr<TensorNetworkMeta> TensorNetworkMetaPtr;
#endif

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

    void MakeLayerStack();

#if ENABLE_CUDA
    TensorNetworkMetaPtr MakeTensorNetwork() { return NULL; }
#endif

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
        if (layerStack) {
            layerStack->SetPermutation(initState, phaseFac);
        }

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

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true);

    bitCapInt MAll()
    {
#if ENABLE_CUDA

#if ENABLE_ENV_VARS
        const bitLenInt maxQb = getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")
            ? (bitLenInt)std::stoi(std::string(getenv("QRACK_QTENSORNETWORK_THRESHOLD_QB")))
            : 27U;
#else
        constexpr bitLenInt maxQb = 27U;
#endif

        bitCapInt toRet;
        if (qubitCount <= maxQb) {
            MakeLayerStack();
            toRet = layerStack->MAll();
        } else {
            TensorNetworkMetaPtr network = MakeTensorNetwork();

            // TODO: Calculate result of measurement with cuTensorNetwork
            throw std::runtime_error("QTensorNetwork doesn't have cuTensorNetwork capabilities yet!");
        }
#else
        MakeLayerStack();
        const bitCapInt toRet = layerStack->MAll();
#endif

        SetPermutation(toRet);

        return toRet;
    }

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
