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

#include "qinterface.hpp"

namespace Qrack {

class QTensorNetwork;
typedef std::shared_ptr<QTensorNetwork> QTensorNetworkPtr;

class QTensorNetwork : public QInterface {
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
        : QTensorNetwork({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
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

    void SetPermutation(bitCapInt initState, complex phaseFac = CMPLX_DEFAULT_ARG) {}

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

    bool ForceM(bitLenInt qubit, bool result, bool doForce = true, bool doApply = true) { return false; }
    bitCapInt MAll() { return 0U; }

    void Mtrx(const complex* mtrx, bitLenInt target) {}
    void MCMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target) {}
    void MACMtrx(const std::vector<bitLenInt>& controls, const complex* mtrx, bitLenInt target) {}
    void MCPhase(const std::vector<bitLenInt>& controls, complex topLeft, complex bottomRight, bitLenInt target) {}
    void MCInvert(const std::vector<bitLenInt>& controls, complex topRight, complex bottomLeft, bitLenInt target) {}

    void FSim(real1_f theta, real1_f phi, bitLenInt qubitIndex1, bitLenInt qubitIndex2) {}
};
} // namespace Qrack
