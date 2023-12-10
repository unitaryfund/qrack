//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#include "qengine_opencl.hpp"
#elif ENABLE_CUDA
#include "common/cudaengine.cuh"
#include "qengine_cuda.hpp"
#endif
#include "qunit.hpp"

namespace Qrack {

struct QEngineInfo {
    QInterfacePtr unit;
    size_t deviceIndex;

    QEngineInfo()
        : unit(NULL)
        , deviceIndex(0U)
    {
    }

    QEngineInfo(QInterfacePtr u, size_t devIndex)
        : unit(u)
        , deviceIndex(devIndex)
    {
    }

    bool operator<(const QEngineInfo& other) const
    {
        const int v = bi_compare(unit->GetMaxQPower(), other.unit->GetMaxQPower());
        if (v == 0) {
            // "Larger" QEngineInfo instances get first scheduling priority, and low device indices have greater
            // capacity, so larger deviceIndices get are "<"
            return other.deviceIndex < deviceIndex;
        } else {
            return v < 0;
        }
    }
};

struct DeviceInfo {
    size_t id;
    bitCapIntOcl maxSize;

    bool operator<(const DeviceInfo& other) const { return maxSize < other.maxSize; }
    bool operator>(const DeviceInfo& other) const { return maxSize > other.maxSize; }
};

class QUnitMulti;
typedef std::shared_ptr<QUnitMulti> QUnitMultiPtr;

class QUnitMulti : public QUnit {

protected:
    bool isRedistributing;
    bool isQEngineOCL;
    size_t defaultDeviceID;
    std::vector<DeviceInfo> deviceList;
    std::vector<bitLenInt> deviceQbList;

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

public:
    QUnitMulti(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = ZERO_BCI,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceID = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F);

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = ZERO_BCI, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QUnitMulti({ QINTERFACE_STABILIZER_HYBRID }, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase,
              useHostMem, deviceID, useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold,
              separation_thresh)
    {
    }

    virtual QInterfacePtr Clone()
    {
        // TODO: Copy buffers instead of flushing?
        for (bitLenInt i = 0U; i < qubitCount; ++i) {
            RevertBasis2Qb(i);
        }

        QUnitMultiPtr copyPtr = std::make_shared<QUnitMulti>(engines, qubitCount, ZERO_BCI, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam, defaultDeviceID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
            deviceIDs, thresholdQubits, separabilityThreshold);

        copyPtr->SetReactiveSeparate(isReactiveSeparate);

        return CloneBody(copyPtr);
    }

protected:
    virtual std::vector<QEngineInfo> GetQInfos();

    virtual bool SeparateBit(bool value, bitLenInt qubit)
    {
        const bool toRet = QUnit::SeparateBit(value, qubit);
        RedistributeQEngines();

        return toRet;
    }

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
    {
        Detach(start, length, std::dynamic_pointer_cast<QUnitMulti>(dest));
    }
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
    {
        if (!length) {
            return;
        }

        QUnit::Detach(start, length, dest);
        RedistributeQEngines();
    }

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
    {
        QInterfacePtr toRet = QUnit::EntangleInCurrentBasis(first, last);
        RedistributeQEngines();

        return toRet;
    }

    virtual void RedistributeQEngines();
};
} // namespace Qrack
