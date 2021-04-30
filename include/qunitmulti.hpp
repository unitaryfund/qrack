//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>

#include "common/oclengine.hpp"
#include "common/parallel_for.hpp"
#include "qengine_opencl.hpp"
#include "qinterface.hpp"
#include "qunit.hpp"

namespace Qrack {

struct QEngineInfo {
    QInterfacePtr unit;
    bitLenInt deviceIndex;

    QEngineInfo()
        : unit(NULL)
        , deviceIndex(0)
    {
    }

    QEngineInfo(QInterfacePtr u, bitLenInt devIndex)
        : unit(u)
        , deviceIndex(devIndex)
    {
    }

    bool operator<(const QEngineInfo& other) const
    {
        if (unit->GetMaxQPower() == other.unit->GetMaxQPower()) {
            // "Larger" QEngineInfo instances get first scheduling priority, and low device indices have greater
            // capacity, so larger deviceIndices get are "<"
            return other.deviceIndex < deviceIndex;
        } else {
            return unit->GetMaxQPower() < other.unit->GetMaxQPower();
        }
    }
};

struct DeviceInfo {
    int id;
    bitCapInt maxSize;

    bool operator<(const DeviceInfo& other) const { return maxSize < other.maxSize; }
    bool operator>(const DeviceInfo& other) const { return maxSize > other.maxSize; }
};

class QUnitMulti;
typedef std::shared_ptr<QUnitMulti> QUnitMultiPtr;

class QUnitMulti : public QUnit, public ParallelFor {

protected:
    int defaultDeviceID;
    std::vector<DeviceInfo> deviceList;

    QInterfacePtr MakeEngine(bitLenInt length, bitCapInt perm);

public:
    QUnitMulti(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {},
        bitLenInt qubitThreshold = 0, real1_f separation_thresh = FP_NORM_EPSILON);

    QUnitMulti(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QUnitMulti(eng, eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceID,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
    {
    }

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int> devList = {}, bitLenInt qubitThreshold = 0,
        real1_f separation_thresh = FP_NORM_EPSILON)
        : QUnitMulti(QINTERFACE_OPTIMAL_G0_CHILD, QINTERFACE_OPTIMAL_G1_CHILD, qBitCount, initState, rgp, phaseFac,
              doNorm, randomGlobalPhase, useHostMem, deviceID, useHardwareRNG, useSparseStateVec, norm_thresh, devList,
              qubitThreshold, separation_thresh)
    {
    }

    using QUnit::TrySeparate;
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);
    virtual bool TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol = REAL1_EPSILON) { return false; }

    virtual QInterfacePtr Clone();
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);

protected:
    virtual std::vector<QEngineInfo> GetQInfos();

    virtual void SeparateBit(bool value, bitLenInt qubit, bool doDispose = true);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
    {
        Detach(start, length, std::dynamic_pointer_cast<QUnitMulti>(dest));
    }
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest);

    virtual void RedistributeQEngines();

    virtual QInterfacePtr EntangleInCurrentBasis(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);
};
} // namespace Qrack
