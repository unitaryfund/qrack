//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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
    QEngineOCLPtr unit;
    bitLenInt deviceIndex;

    QEngineInfo()
        : unit(NULL)
        , deviceIndex(0)
    {
    }

    QEngineInfo(QEngineOCLPtr u, bitLenInt devIndex)
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

class QUnitMulti : public QUnit {

protected:
    int defaultDeviceID;
    std::vector<DeviceInfo> deviceList;

public:
    QUnitMulti(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true,
        bool randomGlobalPhase = true, bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> devList = {})
        : QUnitMulti(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG)
    {
    }
    QUnitMulti(QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> devList = {})
        : QUnitMulti(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG)
    {
    }

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = false, int deviceID = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> devList = {});

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = complex(-999.0, -999.0));
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);

protected:
    virtual std::vector<QEngineInfo> GetQInfos();

    virtual void SeparateBit(bool value, bitLenInt qubit);

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
