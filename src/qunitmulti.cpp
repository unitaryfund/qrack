//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState,
    qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t deviceID,
    bool useHardwareRNG, bool useSparseStateVec, real1_f norm_thresh, std::vector<int64_t> devList,
    bitLenInt qubitThreshold, real1_f sep_thresh)
    : QUnit(eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG,
          useSparseStateVec, norm_thresh, devList, qubitThreshold, sep_thresh)
{
#if ENABLE_ENV_VARS
    isRedistributing = (bool)getenv("QRACK_ENABLE_QUNITMULTI_REDISTRIBUTE");
#else
    isRedistributing = false;
#endif

    if (qubitThreshold) {
        thresholdQubits = qubitThreshold;
    } else {
        const bitLenInt gpuQubits =
            log2(OCLEngine::Instance().GetDeviceContextPtr(devID)->GetPreferredConcurrency()) + 1U;
        const bitLenInt cpuQubits = (GetStride() <= ONE_BCI) ? 0U : (log2(GetStride() - ONE_BCI) + 1U);
        thresholdQubits = gpuQubits < cpuQubits ? gpuQubits : cpuQubits;
    }

    std::vector<DeviceContextPtr> deviceContext = OCLEngine::Instance().GetDeviceContextPtrVector();

    if (!devList.size()) {
        defaultDeviceID = (deviceID < 0) ? OCLEngine::Instance().GetDefaultDeviceID() : (size_t)deviceID;

        for (size_t i = 0U; i < deviceContext.size(); ++i) {
            DeviceInfo deviceInfo;
            deviceInfo.id = i;
            deviceList.push_back(deviceInfo);
        }

        std::swap(deviceList[0U], deviceList[defaultDeviceID]);
    } else {
        defaultDeviceID = (devList[0U] < 0) ? OCLEngine::Instance().GetDefaultDeviceID() : (size_t)devList[0U];

        for (size_t i = 0; i < devList.size(); ++i) {
            DeviceInfo deviceInfo;
            deviceInfo.id = (devList[0U] < 0) ? OCLEngine::Instance().GetDefaultDeviceID() : (size_t)devList[i];
            deviceList.push_back(deviceInfo);
        }
    }

    for (size_t i = 0U; i < deviceList.size(); ++i) {
        deviceList[i].maxSize = deviceContext[deviceList[i].id]->GetMaxAlloc();
    }

    if (!devList.size()) {
        std::sort(deviceList.begin() + 1U, deviceList.end(), std::greater<DeviceInfo>());
    }
}

QInterfacePtr QUnitMulti::MakeEngine(bitLenInt length, bitCapInt perm)
{
    size_t deviceId = defaultDeviceID;
    uint64_t sz = OCLEngine::Instance().GetActiveAllocSize(deviceId);

    for (size_t i = 0U; i < deviceList.size(); ++i) {
        uint64_t tSz = OCLEngine::Instance().GetActiveAllocSize(deviceList[i].id);
        if (sz > tSz) {
            sz = tSz;
            deviceId = deviceList[i].id;
        }
    }

    // Suppress passing device list, since QUnitMulti occupies all devices in the list
    QInterfacePtr toRet = CreateQuantumInterface(engines, length, perm, rand_generator, phaseFactor, doNormalize,
        randGlobalPhase, useHostRam, deviceId, useRDRAND, isSparse, (real1_f)amplitudeFloor, std::vector<int64_t>{},
        thresholdQubits, separabilityThreshold);

    return toRet;
}

std::vector<QEngineInfo> QUnitMulti::GetQInfos()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    std::vector<QEngineInfo> qinfos;

    for (auto&& shard : shards) {
        if (shard.unit && (std::find(qips.begin(), qips.end(), shard.unit) == qips.end())) {
            qips.push_back(shard.unit);
            const size_t deviceIndex = std::distance(
                deviceList.begin(), std::find_if(deviceList.begin(), deviceList.end(), [&](DeviceInfo di) {
                    return di.id == (shard.unit->GetDevice() < 0) ? OCLEngine::Instance().GetDefaultDeviceID()
                                                                  : (size_t)shard.unit->GetDevice();
                }));
            qinfos.push_back(QEngineInfo(shard.unit, deviceIndex));
        }
    }

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    return qinfos;
}

void QUnitMulti::RedistributeQEngines()
{
    // Only redistribute if the env var flag is set and NOT a null string.
    // No need to redistribute, if there is only 1 device
    if (deviceList.size() <= 1U) {
        return;
    }

    // Get shard sizes and devices
    std::vector<QEngineInfo> qinfos = GetQInfos();

    std::vector<bitCapInt> devSizes(deviceList.size(), 0U);

    for (size_t i = 0U; i < qinfos.size(); ++i) {
        // We want to proactively set OpenCL devices for the event they cross threshold.
        if (!isRedistributing &&
            !(!qinfos[i].unit || (qinfos[i].unit->GetMaxQPower() <= 2U) ||
                (qinfos[i].unit->GetQubitCount() < thresholdQubits) || qinfos[i].unit->isClifford())) {
            continue;
        }

        // If the original OpenCL device has equal load to the least, we prefer the original.
        size_t deviceID = qinfos[i].unit->GetDevice();
        size_t devIndex = qinfos[i].deviceIndex;
        bitCapInt sz = devSizes[devIndex];

        // If the original device has 0 determined load, don't switch the unit.
        if (sz) {
            // If the default OpenCL device has equal load to the least, we prefer the default.
            if (devSizes[0U] < sz) {
                deviceID = deviceList[0U].id;
                devIndex = 0U;
                sz = devSizes[0U];
            }

            // Find the device with the lowest load.
            for (size_t j = 0U; j < deviceList.size(); ++j) {
                if ((devSizes[j] < sz) && ((devSizes[j] + qinfos[i].unit->GetMaxQPower()) <= deviceList[j].maxSize)) {
                    deviceID = deviceList[j].id;
                    devIndex = j;
                    sz = devSizes[j];
                }
            }

            // Add this unit to the device with the lowest load.
            qinfos[i].unit->SetDevice(deviceID);
        }

        // Update the size of buffers handles by this device.
        devSizes[devIndex] += qinfos[i].unit->GetMaxQPower();
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
{
    if (!length) {
        return;
    }

    QUnit::Detach(start, length, dest);
    if (!dest || (dest->shards[0U].unit && !dest->shards[0U].unit->isClifford())) {
        RedistributeQEngines();
    }
}

QInterfacePtr QUnitMulti::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    QInterfacePtr unit1 = shards[**first].unit;
    if (unit1) {
        bool isAlreadyEntangled = true;
        // If already fully entangled, just return unit1.
        for (auto bit = first + 1U; bit < last; ++bit) {
            QInterfacePtr unit = shards[**bit].unit;
            if (unit1 != unit) {
                isAlreadyEntangled = false;
                break;
            }
        }

        if (isAlreadyEntangled) {
            return unit1;
        }
    }

    for (auto bit = first; bit < last; ++bit) {
        EndEmulation(**bit);
    }
    unit1 = shards[**first].unit;

    // This does nothing if the first unit is the default device:
    if (deviceList[0U].id !=
        ((unit1->GetDevice() < 0) ? OCLEngine::Instance().GetDefaultDeviceID() : (size_t)unit1->GetDevice())) {
        // Check if size exceeds single device capacity:
        bitLenInt qubitCount = 0U;
        std::map<QInterfacePtr, bool> found;

        for (auto bit = first; bit < last; ++bit) {
            QInterfacePtr unit = shards[**bit].unit;
            if (found.find(unit) == found.end()) {
                found[unit] = true;
                qubitCount += unit->GetQubitCount();
            }
        }

        // If device capacity is exceeded, put on default device:
        if (pow2(qubitCount) > unit1->GetMaxSize()) {
            unit1->SetDevice(deviceList[0U].id);
        }
    }

    QInterfacePtr toRet = QUnit::EntangleInCurrentBasis(first, last);
    RedistributeQEngines();

    return toRet;
}

bool QUnitMulti::SeparateBit(bool value, bitLenInt qubit)
{
    const bool isClifford = shards[qubit].unit->isClifford();
    const bool toRet = QUnit::SeparateBit(value, qubit);
    if (!isClifford && toRet) {
        RedistributeQEngines();
    }
    return toRet;
}

QInterfacePtr QUnitMulti::Clone()
{
    // TODO: Copy buffers instead of flushing?
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        RevertBasis2Qb(i);
    }

    QUnitMultiPtr copyPtr = std::make_shared<QUnitMulti>(engines, qubitCount, 0U, rand_generator, phaseFactor,
        doNormalize, randGlobalPhase, useHostRam, defaultDeviceID, useRDRAND, isSparse, (real1_f)amplitudeFloor,
        deviceIDs, thresholdQubits, separabilityThreshold);

    copyPtr->SetReactiveSeparate(isReactiveSeparate);

    return CloneBody(copyPtr);
}

} // namespace Qrack
