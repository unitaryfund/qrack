//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qunitmulti.hpp"
#include "common/oclengine.hpp"
#include "qfactory.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<bitLenInt> devList)
    : QUnit(QINTERFACE_OPENCL, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1,
          useHardwareRNG, useSparseStateVec, norm_thresh)
    , deviceList(devList)
{
    // Notice that this constructor does not take an engine type parameter, and it always passes QINTERFACE_OPENCL to
    // the QUnit constructor. For QUnitMulti, the "shard" engines are therefore guaranteed to always be QEngineOCL
    // types, and it's safe to assume that they can be cast from QInterfacePtr types to QEngineOCLPtr types in this
    // class.
    deviceCount = OCLEngine::Instance()->GetDeviceCount();
    defaultDeviceID = OCLEngine::Instance()->GetDefaultDeviceID();

    std::vector<DeviceContextPtr> deviceContext = OCLEngine::Instance()->GetDeviceContextPtrVector();

    if (deviceList.size() == 0) {
        for (bitLenInt i = 0; i < deviceContext.size(); i++) {
            deviceList.push_back(i);
        }

        std::swap(deviceList[0], deviceList[defaultDeviceID]);
    }

    for (bitLenInt i = 0; i < deviceList.size(); i++) {
        deviceMaxSizes.push_back(deviceContext[deviceList[i]]->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>());
    }
}

std::vector<QEngineInfo> QUnitMulti::GetQInfos()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    QEngineOCLPtr qOCL;
    std::vector<QEngineInfo> qinfos;

    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            qips.push_back(shard.unit);
            qOCL = std::dynamic_pointer_cast<QEngineOCL>(shard.unit);
            qinfos.push_back(QEngineInfo(qOCL));
        }
    }

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    return qinfos;
}

void QUnitMulti::RedistributeQEngines()
{
    // No need to redistribute, if there is only 1 device
    if (deviceList.size() == 1) {
        return;
    }

    // Get shard sizes and devices
    std::vector<QEngineInfo> qinfos = GetQInfos();

    std::vector<bitCapInt> devSizes(deviceList.size());
    std::fill(devSizes.begin(), devSizes.end(), 0U);
    bitCapInt sz;
    bitLenInt devID, devIndex;
    bitLenInt i, j;

    for (i = 0; i < qinfos.size(); i++) {
        // If the engine adds negligible load, we can let any given unit keep its
        // residency on this device.
        // In fact, single qubit units will be handled entirely by the CPU, anyway.
        if (qinfos[i].unit->GetMaxQPower() <= 2U) {
            continue;
        }

        // If the original OpenCL device has equal load to the least, we prefer the original.
        devID = qinfos[i].unit->GetDeviceID();
        devIndex = std::distance(deviceList.begin(), std::find(deviceList.begin(), deviceList.end(), devID));
        sz = devSizes[devIndex];

        // If the default OpenCL device has equal load to the least, we prefer the default.
        if (devSizes[0] <= sz) {
            devID = deviceList[0];
            devIndex = 0;
            sz = devSizes[0];
        }

        // Find the device with the lowest load.
        for (j = 0; j < deviceList.size(); j++) {
            if (((devSizes[j] + qinfos[i].unit->GetMaxQPower()) <= deviceMaxSizes[j]) && (devSizes[j] < sz)) {
                devID = deviceList[j];
                devIndex = j;
                sz = devSizes[j];
            }
        }

        // Add this unit to the device with the lowest load.
        qinfos[i].unit->SetDevice(devID);

        // Update the size of buffers handles by this device.
        devSizes[devIndex] += qinfos[i].unit->GetMaxQPower();
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
{
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

QInterfacePtr QUnitMulti::EntangleInCurrentBasis(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    // Check if size exceeds single device capacity:
    bitLenInt qubitCount = 0;

    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    std::map<QInterfacePtr, bool> found;

    for (auto bit = first; bit < last; bit++) {
        QInterfacePtr unit = shards[**bit].unit;
        if (found.find(unit) == found.end()) {
            found[unit] = true;
            units.push_back(unit);
            qubitCount += unit->GetQubitCount();
        }
    }

    QInterfacePtr unit1 = shards[**first].unit;

    // If device capacity is exceeded, put on default device:
    if (pow2(qubitCount) > std::dynamic_pointer_cast<QEngineOCL>(unit1)->GetMaxSize()) {
        std::dynamic_pointer_cast<QEngineOCL>(unit1)->SetDevice(deviceList[0]);
    }

    QInterfacePtr toRet = QUnit::EntangleInCurrentBasis(first, last);
    RedistributeQEngines();
    return toRet;
}

void QUnitMulti::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    if (shards.size() > 0) {
        Finish();
    }

    bitLenInt currentDevID = 0;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((1 << i) & perm) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1, (pow2(i) & perm) >> (bitCapInt)i, rand_generator,
            phaseFac, doNormalize, randGlobalPhase, useHostRam, deviceList[currentDevID], useRDRAND);
        shards[i].mapped = 0;
        shards[i].isEmulated = false;
        shards[i].isProbDirty = false;
        shards[i].isPhaseDirty = false;
        shards[i].amp0 = bitState ? ZERO_CMPLX : ONE_CMPLX;
        shards[i].amp1 = bitState ? ONE_CMPLX : ZERO_CMPLX;
        shards[i].isPlusMinus = false;

        currentDevID++;
        if (currentDevID >= deviceList.size()) {
            currentDevID = 0;
        }
    }
}

bool QUnitMulti::TrySeparate(bitLenInt start, bitLenInt length)
{
    bool toRet = QUnit::TrySeparate(start, length);
    if (toRet) {
        RedistributeQEngines();
    }

    return toRet;
}

void QUnitMulti::SeparateBit(bool value, bitLenInt qubit)
{
    QUnit::SeparateBit(value, qubit);
    RedistributeQEngines();
}

} // namespace Qrack
