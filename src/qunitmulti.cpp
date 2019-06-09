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
#include "qfactory.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int ignored, bool useHardwareRNG)
    : QUnit(QINTERFACE_OPENCL, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1,
          useHardwareRNG)
{
    // Notice that this constructor does not take an engine type parameter, and it always passes QINTERFACE_OPENCL to
    // the QUnit constructor. For QUnitMulti, the "shard" engines are therefore guaranteed to always be QEngineOCL
    // types, and it's safe to assume that they can be cast from QInterfacePtr types to QEngineOCLPtr types in this
    // class.
    deviceCount = OCLEngine::Instance()->GetDeviceCount();
    defaultDeviceID = OCLEngine::Instance()->GetDefaultDeviceID();

    bool bitState;
    bitLenInt currentDevID = defaultDeviceID;
    for (bitLenInt i = 0; i < qBitCount; i++) {
        bitState = ((1 << i) & initState) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1, bitState ? 1 : 0, rand_generator, phaseFactor,
            doNormalize, randGlobalPhase, useHostRam, currentDevID, useRDRAND);
        shards[i].mapped = 0;
        shards[i].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i].isProbDirty = false;

        currentDevID++;
        if (currentDevID >= deviceCount) {
            currentDevID = 0;
        }
    }
}

void QUnitMulti::UpdateEngineInfos()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    bitCapInt sz;
    QEngineOCLPtr qOCL;

    qinfos.clear();

    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            sz = (shard.unit)->GetMaxQPower();
            qips.push_back(shard.unit);
            qOCL = std::static_pointer_cast<QEngineOCL>(shard.unit);
            qinfos.push_back(std::make_shared<QEngineInfo>(sz, qOCL->GetDeviceID(), qOCL));
        }
    }
}

QEngineInfoPtr QUnitMulti::FindEngineInfo(QInterfacePtr unit)
{
    for (auto&& qinfo : qinfos) {
        if (qinfo->unit == unit) {
            return qinfo;
        }
    }

    return NULL;
}

void QUnitMulti::RedistributeQEngines()
{
    // No need to redistribute, if there is only 1 device
    if (deviceCount == 1) {
        return;
    }

    // Get shard sizes and devices
    UpdateEngineInfos();
    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    std::vector<bitCapInt> devSizes(deviceCount);
    std::fill(devSizes.begin(), devSizes.end(), 0U);
    bitCapInt sz;
    bitLenInt devID;
    bitLenInt i, j;

    for (i = 0; i < qinfos.size(); i++) {
        devID = i;
        // If the engine adds negligible load, we can let any given unit keep its
        // residency on this device.
        // if (qinfos[i].size <= 2U) {
        //    continue;
        //}
        if (devSizes[qinfos[i]->deviceID] != 0U) {
            // If the original OpenCL device has equal load to the least, we prefer the original.
            sz = devSizes[qinfos[i]->deviceID];
            devID = qinfos[i]->deviceID;

            // If the default OpenCL device has equal load to the least, we prefer the default.
            if (devSizes[defaultDeviceID] < sz) {
                sz = devSizes[defaultDeviceID];
                devID = defaultDeviceID;
            }

            // Find the device with the lowest load.
            for (j = 0; j < deviceCount; j++) {
                if (devSizes[j] < sz) {
                    sz = devSizes[j];
                    devID = j;
                }
            }

            // Add this unit to the device with the lowest load.
            qinfos[i]->unit->SetDevice(devID);
        }
        // Update the size of buffers handles by this device.
        devSizes[devID] += qinfos[i]->size;
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
{
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

QInterfacePtr QUnitMulti::EntangleIterator(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    QInterfacePtr toRet = QUnit::EntangleIterator(first, last);
    RedistributeQEngines();
    return toRet;
}

void QUnitMulti::SetPermutation(bitCapInt perm, complex phaseFac)
{
    bool bitState;

    Finish();

    bitLenInt currentDevID = defaultDeviceID;
    for (bitLenInt i = 0; i < qubitCount; i++) {
        bitState = ((1 << i) & perm) >> i;
        shards[i].unit = CreateQuantumInterface(engine, subengine, 1, ((1 << i) & perm) >> i, rand_generator, phaseFac,
            doNormalize, randGlobalPhase, useHostRam, devID, useRDRAND);
        shards[i].mapped = 0;
        shards[i].prob = bitState ? ONE_R1 : ZERO_R1;
        shards[i].isProbDirty = false;

        currentDevID++;
        if (currentDevID >= deviceCount) {
            currentDevID = 0;
        }
    }

    UpdateEngineInfos();
}

bool QUnitMulti::TrySeparate(bitLenInt start, bitLenInt length)
{
    bool toRet = QUnit::TrySeparate(start, length);
    RedistributeQEngines();

    return toRet;
}

void QUnitMulti::SeparateBit(bool value, bitLenInt qubit)
{
    if (shards[qubit].unit->GetMaxQPower() == 2) {
        return;
    }

    QEngineInfoPtr qinfo = FindEngineInfo(shards[qubit].unit);
    qinfo->size /= 2U;

    QUnit::SeparateBit(value, qubit);

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());
    std::static_pointer_cast<QEngineOCL>(shards[qubit].unit)->SetDevice(qinfos.back()->deviceID);
    qinfos.back()->size += 2U;
}

} // namespace Qrack
