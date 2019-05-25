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
}

void QUnitMulti::RedistributeQEngines()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    std::vector<QEngineInfo> qinfos;
    bitCapInt totSize = 0;
    bitCapInt sz;
    QEngineOCL* qOCL;

    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            sz = 1U << ((shard.unit)->GetQubitCount());
            totSize += sz;
            qips.push_back(shard.unit);
            qOCL = dynamic_cast<QEngineOCL*>(shard.unit.get());
            qinfos.push_back(QEngineInfo(sz, qOCL->GetDeviceID(), qOCL));
        }
    }

    std::vector<bitCapInt> devSizes(deviceCount);
    std::fill(devSizes.begin(), devSizes.end(), 0U);
    bitLenInt devID;
    bitLenInt i, j;

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    for (i = 0; i < qinfos.size(); i++) {
        devID = i;
        // If a given device has 0 load, or if the engine adds negligible load, we can let any given unit keep its
        // residency on this device.
        if (qinfos[i].size <= 2U) {
            break;
        }
        if (devSizes[qinfos[i].deviceID] != 0U) {
            // If two devices have identical load, we prefer the default OpenCL device.
            sz = devSizes[defaultDeviceID];
            devID = defaultDeviceID;

            // Find the device with the lowest load.
            for (j = 0; j < deviceCount; j++) {
                if (devSizes[j] < sz) {
                    sz = devSizes[j];
                    devID = j;
                }
            }

            // Add this unit to the device with the lowest load.
            qinfos[i].unit->SetDevice(devID);
        }
        // Update the size of buffers handles by this device.
        devSizes[devID] += qinfos[i].size;
    }
}

void QUnitMulti::ParFor1(const bitLenInt start, const bitLenInt length, ParallelFunc1 fn)
{
    bitLenInt end = start + length;

    std::map<int, int> devMap;
    std::vector<std::vector<bitLenInt>> devGroups;
    int devID;

    for (bitLenInt i = start; i < end; i++) {
        devID = dynamic_cast<QEngineOCL*>(shards[i].unit.get())->GetDeviceID();
        if (devMap.find(devID) == devMap.end()) {
            devMap[devID] = devGroups.size();
            devGroups.push_back(std::vector<bitLenInt>());
        }
        devGroups[devMap[devID]].push_back(i);
    }

    par_for(0, devMap.size(), [&](bitLenInt dev, bitLenInt cpu) {
        std::vector<bitLenInt> devGroup = devGroups[dev];
        for (bitLenInt b = 0; b < devGroup.size(); b++) {
            fn(devGroup[b], cpu);
        }
    });
}

void QUnitMulti::ParFor2(const bitLenInt start1, const bitLenInt start2, const bitLenInt length, ParallelFunc2 fn)
{
    int d1, d2;
    for (bitLenInt i = 0; i < length; i++) {
        d1 = dynamic_cast<QEngineOCL*>(shards[start1 + i].unit.get())->GetDeviceID();
        d2 = dynamic_cast<QEngineOCL*>(shards[start2 + i].unit.get())->GetDeviceID();
        if (d1 != d2) {
            dynamic_cast<QEngineOCL*>(shards[start2 + i].unit.get())->SetDevice(d1);
        }
    }

    std::map<int, int> devMap;
    std::vector<std::vector<bitLenInt>> devGroups1;
    std::vector<std::vector<bitLenInt>> devGroups2;
    int devID;

    for (bitLenInt i = 0; i < length; i++) {
        devID = dynamic_cast<QEngineOCL*>(shards[start1 + i].unit.get())->GetDeviceID();
        if (devMap.find(devID) == devMap.end()) {
            devMap[devID] = devGroups1.size();
            devGroups1.push_back(std::vector<bitLenInt>());
            devGroups2.push_back(std::vector<bitLenInt>());
        }
        devGroups1[devMap[devID]].push_back(start1 + i);
        devGroups2[devMap[devID]].push_back(start2 + i);
    }

    par_for(0, devMap.size(), [&](bitLenInt dev, bitLenInt cpu) {
        std::vector<bitLenInt> devGroup1 = devGroups1[dev];
        std::vector<bitLenInt> devGroup2 = devGroups2[dev];
        for (bitLenInt b = 0; b < devGroup1.size(); b++) {
            fn(devGroup1[b], devGroup2[b], cpu);
        }
    });
}

void QUnitMulti::ParFor3(
    const bitLenInt start1, const bitLenInt start2, const bitLenInt start3, const bitLenInt length, ParallelFunc3 fn)
{
    int d1, d2, d3;
    for (bitLenInt i = 0; i < length; i++) {
        d1 = dynamic_cast<QEngineOCL*>(shards[start1 + i].unit.get())->GetDeviceID();

        d2 = dynamic_cast<QEngineOCL*>(shards[start2 + i].unit.get())->GetDeviceID();
        if (d1 != d2) {
            dynamic_cast<QEngineOCL*>(shards[start2 + i].unit.get())->SetDevice(d1);
        }

        d3 = dynamic_cast<QEngineOCL*>(shards[start3 + i].unit.get())->GetDeviceID();
        if (d1 != d3) {
            dynamic_cast<QEngineOCL*>(shards[start3 + i].unit.get())->SetDevice(d1);
        }
    }

    std::map<int, int> devMap;
    std::vector<std::vector<bitLenInt>> devGroups1;
    std::vector<std::vector<bitLenInt>> devGroups2;
    std::vector<std::vector<bitLenInt>> devGroups3;
    int devID;

    for (bitLenInt i = 0; i < length; i++) {
        devID = dynamic_cast<QEngineOCL*>(shards[start1 + i].unit.get())->GetDeviceID();
        if (devMap.find(devID) == devMap.end()) {
            devMap[devID] = devGroups1.size();
            devGroups1.push_back(std::vector<bitLenInt>());
            devGroups2.push_back(std::vector<bitLenInt>());
            devGroups3.push_back(std::vector<bitLenInt>());
        }
        devGroups1[devMap[devID]].push_back(start1 + i);
        devGroups2[devMap[devID]].push_back(start2 + i);
        devGroups3[devMap[devID]].push_back(start3 + i);
    }

    par_for(0, devMap.size(), [&](bitLenInt dev, bitLenInt cpu) {
        std::vector<bitLenInt> devGroup1 = devGroups1[dev];
        std::vector<bitLenInt> devGroup2 = devGroups2[dev];
        std::vector<bitLenInt> devGroup3 = devGroups3[dev];
        for (bitLenInt b = 0; b < devGroup1.size(); b++) {
            fn(devGroup1[b], devGroup2[b], devGroup3[b], cpu);
        }
    });
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

/// Set register bits to given permutationParFor
void QUnitMulti::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    QUnit::SetReg(start, length, value);
    MReg(start, length);

    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        shards[bit + start].unit->SetBit(shards[bit + start].mapped, !(!(value & (1 << bit))));
    });
}

// Bit-wise apply measurement gate to a register
bitCapInt QUnitMulti::MReg(bitLenInt start, bitLenInt length)
{
    int numCores = GetConcurrencyLevel();

    bitCapInt* partResults = new bitCapInt[numCores]();

    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { partResults[cpu] |= (M(start + bit) ? (1 << bit) : 0); });

    bitCapInt result = 0;
    for (int i = 0; i < numCores; i++) {
        result |= partResults[i];
    }

    return result;
}

/// "AND" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLAND(qInputStart + bit, cBit, outputStart + bit);
    });
}

/// "OR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLOR(qInputStart + bit, cBit, outputStart + bit);
    });
}

/// "XOR" compare a bit range in QInterface with a classical unsigned integer, and store result in range starting at
/// output
void QUnitMulti::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
        cBit = (1 << bit) & classicalInput;
        CLXOR(qInputStart + bit, cBit, outputStart + bit);
    });
}

} // namespace Qrack
