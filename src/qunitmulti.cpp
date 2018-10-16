//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qunitmulti.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QUnit(QINTERFACE_OPENCL, qBitCount, initState, rgp)
{
    deviceCount = OCLEngine::Instance()->GetDeviceCount();
    defaultDeviceID = OCLEngine::Instance()->GetDefaultDeviceID();

    deviceIDs.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        deviceIDs[i] = i;
    }
    if (defaultDeviceID > 0) {
        std::swap(deviceIDs[0], deviceIDs[defaultDeviceID]);
    }

    RedistributeQEngines();
}

void QUnitMulti::RedistributeQEngines()
{
    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    bitCapInt totSize = 0;
    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            totSize += 1U << ((shard.unit)->GetQubitCount());
            qips.push_back(shard.unit);
        }
    }

    bitCapInt partSize = 0;
    int devicesLeft = deviceCount;
    for (bitLenInt i = 0; i < qips.size(); i++) {
        partSize += 1U << (qips[i]->GetQubitCount());
        if (partSize >= (totSize / devicesLeft)) {
            (dynamic_cast<QEngineOCL*>(qips[i].get()))->SetDevice(deviceIDs[deviceCount - devicesLeft]);
            partSize = 0;
            if (devicesLeft > 1) {
                devicesLeft--;
            }
        }
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

template <class It> QInterfacePtr QUnitMulti::EntangleIterator(It first, It last)
{
    QInterfacePtr toRet = QUnit::EntangleIterator(first, last);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(bitLenInt start, bitLenInt length)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start, length);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start1, length1, start2, length2);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(
    bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3)
{
    QInterfacePtr toRet = QUnit::EntangleRange(start1, length1, start2, length2, start3, length3);
    RedistributeQEngines();
    return toRet;
}

bool QUnitMulti::TrySeparate(std::vector<bitLenInt> bits)
{
    bool didSeparate = QUnit::TrySeparate(bits);
    if (didSeparate) {
        RedistributeQEngines();
    }
    return didSeparate;
}

} // namespace Qrack
