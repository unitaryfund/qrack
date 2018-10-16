//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>

#include "oclengine.hpp"
#include "qengine_opencl.hpp"
#include "qinterface.hpp"
#include "qunit.hpp"

namespace Qrack {

struct QEngineInfo {
    bitCapInt size;
    bitLenInt deviceID;
};

class QUnitMulti;
typedef std::shared_ptr<QUnitMulti> QUnitMultiPtr;

class QUnitMulti : public QUnit {

protected:
    int deviceCount;
    int defaultDeviceID;
    std::vector<int> deviceIDs;

public:
    QUnitMulti(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<std::default_random_engine> rgp = nullptr)
        : QUnitMulti(qBitCount, initState, rgp)
    {
    }

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = 0, std::shared_ptr<std::default_random_engine> rgp = nullptr);

protected:
    QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length);
    QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);
    QInterfacePtr EntangleRange(
        bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2, bitLenInt start3, bitLenInt length3);

    template <class It> QInterfacePtr EntangleIterator(It first, It last);

    void Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    bool TrySeparate(std::vector<bitLenInt> bits);

    void RedistributeQEngines();
};

} // namespace Qrack
