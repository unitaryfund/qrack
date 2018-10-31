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

#include "common/oclengine.hpp"
#include "qengine_opencl.hpp"
#include "qinterface.hpp"
#include "qunit.hpp"

#include "common/parallel_for.hpp"

namespace Qrack {

struct QEngineInfo {
    bitCapInt size;
    bitLenInt deviceID;
};

class QUnitMulti;
typedef std::shared_ptr<QUnitMulti> QUnitMultiPtr;

class QUnitMulti : public QUnit, public ParallelFor {

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

    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    using QUnit::Swap;
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length);
    using QUnit::SqrtSwap;
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length);
    using QUnit::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    using QUnit::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    using QUnit::AntiCNOT;
    virtual void AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CNOT;
    virtual void CNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::T;
    virtual void T(bitLenInt start, bitLenInt length);
    using QUnit::X;
    virtual void X(bitLenInt start, bitLenInt length);
    using QUnit::H;
    virtual void H(bitLenInt start, bitLenInt length);
    using QUnit::Y;
    virtual void Y(bitLenInt start, bitLenInt length);
    using QUnit::Z;
    virtual void Z(bitLenInt start, bitLenInt length);
    using QUnit::CY;
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CZ;
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CLAND;
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QUnit::CLOR;
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QUnit::CLXOR;
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QUnit::RT;
    virtual void RT(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::RTDyad;
    virtual void RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::Exp;
    virtual void Exp(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::ExpDyad;
    virtual void ExpDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::ExpX;
    virtual void ExpX(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::ExpXDyad;
    virtual void ExpXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::ExpY;
    virtual void ExpY(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::ExpYDyad;
    virtual void ExpYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::ExpZ;
    virtual void ExpZ(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::ExpZDyad;
    virtual void ExpZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::RX;
    virtual void RX(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::RXDyad;
    virtual void RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::RY;
    virtual void RY(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::RYDyad;
    virtual void RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::RZ;
    virtual void RZ(real1 radians, bitLenInt start, bitLenInt length);
    using QUnit::RZDyad;
    virtual void RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length);
    using QUnit::CRT;
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRTDyad;
    virtual void CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRX;
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRXDyad;
    virtual void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRY;
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRYDyad;
    virtual void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRZ;
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CRZDyad;
    virtual void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length);

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
