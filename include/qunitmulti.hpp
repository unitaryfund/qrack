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
#include "qengine_opencl.hpp"
#include "qinterface.hpp"
#include "qunit.hpp"

#include "common/parallel_for.hpp"

namespace Qrack {

struct QEngineInfo {
    bitCapInt size;
    bitLenInt deviceID;
    QEngineOCL* unit;

    QEngineInfo(bitCapInt sz, bitLenInt devID, QEngineOCL* u)
        : size(sz)
        , deviceID(devID)
        , unit(u)
    {
        // Intentionally left blank
    }

    bool operator<(const QEngineInfo& other) const
    {
        if (size == other.size) {
            return deviceID < other.deviceID;
        } else {
            return size < other.size;
        }
    }
};

class QUnitMulti;
typedef std::shared_ptr<QUnitMulti> QUnitMultiPtr;

class QUnitMulti : public QUnit, public ParallelFor {

protected:
    int deviceCount;
    int defaultDeviceID;

public:
    QUnitMulti(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<qrack_rand_gen> rgp = nullptr, complex phaseFac = complex(-999.0, -999.0),
        bool doNorm = true, bool randomGlobalPhase = true, bool useHostMem = true)
        : QUnitMulti(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem)
    {
    }

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = 0, std::shared_ptr<qrack_rand_gen> rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = true);

    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    using QUnit::Swap;
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length);
    using QUnit::SqrtSwap;
    virtual void SqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length);
    using QUnit::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length);
    using QUnit::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    using QUnit::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length);
    using QUnit::AntiCNOT;
    virtual void AntiCNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::CNOT;
    virtual void CNOT(bitLenInt control, bitLenInt target, bitLenInt length);
    using QUnit::S;
    virtual void S(bitLenInt start, bitLenInt length);
    using QUnit::IS;
    virtual void IS(bitLenInt start, bitLenInt length);
    using QUnit::T;
    virtual void T(bitLenInt start, bitLenInt length);
    using QUnit::IT;
    virtual void IT(bitLenInt start, bitLenInt length);
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
    virtual QInterfacePtr EntangleIterator(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
    {
        Detach(start, length, std::dynamic_pointer_cast<QUnitMulti>(dest));
    }
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest);

    virtual void RedistributeQEngines();
};

} // namespace Qrack
