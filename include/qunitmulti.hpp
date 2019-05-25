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
    QUnitMulti(QInterfaceEngine eng, QInterfaceEngine subEng, bitLenInt qBitCount, bitCapInt initState = 0,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = complex(-999.0, -999.0), bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = true, int deviceID = -1, bool useHardwareRNG = true)
        : QUnitMulti(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG)
    {
    }
    QUnitMulti(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = true, int ignored = -1, bool useHardwareRNG = true)
        : QUnitMulti(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1, useHardwareRNG)
    {
    }

    QUnitMulti(bitLenInt qBitCount, bitCapInt initState = 0, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = true, int ignored = -1, bool useHardwareRNG = true);

    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    using QUnit::H;
    virtual void H(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::H)); }
    using QUnit::S;
    virtual void S(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::S)); }
    using QUnit::IS;
    virtual void IS(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::IS)); }
    using QUnit::T;
    virtual void T(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::T)); }
    using QUnit::IT;
    virtual void IT(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::IT)); }
    using QUnit::X;
    virtual void X(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::X)); }
    using QUnit::Y;
    virtual void Y(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::Y)); }
    using QUnit::Z;
    virtual void Z(bitLenInt start, bitLenInt length) { OneBitGate(start, length, static_cast<Bit1Fn>(&QUnit::Z)); }
    using QUnit::CNOT;
    virtual void CNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length)
    {
        TwoBitGate(inputBits, targetBits, length, static_cast<Bit2Fn>(&QUnit::CNOT));
    }
    using QUnit::AntiCNOT;
    virtual void AntiCNOT(bitLenInt inputBits, bitLenInt targetBits, bitLenInt length)
    {
        TwoBitGate(inputBits, targetBits, length, static_cast<Bit2Fn>(&QUnit::AntiCNOT));
    }
    using QUnit::CCNOT;
    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
    {
        ThreeBitGate(control1, control2, target, length, static_cast<Bit3Fn>(&QUnit::CCNOT));
    }
    using QUnit::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target, bitLenInt length)
    {
        ThreeBitGate(control1, control2, target, length, static_cast<Bit3Fn>(&QUnit::AntiCCNOT));
    }
    using QUnit::AND;
    virtual void AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
    {
        ThreeBitGate(inputStart1, inputStart2, outputStart, length, static_cast<Bit3Fn>(&QUnit::AND));
    }
    using QUnit::OR;
    virtual void OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
    {
        ThreeBitGate(inputStart1, inputStart2, outputStart, length, static_cast<Bit3Fn>(&QUnit::OR));
    }
    using QUnit::XOR;
    virtual void XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
    {
        ThreeBitGate(inputStart1, inputStart2, outputStart, length, static_cast<Bit3Fn>(&QUnit::XOR));
    }
    using QUnit::RT;
    virtual void RT(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::RT));
    }
    using QUnit::RTDyad;
    virtual void RTDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::RTDyad));
    }
    using QUnit::RX;
    virtual void RX(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::RX));
    }
    using QUnit::RXDyad;
    virtual void RXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::RXDyad));
    }
    using QUnit::CRX;
    virtual void CRX(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRGate(radians, control, target, length, static_cast<Bit2RFn>(&QUnit::CRX));
    }
    using QUnit::CRXDyad;
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRDyadGate(numerator, denomPower, control, target, length, static_cast<Bit2RDyadFn>(&QUnit::CRXDyad));
    }
    using QUnit::RY;
    virtual void RY(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::RY));
    }
    using QUnit::RYDyad;
    virtual void RYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::RYDyad));
    }
    using QUnit::CRY;
    virtual void CRY(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRGate(radians, control, target, length, static_cast<Bit2RFn>(&QUnit::CRY));
    }
    using QUnit::CRYDyad;
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRDyadGate(numerator, denomPower, control, target, length, static_cast<Bit2RDyadFn>(&QUnit::CRYDyad));
    }
    using QUnit::RZ;
    virtual void RZ(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::RZ));
    }
    using QUnit::RZDyad;
    virtual void RZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::RZDyad));
    }
    using QUnit::CRZ;
    virtual void CRZ(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRGate(radians, control, target, length, static_cast<Bit2RFn>(&QUnit::CRZ));
    }
    using QUnit::CRZDyad;
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRDyadGate(numerator, denomPower, control, target, length, static_cast<Bit2RDyadFn>(&QUnit::CRZDyad));
    }
    using QUnit::CRT;
    virtual void CRT(real1 radians, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRGate(radians, control, target, length, static_cast<Bit2RFn>(&QUnit::CRT));
    }
    using QUnit::CRTDyad;
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitRDyadGate(numerator, denomPower, control, target, length, static_cast<Bit2RDyadFn>(&QUnit::CRTDyad));
    }
    using QUnit::Exp;
    virtual void Exp(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::Exp));
    }
    using QUnit::ExpDyad;
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::ExpDyad));
    }
    using QUnit::ExpX;
    virtual void ExpX(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::ExpX));
    }
    using QUnit::ExpXDyad;
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::ExpXDyad));
    }
    using QUnit::ExpY;
    virtual void ExpY(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::ExpY));
    }
    using QUnit::ExpYDyad;
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::ExpYDyad));
    }
    using QUnit::ExpZ;
    virtual void ExpZ(real1 radians, bitLenInt start, bitLenInt length)
    {
        OneBitRGate(radians, start, length, static_cast<Bit1RFn>(&QUnit::ExpZ));
    }
    using QUnit::ExpZDyad;
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        OneBitRDyadGate(numerator, denomPower, start, length, static_cast<Bit1RDyadFn>(&QUnit::ExpZDyad));
    }
    using QUnit::CY;
    virtual void CY(bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitGate(control, target, length, static_cast<Bit2Fn>(&QUnit::CY));
    }
    using QUnit::CZ;
    virtual void CZ(bitLenInt control, bitLenInt target, bitLenInt length)
    {
        TwoBitGate(control, target, length, static_cast<Bit2Fn>(&QUnit::CZ));
    }
    using QUnit::SqrtSwap;
    virtual void SqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        TwoBitGate(start1, start2, length, static_cast<Bit2Fn>(&QUnit::SqrtSwap));
    }
    using QUnit::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        TwoBitGate(start1, start2, length, static_cast<Bit2Fn>(&QUnit::ISqrtSwap));
    }

    using QUnit::CLAND;
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QUnit::CLOR;
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);
    using QUnit::CLXOR;
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length);

protected:
    virtual QInterfacePtr EntangleIterator(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
    {
        Detach(start, length, std::dynamic_pointer_cast<QUnitMulti>(dest));
    }
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest);

    virtual void RedistributeQEngines();

    typedef void (QUnit::*Bit1Fn)(bitLenInt);
    virtual void OneBitGate(bitLenInt start, bitLenInt length, Bit1Fn fn)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(start + bit); });
    }

    typedef void (QUnit::*Bit2Fn)(bitLenInt, bitLenInt);
    virtual void TwoBitGate(bitLenInt start1, bitLenInt start2, bitLenInt length, Bit2Fn fn)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(start1 + bit, start2 + bit); });
    }

    typedef void (QUnit::*Bit3Fn)(bitLenInt, bitLenInt, bitLenInt);
    virtual void ThreeBitGate(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length, Bit3Fn fn)
    {
        par_for(
            0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(start1 + bit, start2 + bit, start3 + bit); });
    }

    typedef void (QUnit::*Bit1RFn)(real1, bitLenInt);
    virtual void OneBitRGate(real1 angle, bitLenInt start, bitLenInt length, Bit1RFn fn)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(angle, start + bit); });
    }

    typedef void (QUnit::*Bit1RDyadFn)(int, int, bitLenInt);
    virtual void OneBitRDyadGate(int numerator, int denomPower, bitLenInt start, bitLenInt length, Bit1RDyadFn fn)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(numerator, denomPower, start + bit); });
    }

    typedef void (QUnit::*Bit2RFn)(real1, bitLenInt, bitLenInt);
    virtual void TwoBitRGate(real1 angle, bitLenInt start1, bitLenInt start2, bitLenInt length, Bit2RFn fn)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(angle, start1 + bit, start2 + bit); });
    }

    typedef void (QUnit::*Bit2RDyadFn)(int, int, bitLenInt, bitLenInt);
    virtual void TwoBitRDyadGate(
        int numerator, int denomPower, bitLenInt start1, bitLenInt start2, bitLenInt length, Bit2RDyadFn fn)
    {
        par_for(0, length,
            [&](bitLenInt bit, bitLenInt cpu) { (this->*fn)(numerator, denomPower, start1 + bit, start2 + bit); });
    }
};

} // namespace Qrack
