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
    bitCapInt size;
    bitLenInt deviceID;
    QEngineOCLPtr unit;

    QEngineInfo(bitCapInt sz, bitLenInt devID, QEngineOCLPtr u)
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

    virtual void SetQubitCount(bitLenInt qb)
    {
        for (bitLenInt i = qb; i < shards.size(); i++) {
            WaitBit(i);
        }
        shards.resize(qb);
        QInterface::SetQubitCount(qb);
    }

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

    virtual ~QUnitMulti() { WaitAllBits(); }

    /**
     * \defgroup QUnit overrides
     *
     * All methods overridden to be useful and safe for MP
     *
     * @{
     */

    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit);
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = complex(-999.0, -999.0));
    virtual void CopyState(QUnitMultiPtr orig);
    virtual void CopyState(QInterfacePtr orig);
    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void Compose(QUnitMultiPtr toCopy, bool isMid, bitLenInt start);
    virtual bool TrySeparate(bitLenInt start, bitLenInt length = 1);
    virtual void DumpShards();
    virtual real1 Prob(bitLenInt qubit);
    virtual real1 ProbAll(bitCapInt fullRegister);
    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true);
    virtual void PhaseFlip();
    virtual void UpdateRunningNorm();
    virtual void Finish();
    virtual bool ApproxCompare(QUnitMultiPtr toCompare);
    virtual QInterfacePtr Clone();

    /** @} */

    /**
     * \defgroup RegGates Register-spanning gates
     *
     * Convienence and optimized functions implementing gates are applied from
     * the bit 'start' for 'length' bits for the register.
     *
     * @{
     */

    /** Bitwise Hadamard */
    using QUnit::H;
    virtual void H(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { H(start + bit); });
    }

    /** Bitwise S operator (1/4 phase rotation) */
    using QUnit::S;
    virtual void S(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { S(start + bit); });
    }

    /** Bitwise inverse S operator (1/4 phase rotation) */
    using QUnit::IS;
    virtual void IS(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { IS(start + bit); });
    }

    /** Bitwise T operator (1/8 phase rotation) */
    using QUnit::T;
    virtual void T(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { T(start + bit); });
    }

    /** Bitwise inverse T operator (1/8 phase rotation) */
    using QUnit::IT;
    virtual void IT(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { IT(start + bit); });
    }

    /** Bitwise Pauli X (or logical "NOT") operator */
    using QUnit::X;
    virtual void X(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { X(start + bit); });
    }

    /** Bitwise Pauli Y operator */
    using QUnit::Y;
    virtual void Y(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Y(start + bit); });
    }

    /** Bitwise Pauli Z operator */
    using QUnit::Z;
    virtual void Z(bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Z(start + bit); });
    }

    /** Bitwise controlled-not */
    using QUnit::CNOT;
    virtual void CNOT(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CNOT(start1 + bit, start2 + bit); });
    }

    /** Bitwise "anti-"controlled-not */
    using QUnit::AntiCNOT;
    virtual void AntiCNOT(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { AntiCNOT(start1 + bit, start2 + bit); });
    }

    /** Bitwise swap */
    using QUnit::Swap;
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Swap(start1 + bit, start2 + bit); });
    }

    /** Bitwise square root of swap */
    using QUnit::SqrtSwap;
    virtual void SqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { SqrtSwap(start1 + bit, start2 + bit); });
    }

    /** Bitwise inverse square root of swap */
    using QUnit::ISqrtSwap;
    virtual void ISqrtSwap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ISqrtSwap(start1 + bit, start2 + bit); });
    }

    /** Bitwise doubly controlled-not */
    using QUnit::CCNOT;
    virtual void CCNOT(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CCNOT(start1 + bit, start2 + bit, start3 + bit); });
    }

    /** Bitwise doubly "anti-"controlled-not */
    using QUnit::AntiCCNOT;
    virtual void AntiCCNOT(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { AntiCCNOT(start1 + bit, start2 + bit, start3 + bit); });
    }

    /**
     * Bitwise "AND"
     *
     * "AND" registers at "inputStart1" and "inputStart2," of "length" bits,
     * placing the result in "outputStart".
     */
    using QUnit::AND;
    virtual void AND(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { AND(start1 + bit, start2 + bit, start3 + bit); });
    }

    /**
     * Classical bitwise "AND"
     *
     * "AND" registers at "inputStart1" and the classic bits of "classicalInput," of "length" bits,
     * placing the result in "outputStart".
     */
    using QUnit::CLAND;
    virtual void CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
            bool cBit = (1 << bit) & classicalInput;
            CLAND(qInputStart + bit, cBit, outputStart + bit);
        });
    }

    /** Bitwise "OR" */
    using QUnit::OR;
    virtual void OR(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { OR(start1 + bit, start2 + bit, start3 + bit); });
    }

    /** Classical bitwise "OR" */
    using QUnit::CLOR;
    virtual void CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
            bool cBit = (1 << bit) & classicalInput;
            CLOR(qInputStart + bit, cBit, outputStart + bit);
        });
    }

    /** Bitwise "XOR" */
    using QUnit::XOR;
    virtual void XOR(bitLenInt start1, bitLenInt start2, bitLenInt start3, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { XOR(start1 + bit, start2 + bit, start3 + bit); });
    }

    /** Classical bitwise "XOR" */
    using QUnit::CLXOR;
    virtual void CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) {
            bool cBit = (1 << bit) & classicalInput;
            CLXOR(qInputStart + bit, cBit, outputStart + bit);
        });
    }

    /**
     * Bitwise phase shift gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around |1> state
     */
    using QUnit::RT;
    virtual void RT(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RT(radians, start + bit); });
    }

    /**
     * Bitwise dyadic fraction phase shift gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around |1>
     * state.
     */
    using QUnit::RTDyad;
    virtual void RTDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RTDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise X axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli X axis
     */
    using QUnit::RX;
    virtual void RX(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RX(radians, start + bit); });
    }

    /**
     * Bitwise dyadic fraction X axis rotation gate
     *
     * Rotates \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ on Pauli x axis.
     */
    using QUnit::RXDyad;
    virtual void RXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RXDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise controlled X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ e^{-i*\theta/2} \f$ on Pauli x axis.
     */
    using QUnit::CRX;
    virtual void CRX(real1 radians, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRX(radians, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled dyadic fraction X axis rotation gate
     *
     * If "control" is 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli x axis.
     */
    using QUnit::CRXDyad;
    virtual void CRXDyad(int numerator, int denomPower, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length,
            [&](bitLenInt bit, bitLenInt cpu) { CRXDyad(numerator, denomPower, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise Y axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli y axis.
     */
    using QUnit::RY;
    virtual void RY(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RY(radians, start + bit); });
    }

    /**
     * Bitwise dyadic fraction Y axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    using QUnit::RYDyad;
    virtual void RYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RYDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise controlled Y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Y axis.
     */
    using QUnit::CRY;
    virtual void CRY(real1 radians, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRY(radians, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled dyadic fraction y axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Y
     * axis.
     */
    using QUnit::CRYDyad;
    virtual void CRYDyad(int numerator, int denomPower, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length,
            [&](bitLenInt bit, bitLenInt cpu) { CRYDyad(numerator, denomPower, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise Z axis rotation gate
     *
     * Rotates as \f$ e^{-i*\theta/2} \f$ around Pauli Z axis.
     */
    using QUnit::RZ;
    virtual void RZ(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RZ(radians, start + bit); });
    }

    /**
     * Bitwise dyadic fraction Z axis rotation gate
     *
     * Rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z axis.
     */
    using QUnit::RZDyad;
    virtual void RZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { RZDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise controlled Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ e^{-i*\theta/2} \f$ around
     * Pauli Zaxis.
     */
    using QUnit::CRZ;
    virtual void CRZ(real1 radians, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRZ(radians, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled dyadic fraction Z axis rotation gate
     *
     * If "control" is set to 1, rotates as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$ around Pauli Z
     * axis.
     */
    using QUnit::CRZDyad;
    virtual void CRZDyad(int numerator, int denomPower, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length,
            [&](bitLenInt bit, bitLenInt cpu) { CRZDyad(numerator, denomPower, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ e^{-i*\theta/2}
     * \f$ around |1> state.
     */
    using QUnit::CRT;
    virtual void CRT(real1 radians, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CRT(radians, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled dyadic fraction "phase shift gate"
     *
     * If control bit is set to 1, rotates target bit as \f$ \exp\left(i*{\pi * numerator} / 2^{denomPower}\right) \f$
     * around |1> state.
     */
    using QUnit::CRTDyad;
    virtual void CRTDyad(int numerator, int denomPower, bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length,
            [&](bitLenInt bit, bitLenInt cpu) { CRTDyad(numerator, denomPower, start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise (identity) exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*I} \f$, exponentiation of the identity operator
     */
    using QUnit::Exp;
    virtual void Exp(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { Exp(radians, start + bit); });
    }

    /**
     * Bitwise Dyadic fraction (identity) exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * I / 2^{denomPower}\right) \f$, exponentiation of the identity
     * operator
     */
    using QUnit::ExpDyad;
    virtual void ExpDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise Pauli X exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_x} \f$, exponentiation of the Pauli X operator
     */
    using QUnit::ExpX;
    virtual void ExpX(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpX(radians, start + bit); });
    }

    /**
     * Bitwise Dyadic fraction Pauli X exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_x / 2^{denomPower}\right) \f$, exponentiation of the Pauli X
     * operator
     */
    using QUnit::ExpXDyad;
    virtual void ExpXDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpXDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise Pauli Y exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_y} \f$, exponentiation of the Pauli Y operator
     */
    using QUnit::ExpY;
    virtual void ExpY(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpY(radians, start + bit); });
    }

    /**
     * Bitwise Dyadic fraction Pauli Y exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_y / 2^{denomPower}\right) \f$, exponentiation of the Pauli Y
     * operator
     */
    using QUnit::ExpYDyad;
    virtual void ExpYDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpYDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise Pauli Z exponentiation gate
     *
     * Applies \f$ e^{-i*\theta*\sigma_z} \f$, exponentiation of the Pauli Z operator
     */
    using QUnit::ExpZ;
    virtual void ExpZ(real1 radians, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpZ(radians, start + bit); });
    }

    /**
     * Bitwise Dyadic fraction Pauli Z exponentiation gate
     *
     * Applies \f$ \exp\left(-i * \pi * numerator * \sigma_z / 2^{denomPower}\right) \f$, exponentiation of the Pauli Z
     * operator
     */
    using QUnit::ExpZDyad;
    virtual void ExpZDyad(int numerator, int denomPower, bitLenInt start, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { ExpZDyad(numerator, denomPower, start + bit); });
    }

    /**
     * Bitwise controlled Y gate
     *
     * If the "control" bit is set to 1, then the Pauli "Y" operator is applied
     * to "target."
     */
    using QUnit::CY;
    virtual void CY(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CY(start1 + bit, start2 + bit); });
    }

    /**
     * Bitwise controlled Z gate
     *
     * If the "control" bit is set to 1, then the Pauli "Z" operator is applied
     * to "target."
     */
    using QUnit::CZ;
    virtual void CZ(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        par_for(0, length, [&](bitLenInt bit, bitLenInt cpu) { CZ(start1 + bit, start2 + bit); });
    }

    /** @} */

protected:
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitPtr dest)
    {
        Detach(start, length, std::dynamic_pointer_cast<QUnitMulti>(dest));
    }
    virtual void Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest);

    virtual void RedistributeQEngines();

    virtual QInterfacePtr EntangleIterator(
        std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last);

    virtual QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length);

    virtual QInterfacePtr EntangleAll();

    virtual void CopyState(QUnitMulti* orig);

    virtual void WaitUnit(const QInterfacePtr& unit);

    virtual void WaitBit(const bitLenInt& qubit) { WaitUnit(shards[qubit].unit); }

    virtual void WaitAllBits();
};
} // namespace Qrack
