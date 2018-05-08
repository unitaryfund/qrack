//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright 2017-2018, Daniel Strano and the Qrack and VM6502Q contributors.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include <random>

#include "qinterface.hpp"

namespace Qrack {

/** Associates a QInterface object with a set of bits. */
struct QEngineShard {
    QInterfacePtr unit;
    bitLenInt mapped;
};

class QUnit;
typedef std::shared_ptr<QUnit> QUnitPtr;

class QUnit : public QInterface
{
protected:
    QInterfaceEngine engine;
    std::vector<QEngineShard> shards;

    std::shared_ptr<std::default_random_engine> rand_generator;

    virtual void SetQubitCount(bitLenInt qb)
    {
        shards.resize(qb);
        QInterface::SetQubitCount(qb);
    }

public:
    QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0, std::shared_ptr<std::default_random_engine> rgp = nullptr);

    virtual void SetQuantumState(complex* inputState);
    virtual void SetPermutation(bitCapInt perm) { SetReg(0, qubitCount, perm); }
    virtual bitLenInt Cohere(QInterfacePtr toCopy);
    virtual std::map<QInterfacePtr, bitLenInt> Cohere(std::vector<QInterfacePtr> toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);

    /**
     * \defgroup BasicGates Basic quantum gate primitives
     *@{
     */

    virtual void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target);
    virtual void CNOT(bitLenInt control, bitLenInt target);
    virtual void AntiCNOT(bitLenInt control, bitLenInt target);
    virtual void H(bitLenInt qubit);
    virtual bool M(bitLenInt qubit);
    virtual void X(bitLenInt qubit);
    virtual void Y(bitLenInt qubit);
    virtual void Z(bitLenInt qubit);
    virtual void CY(bitLenInt control, bitLenInt target);
    virtual void CZ(bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup LogicGates Logic Gates
     *
     * Each bit is paired with a CL* variant that utilizes a classical bit as
     * an input.
     *
     * @{
     */

    virtual void AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit);
    virtual void CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);
    virtual void CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit);

    /** @} */

    /**
     * \defgroup RotGates Rotational gates:
     *
     * NOTE: Dyadic operation angle sign is reversed from radian rotation
     * operators and lacks a division by a factor of two.
     *
     * @{
     */

    virtual void RT(double radians, bitLenInt qubit);
    virtual void RX(double radians, bitLenInt qubit);
    virtual void CRX(double radians, bitLenInt control, bitLenInt target);
    virtual void RY(double radians, bitLenInt qubit);
    virtual void CRY(double radians, bitLenInt control, bitLenInt target);
    virtual void RZ(double radians, bitLenInt qubit);
    virtual void CRZ(double radians, bitLenInt control, bitLenInt target);
    virtual void CRT(double radians, bitLenInt control, bitLenInt target);

    /** @} */

    /**
     * \defgroup ArithGate Arithmetic and other opcode-like gate implemenations.
     *
     * @{
     */

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

    /** @} */

    /**
     * \defgroup ExtraOps Extra operations and capabilities
     *
     * @{
     */

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlip();
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual bitCapInt MReg(bitLenInt start, bitLenInt length);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2);

    /** @} */

    /**
     * \defgroup UtilityFunc Utility functions
     *
     * @{
     */

    virtual void CopyState(QUnitPtr orig);
    virtual void CopyState(QInterfacePtr orig);
    virtual double Prob(bitLenInt qubit);
    virtual double ProbAll(bitCapInt fullRegister);
    virtual void SetBit(bitLenInt qubit1, bool value);

    /** @} */

protected:
    typedef void (QInterface::*INCxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt);
    typedef void (QInterface::*INCxxFn)(bitCapInt, bitLenInt, bitLenInt, bitLenInt, bitLenInt);
    void INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void INCxx(INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index);

    QInterfacePtr Entangle(std::initializer_list<bitLenInt *> bits);
    QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length);
    QInterfacePtr EntangleRange(bitLenInt start, bitLenInt length, bitLenInt start2, bitLenInt length2);

    template <class It> QInterfacePtr EntangleIterator(It first, It last);

    template <typename F, typename ... B>
    void EntangleAndCallMember(F fn, B ... bits);
    template <typename F, typename ... B>
    void EntangleAndCall(F fn, B ... bits);

    void OrderContiguous(QInterfacePtr unit);

    void Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest);

    struct QSortEntry
    {
        bitLenInt bit;
        bitLenInt mapped;
        bool operator<(const QSortEntry &rhs) {
            return mapped < rhs.mapped;
        }
        bool operator>(const QSortEntry &rhs) {
            return mapped > rhs.mapped;
        }
    };
    void SortUnit(QInterfacePtr unit, std::vector<QSortEntry> &bits, bitLenInt low, bitLenInt high);

    /* Debugging and diagnostic routines. */
    void DumpShards();
    QInterfacePtr GetUnit(bitLenInt bit) { return shards[bit].unit; }
};

} // namespace Qrack
