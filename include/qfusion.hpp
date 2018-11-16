//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

namespace Qrack {

struct BitBuffer;
typedef std::shared_ptr<BitBuffer> BitBufferPtr;
typedef std::shared_ptr<complex[4]> BitOp;

// This is a buffer struct that's capable of representing both single bit and controlled single bit gates.
struct BitBuffer {
    bool anti;
    std::vector<bitLenInt> controls;
    BitOp matrix;

    BitBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, const complex* mtrx)
        : anti(antiCtrl)
        , controls(cntrlLen)
        , matrix(new complex[4])
    {
        if (cntrlLen > 0) {
            std::copy(cntrls, cntrls + cntrlLen, controls.begin());
            std::sort(controls.begin(), controls.end());
        }

        std::copy(mtrx, mtrx + 4, matrix.get());
    }

    bool CompareControls(BitBufferPtr toCmp)
    {
        if (toCmp == NULL) {
            // If a bit buffer is empty, it's fine to overwrite it.
            return true;
        }

        // Otherwise, we return "false" if we need to flush, and true if we can keep buffering.

        if (anti != toCmp->anti) {
            return false;
        }

        if (controls.size() != toCmp->controls.size()) {
            return false;
        }

        for (bitLenInt i = 0; i < controls.size(); i++) {
            if (controls[i] != toCmp->controls[i]) {
                return false;
            }
        }

        return true;
    }
};

class QFusion;
typedef std::shared_ptr<QFusion> QFusionPtr;

class QFusion : public QInterface {
protected:
    static const bitLenInt MIN_FUSION_BITS = 3U;
    QInterfacePtr qReg;
    std::shared_ptr<std::default_random_engine> rand_generator;

    std::vector<BitBufferPtr> bitBuffers;
    std::vector<std::vector<bitLenInt>> bitControls;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
        bitBuffers.resize(qb);
        bitControls.resize(qb);
    }

    virtual void NormalizeState(real1 nrm = -999.0)
    {
        // Intentionally left blank
    }

public:
    QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<std::default_random_engine> rgp = nullptr);

    virtual void SetQuantumState(complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetPermutation(bitCapInt perm);
    virtual void SetReg(bitLenInt start, bitLenInt length, bitCapInt value);
    virtual void SetBit(bitLenInt qubitIndex, bool value);
    using QInterface::Cohere;
    virtual bitLenInt Cohere(QInterfacePtr toCopy) { return Cohere(std::dynamic_pointer_cast<QFusion>(toCopy)); }
    virtual bitLenInt Cohere(QFusionPtr toCopy);
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        Decohere(start, length, std::dynamic_pointer_cast<QFusion>(dest));
    }
    virtual void Decohere(bitLenInt start, bitLenInt length, QFusionPtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);
    virtual void ApplyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void ApplyAntiControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx);
    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2);
    virtual bool ForceM(bitLenInt qubit, bool result, bool doForce = true, real1 nrmlzr = ONE_R1);
    virtual bitCapInt ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values);
    virtual bitCapInt ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce = true);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
    virtual void PhaseFlip();

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    virtual void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void SqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);
    virtual void ISqrtSwap(bitLenInt qubitIndex1, bitLenInt qubitIndex2);

    virtual void CopyState(QInterfacePtr orig) { return CopyState(std::dynamic_pointer_cast<QFusion>(orig)); }
    virtual void CopyState(QFusionPtr orig);
    virtual bool IsPhaseSeparable(bool forceCheck = false);
    virtual real1 Prob(bitLenInt qubitIndex);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    virtual real1 ProbAll(bitCapInt fullRegister);

protected:
    BitOp Mul2x2(BitOp left, BitOp right);

    /** Buffer flush methods, to apply accumulated buffers when bits are checked for output or become involved in
     * nonbufferable operations */

    void FlushBit(const bitLenInt& qubitIndex);

    inline void FlushReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0U; i < length; i++) {
            FlushBit(start + i);
        }
    }

    inline void FlushList(const bitLenInt* bitList, const bitLenInt& bitListLen)
    {
        for (bitLenInt i = 0; i < bitListLen; i++) {
            FlushBit(bitList[i]);
        }
    }

    inline void FlushMask(const bitCapInt mask)
    {
        bitCapInt v = mask; // count the number of bits set in v
        bitCapInt oldV;
        bitCapInt power;
        bitLenInt length; // c accumulates the total bits set in v
        for (length = 0; v; length++) {
            oldV = v;
            v &= v - 1; // clear the least significant bit set
            power = (v ^ oldV) & oldV;
            if (power) {
                FlushBit(log2(power));
            }
        }
    }

    inline void FlushAll() { FlushReg(0, qubitCount); }

    /** Buffer discard methods, for when the state of a bit becomes irrelevant before a buffer flush */

    void DiscardBit(const bitLenInt& qubitIndex);

    inline void DiscardReg(const bitLenInt& start, const bitLenInt& length)
    {
        for (bitLenInt i = 0; i < length; i++) {
            DiscardBit(start + i);
        }
    }

    inline void DiscardAll() { DiscardReg(0, qubitCount); }
};
} // namespace Qrack
