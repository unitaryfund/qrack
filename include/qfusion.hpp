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

#include <future>

#include "qinterface.hpp"

namespace Qrack {

struct BitBuffer;
struct GateBuffer;
struct ArithmeticBuffer;
typedef std::shared_ptr<BitBuffer> BitBufferPtr;
typedef std::shared_ptr<GateBuffer> GateBufferPtr;
typedef std::shared_ptr<ArithmeticBuffer> ArithmeticBufferPtr;
typedef std::shared_ptr<complex> BitOp;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
struct BitBuffer {
    bool anti;
    bool isArithmetic;
    std::vector<bitLenInt> controls;

    BitBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, bool isArith)
        : anti(antiCtrl)
        , isArithmetic(isArith)
        , controls(cntrlLen)
    {
        if (cntrlLen > 0) {
            std::copy(cntrls, cntrls + cntrlLen, controls.begin());
            std::sort(controls.begin(), controls.end());
        }
    }

    BitBuffer(BitBuffer* toCopy)
        : anti(toCopy->anti)
        , isArithmetic(toCopy->isArithmetic)
        , controls(toCopy->controls)
    {
        // Intentionally left blank.
    }

    virtual bool CompareControls(BitBufferPtr toCmp)
    {
        if (toCmp == NULL) {
            // If a bit buffer is empty, it's fine to overwrite it.
            return true;
        }

        // Otherwise, we return "false" if we need to flush, and true if we can keep buffering.

        if (anti != toCmp->anti) {
            return false;
        }

        if (isArithmetic != toCmp->isArithmetic) {
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

struct GateBuffer : public BitBuffer {
    BitOp matrix;

    GateBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, const complex* mtrx)
        : BitBuffer(antiCtrl, cntrls, cntrlLen, false)
        , matrix(new complex[4], std::default_delete<complex[]>())
    {
        std::copy(mtrx, mtrx + 4, matrix.get());
    }

    GateBuffer(GateBuffer* toCopy, BitOp mtrx)
        : BitBuffer(toCopy)
        , matrix(mtrx)
    {
        // Intentionally left blank.
    }

    GateBufferPtr LeftMul(BitBufferPtr rightBuffer)
    {
        // If we pass the threshold number of qubits for buffering, we just do 2x2 complex matrix multiplication.
        // We parallelize this, since we can.
        // If a matrix component is very close to zero, we assume it's floating-point-error on a composition that has an
        // exactly 0 component, number theoretically. (If it's not exactly 0 by number theory, it's numerically
        // negligible, and we're safe.)

        BitOp outBuffer(new complex[4], std::default_delete<complex[]>());

        if (rightBuffer != NULL) {
            GateBuffer* rightGate = dynamic_cast<GateBuffer*>(rightBuffer.get());
            BitOp right = rightGate->matrix;

            std::vector<std::future<void>> futures(4);

            futures[0] = std::async(std::launch::async, [&]() {
                outBuffer.get()[0] = (matrix.get()[0] * right.get()[0]) + (matrix.get()[1] * right.get()[2]);
                if (norm(outBuffer.get()[0]) < min_norm) {
                    outBuffer.get()[0] = complex(ZERO_R1, ZERO_R1);
                }
            });
            futures[1] = std::async(std::launch::async, [&]() {
                outBuffer.get()[1] = (matrix.get()[0] * right.get()[1]) + (matrix.get()[1] * right.get()[3]);
                if (norm(outBuffer.get()[1]) < min_norm) {
                    outBuffer.get()[1] = complex(ZERO_R1, ZERO_R1);
                }
            });
            futures[2] = std::async(std::launch::async, [&]() {
                outBuffer.get()[2] = (matrix.get()[2] * right.get()[0]) + (matrix.get()[3] * right.get()[2]);
                if (norm(outBuffer.get()[2]) < min_norm) {
                    outBuffer.get()[2] = complex(ZERO_R1, ZERO_R1);
                }
            });
            futures[3] = std::async(std::launch::async, [&]() {
                outBuffer.get()[3] = (matrix.get()[2] * right.get()[1]) + (matrix.get()[3] * right.get()[3]);
                if (norm(outBuffer.get()[3]) < min_norm) {
                    outBuffer.get()[3] = complex(ZERO_R1, ZERO_R1);
                }
            });

            for (int i = 0; i < 4; i++) {
                futures[i].get();
            }
        } else {
            std::copy(matrix.get(), matrix.get() + 4, outBuffer.get());
        }

        return std::make_shared<GateBuffer>(this, outBuffer);
    }
};

struct ArithmeticBuffer : public BitBuffer {
    bitLenInt start;
    bitLenInt length;
    int toAdd;

    ArithmeticBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, const bitLenInt& strt,
        const bitLenInt& len, int intToAdd)
        : BitBuffer(antiCtrl, cntrls, cntrlLen, true)
        , start(strt)
        , length(len)
        , toAdd(intToAdd)
    {
        // Intentionally left blank.
    }

    virtual bool CompareControls(BitBufferPtr toCmp)
    {
        if (toCmp == NULL) {
            return true;
        }

        if (BitBuffer::CompareControls(toCmp) == false) {
            return false;
        }

        ArithmeticBuffer* toCmpArith = dynamic_cast<ArithmeticBuffer*>(toCmp.get());
        if (start != toCmpArith->start) {
            return false;
        }

        if (length != toCmpArith->length) {
            return false;
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

    std::vector<BitBufferPtr> bitBuffers;
    std::vector<std::vector<bitLenInt>> bitControls;

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
        bitBuffers.resize(qb);
        bitControls.resize(qb);
    }

public:
    QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<std::default_random_engine> rgp = nullptr);
    QFusion(QInterfacePtr target);

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

    virtual QInterfacePtr ReleaseEngine()
    {
        FlushAll();
        QInterfacePtr toRet = qReg;
        qReg = NULL;
        SetQubitCount(0);
        return toRet;
    }

protected:
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

    /** Method to compose arithmetic gates */
    void BufferArithmetic(bitLenInt* controls, bitLenInt controlLen, int toAdd, bitLenInt inOutStart, bitLenInt length);
};
} // namespace Qrack
