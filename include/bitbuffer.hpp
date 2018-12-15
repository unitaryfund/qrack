//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This header defines buffers for Qrack::QFusion.
// QFusion adds an optional "gate fusion" layer on top of a QEngine or QUnit.
// Single bit gates are buffered in per-bit 2x2 complex matrices, to reduce the cost
// of successive application of single bit gates to the same bit.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <algorithm>
#include <vector>

#include "qinterface.hpp"

namespace Qrack {

struct BitBuffer;
struct GateBuffer;
struct ArithmeticBuffer;
typedef std::shared_ptr<BitBuffer> BitBufferPtr;
typedef std::shared_ptr<GateBuffer> GateBufferPtr;
typedef std::shared_ptr<ArithmeticBuffer> ArithmeticBufferPtr;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
struct BitBuffer {
    bool anti;
    bool isArithmetic;
    std::vector<bitLenInt> controls;

    BitBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, bool isArith);

    BitBuffer(BitBuffer* toCopy)
        : anti(toCopy->anti)
        , isArithmetic(toCopy->isArithmetic)
        , controls(toCopy->controls)
    {
        // Intentionally left blank.
    }

    virtual void Apply(QInterfacePtr qReg, const bitLenInt& qubitIndex, std::vector<BitBufferPtr>* bitBuffers) = 0;

    virtual bool Combinable(BitBufferPtr toCmp);

    virtual BitBufferPtr LeftRightCompose(BitBufferPtr rightBuffer) = 0;

    virtual bool IsIdentity() = 0;
};

struct GateBuffer : public BitBuffer {
    BitOp matrix;

    GateBuffer(bool antiCtrl, const bitLenInt* cntrls, const bitLenInt& cntrlLen, const complex* mtrx);

    GateBuffer(GateBuffer* toCopy, BitOp mtrx)
        : BitBuffer(toCopy)
        , matrix(mtrx)
    {
        // Intentionally left blank.
    }

    virtual void Apply(QInterfacePtr qReg, const bitLenInt& qubitIndex, std::vector<BitBufferPtr>* bitBuffers);

    virtual BitBufferPtr LeftRightCompose(BitBufferPtr rightBuffer);

    virtual bool IsIdentity();
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

    ArithmeticBuffer(ArithmeticBuffer* toCopy, int add)
        : BitBuffer(toCopy)
        , start(toCopy->start)
        , length(toCopy->length)
        , toAdd(toCopy->toAdd + add)
    {
        // Intentionally left blank.
    }

    virtual bool Combinable(BitBufferPtr toCmp);

    virtual void Apply(QInterfacePtr qReg, const bitLenInt& qubitIndex, std::vector<BitBufferPtr>* bitBuffers);

    virtual BitBufferPtr LeftRightCompose(BitBufferPtr rightBuffer);

    virtual bool IsIdentity();
};
} // namespace Qrack
