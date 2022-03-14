//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateVec) {                                                                                                   \
        return;                                                                                                        \
    }

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineCPU::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    shift %= length;
    if (shift == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ regMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl regInt = (lcv & regMask) >> start;
        const bitCapIntOcl outInt = (regInt >> (length - shift)) | ((regInt << shift) & lengthMask);
        nStateVec->write((outInt << start) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

#if ENABLE_ALU
/// Arithmetic shift left, with last 2 bits as sign and carry
void QInterface::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROL(shift, start, length);
            SetReg(start, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void QInterface::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROR(shift, start, length);
            SetReg(end - shift - 1, shift, 0);
            Swap(end - 1, end - 2);
        }
    }
}

/// Logical shift left, filling the extra bits with |0>
void QInterface::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROL(shift, start, length);
            SetReg(start, shift, 0);
        }
    }
}

/// Logical shift right, filling the extra bits with |0>
void QInterface::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            SetReg(start, shift, 0);
            ROR(shift, start, length);
        }
    }
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthMask = pow2MaskOcl(length);
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd;
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = (inOutInt + toAddOcl) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with controls)
void QEngineCPU::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    if (length == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controlLen);

    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd;
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->copy(stateVec);
    stateVec->isReadLocked = false;

    par_for_mask(0, maxQPowerOcl, controlPowers.get(), controlLen, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = (inOutInt + toAddOcl) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes | controlMask, stateVec->read(lcv | controlMask));
    });

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with carry)
void QEngineCPU::INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryIndex), ONE_BCI, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        bitCapIntOcl outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        nStateVec->write(outRes, stateVec->read(lcv));
    });
    ResetStateVec(nStateVec);
}

/**
 * Add an integer to the register, with sign and without carry. Because the
 * register length is an arbitrary number of bits, the sign bit position on the
 * integer to add is variable. Hence, the integer to add is specified as cast
 * to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QEngineCPU::INCS(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd;
    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl signMask = pow2Ocl(length - ONE_BCI);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAddOcl;
        bitCapIntOcl outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes;
        }
        bool isOverflow = isOverflowAdd(inOutInt, toAddOcl, signMask, lengthPower);
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    const bitCapIntOcl signMask = pow2Ocl(length - ONE_BCI);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, carryMask, ONE_BCI, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl inInt = toModOcl;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        bitCapIntOcl outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        if (isOverflow) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(
    bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl signMask = pow2Ocl(length - ONE_BCI);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << inOutStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, carryMask, ONE_BCI, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        const bitCapIntOcl inInt = toModOcl;
        const bitCapIntOcl outInt = inOutInt + toModOcl;
        bitCapIntOcl outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = isOverflowAdd(inOutInt, inInt, signMask, lengthPower);
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::MULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl toMulOcl = (bitCapIntOcl)toMul;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl highMask = lowMask << length;
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryStart), length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl mulInt = ((lcv & inOutMask) >> inOutStart) * toMulOcl;
        const bitCapIntOcl mulRes =
            ((mulInt & lowMask) << inOutStart) | (((mulInt & highMask) >> length) << carryStart) | otherRes;
        nStateVec->write(outFn(lcv, mulRes), stateVec->read(inFn(lcv, mulRes)));
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    SetReg(carryStart, length, 0);

    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }
    if (toMul == ONE_BCI) {
        return;
    }

    MULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; }, toMul, inOutStart, carryStart, length);
}

void QEngineCPU::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
        throw std::invalid_argument("DIV by zero");
    }
    if (toDiv == ONE_BCI) {
        return;
    }

    MULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; }, toDiv, inOutStart, carryStart, length);
}

void QEngineCPU::CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl toMulOcl = (bitCapIntOcl)toMul;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl highMask = lowMask << length;
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_mask(
        0, maxQPowerOcl, skipPowers.get(), controlLen + length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl otherRes = lcv & otherMask;
            const bitCapIntOcl mulInt = ((lcv & inOutMask) >> inOutStart) * toMulOcl;
            const bitCapIntOcl mulRes = ((mulInt & lowMask) << inOutStart) |
                (((mulInt & highMask) >> length) << carryStart) | otherRes | controlMask;
            const bitCapIntOcl origRes = lcv | controlMask;
            nStateVec->write(outFn(origRes, mulRes), stateVec->read(inFn(origRes, mulRes)));

            nStateVec->write(lcv, stateVec->read(lcv));
            bitCapIntOcl partControlMask;
            for (bitCapIntOcl j = ONE_BCI; j < pow2Mask(controlLen); j++) {
                partControlMask = 0;
                for (bitLenInt k = 0; k < controlLen; k++) {
                    if ((j >> k) & ONE_BCI) {
                        partControlMask |= controlPowers[k];
                    }
                }
                nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
            }
        });

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0);

    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }
    if (toMul == ONE_BCI) {
        return;
    }

    CMULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; }, toMul, inOutStart, carryStart, length,
        controls, controlLen);
}

void QEngineCPU::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (toDiv == 0) {
        throw std::invalid_argument("DIV by zero");
    }
    if (toDiv == ONE_BCI) {
        return;
    }

    CMULDIV([](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return mul; },
        [](const bitCapIntOcl& orig, const bitCapIntOcl& mul) { return orig; }, toDiv, inOutStart, carryStart, length,
        controls, controlLen);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bool& inverse)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl modNOcl = (bitCapIntOcl)modN;
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inMask = lowMask << inStart;
    const bitCapIntOcl outMask = lowMask << outStart;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, pow2Ocl(outStart), length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inRes = lcv & inMask;
        const bitCapIntOcl outRes = (kernelFn(inRes >> inStart) % modNOcl) << outStart;
        if (inverse) {
            nStateVec->write(lcv, stateVec->read(inRes | outRes | otherRes));
        } else {
            nStateVec->write(inRes | outRes | otherRes, stateVec->read(lcv));
        }
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    SetReg(outStart, length, 0);

    if (toMod == 0) {
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    ModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length);
}

void QEngineCPU::IMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    ModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length, true);
}

void QEngineCPU::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    ModNOut(
        [&toModOcl](const bitCapIntOcl& inInt) { return intPowOcl(toModOcl, inInt); }, modN, inStart, outStart, length);
}

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen,
    const bool& inverse)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl modNOcl = (bitCapIntOcl)modN;
    const bitCapIntOcl lowPower = pow2Ocl(length);
    const bitCapIntOcl lowMask = lowPower - ONE_BCI;
    const bitCapIntOcl inMask = lowMask << inStart;
    const bitCapIntOcl outMask = lowMask << outStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(outStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask | controlMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_mask(
        0, maxQPowerOcl, skipPowers.get(), controlLen + length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl otherRes = lcv & otherMask;
            const bitCapIntOcl inRes = lcv & inMask;
            const bitCapIntOcl outRes = (kernelFn(inRes >> inStart) % modNOcl) << outStart;

            if (inverse) {
                nStateVec->write(lcv | controlMask, stateVec->read(inRes | outRes | otherRes | controlMask));
            } else {
                nStateVec->write(inRes | outRes | otherRes | controlMask, stateVec->read(lcv | controlMask));
            }
            nStateVec->write(lcv, stateVec->read(lcv));

            for (bitCapIntOcl j = ONE_BCI; j < pow2Mask(controlLen); j++) {
                bitCapIntOcl partControlMask = 0;
                for (bitLenInt k = 0; k < controlLen; k++) {
                    if ((j >> k) & ONE_BCI) {
                        partControlMask |= controlPowers[k];
                    }
                }
                nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
            }
        });

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    SetReg(outStart, length, 0);

    CModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length,
        controls, controlLen);
}

void QEngineCPU::CIMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        IMULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    CModNOut([&toModOcl](const bitCapIntOcl& inInt) { return inInt * toModOcl; }, modN, inStart, outStart, length,
        controls, controlLen, true);
}

void QEngineCPU::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;

    CModNOut([&toModOcl](const bitCapIntOcl& inInt) { return intPowOcl(toModOcl, inInt); }, modN, inStart, outStart,
        length, controls, controlLen);
}

#if ENABLE_BCD
/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    int nibbleCount = length / 4;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toAdd %= maxPow;
    if (toAdd == 0) {
        return;
    }

    const bitCapIntOcl toAddOcl = (bitCapIntOcl)toAdd;
    const bitCapIntOcl inOutMask = bitRegMaskOcl(inOutStart, length);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl partToAdd = toAddOcl;
        bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        std::unique_ptr<int8_t[]> nibbles(new int8_t[nibbleCount]);
        bool isValid = true;
        for (int j = 0; j < nibbleCount; j++) {
            int8_t test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4UL;
            int8_t test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapIntOcl outInt = 0;
            for (int j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= (bitCapIntOcl)nibbles[j] << (j * 4U * ONE_BCI);
            }
            nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add BCD integer (without sign, with carry)
void QEngineCPU::INCDECBCDC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    CHECK_ZERO_SKIP();

    if (length == 0) {
        return;
    }

    int nibbleCount = length / 4;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    const bitCapIntOcl toModOcl = (bitCapIntOcl)toMod;
    const bitCapIntOcl inOutMask = bitRegMaskOcl(inOutStart, length);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPowerOcl, pow2Ocl(carryIndex), ONE_BCI, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl partToAdd = toModOcl;
        bitCapIntOcl inOutInt = (lcv & inOutMask) >> inOutStart;
        int test1, test2;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;

        test1 = (int)(inOutInt & 15UL);
        inOutInt >>= 4U * ONE_BCI;
        test2 = (int)(partToAdd % 10);
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (int j = 1; j < nibbleCount; j++) {
            test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4U * ONE_BCI;
            test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapIntOcl outInt = 0;
            bitCapIntOcl outRes = 0;
            bitCapIntOcl carryRes = 0;
            for (int j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= (bitCapIntOcl)nibbles[j] << (j * 4U * ONE_BCI);
            }
            outRes = (outInt << inOutStart) | otherRes | carryRes;
            nStateVec->write(outRes, stateVec->read(lcv));
            outRes ^= carryMask;
            nStateVec->write(outRes, stateVec->read(lcv | carryMask));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
            nStateVec->write(lcv | carryMask, stateVec->read(lcv | carryMask));
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}
#endif

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values, bool resetValue)
{
    if (!stateVec) {
        return 0U;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, 0);
    }

    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl skipPower = pow2Ocl(valueStart);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn;
    if (valueBytes == 1) {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)values[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else if (valueBytes == 2) {
        uint16_t* inputIntPtr = (uint16_t*)values;
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)inputIntPtr[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else if (valueBytes == 4) {
        uint32_t* inputIntPtr = (uint32_t*)values;
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            nStateVec->write(
                lcv | ((bitCapIntOcl)inputIntPtr[(lcv & inputMask) >> indexStart] << valueStart), stateVec->read(lcv));
        };
    } else {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl inputInt = (lcv & inputMask) >> indexStart;
            bitCapIntOcl outputInt = 0;
            for (bitCapIntOcl j = 0; j < valueBytes; j++) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
            bitCapIntOcl outputRes = outputInt << valueStart;
            nStateVec->write(outputRes | lcv, stateVec->read(lcv));
        };
    }

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, valueLength, fn);
    }

    ResetStateVec(nStateVec);

    real1_f average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    return (bitCapInt)(average + (real1_f)0.5f);
}

/// Add based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (!stateVec) {
        return 0U;
    }

    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are ADded with Carry (ADC) to values entangled in the "outputStart" register with the "inputStart" register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry has to first to be measured for its input value.
    bitCapIntOcl carryIn = 0;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 1;
        X(carryIndex);
    }

    Finish();

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl skipPower = pow2Ocl(carryIndex);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapIntOcl otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapIntOcl inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapIntOcl inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapIntOcl outputRes = lcv & outputMask;

        // Maintaining the entanglement, we add the classical input value
        // corresponding with the state of the "inputStart" register to
        // "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapIntOcl outputInt = 0;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (valueBytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < valueBytes; j++) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt += (outputRes >> valueStart) + carryIn;

        // If we exceed max char, we subtract 256 and entangle the carry as
        // set.
        bitCapIntOcl carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }
        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << valueStart;

        nStateVec->write(outputRes | inputRes | otherRes | carryRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, skipPower, 0), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, 1, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    real1_f average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapInt)(average + (real1_f)0.5f);
}

/// Subtract based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (!stateVec) {
        return 0U;
    }

    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are SuBtracted with Carry (SBC) from values entangled in the "outputStart" register with the "inputStart"
    // register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry (or "borrow") has to first to be measured for its input value.
    bitCapIntOcl carryIn = 1;
    if (M(carryIndex)) {
        // If the carry is set, we borrow 1 going in. We always initially clear the carry after testing for borrow in.
        carryIn = 0;
        X(carryIndex);
    }

    Finish();

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    const bitLenInt valueBytes = (valueLength + 7U) / 8U;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl skipPower = pow2Ocl(carryIndex);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapIntOcl otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapIntOcl inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapIntOcl inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapIntOcl outputRes = lcv & outputMask;

        // Maintaining the entanglement, we subtract the classical input
        // value corresponding with the state of the "inputStart" register
        // from "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapIntOcl outputInt = 0;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (valueBytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < valueBytes; j++) {
                outputInt |= (bitCapIntOcl)values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt = (outputRes >> valueStart) + (lengthPower - (outputInt + carryIn));

        // If our subtractions results in less than 0, we add 256 and
        // entangle the carry as set.  (Since we're using unsigned types,
        // we start by adding 256 with the carry, and then subtract 256 and
        // clear the carry if we don't have a borrow-out.)
        bitCapIntOcl carryRes = 0;

        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << valueStart;

        nStateVec->write(outputRes | inputRes | otherRes | carryRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, skipPower, 0), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, skipPower, valueLength, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    real1_f average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapInt)(average + (real1_f)0.5f);
}

/// Transform a length of qubit register via lookup through a hash table.
void QEngineCPU::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    CHECK_ZERO_SKIP();

    bitLenInt bytes = (length + 7U) / 8U;
    bitCapIntOcl inputMask = bitRegMaskOcl(start, length);

    Finish();

    StateVectorPtr nStateVec = AllocStateVec(maxQPowerOcl);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        const bitCapIntOcl inputRes = lcv & inputMask;
        const bitCapIntOcl inputInt = inputRes >> start;
        bitCapIntOcl outputInt = 0;
        if (bytes == 1) {
            outputInt = values[inputInt];
        } else if (bytes == 2) {
            outputInt = ((uint16_t*)values)[inputInt];
        } else if (bytes == 4) {
            outputInt = ((uint32_t*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0; j < bytes; j++) {
                outputInt |= (bitCapIntOcl)values[inputInt * bytes + j] << (8U * j);
            }
        }
        bitCapIntOcl outputRes = outputInt << start;
        nStateVec->write(outputRes | (lcv & ~inputRes), stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl input1Mask = pow2Ocl(inputBit1);
    const bitCapIntOcl input2Mask = pow2Ocl(inputBit2);
    const bitCapIntOcl carryInSumOutMask = pow2Ocl(carryInSumOut);
    const bitCapIntOcl carryOutMask = pow2Ocl(carryOut);

    bitCapIntOcl qPowers[2] = { carryInSumOutMask, carryOutMask };
    std::sort(qPowers, qPowers + 2);

    Finish();

    par_for_mask(0, maxQPowerOcl, qPowers, 2, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // Carry-in, sum bit in
        const complex ins0c0 = stateVec->read(lcv);
        const complex ins0c1 = stateVec->read(lcv | carryInSumOutMask);
        const complex ins1c0 = stateVec->read(lcv | carryOutMask);
        const complex ins1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        const bool aVal = (lcv & input1Mask) != 0;
        const bool bVal = (lcv & input2Mask) != 0;

        // Carry-out, sum bit out
        complex outs0c0, outs0c1, outs1c0, outs1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                outs0c0 = ins0c0;
                outs1c0 = ins0c1;
                // Non-coding:
                outs0c1 = ins1c0;
                outs1c1 = ins1c1;
            } else {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            }
        } else {
            if (!bVal) {
                // Coding:
                outs1c0 = ins0c0;
                outs0c1 = ins0c1;
                // Non-coding:
                outs1c1 = ins1c0;
                outs0c0 = ins1c1;
            } else {
                // Coding:
                outs0c1 = ins0c0;
                outs1c1 = ins0c1;
                // Non-coding:
                outs0c0 = ins1c0;
                outs1c0 = ins1c1;
            }
        }

        stateVec->write(lcv, outs0c0);
        stateVec->write(lcv | carryOutMask, outs0c1);
        stateVec->write(lcv | carryInSumOutMask, outs1c0);
        stateVec->write(lcv | carryInSumOutMask | carryOutMask, outs1c1);
    });
}

void QEngineCPU::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl input1Mask = pow2Ocl(inputBit1);
    const bitCapIntOcl input2Mask = pow2Ocl(inputBit2);
    const bitCapIntOcl carryInSumOutMask = pow2Ocl(carryInSumOut);
    const bitCapIntOcl carryOutMask = pow2Ocl(carryOut);

    bitCapIntOcl qPowers[2] = { carryInSumOutMask, carryOutMask };
    std::sort(qPowers, qPowers + 2);

    Finish();

    par_for_mask(0, maxQPowerOcl, qPowers, 2, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        // Carry-in, sum bit out
        const complex outs0c0 = stateVec->read(lcv);
        const complex outs0c1 = stateVec->read(lcv | carryOutMask);
        const complex outs1c0 = stateVec->read(lcv | carryInSumOutMask);
        const complex outs1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        const bool aVal = (lcv & input1Mask) != 0;
        const bool bVal = (lcv & input2Mask) != 0;

        // Carry-out, sum bit in
        complex ins0c0, ins0c1, ins1c0, ins1c1;

        if (!aVal) {
            if (!bVal) {
                // Coding:
                ins0c0 = outs0c0;
                ins0c1 = outs1c0;
                // Non-coding:
                ins1c0 = outs0c1;
                ins1c1 = outs1c1;
            } else {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            }
        } else {
            if (!bVal) {
                // Coding:
                ins0c0 = outs1c0;
                ins0c1 = outs0c1;
                // Non-coding:
                ins1c0 = outs1c1;
                ins1c1 = outs0c0;
            } else {
                // Coding:
                ins0c0 = outs0c1;
                ins0c1 = outs1c1;
                // Non-coding:
                ins1c0 = outs0c0;
                ins1c1 = outs1c0;
            }
        }

        stateVec->write(lcv, ins0c0);
        stateVec->write(lcv | carryInSumOutMask, ins0c1);
        stateVec->write(lcv | carryOutMask, ins1c0);
        stateVec->write(lcv | carryInSumOutMask | carryOutMask, ins1c1);
    });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, greaterPerm, start, length, flagIndex] {
        const bitCapIntOcl regMask = bitRegMaskOcl(start, length);
        const bitCapIntOcl flagMask = pow2Ocl(flagIndex);
        const bitCapIntOcl greaterPermOcl = (bitCapIntOcl)greaterPerm;

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if ((((lcv & regMask) >> start) < greaterPermOcl) & ((lcv & flagMask) == flagMask))
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, greaterPerm, start, length] {
        const bitCapIntOcl regMask = bitRegMaskOcl(start, length);
        const bitCapIntOcl greaterPermOcl = (bitCapIntOcl)greaterPerm;

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if (((lcv & regMask) >> start) < greaterPermOcl)
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}
#endif
}; // namespace Qrack
