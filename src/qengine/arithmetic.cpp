//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineCPU::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    shift %= length;
    if (shift == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    bitCapInt regMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ regMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt regInt = (lcv & regMask) >> start;
        bitCapInt outInt = (regInt >> (length - shift)) | ((regInt << shift) & lengthMask);
        nStateVec->write((outInt << start) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthMask = pow2Mask(length);
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ inOutMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with controls)
void QEngineCPU::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inOutMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->copy(stateVec);
    stateVec->isReadLocked = false;

    par_for_mask(0, maxQPower, controlPowers, controlLen, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes | controlMask, stateVec->read(lcv | controlMask));
    });

    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with carry)
void QEngineCPU::INCDECC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt carryMask = pow2(carryIndex);
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = maxQPower - ONE_BCI;

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, pow2(carryIndex), ONE_BCI, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt outInt = inOutInt + toMod;
        bitCapInt outRes;
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
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt overflowMask = pow2(overflowIndex);
    bitCapInt signMask = pow2(length - ONE_BCI);
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ inOutMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes;
        }
        bool isOverflow = isOverflowAdd(inOutInt, toAdd, signMask, lengthPower);
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt signMask = pow2(length - ONE_BCI);
    bitCapInt carryMask = pow2(carryIndex);
    bitCapInt otherMask = maxQPower - ONE_BCI;
    bitCapInt inOutMask = lengthMask << inOutStart;

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, carryMask, ONE_BCI, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt inInt = toMod;
        bitCapInt outInt = inOutInt + toMod;
        bitCapInt outRes;
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

void QEngineCPU::INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
    const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = pow2(length);
    bitCapInt lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt overflowMask = pow2(overflowIndex);
    bitCapInt signMask = pow2(length - ONE_BCI);
    bitCapInt carryMask = pow2(carryIndex);
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inOutMask | carryMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, carryMask, ONE_BCI, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        bitCapInt inInt = toMod;
        bitCapInt outInt = inOutInt + toMod;
        bitCapInt outRes;
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
    bitCapInt lowMask = pow2Mask(length);
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inOutMask | carryMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, pow2(carryStart), length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt mulInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        bitCapInt mulRes =
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

    MULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return orig; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return mul; }, toMul, inOutStart, carryStart, length);
}

void QEngineCPU::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
        throw "DIV by zero";
    }
    if (toDiv == ONE_BCI) {
        return;
    }

    MULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return mul; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return orig; }, toDiv, inOutStart, carryStart, length);
}

void QEngineCPU::CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lowMask = pow2Mask(length);
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2(carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inOutMask | carryMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt mulInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        bitCapInt mulRes = ((mulInt & lowMask) << inOutStart) | (((mulInt & highMask) >> length) << carryStart) |
            otherRes | controlMask;
        bitCapInt origRes = lcv | controlMask;
        nStateVec->write(outFn(origRes, mulRes), stateVec->read(inFn(origRes, mulRes)));

        nStateVec->write(lcv, stateVec->read(lcv));
        bitCapInt partControlMask;
        for (bitCapInt j = ONE_BCI; j < pow2Mask(controlLen); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
                if ((j >> k) & ONE_BCI) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
        }
    });

    delete[] skipPowers;
    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
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

    CMULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return orig; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return mul; }, toMul, inOutStart, carryStart, length,
        controls, controlLen);
}

void QEngineCPU::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (toDiv == 0) {
        throw "DIV by zero";
    }
    if (toDiv == ONE_BCI) {
        return;
    }

    CMULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return mul; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return orig; }, toDiv, inOutStart, carryStart, length,
        controls, controlLen);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bool& inverse)
{
    bitCapInt lowMask = pow2Mask(length);
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;
    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inMask | outMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, pow2(outStart), length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;
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

    ModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length);
}

void QEngineCPU::IMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0) {
        return;
    }

    ModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length, true);
}

void QEngineCPU::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    ModNOut([&toMod](const bitCapInt& inInt) { return intPow(toMod, inInt); }, modN, inStart, outStart, length);
}

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen,
    const bool& inverse)
{
    bitCapInt lowPower = pow2(length);
    bitCapInt lowMask = lowPower - ONE_BCI;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2(outStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - ONE_BCI) ^ (inMask | outMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;

        if (inverse) {
            nStateVec->write(lcv | controlMask, stateVec->read(inRes | outRes | otherRes | controlMask));
        } else {
            nStateVec->write(inRes | outRes | otherRes | controlMask, stateVec->read(lcv | controlMask));
        }
        nStateVec->write(lcv, stateVec->read(lcv));

        bitCapInt partControlMask;
        for (bitCapInt j = ONE_BCI; j < pow2Mask(controlLen); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
                if ((j >> k) & ONE_BCI) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec->write(lcv | partControlMask, stateVec->read(lcv | partControlMask));
        }
    });

    delete[] skipPowers;
    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0);

    CModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length, controls,
        controlLen);
}

void QEngineCPU::CIMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        IMULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length, controls,
        controlLen, true);
}

void QEngineCPU::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&toMod](const bitCapInt& inInt) { return intPow(toMod, inInt); }, modN, inStart, outStart, length,
        controls, controlLen);
}

/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    int nibbleCount = length / 4;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toAdd %= maxPow;
    if (toAdd == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(inOutStart, length);
    bitCapInt otherMask = maxQPower - ONE_BCI;
    otherMask ^= inOutMask;
    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4UL;
            test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U * ONE_BCI);
            }
            nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
        }
        delete[] nibbles;
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    ResetStateVec(nStateVec);
}

/// Add BCD integer (without sign, with carry)
void QEngineCPU::INCDECBCDC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    if (length == 0) {
        return;
    }

    int nibbleCount = length / 4;
    if (nibbleCount * 4 != (int)length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(inOutStart, length);
    bitCapInt otherMask = maxQPower - ONE_BCI;
    bitCapInt carryMask = pow2(carryIndex);

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    par_for_skip(0, maxQPower, pow2(carryIndex), ONE_BCI, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToAdd = toMod;
        bitCapInt inOutInt = (lcv & inOutMask) >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;

        test1 = (int)(inOutInt & 15UL);
        inOutInt >>= 4U * ONE_BCI;
        test2 = (int)(partToAdd % 10);
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = (int)(inOutInt & 15UL);
            inOutInt >>= 4U * ONE_BCI;
            test2 = (int)(partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if ((test1 > 9) || (test2 > 9)) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            bitCapInt outRes = 0;
            bitCapInt carryRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U * ONE_BCI);
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

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, unsigned char* values, bool resetValue)
{
    if (resetValue) {
        SetReg(valueStart, valueLength, 0);
    }

    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt skipPower = pow2(valueStart);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn;
    if (valueBytes == 1) {
        fn = [&](const bitCapInt lcv, const int cpu) {
            nStateVec->write(
                lcv | (values[(bitCapIntOcl)((lcv & inputMask) >> indexStart)] << valueStart), stateVec->read(lcv));
        };
    } else {
        fn = [&](const bitCapInt lcv, const int cpu) {
            bitCapIntOcl inputInt = (bitCapIntOcl)((lcv & inputMask) >> indexStart);
            bitCapInt outputInt = 0;
            for (bitCapIntOcl j = 0; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
            bitCapInt outputRes = outputInt << valueStart;
            nStateVec->write(outputRes | lcv, stateVec->read(lcv));
        };
    }

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for_skip(0, maxQPower, skipPower, valueLength, fn);
    }

    ResetStateVec(nStateVec);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    return (bitCapInt)(average + (ONE_R1 / 2));
}

/// Add based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{

    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are ADded with Carry (ADC) to values entangled in the "outputStart" register with the "inputStart" register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry has to first to be measured for its input value.
    bitCapInt carryIn = 0;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 1;
        X(carryIndex);
    }

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = pow2(valueLength);
    bitCapInt carryMask = pow2(carryIndex);
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    bitCapInt otherMask = (maxQPower - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = pow2(carryIndex);

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapInt inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapIntOcl inputInt = (bitCapIntOcl)(inputRes >> indexStart);

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & outputMask;

        // Maintaining the entanglement, we add the classical input value
        // corresponding with the state of the "inputStart" register to
        // "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = 0;
        for (bitCapIntOcl j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        outputInt += (outputRes >> valueStart) + carryIn;

        // If we exceed max char, we subtract 256 and entangle the carry as
        // set.
        bitCapInt carryRes = 0;
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
        par_for_skip(0, maxQPower, skipPower, 1, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapInt)(average + (ONE_R1 / 2));
}

/// Subtract based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    // This a quantum/classical interface method, similar to IndexedLDA.
    // Like IndexedLDA, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are SuBtracted with Carry (SBC) from values entangled in the "outputStart" register with the "inputStart"
    // register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry (or "borrow") has to first to be measured for its input value.
    bitCapInt carryIn = 1;
    if (M(carryIndex)) {
        // If the carry is set, we borrow 1 going in. We always initially clear the carry after testing for borrow in.
        carryIn = 0;
        X(carryIndex);
    }

    // We calloc a new stateVector for output.
    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = pow2(valueLength);
    bitCapInt carryMask = pow2(carryIndex);
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    bitCapInt otherMask = (maxQPower - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = pow2(carryIndex);

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapInt inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapIntOcl inputInt = (bitCapIntOcl)(inputRes >> indexStart);

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & outputMask;

        // Maintaining the entanglement, we subtract the classical input
        // value corresponding with the state of the "inputStart" register
        // from "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = 0;
        for (bitCapIntOcl j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        outputInt = (outputRes >> valueStart) + (lengthPower - (outputInt + carryIn));

        // If our subtractions results in less than 0, we add 256 and
        // entangle the carry as set.  (Since we're using unsigned types,
        // we start by adding 256 with the carry, and then subtract 256 and
        // clear the carry if we don't have a borrow-out.)
        bitCapInt carryRes = 0;

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
        par_for_skip(0, maxQPower, skipPower, valueLength, fn);
    }

    // We dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapInt)(average + (ONE_R1 / 2));
}

/// Transform a length of qubit register via lookup through a hash table.
void QEngineCPU::Hash(bitLenInt start, bitLenInt length, unsigned char* values)
{
    bitLenInt bytes = (length + 7U) / 8U;
    bitCapInt inputMask = bitRegMask(start, length);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt inputRes = lcv & inputMask;
        bitCapIntOcl inputInt = (bitCapIntOcl)(inputRes >> start);
        bitCapInt outputInt = 0;
        for (bitCapIntOcl j = 0; j < bytes; j++) {
            outputInt |= values[inputInt * bytes + j] << (8U * j);
        }
        bitCapInt outputRes = outputInt << start;
        nStateVec->write(outputRes | (lcv & ~inputRes), stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }

    ResetStateVec(nStateVec);
}

void QEngineCPU::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    bitCapInt input1Mask = pow2(inputBit1);
    bitCapInt input2Mask = pow2(inputBit2);
    bitCapInt carryInSumOutMask = pow2(carryInSumOut);
    bitCapInt carryOutMask = pow2(carryOut);

    bitCapInt qPowers[2] = { carryInSumOutMask, carryOutMask };
    std::sort(qPowers, qPowers + 2);

    par_for_mask(0, maxQPower, qPowers, 2, [&](const bitCapInt lcv, const int cpu) {

        // Carry-in, sum bit in
        complex ins0c0 = stateVec->read(lcv);
        complex ins0c1 = stateVec->read(lcv | carryInSumOutMask);
        complex ins1c0 = stateVec->read(lcv | carryOutMask);
        complex ins1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        bool aVal = (lcv & input1Mask) != 0;
        bool bVal = (lcv & input2Mask) != 0;

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
    bitCapInt input1Mask = pow2(inputBit1);
    bitCapInt input2Mask = pow2(inputBit2);
    bitCapInt carryInSumOutMask = pow2(carryInSumOut);
    bitCapInt carryOutMask = pow2(carryOut);

    bitCapInt qPowers[2] = { carryInSumOutMask, carryOutMask };
    std::sort(qPowers, qPowers + 2);

    par_for_mask(0, maxQPower, qPowers, 2, [&](const bitCapInt lcv, const int cpu) {

        // Carry-in, sum bit out
        complex outs0c0 = stateVec->read(lcv);
        complex outs0c1 = stateVec->read(lcv | carryOutMask);
        complex outs1c0 = stateVec->read(lcv | carryInSumOutMask);
        complex outs1c1 = stateVec->read(lcv | carryInSumOutMask | carryOutMask);

        bool aVal = (lcv & input1Mask) != 0;
        bool bVal = (lcv & input2Mask) != 0;

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

}; // namespace Qrack
