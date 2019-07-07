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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    bitCapInt regMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - 1U) ^ regMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt regRes = lcv & regMask;
        bitCapInt regInt = regRes >> start;
        bitCapInt outInt = (regInt >> (length - shift)) | ((regInt << shift) & lengthMask);
        nStateVec->write((outInt << start) | otherRes, stateVec->read(lcv));
    });
    ResetStateVec(nStateVec);
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthMask = (1U << length) - 1U;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ inOutMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
        nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
    });
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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->copy(*stateVec);

    par_for_mask(0, maxQPower, controlPowers, controlLen, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = maxQPower - 1U;

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, 1U << carryIndex, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toAdd &= lengthMask;
    if (toAdd == 0) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ inOutMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes;
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & lengthMask) + 1U;
            inInt = ((~inInt) & lengthMask) + 1U;
            if ((inOutInt + inInt) > signMask) {
                isOverflow = true;
            }
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask) {
                isOverflow = true;
            }
        }
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec->write(outRes, -stateVec->read(lcv));
        } else {
            nStateVec->write(outRes, stateVec->read(lcv));
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt inOutMask = lengthMask << inOutStart;

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, carryMask, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt inInt = toMod;
        bitCapInt outInt = inOutInt + toMod;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & lengthMask) + 1U;
            inInt = ((~inInt) & lengthMask) + 1U;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, carryMask, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt inInt = toMod;
        bitCapInt outInt = inOutInt + toMod;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & lengthMask) + 1U;
            inInt = ((~inInt) & lengthMask) + 1U;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
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
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, 1U << carryStart, length, [&](const bitCapInt lcv, const int cpu) {
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
    if (toMul == 1U) {
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
    if (toDiv == 1U) {
        return;
    }

    MULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return mul; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return orig; }, toDiv, inOutStart, carryStart, length);
}

void QEngineCPU::CMULDIV(const IOFn& inFn, const IOFn& outFn, const bitCapInt& toMul, const bitLenInt& inOutStart,
    const bitLenInt& carryStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt mulInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        bitCapInt mulRes = ((mulInt & lowMask) << inOutStart) | (((mulInt & highMask) >> length) << carryStart) |
            otherRes | controlMask;
        bitCapInt origRes = lcv | controlMask;
        nStateVec->write(outFn(origRes, mulRes), stateVec->read(inFn(origRes, mulRes)));

        nStateVec->write(lcv, stateVec->read(lcv));
        bitCapInt partControlMask;
        for (bitCapInt j = 1U; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
                if (j & (1U << k)) {
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
    if (toMul == 1U) {
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
    if (toDiv == 1U) {
        return;
    }

    CMULDIV([](const bitCapInt& orig, const bitCapInt& mul) { return mul; },
        [](const bitCapInt& orig, const bitCapInt& mul) { return orig; }, toDiv, inOutStart, carryStart, length,
        controls, controlLen);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length)
{
    SetReg(outStart, length, 0);

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, 1U << outStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;
        nStateVec->write(inRes | outRes | otherRes, stateVec->read(lcv));
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0) {
        SetReg(outStart, length, 0);
        return;
    }

    ModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length);
}

void QEngineCPU::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    ModNOut([&toMod](const bitCapInt& inInt) { return intPow(toMod, inInt); }, modN, inStart, outStart, length);
}

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen)
{
    SetReg(outStart, length, 0);

    bitCapInt lowPower = 1U << length;
    bitCapInt lowMask = lowPower - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (outStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask | controlMask);

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;

        nStateVec->write(inRes | outRes | otherRes, stateVec->read(lcv | controlMask));
        nStateVec->write(lcv, stateVec->read(lcv));

        bitCapInt partControlMask;
        for (bitCapInt j = 1U; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
                if (j & (1U << k)) {
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

    CModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length, controls,
        controlLen);
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
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toAdd %= maxPow;
    if (toAdd == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(inOutStart, length);
    bitCapInt otherMask = maxQPower - 1U;
    otherMask ^= inOutMask;
    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = (partToAdd % 10);
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
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U);
            }
            nStateVec->write((outInt << inOutStart) | otherRes, stateVec->read(lcv));
        } else {
            nStateVec->write(lcv, stateVec->read(lcv));
        }
        delete[] nibbles;
    });
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
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(inOutStart, length);
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt carryMask = 1U << carryIndex;

    otherMask ^= inOutMask | carryMask;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    par_for_skip(0, maxQPower, 1U << carryIndex, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToAdd = toMod;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = partToAdd % 10;
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
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U);
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
bitCapInt QEngineCPU::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    SetReg(valueStart, valueLength, 0);

    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt skipPower = 1U << valueStart;

    StateVectorPtr nStateVec = AllocStateVec(maxQPower);
    nStateVec->clear();

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bitCapInt inputRes = lcv & inputMask;
        bitCapInt inputInt = inputRes >> indexStart;
        bitCapInt outputInt = 0;
        for (bitLenInt j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        bitCapInt outputRes = outputInt << valueStart;
        nStateVec->write(outputRes | lcv, stateVec->read(lcv));
    };

    if (stateVec->is_sparse()) {
        par_for_set(stateVec->iterable(0, bitRegMask(valueStart, valueLength), 0), fn);
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

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = 1U << valueLength;
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    bitCapInt otherMask = (maxQPower - 1U) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1U << carryIndex;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapInt inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapInt inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & outputMask;

        // Maintaining the entanglement, we add the classical input value
        // corresponding with the state of the "inputStart" register to
        // "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = 0;
        for (bitLenInt j = 0; j < valueBytes; j++) {
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
        par_for_set(stateVec->iterable(0, skipPower, 0), fn);
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

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = 1U << valueLength;
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    bitCapInt otherMask = (maxQPower - 1U) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1U << carryIndex;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & otherMask;

        // These are bits that index the classical memory we're loading from:
        bitCapInt inputRes = lcv & inputMask;

        // If we read these as a char type, this is their value as a char:
        bitCapInt inputInt = inputRes >> indexStart;

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & outputMask;

        // Maintaining the entanglement, we subtract the classical input
        // value corresponding with the state of the "inputStart" register
        // from "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = 0;
        for (bitLenInt j = 0; j < valueBytes; j++) {
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
        par_for_set(stateVec->iterable(0, skipPower, 0), fn);
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

}; // namespace Qrack
