//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"
#include "qfactory.hpp"

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineCPU::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if (shift == 0U || length == 0U) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    bitCapInt regMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - 1U) ^ regMask;

    complex* nStateVec = AllocStateVec(maxQPower);

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt regRes = lcv & regMask;
        bitCapInt regInt = regRes >> start;
        bitCapInt outInt = (regInt >> (length - shift)) | ((regInt << shift) & lengthMask);
        nStateVec[(outInt << start) | otherRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt lengthMask = (1U << length) - 1U;
    toAdd &= lengthMask;
    if ((length == 0U) || (toAdd == 0U)) {
        return;
    }

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
        nStateVec[(outInt << inOutStart) | otherRes] = stateVec[lcv];
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

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toAdd &= lengthMask;
    if ((length == 0U) || (toAdd == 0U)) {
        return;
    }

    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | controlMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::copy(stateVec, stateVec + maxQPower, nStateVec);

    par_for_mask(0U, maxQPower, controlPowers, controlLen, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
        nStateVec[(outInt << inOutStart) | otherRes | controlMask] = stateVec[lcv | controlMask];
    });

    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with carry)
void QEngineCPU::INCDECC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if ((length == 0U) || (toMod == 0U)) {
        return;
    }

    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = maxQPower - 1U;

    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << carryIndex, 1U, [&](const bitCapInt lcv, const int cpu) {
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
        nStateVec[outRes] = stateVec[lcv];
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
    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toAdd &= lengthMask;
    if ((length == 0U) || (toAdd == 0U)) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
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
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDECSC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if ((length == 0U) || (toMod == 0U)) {
        return;
    }

    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt inOutMask = lengthMask << inOutStart;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, carryMask, 1U, [&](const bitCapInt lcv, const int cpu) {
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
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(nStateVec);
}

/**
 * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is an
 * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QEngineCPU::INCSC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and with carry. Flip phase on overflow. Because the register length
 * is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QEngineCPU::DECSC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = (1U << length) - toSub;
    INCDECSC(invToSub, inOutStart, length, carryIndex);
}

void QEngineCPU::INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
    const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
{
    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if ((length == 0U) || (toMod == 0U)) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt signMask = 1U << (length - 1U);
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, carryMask, 1U, [&](const bitCapInt lcv, const int cpu) {
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
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(nStateVec);
}

/**
 * Add an integer to the register, with sign and with carry. If the overflow is set, flip phase on overflow. Because the
 * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
 * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
 * position before the cast.
 */
void QEngineCPU::INCSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, overflowIndex, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and with carry. If the overflow is set, flip phase on overflow.
 * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is variable.
 * Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QEngineCPU::DECSC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = (1U << length) - toSub;
    INCDECSC(invToSub, inOutStart, length, overflowIndex, carryIndex);
}

void QEngineCPU::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    SetReg(carryStart, length, 0U);

    if (toMul == 0U) {
        SetReg(inOutStart, length, 0U);
        return;
    }
    if (toMul == 1U) {
        return;
    }

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << carryStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes] =
            stateVec[lcv];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0U) {
        throw "DIV by zero (or modulo 0 to register size)";
    }
    if (toDiv == 1U) {
        return;
    }

    bitCapInt lowPower = 1U << length;
    bitCapInt lowMask = lowPower - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << carryStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = (((lcv & inOutMask) >> inOutStart) * toDiv);
        nStateVec[lcv] =
            stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length)
{
    SetReg(outStart, length, 0U);

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << outStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;
        nStateVec[inRes | outRes | otherRes] = stateVec[lcv];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0U) {
        SetReg(outStart, length, 0U);
        return;
    }

    ModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length);
}

void QEngineCPU::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    ModNOut([&toMod](const bitCapInt& inInt) { return intPow(toMod, inInt); }, modN, inStart, outStart, length);
}

void QEngineCPU::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0U);

    bitCapInt lowPower = 1U << length;
    toMul %= lowPower;
    if (toMul == 0U) {
        SetReg(inOutStart, length, 0U);
        return;
    }
    if (toMul == 1U) {
        return;
    }

    bitCapInt lowMask = lowPower - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0U; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask | controlMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_mask(0U, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes |
            controlMask] = stateVec[lcv | controlMask];

        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1U; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0U;
            for (bitLenInt k = 0U; k < controlLen; k++) {
                if (j & (1U << k)) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec[lcv | partControlMask] = stateVec[lcv | partControlMask];
        }
    });

    delete[] skipPowers;
    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

void QEngineCPU::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    bitCapInt lowPower = 1U << length;
    if ((toDiv == 0U) || (toDiv >= lowPower)) {
        throw "DIV by zero (or modulo 0 to register size)";
    }
    if (toDiv == 1U) {
        return;
    }

    bitCapInt lowMask = lowPower - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0U; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask | controlMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_mask(0U, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = (((lcv & inOutMask) >> inOutStart) * toDiv);
        nStateVec[lcv | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> length) << carryStart) | otherRes | controlMask];

        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1U; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0U;
            for (bitLenInt k = 0U; k < controlLen; k++) {
                if (j & (1U << k)) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec[lcv | partControlMask] = stateVec[lcv | partControlMask];
        }
    });

    delete[] skipPowers;
    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen)
{
    SetReg(outStart, length, 0U);

    bitCapInt lowPower = 1U << length;
    bitCapInt lowMask = lowPower - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0U; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (outStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask | controlMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_mask(0U, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = (kernelFn(inRes >> inStart) % modN) << outStart;

        nStateVec[inRes | outRes | otherRes] = stateVec[lcv | controlMask];
        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1U; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0U;
            for (bitLenInt k = 0U; k < controlLen; k++) {
                if (j & (1U << k)) {
                    partControlMask |= controlPowers[k];
                }
            }
            nStateVec[lcv | partControlMask] = stateVec[lcv | partControlMask];
        }
    });

    delete[] skipPowers;
    delete[] controlPowers;

    ResetStateVec(nStateVec);
}

void QEngineCPU::CMULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
        MULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&toMod](const bitCapInt& inInt) { return inInt * toMod; }, modN, inStart, outStart, length, controls,
        controlLen);
}

void QEngineCPU::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0U) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&toMod](const bitCapInt& inInt) { return intPow(toMod, inInt); }, modN, inStart, outStart, length,
        controls, controlLen);
}

/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1U << length) - 1U) << inOutStart;
    bitCapInt otherMask = maxQPower - 1U;
    otherMask ^= inOutMask;
    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
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
            nStateVec[(outInt << (inOutStart)) | otherRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}

/// Subtract BCD integer (without sign)
void QEngineCPU::DECBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt inOutMask = ((1U << length) - 1U) << inOutStart;
    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToSub = toAdd;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            if (test1 > 9) {
                isValid = false;
                break;
            }
            inOutInt >>= 4U;
            test2 = (partToSub % 10);
            partToSub /= 10;
            nibbles[j] = test1 - test2;
        }
        if (isValid) {
            bitCapInt outInt = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1U]--;
                    }
                }
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U);
            }
            nStateVec[(outInt << (inOutStart)) | otherRes] += stateVec[lcv];
        } else {
            nStateVec[lcv] += stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}

/// Add BCD integer (without sign, with carry)
void QEngineCPU::INCBCDC(
    bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1U << length) - 1U) << inOutStart;
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt carryMask = 1U << carryIndex;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << carryIndex, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToAdd = toAdd;
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
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
            outRes ^= carryMask;
            nStateVec[outRes] = stateVec[lcv | carryMask];
        } else {
            nStateVec[lcv] = stateVec[lcv];
            nStateVec[lcv | carryMask] = stateVec[lcv | carryMask];
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}

/// Subtract BCD integer (without sign, with carry)
void QEngineCPU::DECBCDC(
    bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1U << length) - 1U) << inOutStart;
    bitCapInt otherMask = maxQPower - 1U;
    bitCapInt carryMask = 1U << carryIndex;
    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, 1U << carryIndex, 1U, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt partToSub = toSub;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        int test1, test2;
        int j;
        int* nibbles = new int[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToSub % 10;
        partToSub /= 10;
        nibbles[0] = test1 - test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = partToSub % 10;
            partToSub /= 10;
            nibbles[j] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            bitCapInt outRes = 0;
            bitCapInt carryRes = carryMask;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    } else {
                        carryRes = 0;
                    }
                }
                outInt |= ((bitCapInt)nibbles[j]) << (j * 4U);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
            outRes ^= carryMask;
            nStateVec[outRes] = stateVec[lcv | carryMask];
        } else {
            nStateVec[lcv] = stateVec[lcv];
            nStateVec[lcv | carryMask] = stateVec[lcv | carryMask];
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineCPU::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    par_for_skip(0U, maxQPower, 1U << start, length,
        [&](const bitCapInt lcv, const int cpu) { stateVec[lcv] = -stateVec[lcv]; });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapInt regMask = ((1U << length) - 1U) << start;
    bitCapInt flagMask = 1U << flagIndex;

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((((lcv & regMask) >> start) < greaterPerm) & ((lcv & flagMask) == flagMask))
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = ((1U << length) - 1U) << start;

    par_for(0U, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    bitCapInt i, outputInt;
    SetReg(valueStart, valueLength, 0U);

    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt inputMask = ((1U << indexLength) - 1U) << indexStart;
    bitCapInt outputMask = ((1U << valueLength) - 1U) << valueStart;
    bitCapInt skipPower = 1U << valueStart;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0U, maxQPower, skipPower, valueLength, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt inputRes = lcv & inputMask;
        bitCapInt inputInt = inputRes >> indexStart;
        bitCapInt outputInt = 0U;
        for (bitLenInt j = 0U; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        bitCapInt outputRes = outputInt << valueStart;
        nStateVec[outputRes | lcv] = stateVec[lcv];
    });

    real1 prob;
    real1 totProb = ZERO_R1;
    real1 average = ZERO_R1;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    ResetStateVec(nStateVec);

    return (bitCapInt)(average + 0.5);
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
    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = 1U << valueLength;
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inputMask = ((1U << indexLength) - 1U) << indexStart;
    bitCapInt outputMask = ((1U << valueLength) - 1U) << valueStart;
    bitCapInt otherMask = (maxQPower - 1U) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1U << carryIndex;

    par_for_skip(0U, maxQPower, skipPower, 1U, [&](const bitCapInt lcv, const int cpu) {
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
        bitCapInt outputInt = 0U;
        for (bitLenInt j = 0U; j < valueBytes; j++) {
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

        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    });

    // At the end, just as a convenience, we return the expectation value for
    // the addition result.
    bitCapInt i, outputInt;
    real1 prob;
    real1 totProb = ZERO_R1;
    real1 average = ZERO_R1;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
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
    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    bitLenInt valueBytes = (valueLength + 7U) / 8U;
    bitCapInt lengthPower = 1U << valueLength;
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inputMask = ((1U << indexLength) - 1U) << indexStart;
    bitCapInt outputMask = ((1U << valueLength) - 1U) << valueStart;
    bitCapInt otherMask = (maxQPower - 1U) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1U << carryIndex;

    par_for_skip(0U, maxQPower, skipPower, 1U, [&](const bitCapInt lcv, const int cpu) {
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
        bitCapInt outputInt = 0U;
        for (bitLenInt j = 0U; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        outputInt = (outputRes >> valueStart) + (lengthPower - (outputInt + carryIn));

        // If our subtractions results in less than 0, we add 256 and
        // entangle the carry as set.  (Since we're using unsigned types,
        // we start by adding 256 with the carry, and then subtract 256 and
        // clear the carry if we don't have a borrow-out.)
        bitCapInt carryRes = 0U;

        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << valueStart;

        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    });

    // At the end, just as a convenience, we return the expectation value for
    // the addition result.
    bitCapInt i, outputInt;
    real1 prob;
    real1 totProb = ZERO_R1;
    real1 average = ZERO_R1;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}

QInterfacePtr QEngineCPU::Clone()
{
    QInterfacePtr clone = CreateQuantumInterface(
        QINTERFACE_CPU, qubitCount, 0, rand_generator, complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, true);
    clone->SetQuantumState(stateVec);
    return clone;
}
}; // namespace Qrack
