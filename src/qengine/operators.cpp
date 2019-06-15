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
    if (shift == 0 || length == 0) {
        return;
    }

    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask |= 1 << (start + i);
    }
    otherMask -= regMask;

    complex* nStateVec = AllocStateVec(maxQPower);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt regRes = lcv & regMask;
        bitCapInt regInt = regRes >> start;
        bitCapInt outInt = (regInt >> (length - shift)) | ((regInt << shift) & (lengthPower - 1));
        nStateVec[(outInt << start) + otherRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QEngineCPU::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (shift == 0 || length == 0) {
        return;
    }

    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;

    for (i = 0; i < length; i++) {
        regMask |= 1 << (start + i);
    }
    otherMask -= regMask;

    complex* nStateVec = AllocStateVec(maxQPower);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt regRes = lcv & regMask;
        bitCapInt regInt = regRes >> start;
        bitCapInt outInt = (regInt >> shift) | ((regInt << (length - shift)) & (lengthPower - 1));
        nStateVec[(outInt << start) + otherRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

void QEngineCPU::INCDEC(
    const MFn& kernelFn, const bitCapInt& toAdd, const bitLenInt& inOutStart, const bitLenInt& length)
{
    bitCapInt lengthMask = (1U << length) - 1U;
    bitCapInt inOutMask = lengthMask << inOutStart;
    bitCapInt otherMask = (1U << qubitCount) - 1U;

    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inOutRes = lcv & inOutMask;
        bitCapInt inOutInt = inOutRes >> inOutStart;
        bitCapInt outInt = kernelFn(inOutInt) & lengthMask;
        nStateVec[(outInt << inOutStart) | otherRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    INCDEC([&toAdd](const bitCapInt& inOutInt) { return inOutInt + toAdd; }, toAdd, inOutStart, length);
}

/// Subtract integer (without sign)
void QEngineCPU::DEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    INCDEC([&toSub](const bitCapInt& inOutInt) { return inOutInt - toSub; }, toSub, inOutStart, length);
}

void QEngineCPU::INCDECC(
    const MFn& kernelFn, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;

    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt outInt = kernelFn(inOutInt);
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        nStateVec[outRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

/// Add integer (without sign, with carry)
void QEngineCPU::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECC([&toAdd](const bitCapInt& inOutInt) { return inOutInt + toAdd; }, inOutStart, length, carryIndex);
}

/// Subtract integer (without sign, with carry)
void QEngineCPU::DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt lengthPower = 1 << length;

    INCDECC([&toSub, &lengthPower](const bitCapInt& inOutInt) { return (inOutInt + lengthPower) - toSub; }, inOutStart,
        length, carryIndex);
}

/// Add integer (without sign, with controls)
void QEngineCPU::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    bitCapInt lengthPower = 1 << length;
    bitCapInt lengthMask = lengthPower - 1;
    toAdd &= lengthMask;
    if ((length > 0) && (toAdd > 0)) {
        bitCapInt inOutMask = lengthMask << inOutStart;
        bitCapInt otherMask = maxQPower - 1;

        bitCapInt* controlPowers = new bitCapInt[controlLen];
        bitCapInt controlMask = 0U;
        for (bitLenInt i = 0; i < controlLen; i++) {
            controlPowers[i] = 1U << controls[i];
            controlMask |= controlPowers[i];
        }
        std::sort(controlPowers, controlPowers + controlLen);

        otherMask ^= inOutMask | controlMask;

        complex* nStateVec = AllocStateVec(maxQPower);
        std::copy(stateVec, stateVec + maxQPower, nStateVec);

        par_for_mask(0, maxQPower, controlPowers, controlLen, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = lcv & otherMask;
            bitCapInt inOutRes = lcv & inOutMask;
            bitCapInt inOutInt = inOutRes >> inOutStart;
            bitCapInt outInt = (inOutInt + toAdd) & lengthMask;
            nStateVec[(outInt << inOutStart) | otherRes | controlMask] = stateVec[lcv | controlMask];
        });

        delete[] controlPowers;

        ResetStateVec(nStateVec);
    }
}

/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

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
            nStateVec[(outInt << (inOutStart)) | otherRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
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
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
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

/**
 * Add an integer to the register, with sign and without carry. Because the
 * register length is an arbitrary number of bits, the sign bit position on the
 * integer to add is variable. Hence, the integer to add is specified as cast
 * to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QEngineCPU::INCS(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << (inOutStart)) | otherRes;
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
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
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
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
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
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

/// Add integer (without sign, with controls)
void QEngineCPU::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DEC(toSub, inOutStart, length);
        return;
    }

    bitCapInt lengthPower = 1 << length;
    bitCapInt lengthMask = lengthPower - 1;
    toSub &= lengthMask;
    if ((length > 0) && (toSub > 0)) {
        bitCapInt inOutMask = lengthMask << inOutStart;
        bitCapInt otherMask = maxQPower - 1;

        bitCapInt* controlPowers = new bitCapInt[controlLen];
        bitCapInt controlMask = 0U;
        for (bitLenInt i = 0; i < controlLen; i++) {
            controlPowers[i] = 1U << controls[i];
            controlMask |= controlPowers[i];
        }
        std::sort(controlPowers, controlPowers + controlLen);

        otherMask ^= inOutMask | controlMask;

        complex* nStateVec = AllocStateVec(maxQPower);
        std::copy(stateVec, stateVec + maxQPower, nStateVec);

        par_for_mask(0, maxQPower, controlPowers, controlLen, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = lcv & otherMask;
            bitCapInt inOutRes = lcv & inOutMask;
            bitCapInt inOutInt = inOutRes >> inOutStart;
            bitCapInt outInt = ((lengthPower + inOutInt) - toSub) & lengthMask;
            nStateVec[(outInt << inOutStart) | otherRes | controlMask] = stateVec[lcv | controlMask];
        });

        delete[] controlPowers;

        ResetStateVec(nStateVec);
    }
}

/// Subtract BCD integer (without sign)
void QEngineCPU::DECBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    int nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
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

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QEngineCPU::DECS(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = overflowMask;
        bitCapInt outInt = inOutInt - toSub + lengthPower;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << (inOutStart)) | otherRes;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toSub;
        bitCapInt outInt = (inOutInt - toSub) + (lengthPower);
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toSub;
        bitCapInt outInt = (inOutInt - toSub) + (lengthPower);
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;
    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
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

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt highMask = lowMask << length;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, 1U << carryStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes] =
            stateVec[lcv];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
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

    par_for_skip(0, maxQPower, 1U << carryStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = (((lcv & inOutMask) >> inOutStart) * toDiv);
        nStateVec[lcv] =
            stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::ModNOut(const MFn& kernelFn, const bitCapInt& toMod, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length)
{
    SetReg(outStart, length, 0);

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask);

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, 1U << outStart, length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = kernelFn(inRes);
        nStateVec[inRes | outRes | otherRes] = stateVec[lcv];
    });

    ResetStateVec(nStateVec);
}

void QEngineCPU::MULModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0) {
        SetReg(outStart, length, 0);
        return;
    }

    ModNOut([&inStart, &outStart, &toMod, &modN](
                const bitCapInt& inRes) { return (((inRes >> inStart) * toMod) % modN) << outStart; },
        toMod, modN, inStart, outStart, length);
}

void QEngineCPU::POWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    ModNOut([&inStart, &outStart, &toMod, &modN](
                const bitCapInt& inRes) { return (intPow(toMod, inRes >> inStart) % modN) << outStart; },
        toMod, modN, inStart, outStart, length);
}

void QEngineCPU::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0);

    bitCapInt lowPower = 1U << length;
    toMul %= lowPower;
    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
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

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = ((lcv & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> length) << carryStart) | otherRes |
            controlMask] = stateVec[lcv | controlMask];

        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
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
    if (controlLen == 0) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    bitCapInt lowPower = 1U << length;
    if ((toDiv == 0) || (toDiv >= lowPower)) {
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

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt outInt = (((lcv & inOutMask) >> inOutStart) * toDiv);
        nStateVec[lcv | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> length) << carryStart) | otherRes | controlMask];

        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
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

void QEngineCPU::CModNOut(const MFn& kernelFn, const bitCapInt& toMod, const bitCapInt& modN, const bitLenInt& inStart,
    const bitLenInt& outStart, const bitLenInt& length, const bitLenInt* controls, const bitLenInt& controlLen)
{
    SetReg(outStart, length, 0);

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

    par_for_mask(0, maxQPower, skipPowers, controlLen + length, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = lcv & otherMask;
        bitCapInt inRes = lcv & inMask;
        bitCapInt outRes = kernelFn(inRes);

        nStateVec[inRes | outRes | otherRes] = stateVec[lcv | controlMask];
        nStateVec[lcv] = stateVec[lcv];

        bitCapInt partControlMask;
        for (bitCapInt j = 1; j < ((1U << controlLen) - 1U); j++) {
            partControlMask = 0;
            for (bitLenInt k = 0; k < controlLen; k++) {
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
    if (controlLen == 0) {
        MULModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&inStart, &outStart, &toMod, &modN](
                 const bitCapInt& inRes) { return (((inRes >> inStart) * toMod) % modN) << outStart; },
        toMod, modN, inStart, outStart, length, controls, controlLen);
}

void QEngineCPU::CPOWModNOut(bitCapInt toMod, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        POWModNOut(toMod, modN, inStart, outStart, length);
        return;
    }

    CModNOut([&inStart, &outStart, &toMod, &modN](
                 const bitCapInt& inRes) { return (intPow(toMod, inRes >> inStart) % modN) << outStart; },
        toMod, modN, inStart, outStart, length, controls, controlLen);
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineCPU::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    par_for_skip(
        0, maxQPower, 1 << start, length, [&](const bitCapInt lcv, const int cpu) { stateVec[lcv] = -stateVec[lcv]; });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt flagMask = 1 << flagIndex;

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((((lcv & regMask) >> start) < greaterPerm) & ((lcv & flagMask) == flagMask))
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = ((1 << length) - 1) << start;

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    bitCapInt i, outputInt;
    SetReg(valueStart, valueLength, 0);

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt skipPower = 1 << valueStart;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    par_for_skip(0, maxQPower, skipPower, valueLength, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt inputRes = lcv & inputMask;
        bitCapInt inputInt = inputRes >> indexStart;
        bitCapInt outputInt = 0;
        for (bitLenInt j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
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
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt lengthPower = 1 << valueLength;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1 << carryIndex;

    par_for_skip(0, maxQPower, skipPower, 1, [&](const bitCapInt lcv, const int cpu) {
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
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
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
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt lengthPower = 1 << valueLength;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt skipPower = 1 << carryIndex;

    par_for_skip(0, maxQPower, skipPower, 1, [&](const bitCapInt lcv, const int cpu) {
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
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
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
