//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

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

/// Add integer (without sign, with carry)
void QEngineCPU::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;

    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt outInt = inOutInt + toAdd;
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

/// Subtract integer (without sign, with carry)
void QEngineCPU::DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;

    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt outInt = (inOutInt + lengthPower) - toSub;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | carryMask;
        }
        nStateVec[outRes] = stateVec[lcv];
    });
    ResetStateVec(nStateVec);
}

/// Add integer (without sign)
void QEngineCPU::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    toAdd %= lengthPower;
    if ((length > 0) && (toAdd > 0)) {
        bitCapInt i, j;
        bitLenInt end = start + length;
        bitCapInt startPower = 1 << start;
        bitCapInt endPower = 1 << end;
        bitCapInt iterPower = 1 << (qubitCount - end);
        bitCapInt maxLCV = iterPower * endPower;

        for (i = 0; i < startPower; i++) {
            for (j = 0; j < maxLCV; j += endPower) {
                rotate(stateVec + i + j, stateVec + ((lengthPower - toAdd) * startPower) + i + j,
                    stateVec + endPower + i + j, startPower);
            }
        }
    }
}

/// Add BCD integer (without sign)
void QEngineCPU::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
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
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
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
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
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
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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

/// Subtract integer (without sign)
void QEngineCPU::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    toSub %= lengthPower;
    if ((length > 0) && (toSub > 0)) {
        bitCapInt i, j;
        bitLenInt end = start + length;
        bitCapInt startPower = 1 << start;
        bitCapInt endPower = 1 << end;
        bitCapInt iterPower = 1 << (qubitCount - end);
        bitCapInt maxLCV = iterPower * endPower;
        for (i = 0; i < startPower; i++) {
            for (j = 0; j < maxLCV; j += endPower) {
                rotate(stateVec + i + j, stateVec + (toSub * startPower) + i + j,
                    stateVec + endPower + i + j, startPower);
            }
        }
    }
}

/// Subtract BCD integer (without sign)
void QEngineCPU::DECBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    otherMask ^= inOutMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToSub = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = (partToSub % 10);
            partToSub /= 10;
            nibbles[j] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            nStateVec[(outInt << (inOutStart)) | otherRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    }
    else {
        toSub++;
    }
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
        toSub++;
    }
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;
    otherMask ^= inOutMask | carryMask;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToSub = toSub;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15;
        test2 = partToSub % 10;
        partToSub /= 10;
        nibbles[0] = test1 - test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
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
            bitCapInt carryRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(nStateVec);
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineCPU::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    bitCapInt otherMask = (~(((1 << length) - 1) << start)) & (maxQPower - 1);
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((lcv & otherMask) == lcv)
            stateVec[lcv] = -stateVec[lcv];
    });
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

/// Set register bits to given permutation
void QEngineCPU::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (value == 1));
    } else if ((start == 0) && (length == qubitCount)) {
        double angle = Rand() * 2.0 * M_PI;

        runningNorm = 1.0;
        std::fill(stateVec, stateVec + maxQPower, complex(0.0, 0.0));
        stateVec[value] = complex(cos(angle), sin(angle));
    } else {
        bool bitVal;
        bitCapInt regVal = MReg(start, length);
        for (bitLenInt i = 0; i < length; i++) {
            bitVal = regVal & (1 << i);
            if ((bitVal && !(value & (1 << i))) || (!bitVal && (value & (1 << i))))
                X(start + i);
        }
    }
}

/// Measure permutation state of a register
bitCapInt QEngineCPU::MReg(bitLenInt start, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        if (M(start)) {
            return 1;
        } else {
            return 0;
        }
    }

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bool foundPerm;
    double prob = Rand();
    double angle = Rand() * 2.0 * M_PI;
    double cosine = cos(angle);
    double sine = sin(angle);
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    double probArray[lengthPower];
    double lowerProb, largestProb, nrmlzr;
    bitCapInt lcv, result;

    for (lcv = 0; lcv < lengthPower; lcv++) {
        probArray[lcv] = 0.0;
    }

    for (lcv = 0; lcv < maxQPower; lcv++) {
        probArray[(lcv & regMask) >> start] += norm(stateVec[lcv]);
    }

    lcv = 0;
    foundPerm = false;
    lowerProb = 0.0;
    largestProb = 0.0;
    result = lengthPower - 1;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in an bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    while ((!foundPerm) && (lcv < maxQPower)) {
        if ((probArray[lcv] + lowerProb) > prob) {
            foundPerm = true;
            result = lcv;
            nrmlzr = probArray[lcv];
        } else {
            if (largestProb <= probArray[lcv]) {
                largestProb = probArray[lcv];
                result = lcv;
                nrmlzr = largestProb;
            }
            lowerProb += probArray[lcv];
            lcv++;
        }
    }

    bitCapInt resultPtr = result << start;
    complex nrm = complex(cosine, sine) / nrmlzr;

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((lcv & resultPtr) == resultPtr) {
            stateVec[lcv] = nrm * stateVec[lcv];
        } else {
            stateVec[lcv] = complex(0.0, 0.0);
        }
    });

    UpdateRunningNorm();

    return result;
}

/// Set 8 bit register bits based on read from classical memory
bitCapInt QEngineCPU::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    bitCapInt i, outputInt;
    SetReg(valueStart, valueLength, 0);

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt skipPower = 1 << valueStart;

    complex* nStateVec = AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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

    double prob, average, totProb;
    totProb = 0.0;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    average /= totProb;

    ResetStateVec(nStateVec);

    return (bitCapInt)(average + 0.5);
}

/// Add based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{

    // This a quantum/classical interface method, similar to SuperposeReg8.
    // Like SuperposeReg8, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    double prob, average, totProb;
    totProb = 0.0;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    average /= totProb;

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}

/// Subtract based on an indexed load from classical memory
bitCapInt QEngineCPU::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    // This a quantum/classical interface method, similar to SuperposeReg8.
    // Like SuperposeReg8, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
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
    std::fill(nStateVec, nStateVec + maxQPower, complex(0.0, 0.0));

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
    double prob, average, totProb;
    totProb = 0.0;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    average /= totProb;

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}

};
