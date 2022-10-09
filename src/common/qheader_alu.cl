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

void kernel inc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[4];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    for (bitCapIntOcl i = ID; i < maxI; i += Nthreads) {
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | (i & otherMask)] =
            stateVec[i];
    }
}

void kernel cinc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[4];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[7];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < controlLen; p++) {
            bitCapIntOcl iLow = iHigh & (controlPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        bitCapIntOcl otherRes = i & otherMask;
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | otherRes | controlMask] =
            stateVec[i | controlMask];
    }
}

void kernel incdecc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    const bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toMod = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl outInt = (inOutRes >> inOutStart) + toMod;
        bitCapIntOcl outRes = 0U;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = carryMask;
        }
        outRes |= outInt << inOutStart;
        nStateVec[outRes | otherRes] = stateVec[i];
    }
}

void kernel incs(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    const bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        const bitCapIntOcl inOutRes = lcv & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? (outRes = (outInt << inOutStart) | otherRes) : (((outInt - lengthPower) << inOutStart) | otherRes);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
            inInt = ((~inInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
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
        cmplx amp = stateVec[lcv];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incdecsc1(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    const bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[7];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? (outRes = (outInt << inOutStart) | otherRes) : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
            inInt = ((~inInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        cmplx amp = stateVec[i];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incdecsc2(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    const bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    const bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inOutRes = i & inOutMask;
        bitCapIntOcl inOutInt = inOutRes >> inOutStart;
        const bitCapIntOcl outInt = inOutInt + toAdd;
        bitCapIntOcl outRes = (outInt < lengthPower) ? ((outInt << inOutStart) | otherRes) : (((outInt - lengthPower) << inOutStart) | otherRes | carryMask);
        bitCapIntOcl inInt = toAdd;

        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
            inInt = ((~inInt) & (lengthPower - ONE_BCI)) + ONE_BCI;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & signMask) {
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        cmplx amp = stateVec[i];
        if (isOverflow) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel mul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt carryStart = bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes] =
            stateVec[i];
    }
}

void kernel div(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt carryStart = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toDiv;
        nStateVec[i] =
            stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes];
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define MODNOUT(indexIn, indexOut)                                                                                     \
    const bitCapIntOcl Nthreads = get_global_size(0);                                                                  \
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                      \
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                     \
    const bitCapIntOcl inMask = bitCapIntOclPtr[2];                                                                    \
    /* bitCapIntOcl outMask = bitCapIntOclPtr[3]; */                                                                   \
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];                                                                 \
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];                                                               \
    /* bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI; */                                                           \
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[6];                                                           \
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[7];                                                          \
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];                                                                  \
    const bitCapIntOcl modN = bitCapIntOclPtr[9];                                                                      \
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {                                                         \
        const bitCapIntOcl iHigh = lcv;                                                                                \
        const bitCapIntOcl iLow = iHigh & skipMask;                                                                    \
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;                                                           \
                                                                                                                       \
        const bitCapIntOcl otherRes = i & otherMask;                                                                   \
        const bitCapIntOcl inRes = i & inMask;                                                                         \
        const bitCapIntOcl outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                 \
        nStateVec[indexOut] = stateVec[indexIn];                                                                       \
    }

void kernel mulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    MODNOUT(i, (inRes | outRes | otherRes));
}

void kernel imulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    MODNOUT((inRes | outRes | otherRes), i);
}

void kernel powmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl base = bitCapIntOclPtr[1];
    const bitCapIntOcl inMask = bitCapIntOclPtr[2];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[5];
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[6];
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    const bitCapIntOcl modN = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & skipMask;
        const bitCapIntOcl i = iLow | (iHigh ^ iLow) << len;

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inRes = i & inMask;
        const bitCapIntOcl inInt = inRes >> inStart;

        bitCapIntOcl powRes = base;
        if (inInt == 0) {
            powRes = 1;
        } else {
            for (bitCapIntOcl pw = 1; pw < inInt; pw++) {
                powRes *= base;
            }
        }

        const bitCapIntOcl outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes] = stateVec[i];
    }
}

void kernel fulladd(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    const bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    const bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    const bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - ONE_BCI;
        qMask2 = carryOutMask - ONE_BCI;
    } else {
        qMask1 = carryOutMask - ONE_BCI;
        qMask2 = carryInSumOutMask - ONE_BCI;
    }

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit in
        const cmplx ins0c0 = stateVec[i];
        const cmplx ins0c1 = stateVec[i | carryInSumOutMask];
        const cmplx ins1c0 = stateVec[i | carryOutMask];
        const cmplx ins1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        const bool aVal = (i & input1Mask);
        const bool bVal = (i & input2Mask);

        cmplx outs0c0, outs0c1, outs1c0, outs1c1;

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

        stateVec[i] = outs0c0;
        stateVec[i | carryOutMask] = outs0c1;
        stateVec[i | carryInSumOutMask] = outs1c0;
        stateVec[i | carryInSumOutMask | carryOutMask] = outs1c1;
    }
}

void kernel ifulladd(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    const bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    const bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    const bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - ONE_BCI;
        qMask2 = carryOutMask - ONE_BCI;
    } else {
        qMask1 = carryOutMask - ONE_BCI;
        qMask2 = carryInSumOutMask - ONE_BCI;
    }

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit out
        const cmplx outs0c0 = stateVec[i];
        const cmplx outs0c1 = stateVec[i | carryOutMask];
        const cmplx outs1c0 = stateVec[i | carryInSumOutMask];
        const cmplx outs1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        const bool aVal = (i & input1Mask);
        const bool bVal = (i & input2Mask);

        cmplx ins0c0, ins0c1, ins1c0, ins1c1;

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

        stateVec[i] = ins0c0;
        stateVec[i | carryInSumOutMask] = ins0c1;
        stateVec[i | carryOutMask] = ins1c0;
        stateVec[i | carryInSumOutMask | carryOutMask] = ins1c1;
    }
}

#define CMOD_START()                                                                                                   \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    bitCapIntOcl i = 0U;                                                                                               \
    for (bitLenInt p = 0U; p < (controlLen + len); p++) {                                                              \
        bitCapIntOcl iLow = iHigh & (controlPowers[p] - ONE_BCI);                                                      \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define CMOD_FINISH()                                                                                                  \
    nStateVec[i] = stateVec[i];                                                                                        \
    for (bitCapIntOcl j = ONE_BCI; j < ((ONE_BCI << controlLen) - ONE_BCI); j++) {                                     \
        bitCapIntOcl partControlMask = 0U;                                                                             \
        for (bitLenInt k = 0U; k < controlLen; k++) {                                                                  \
            if (j & (ONE_BCI << k)) {                                                                                  \
                partControlMask |= controlPowers[controlLen + len + k];                                                \
            }                                                                                                          \
        }                                                                                                              \
        nStateVec[i | partControlMask] = stateVec[i | partControlMask];                                                \
    }

void kernel cmul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    const bitCapIntOcl highMask = lowMask << len;
    const bitLenInt inOutStart = (bitLenInt)bitCapIntOclPtr[8];
    const bitLenInt carryStart = (bitLenInt)bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes |
            controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

void kernel cdiv(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    const bitCapIntOcl highMask = lowMask << len;
    const bitCapIntOcl inOutStart = bitCapIntOclPtr[8];
    const bitCapIntOcl carryStart = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl outInt = (((i & inOutMask) >> inOutStart) * toDiv);
        nStateVec[i | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask];

        CMOD_FINISH();
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define CMODNOUT(indexIn, indexOut)                                                                                    \
    const bitCapIntOcl Nthreads = get_global_size(0);                                                                  \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                            \
    const bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                     \
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];                                                        \
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];                                                               \
    const bitCapIntOcl inMask = bitCapIntOclPtr[4];                                                                    \
    const bitCapIntOcl outMask = bitCapIntOclPtr[5];                                                                   \
    const bitCapIntOcl modN = bitCapIntOclPtr[6];                                                                      \
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];                                                               \
    /* bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI; */                                                           \
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[8];                                                           \
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[9];                                                          \
                                                                                                                       \
    const bitCapIntOcl otherMask = (maxI - ONE_BCI) ^ (inMask | outMask | controlMask);                                \
    maxI >>= (controlLen + len);                                                                                       \
                                                                                                                       \
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {                                                         \
        CMOD_START();                                                                                                  \
                                                                                                                       \
        const bitCapIntOcl otherRes = i & otherMask;                                                                   \
        const bitCapIntOcl inRes = i & inMask;                                                                         \
        const bitCapIntOcl outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                 \
        nStateVec[indexOut] = stateVec[indexIn];                                                                       \
                                                                                                                       \
        CMOD_FINISH();                                                                                                 \
    }

void kernel cmulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    CMODNOUT((i | controlMask), (inRes | outRes | otherRes | controlMask));
}

void kernel cimulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    CMODNOUT((inRes | outRes | otherRes | controlMask), (i | controlMask));
}

void kernel cpowmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl base = bitCapIntOclPtr[1];
    const bitLenInt controlLen = (bitLenInt)bitCapIntOclPtr[2];
    const bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    const bitCapIntOcl inMask = bitCapIntOclPtr[4];
    const bitCapIntOcl outMask = bitCapIntOclPtr[5];
    const bitCapIntOcl modN = bitCapIntOclPtr[6];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[7];
    const bitLenInt inStart = (bitLenInt)bitCapIntOclPtr[8];
    const bitLenInt outStart = (bitLenInt)bitCapIntOclPtr[9];
    const bitCapIntOcl otherMask = (maxI - ONE_BCI) ^ (inMask | outMask | controlMask);
    maxI >>= (controlLen + len);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inRes = i & inMask;
        const bitCapIntOcl inInt = inRes >> inStart;

        bitCapIntOcl powRes = base;
        if (inInt == 0) {
            powRes = 1;
        } else {
            for (bitCapIntOcl pw = 1; pw < inInt; pw++) {
                powRes *= base;
            }
        }

        const bitCapIntOcl outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes | controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

void kernel indexedLda(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[4];
    const bitLenInt valueLength = (bitLenInt)bitCapIntOclPtr[5];
    const bitCapIntOcl lowMask = (ONE_BCI << outputStart) - ONE_BCI;
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & lowMask;
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << valueLength);

        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        const bitCapIntOcl outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

void kernel indexedAdc(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    const bitLenInt carryIn = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (carryMask - ONE_BCI);
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputRes = i & outputMask;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt += (outputRes >> outputStart) + carryIn;

        bitCapIntOcl carryRes = 0U;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    }
}

void kernel indexedSbc(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt inputStart = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitLenInt outputStart = (bitLenInt)bitCapIntOclPtr[3];
    const bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    const bitLenInt carryIn = (bitLenInt)bitCapIntOclPtr[6];
    const bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    const bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    const bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (carryMask - ONE_BCI);
        const bitCapIntOcl i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        const bitCapIntOcl otherRes = i & otherMask;
        const bitCapIntOcl inputRes = i & inputMask;
        const bitCapIntOcl inputInt = inputRes >> inputStart;
        bitCapIntOcl outputRes = i & outputMask;
        bitCapIntOcl outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt = (outputRes >> outputStart) + (lengthPower - (outputInt + carryIn));

        bitCapIntOcl carryRes = 0U;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[i];
    }
}

void kernel hash(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    const bitCapIntOcl bytes = bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl inputRes = lcv & inputMask;
        const bitCapIntOcl inputInt = inputRes >> start;
        bitCapIntOcl outputInt = 0U;
        if (bytes == 1) {
            outputInt = values[inputInt];
        } else if (bytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (bitCapIntOcl j = 0U; j < bytes; j++) {
                outputInt |= values[inputInt * bytes + j] << (8U * j);
            }
        }
        const bitCapIntOcl outputRes = outputInt << start;
        nStateVec[outputRes | (lcv & ~inputRes)] = stateVec[lcv];
    }
}

void kernel cphaseflipifless(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl skipPower = bitCapIntOclPtr[2];
    const bitCapIntOcl greaterPerm = bitCapIntOclPtr[3];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[4];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iHigh = lcv;
        const bitCapIntOcl iLow = iHigh & (skipPower - ONE_BCI);
        const bitCapIntOcl i = (iLow | ((iHigh ^ iLow) << ONE_BCI)) | skipPower;

        if (((i & regMask) >> start) < greaterPerm)
            stateVec[i] = -stateVec[i];
    }
}

void kernel phaseflipifless(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl greaterPerm = bitCapIntOclPtr[2];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    }
}
