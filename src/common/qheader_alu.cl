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
    bitCapIntOcl Nthreads, i;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[4];
    bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    for (i = ID; i < maxI; i += Nthreads) {
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | (i & otherMask)] =
            stateVec[i];
    }
}

void kernel cinc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, i, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[4];
    bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    bitCapIntOcl controlLen = bitCapIntOclPtr[6];
    bitCapIntOcl controlMask = bitCapIntOclPtr[7];
    bitCapIntOcl otherRes;
    bitCapIntOcl iHigh, iLow;
    bitLenInt p;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0U;
        for (p = 0U; p < controlLen; p++) {
            iLow = iHigh & (controlPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        otherRes = i & otherMask;
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | otherRes | controlMask] =
            stateVec[i | controlMask];
    }
}

void kernel incdecc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[5];
    bitCapIntOcl toMod = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, inOutRes, outInt, outRes, i;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        otherRes = i & otherMask;
        inOutRes = i & inOutMask;
        outInt = (inOutRes >> inOutStart) + toMod;
        outRes = 0U;
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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[5];
    bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, inOutInt, inOutRes, inInt, outInt, outRes;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        inOutRes = lcv & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = toAdd;
        outInt = inOutInt + toAdd;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes;
        }
        isOverflow = false;
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
        amp = stateVec[lcv];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incdecsc1(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[6];
    bitCapIntOcl toAdd = bitCapIntOclPtr[7];
    bitCapIntOcl otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        otherRes = i & otherMask;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = toAdd;
        outInt = inOutInt + toAdd;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        isOverflow = false;
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
        amp = stateVec[i];
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incdecsc2(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[5];
    bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        otherRes = i & otherMask;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = toAdd;
        outInt = inOutInt + toAdd;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        isOverflow = false;
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
        amp = stateVec[i];
        if (isOverflow) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel mul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toMul = bitCapIntOclPtr[1];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    bitCapIntOcl len = bitCapIntOclPtr[5];
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    bitCapIntOcl highMask = lowMask << len;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[6];
    bitCapIntOcl carryStart = bitCapIntOclPtr[7];
    bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    bitCapIntOcl otherRes, outInt;
    bitCapIntOcl i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | (iHigh ^ iLow) << len;

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes] =
            stateVec[i];
    }
}

void kernel div(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    bitCapIntOcl len = bitCapIntOclPtr[5];
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    bitCapIntOcl highMask = lowMask << len;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[6];
    bitCapIntOcl carryStart = bitCapIntOclPtr[7];
    bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    bitCapIntOcl otherRes, outInt;
    bitCapIntOcl i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | (iHigh ^ iLow) << len;

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toDiv;
        nStateVec[i] =
            stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes];
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define MODNOUT(indexIn, indexOut)                                                                                     \
    bitCapIntOcl Nthreads, lcv;                                                                                        \
                                                                                                                       \
    Nthreads = get_global_size(0);                                                                                     \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                            \
    bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                           \
    bitCapIntOcl inMask = bitCapIntOclPtr[2];                                                                          \
    /* bitCapIntOcl outMask = bitCapIntOclPtr[3]; */                                                                   \
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];                                                                       \
    bitCapIntOcl len = bitCapIntOclPtr[5];                                                                             \
    /* bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI; */                                                           \
    bitCapIntOcl inStart = bitCapIntOclPtr[6];                                                                         \
    bitCapIntOcl outStart = bitCapIntOclPtr[7];                                                                        \
    bitCapIntOcl skipMask = bitCapIntOclPtr[8];                                                                        \
    bitCapIntOcl modN = bitCapIntOclPtr[9];                                                                            \
    bitCapIntOcl otherRes, inRes, outRes;                                                                              \
    bitCapIntOcl i, iHigh, iLow;                                                                                       \
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {                                                                      \
        iHigh = lcv;                                                                                                   \
        iLow = iHigh & skipMask;                                                                                       \
        i = iLow | (iHigh ^ iLow) << len;                                                                              \
                                                                                                                       \
        otherRes = i & otherMask;                                                                                      \
        inRes = i & inMask;                                                                                            \
        outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                                    \
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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl base = bitCapIntOclPtr[1];
    bitCapIntOcl inMask = bitCapIntOclPtr[2];
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    bitCapIntOcl len = bitCapIntOclPtr[5];
    bitCapIntOcl inStart = bitCapIntOclPtr[6];
    bitCapIntOcl outStart = bitCapIntOclPtr[7];
    bitCapIntOcl skipMask = bitCapIntOclPtr[8];
    bitCapIntOcl modN = bitCapIntOclPtr[9];
    bitCapIntOcl otherRes, inRes, outRes, inInt;
    bitCapIntOcl i, iHigh, iLow, powRes, pw;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | (iHigh ^ iLow) << len;

        otherRes = i & otherMask;
        inRes = i & inMask;
        inInt = inRes >> inStart;

        powRes = base;
        for (pw = 1; pw < inInt; pw++) {
            powRes *= base;
        }
        if (inInt == 0) {
            powRes = 1;
        }

        outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes] = stateVec[i];
    }
}

void kernel fulladd(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - ONE_BCI;
        qMask2 = carryOutMask - ONE_BCI;
    } else {
        qMask1 = carryOutMask - ONE_BCI;
        qMask2 = carryInSumOutMask - ONE_BCI;
    }

    cmplx ins0c0, ins0c1, ins1c0, ins1c1;
    cmplx outs0c0, outs0c1, outs1c0, outs1c1;

    bitCapIntOcl i;

    bool aVal, bVal;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit in
        ins0c0 = stateVec[i];
        ins0c1 = stateVec[i | carryInSumOutMask];
        ins1c0 = stateVec[i | carryOutMask];
        ins1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        aVal = (i & input1Mask);
        bVal = (i & input2Mask);

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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl input1Mask = bitCapIntOclPtr[1];
    bitCapIntOcl input2Mask = bitCapIntOclPtr[2];
    bitCapIntOcl carryInSumOutMask = bitCapIntOclPtr[3];
    bitCapIntOcl carryOutMask = bitCapIntOclPtr[4];

    bitCapIntOcl qMask1, qMask2;
    if (carryInSumOutMask < carryOutMask) {
        qMask1 = carryInSumOutMask - ONE_BCI;
        qMask2 = carryOutMask - ONE_BCI;
    } else {
        qMask1 = carryOutMask - ONE_BCI;
        qMask2 = carryInSumOutMask - ONE_BCI;
    }

    cmplx ins0c0, ins0c1, ins1c0, ins1c1;
    cmplx outs0c0, outs0c1, outs1c0, outs1c1;

    bitCapIntOcl i;

    bool aVal, bVal;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2();

        // Carry-in, sum bit out
        outs0c0 = stateVec[i];
        outs0c1 = stateVec[i | carryOutMask];
        outs1c0 = stateVec[i | carryInSumOutMask];
        outs1c1 = stateVec[i | carryInSumOutMask | carryOutMask];

        aVal = (i & input1Mask);
        bVal = (i & input2Mask);

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
    iHigh = lcv;                                                                                                       \
    i = 0U;                                                                                                            \
    for (p = 0U; p < (controlLen + len); p++) {                                                                        \
        iLow = iHigh & (controlPowers[p] - ONE_BCI);                                                                   \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define CMOD_FINISH()                                                                                                  \
    nStateVec[i] = stateVec[i];                                                                                        \
    for (j = ONE_BCI; j < ((ONE_BCI << controlLen) - ONE_BCI); j++) {                                                  \
        partControlMask = 0U;                                                                                          \
        for (k = 0U; k < controlLen; k++) {                                                                            \
            if (j & (ONE_BCI << k)) {                                                                                  \
                partControlMask |= controlPowers[controlLen + len + k];                                                \
            }                                                                                                          \
        }                                                                                                              \
        nStateVec[i | partControlMask] = stateVec[i | partControlMask];                                                \
    }

void kernel cmul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toMul = bitCapIntOclPtr[1];
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    bitCapIntOcl len = bitCapIntOclPtr[7];
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    bitCapIntOcl highMask = lowMask << len;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[8];
    bitCapIntOcl carryStart = bitCapIntOclPtr[9];
    bitCapIntOcl otherRes, outInt;
    bitCapIntOcl i, iHigh, iLow, j;
    bitLenInt p, k;
    bitCapIntOcl partControlMask;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes |
            controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

void kernel cdiv(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec,
    constant bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    // bitCapIntOcl carryMask = bitCapIntOclPtr[5];
    bitCapIntOcl otherMask = bitCapIntOclPtr[6];
    bitCapIntOcl len = bitCapIntOclPtr[7];
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
    bitCapIntOcl highMask = lowMask << len;
    bitCapIntOcl inOutStart = bitCapIntOclPtr[8];
    bitCapIntOcl carryStart = bitCapIntOclPtr[9];
    bitCapIntOcl otherRes, outInt;
    bitCapIntOcl i, iHigh, iLow, j;
    bitLenInt p, k;
    bitCapIntOcl partControlMask;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        otherRes = i & otherMask;
        outInt = (((i & inOutMask) >> inOutStart) * toDiv);
        nStateVec[i | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask];

        CMOD_FINISH();
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define CMODNOUT(indexIn, indexOut)                                                                                    \
    bitCapIntOcl Nthreads, lcv;                                                                                        \
                                                                                                                       \
    Nthreads = get_global_size(0);                                                                                     \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                            \
    bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                           \
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];                                                                      \
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];                                                                     \
    bitCapIntOcl inMask = bitCapIntOclPtr[4];                                                                          \
    bitCapIntOcl outMask = bitCapIntOclPtr[5];                                                                         \
    bitCapIntOcl modN = bitCapIntOclPtr[6];                                                                            \
    bitCapIntOcl len = bitCapIntOclPtr[7];                                                                             \
    /* bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI; */                                                           \
    bitCapIntOcl inStart = bitCapIntOclPtr[8];                                                                         \
    bitCapIntOcl outStart = bitCapIntOclPtr[9];                                                                        \
                                                                                                                       \
    bitCapIntOcl otherMask = (maxI - ONE_BCI) ^ (inMask | outMask | controlMask);                                      \
    maxI >>= (controlLen + len);                                                                                       \
                                                                                                                       \
    bitCapIntOcl otherRes, outRes, inRes;                                                                              \
    bitCapIntOcl i, iHigh, iLow, j;                                                                                    \
    bitLenInt p, k;                                                                                                    \
    bitCapIntOcl partControlMask;                                                                                      \
                                                                                                                       \
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {                                                                      \
        CMOD_START();                                                                                                  \
                                                                                                                       \
        otherRes = i & otherMask;                                                                                      \
        inRes = i & inMask;                                                                                            \
        outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                                    \
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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl base = bitCapIntOclPtr[1];
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    bitCapIntOcl inMask = bitCapIntOclPtr[4];
    bitCapIntOcl outMask = bitCapIntOclPtr[5];
    bitCapIntOcl modN = bitCapIntOclPtr[6];
    bitCapIntOcl len = bitCapIntOclPtr[7];
    bitCapIntOcl inStart = bitCapIntOclPtr[8];
    bitCapIntOcl outStart = bitCapIntOclPtr[9];

    bitCapIntOcl otherMask = (maxI - ONE_BCI) ^ (inMask | outMask | controlMask);
    maxI >>= (controlLen + len);

    bitCapIntOcl otherRes, outRes, inRes, inInt;
    bitCapIntOcl i, iHigh, iLow, j, powRes, pw;
    bitLenInt p, k;
    bitCapIntOcl partControlMask;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        CMOD_START();

        otherRes = i & otherMask;
        inRes = i & inMask;
        inInt = inRes >> inStart;

        powRes = base;
        for (pw = 1; pw < inInt; pw++) {
            powRes *= base;
        }
        if (inInt == 0) {
            powRes = 1;
        }

        outRes = (powRes % modN) << outStart;

        nStateVec[inRes | outRes | otherRes | controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

void kernel indexedLda(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inputStart = bitCapIntOclPtr[1];
    bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    bitCapIntOcl outputStart = bitCapIntOclPtr[3];
    bitCapIntOcl valueBytes = bitCapIntOclPtr[4];
    bitCapIntOcl valueLength = bitCapIntOclPtr[5];
    bitCapIntOcl lowMask = (ONE_BCI << outputStart) - ONE_BCI;
    bitCapIntOcl inputRes, inputInt, outputRes, outputInt;
    bitCapIntOcl i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & lowMask;
        i = iLow | ((iHigh ^ iLow) << valueLength);

        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

void kernel indexedAdc(
    global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, global uchar* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inputStart = bitCapIntOclPtr[1];
    bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    bitCapIntOcl outputStart = bitCapIntOclPtr[3];
    bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    bitCapIntOcl carryIn = bitCapIntOclPtr[6];
    bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    bitCapIntOcl otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    bitCapIntOcl i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - ONE_BCI);
        i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt += (outputRes >> outputStart) + carryIn;

        carryRes = 0U;
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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inputStart = bitCapIntOclPtr[1];
    bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    bitCapIntOcl outputStart = bitCapIntOclPtr[3];
    bitCapIntOcl outputMask = bitCapIntOclPtr[4];
    bitCapIntOcl otherMask = bitCapIntOclPtr[5];
    bitCapIntOcl carryIn = bitCapIntOclPtr[6];
    bitCapIntOcl carryMask = bitCapIntOclPtr[7];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[8];
    bitCapIntOcl valueBytes = bitCapIntOclPtr[9];
    bitCapIntOcl otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    bitCapIntOcl i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - ONE_BCI);
        i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = 0U;
        if (valueBytes == 1) {
            outputInt = values[inputInt];
        } else if (valueBytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (j = 0U; j < valueBytes; j++) {
                outputInt |= values[inputInt * valueBytes + j] << (8U * j);
            }
        }
        outputInt = (outputRes >> outputStart) + (lengthPower - (outputInt + carryIn));

        carryRes = 0U;
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
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl start = bitCapIntOclPtr[1];
    bitCapIntOcl inputMask = bitCapIntOclPtr[2];
    bitCapIntOcl bytes = bitCapIntOclPtr[3];
    bitCapIntOcl inputRes, inputInt, outputRes, outputInt;
    bitCapIntOcl j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        inputRes = lcv & inputMask;
        inputInt = inputRes >> start;
        outputInt = 0U;
        if (bytes == 1) {
            outputInt = values[inputInt];
        } else if (bytes == 2) {
            outputInt = ((global ushort*)values)[inputInt];
        } else {
            for (j = 0U; j < bytes; j++) {
                outputInt |= values[inputInt * bytes + j] << (8U * j);
            }
        }
        outputRes = outputInt << start;
        nStateVec[outputRes | (lcv & ~inputRes)] = stateVec[lcv];
    }
}

void kernel cphaseflipifless(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;
    bitCapIntOcl i, iLow, iHigh;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl regMask = bitCapIntOclPtr[1];
    bitCapIntOcl skipPower = bitCapIntOclPtr[2];
    bitCapIntOcl greaterPerm = bitCapIntOclPtr[3];
    bitCapIntOcl start = bitCapIntOclPtr[4];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (skipPower - ONE_BCI);
        i = (iLow | ((iHigh ^ iLow) << ONE_BCI)) | skipPower;

        if (((i & regMask) >> start) < greaterPerm)
            stateVec[i] = -stateVec[i];
    }
}

void kernel phaseflipifless(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl regMask = bitCapIntOclPtr[1];
    bitCapIntOcl greaterPerm = bitCapIntOclPtr[2];
    bitCapIntOcl start = bitCapIntOclPtr[3];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    }
}
