#define bitCapInt ulong
#define bitLenInt unsigned char

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return atan2(cmp.y, cmp.x);
}

void kernel apply2x2(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapInt* bitCapIntPtr)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant cmplx* mtrx = cmplxPtr;

    real1 nrm = cmplxPtr[4].x;
    bitCapInt bitCount = bitCapIntPtr[0];
    bitCapInt maxI = bitCapIntPtr[1];
    bitCapInt offset1 = bitCapIntPtr[2];
    bitCapInt offset2 = bitCapIntPtr[3];
    constant bitCapInt* qPowersSorted = (bitCapIntPtr + 4);

    cmplx Y0, Y1;
    bitCapInt i, iLow, iHigh;
    bitLenInt p;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0;
        for (p = 0; p < bitCount; p++) {
            iLow = iHigh & (qPowersSorted[p] - 1);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1;
        }
        i |= iHigh;

        Y0 = stateVec[i | offset1];
        Y1 = stateVec[i | offset2]; 

        stateVec[i | offset1] = nrm * (zmul(mtrx[0], Y0) + zmul(mtrx[1], Y1));
        stateVec[i | offset2] = nrm * (zmul(mtrx[2], Y0) + zmul(mtrx[3], Y1));
    }
}

void kernel apply2x2norm(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapInt* bitCapIntPtr, global real1* nrmParts)
{
    bitCapInt ID, Nthreads, lcv;
    real1 nrm1, nrm2;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant cmplx* mtrx = cmplxPtr;

    real1 nrm = cmplxPtr[4].x;
    bitCapInt bitCount = bitCapIntPtr[0];
    bitCapInt maxI = bitCapIntPtr[1];
    bitCapInt offset1 = bitCapIntPtr[2];
    bitCapInt offset2 = bitCapIntPtr[3];
    constant bitCapInt* qPowersSorted = (bitCapIntPtr + 4);

    cmplx Y0, Y1, YT;
    bitCapInt i, iLow, iHigh;
    bitLenInt p;
    real1 partNrm = ZERO_R1;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0;
        for (p = 0; p < bitCount; p++) {
            iLow = iHigh & (qPowersSorted[p] - 1);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1;
        }
        i |= iHigh;

        YT = stateVec[i | offset1];
        Y1 = stateVec[i | offset2];

        Y0 = nrm * (zmul(mtrx[0], YT) + zmul(mtrx[1], Y1));
        Y1 = nrm * (zmul(mtrx[2], YT) + zmul(mtrx[3], Y1));

        stateVec[i | offset1] = Y0;
        stateVec[i | offset2] = Y1;
        nrm1 = dot(Y0, Y0);
        nrm2 = dot(Y1, Y1);
        if (nrm1 < min_norm) {
            nrm1 = ZERO_R1;
        }
        if (nrm2 >= min_norm) {
            nrm1 += nrm2;
        }
        partNrm += nrm1;
    }
    nrmParts[ID] = partNrm;
}

void kernel cohere(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt nMaxQPower = bitCapIntPtr[0];
    bitCapInt startMask = bitCapIntPtr[1];
    bitCapInt endMask = bitCapIntPtr[2];
    bitCapInt qubitCount = bitCapIntPtr[3];
    for (lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & startMask], stateVec2[(lcv & endMask) >> qubitCount]);
    }
}

void kernel decohereprob(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* remainderStateProb, global real1* remainderStateAngle, global real1* partStateProb, global real1* partStateAngle)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt partPower = bitCapIntPtr[0];
    bitCapInt remainderPower = bitCapIntPtr[1];
    bitCapInt start = bitCapIntPtr[2];
    bitCapInt len = bitCapIntPtr[3];
    bitCapInt j, k, l;
    cmplx amp;
    real1 partProb;

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv % (1 << start);
        j = j | ((lcv ^ j) << len);
        partProb = ZERO_R1;
        for (k = 0; k < partPower; k++) {
            l = j | (k << start);
            amp = stateVec[l];
            partProb += dot(amp, amp);
        }
        remainderStateProb[lcv] = partProb;
        remainderStateAngle[lcv] = arg(amp);
    }

    for (lcv = ID; lcv < partPower; lcv += Nthreads) {
        j = lcv << start;
        partProb = ZERO_R1;
        for (k = 0; k < remainderPower; k++) {
            l = k % (1 << start);
            l = l | ((k ^ l) << len);
            l = j | l;
            amp = stateVec[l];
            partProb += dot(amp, amp);
        }
        partStateProb[lcv] = partProb;
        partStateAngle[lcv] = arg(amp);
    }
}

void kernel disposeprob(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* remainderStateProb, global real1* remainderStateAngle)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt partPower = bitCapIntPtr[0];
    bitCapInt remainderPower = bitCapIntPtr[1];
    bitCapInt start = bitCapIntPtr[2];
    bitCapInt len = bitCapIntPtr[3];
    bitCapInt j, k, l;
    cmplx amp;
    real1 partProb;

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv % (1 << start);
        j = j | ((lcv ^ j) << len);
        partProb = ZERO_R1;
        for (k = 0; k < partPower; k++) {
            l = j | (k << start);
            amp = stateVec[l];
            partProb += dot(amp, amp);
        }
        remainderStateProb[lcv] = partProb;
        remainderStateAngle[lcv] = arg(amp);
    }
}

void kernel decohereamp(global real1* stateProb, global real1* stateAngle, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxQPower = bitCapIntPtr[0];
    real1 angle;
    for (lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((cmplx)(angle + SineShift, angle));
    }
}

void kernel isphaseseparable(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* phases, global bitLenInt* isAllSame)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    real1 nrm;
    cmplx amp;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        nrm = dot(amp, amp);
        if (nrm > min_norm) {
            real1 nPhase = arg(stateVec[lcv]);
            if (phases[ID] < (-PI_R1)) {
                phases[ID] = nPhase;
            } else {
                real1 diff = phases[ID] - nPhase;
                if (diff < ZERO_R1) {
                    diff = -diff;
                }
                if (diff > PI_R1) {
                    diff = (2 * PI_R1) - diff;
                }
                if (diff > min_norm) {
                    isAllSame[ID] = 0;
                }
            }
        }
    }
}

void kernel prob(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* oneChanceBuffer)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt qPower = bitCapIntPtr[1];
    bitCapInt qMask = qPower - 1;
    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapInt i;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << 1) | qPower;
        amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    oneChanceBuffer[ID] = oneChancePart;
}

void kernel x(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt regMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        nStateVec[lcv] = stateVec[(lcv & otherMask) | ((~lcv) & regMask)];
    }
}

void kernel swap(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt reg1Mask = bitCapIntPtr[1];
    bitCapInt reg2Mask = bitCapIntPtr[2];
    bitCapInt otherMask = bitCapIntPtr[3];
    bitCapInt start1 = bitCapIntPtr[4];
    bitCapInt start2 = bitCapIntPtr[5];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        nStateVec[lcv] = stateVec[ 
                                  (((lcv & reg2Mask) >> start2) << start1) |
                                  (((lcv & reg1Mask) >> start1) << start2) |
                                  (lcv & otherMask)
                                 ];
    }
}

void kernel rol(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt regMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt start = bitCapIntPtr[4];
    bitCapInt shift = bitCapIntPtr[5];
    bitCapInt length = bitCapIntPtr[6];
    bitCapInt otherRes, regRes, regInt, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        inInt = ((regInt >> shift) | (regInt << (length - shift))) & lengthMask;
        nStateVec[lcv] = stateVec[(inInt << start) | otherRes];
    }
}

void kernel ror(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt regMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt start = bitCapIntPtr[4];
    bitCapInt shift = bitCapIntPtr[5];
    bitCapInt length = bitCapIntPtr[6];
    bitCapInt otherRes, regRes, regInt, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        inInt = ((regInt >> (length - shift)) | (regInt << shift)) & lengthMask;
        nStateVec[lcv] = stateVec[(inInt << start) | otherRes];
    }
}

void kernel inc(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, i;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt inOutStart = bitCapIntPtr[4];
    bitCapInt toAdd = bitCapIntPtr[5];
    bitCapInt otherRes, inRes;
    for (i = ID; i < maxI; i += Nthreads) {
        otherRes = (i & otherMask);
        inRes = (i & inOutMask);
        inRes = (((lengthMask + 1 + (inRes >> inOutStart)) - toAdd) & lengthMask) << inOutStart;
        nStateVec[i] = stateVec[inRes | otherRes];
    }
}

void kernel dec(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, i;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt inOutStart = bitCapIntPtr[4];
    bitCapInt toSub = bitCapIntPtr[5];
    bitCapInt otherRes, inRes;
    for (i = ID; i < maxI; i += Nthreads) {
        otherRes = (i & otherMask);
        inRes = (i & inOutMask);
        inRes = (((inRes >> inOutStart) + toSub) & lengthMask) << inOutStart;
        nStateVec[i] = stateVec[inRes | otherRes];
    }
}

void kernel incc(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt carryMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toAdd = bitCapIntPtr[6];
    bitCapInt otherRes, inOutRes, outInt, outRes, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);
        otherRes = (i & otherMask);
        inOutRes = (i & inOutMask);
        outInt = (inOutRes >> inOutStart) + toAdd;
        outRes = 0;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = carryMask;
        }
        outRes |= outInt << inOutStart;
        nStateVec[outRes | otherRes] = stateVec[i];
    }
}

void kernel decc(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthMask = bitCapIntPtr[3] - 1;
    bitCapInt carryMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toSub = bitCapIntPtr[6];
    bitCapInt otherRes, inOutRes, outInt, outRes, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);
        otherRes = (i & otherMask);
        inOutRes = (i & inOutMask);
        outInt = (lengthMask + 1 + (inOutRes >> inOutStart)) - toSub;
        outRes = 0;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = carryMask;
        }
        outRes |= outInt << inOutStart;
        nStateVec[outRes | otherRes] = stateVec[i];
    }
}

void kernel incs(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt overflowMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toAdd = bitCapIntPtr[6];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes;
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
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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

void kernel decs(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt overflowMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toSub = bitCapIntPtr[6];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        inOutRes = lcv & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = overflowMask;
        outInt = inOutInt - toSub + lengthPower;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes;
        }
        isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask) {
                isOverflow = true;
            }
        }
        // First positive:
        else if (inOutInt & (~inInt) & signMask) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask) {
                isOverflow = true;
            }
        }
        amp = stateVec[lcv];
        if (isOverflow && ((outRes & overflowMask) == overflowMask))  {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incsc1(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt overflowMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt inOutStart = bitCapIntPtr[6];
    bitCapInt toAdd = bitCapIntPtr[7];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i, iHigh, iLow;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

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
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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

void kernel decsc1(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt overflowMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt inOutStart = bitCapIntPtr[6];
    bitCapInt toSub = bitCapIntPtr[7];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i, iHigh, iLow;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = toSub;
        outInt = (inOutInt - toSub) + lengthPower;
        if (outInt < lengthPower) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & signMask) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        amp = stateVec[i];
        if (isOverflow && ((outRes & overflowMask) == overflowMask))  {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incsc2(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt carryMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toAdd = bitCapIntPtr[6];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i, iHigh, iLow;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

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
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
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

void kernel decsc2(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt lengthPower = bitCapIntPtr[3];
    bitCapInt signMask = lengthPower >> 1;
    bitCapInt carryMask = bitCapIntPtr[4];
    bitCapInt inOutStart = bitCapIntPtr[5];
    bitCapInt toSub = bitCapIntPtr[6];
    bitCapInt otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i, iHigh, iLow;
    cmplx amp;
    bool isOverflow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        inInt = toSub;
        outInt = (inOutInt - toSub) + lengthPower;
        if (outInt < (lengthPower)) {
            outRes = (outInt << inOutStart) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << inOutStart) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & signMask) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & signMask) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        amp = stateVec[i];
        if (isOverflow)  {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

void kernel incbcd(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt inOutStart = bitCapIntPtr[3];
    bitCapInt toAdd = bitCapIntPtr[4];
    bitCapInt nibbleCount = bitCapIntPtr[5];
    bitCapInt otherRes, partToAdd, inOutRes, inOutInt;
    char test1, test2;
    unsigned char j;
    // For 128 qubits, we would have 32 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    char nibbles[32];
    bool isValid;
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & (otherMask));
        partToAdd = toAdd;
        inOutRes = (lcv & (inOutMask));
        inOutInt = inOutRes >> (inOutStart);
        isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = (partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        amp = stateVec[lcv];
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
            nStateVec[(outInt << (inOutStart)) | otherRes] = amp;
        } else {
            nStateVec[lcv] = amp;
        }
    }
}

void kernel decbcd(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt inOutStart = bitCapIntPtr[3];
    bitCapInt toSub = bitCapIntPtr[4];
    bitCapInt nibbleCount = bitCapIntPtr[5];
    bitCapInt otherRes, partToSub, inOutRes, inOutInt;
    char test1, test2;
    unsigned char j;
    // For 128 qubits, we would have 32 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    char nibbles[32];
    bool isValid;
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        partToSub = toSub;
        inOutRes = lcv & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = (partToSub % 10);
            partToSub /= 10;
            nibbles[j] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        amp = stateVec[lcv];
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
            nStateVec[(outInt << (inOutStart)) | otherRes] = amp;
        } else {
            nStateVec[lcv] = amp;
        }
    }
}

void kernel incbcdc(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt carryMask = bitCapIntPtr[3];
    bitCapInt inOutStart = bitCapIntPtr[4];
    bitCapInt toAdd = bitCapIntPtr[5];
    bitCapInt nibbleCount = bitCapIntPtr[6];
    bitCapInt otherRes, partToAdd, inOutRes, inOutInt, outInt, outRes, carryRes;
    char test1, test2;
    unsigned char j;
    // For 128 qubits, we would have 32 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    char nibbles[32];
    bool isValid;
    cmplx amp;
    bitCapInt i, iLow, iHigh;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        partToAdd = toAdd;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

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
        amp = stateVec[i];
        if (isValid) {
            outInt = 0;
            outRes = 0;
            carryRes = 0;
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
            nStateVec[outRes] = amp;
        } else {
            nStateVec[i] = amp;
        }
    }
}

void kernel decbcdc(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inOutMask = bitCapIntPtr[1];
    bitCapInt otherMask = bitCapIntPtr[2];
    bitCapInt carryMask = bitCapIntPtr[3];
    bitCapInt inOutStart = bitCapIntPtr[4];
    bitCapInt toSub = bitCapIntPtr[5];
    bitCapInt nibbleCount = bitCapIntPtr[6];
    bitCapInt otherRes, partToSub, inOutRes, inOutInt, outInt, outRes, carryRes;
    char test1, test2;
    unsigned char j;
    // For 128 qubits, we would have 32 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    char nibbles[32];
    bool isValid;
    cmplx amp;
    bitCapInt i, iLow, iHigh;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        partToSub = toSub;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

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
        amp = stateVec[i];
        if (isValid) {
            outInt = 0;
            outRes = 0;
            carryRes = carryMask;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    } else {
                        carryRes = 0;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = amp;
        } else {
            nStateVec[i] = amp;
        }
    }
}

void kernel mul(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt toMul = bitCapIntPtr[1];
    bitCapInt lowMask = bitCapIntPtr[2];
    bitCapInt highMask = bitCapIntPtr[3];
    bitCapInt inOutMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt otherMask = bitCapIntPtr[6];
    bitCapInt len = bitCapIntPtr[7];
    bitCapInt inOutStart = bitCapIntPtr[8];
    bitCapInt carryStart = bitCapIntPtr[9];
    bitCapInt iLowMask = (1 << carryStart) - 1;
    bitCapInt otherRes, outInt, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & iLowMask;
        i = iLow | ((iHigh ^ iLow) << len);

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes] = stateVec[i];
    }
}

void kernel div(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt toDiv = bitCapIntPtr[1];
    bitCapInt lowMask = bitCapIntPtr[2];
    bitCapInt highMask = bitCapIntPtr[3];
    bitCapInt inOutMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt otherMask = bitCapIntPtr[6];
    bitCapInt len = bitCapIntPtr[7];
    bitCapInt inOutStart = bitCapIntPtr[8];
    bitCapInt carryStart = bitCapIntPtr[9];
    bitCapInt iLowMask = (1 << carryStart) - 1;
    bitCapInt otherRes, outInt, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & iLowMask;
        i = iLow | ((iHigh ^ iLow) << len);

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toDiv;
        nStateVec[i] = stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes];
    }
}

void kernel cmul(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt toMul = bitCapIntPtr[1];
    bitCapInt lowMask = bitCapIntPtr[2];
    bitCapInt controlPower = bitCapIntPtr[3];
    bitCapInt inOutMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt otherMask = bitCapIntPtr[6];
    bitCapInt len = bitCapIntPtr[7];
    bitCapInt inOutStart = bitCapIntPtr[8];
    bitCapInt carryStart = bitCapIntPtr[9];
    bitCapInt iLowMask = (1 << carryStart) - 1;
    bitCapInt highMask = lowMask << len;
    bitCapInt otherRes, outInt, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & iLowMask;
        i = iLow | ((iHigh ^ iLow) << len);

        iHigh = i;
        iLow = iHigh & (controlPower - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toMul;
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes | controlPower] = stateVec[i | controlPower];
        nStateVec[i] = stateVec[i];
    }
}

void kernel cdiv(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt toDiv = bitCapIntPtr[1];
    bitCapInt lowMask = bitCapIntPtr[2];
    bitCapInt controlPower = bitCapIntPtr[3];
    bitCapInt inOutMask = bitCapIntPtr[4];
    bitCapInt carryMask = bitCapIntPtr[5];
    bitCapInt otherMask = bitCapIntPtr[6];
    bitCapInt len = bitCapIntPtr[7];
    bitCapInt inOutStart = bitCapIntPtr[8];
    bitCapInt carryStart = bitCapIntPtr[9];
    bitCapInt iLowMask = (1 << carryStart) - 1;
    bitCapInt highMask = lowMask << len;
    bitCapInt otherRes, outInt, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & iLowMask;
        i = iLow | ((iHigh ^ iLow) << len);

        iHigh = i;
        iLow = iHigh & (controlPower - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        outInt = ((i & inOutMask) >> inOutStart) * toDiv;
        nStateVec[i | controlPower] = stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes | controlPower];
        nStateVec[i] = stateVec[i];
    }
}

void kernel indexedLda(
    global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec, constant bitLenInt* values)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inputStart = bitCapIntPtr[1];
    bitCapInt inputMask = bitCapIntPtr[2];
    bitCapInt outputStart = bitCapIntPtr[3];
    bitCapInt valueBytes = bitCapIntPtr[4];
    bitCapInt valueLength = bitCapIntPtr[5];
    bitCapInt lowMask = (1 << outputStart) - 1;
    bitCapInt inputRes, inputInt, outputRes, outputInt;
    bitCapInt i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & lowMask;
        i = iLow | ((iHigh ^ iLow) << valueLength);

        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputInt = 0;
        for (j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
        }
        outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

void kernel indexedAdc(
    global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec, constant bitLenInt* values)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inputStart = bitCapIntPtr[1];
    bitCapInt inputMask = bitCapIntPtr[2];
    bitCapInt outputStart = bitCapIntPtr[3];
    bitCapInt outputMask = bitCapIntPtr[4];
    bitCapInt otherMask = bitCapIntPtr[5];
    bitCapInt carryIn = bitCapIntPtr[6];
    bitCapInt carryMask = bitCapIntPtr[7];
    bitCapInt lengthPower = bitCapIntPtr[8];
    bitCapInt valueBytes = bitCapIntPtr[9];
    bitCapInt otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    bitCapInt i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = 0;
        for (j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
        }
        outputInt += (outputRes >> outputStart) + carryIn;

        carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    }
}

void kernel indexedSbc(
    global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global cmplx* nStateVec, constant bitLenInt* values)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt inputStart = bitCapIntPtr[1];
    bitCapInt inputMask = bitCapIntPtr[2];
    bitCapInt outputStart = bitCapIntPtr[3];
    bitCapInt outputMask = bitCapIntPtr[4];
    bitCapInt otherMask = bitCapIntPtr[5];
    bitCapInt carryIn = bitCapIntPtr[6];
    bitCapInt carryMask = bitCapIntPtr[7];
    bitCapInt lengthPower = bitCapIntPtr[8];
    bitCapInt valueBytes = bitCapIntPtr[9];
    bitCapInt otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    bitCapInt i, iLow, iHigh, j;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow | ((iHigh ^ iLow) << 1);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = 0;
        for (j = 0; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8 * j);
        }
        outputInt = (outputRes >> outputStart) + (lengthPower - (outputInt + carryIn));

        carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[i];
    }
}

void kernel nrmlze(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, constant real1* args_ptr) {
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    real1 nrm = args_ptr[1];
    cmplx amp;
    
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv] / nrm;
        //"min_norm" is defined in qinterface.hpp
        if (dot(amp, amp) < min_norm) {
            amp = (cmplx)(ZERO_R1, ZERO_R1);
        }
        stateVec[lcv] = amp;
    }
}

void kernel updatenorm(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* norm_ptr) {
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    cmplx amp;
    real1 nrm;
    real1 partNrm = ZERO_R1;


    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        nrm = dot(amp, amp);
        if (nrm < min_norm) {
            nrm = ZERO_R1;
        }
        partNrm += nrm;
    }
    norm_ptr[ID] = partNrm;
}

void kernel applym(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, constant cmplx* cmplx_ptr) {
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt qPower = bitCapIntPtr[1];
    bitCapInt qMask = qPower - 1;
    bitCapInt savePower = bitCapIntPtr[2];
    bitCapInt discardPower = qPower ^ savePower;
    cmplx nrm = cmplx_ptr[0];
    bitCapInt i, iLow, iHigh, j;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & qMask;
        i = iLow | ((iHigh ^ iLow) << 1);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel phaseflip(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = -stateVec[lcv];
    }
}

void kernel zerophaseflip(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr)
{
    bitCapInt ID, Nthreads, lcv;
    bitCapInt i, iLow, iHigh;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt skipMask = bitCapIntPtr[1] - 1;
    bitCapInt skipLength = bitCapIntPtr[2];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | ((iHigh ^ iLow) << skipLength);

        stateVec[i] = -stateVec[i];
    }
}

void kernel cphaseflipifless(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr)
{
    bitCapInt ID, Nthreads, lcv;
    bitCapInt i, iLow, iHigh;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt regMask = bitCapIntPtr[1];
    bitCapInt skipPower = bitCapIntPtr[2];
    bitCapInt greaterPerm = bitCapIntPtr[3];
    bitCapInt start = bitCapIntPtr[4];
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (skipPower - 1);
        i = (iLow | ((iHigh ^ iLow) << 1)) | skipPower;

        if (((i & regMask) >> start) < greaterPerm)
            stateVec[i] = -stateVec[i];
    }
}

void kernel phaseflipifless(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr)
{
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt regMask = bitCapIntPtr[1];
    bitCapInt greaterPerm = bitCapIntPtr[2];
    bitCapInt start = bitCapIntPtr[3];
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    }
}
