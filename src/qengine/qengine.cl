#define bitCapInt ulong
#define bitLenInt unsigned char

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == 0.0 && cmp.y == 0.0)
        return 0.0;
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

    cmplx Y0;
    bitCapInt i, iLow, iHigh;
    cmplx qubit[2];
    bitLenInt p;
    lcv = ID;
    iHigh = lcv;
    i = 0;
    for (p = 0; p < bitCount; p++) {
        iLow = iHigh % qPowersSorted[p];
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
    }
    i += iHigh;
    while (i < maxI) {
        Y0 = stateVec[i + offset1];
        qubit[1] = stateVec[i + offset2];
 
        qubit[0] = nrm * (zmul(mtrx[0], Y0) + zmul(mtrx[1], qubit[1]));
        qubit[1] = nrm * (zmul(mtrx[2], Y0) + zmul(mtrx[3], qubit[1]));

        stateVec[i + offset1] = qubit[0];
        stateVec[i + offset2] = qubit[1];

        lcv += Nthreads;
        iHigh = lcv;
        i = 0;
        for (p = 0; p < bitCount; p++) {
            iLow = iHigh % qPowersSorted[p];
            i += iLow;
            iHigh = (iHigh - iLow) << 1;
        }
        i += iHigh;
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

    cmplx Y0;
    bitCapInt i, iLow, iHigh;
    cmplx qubit[2];
    bitLenInt p;
    lcv = ID;
    iHigh = lcv;
    i = 0;
    for (p = 0; p < bitCount; p++) {
        iLow = iHigh % qPowersSorted[p];
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
    }
    i += iHigh;
    while (i < maxI) {
        Y0 = stateVec[i + offset1];
        qubit[1] = stateVec[i + offset2];

        qubit[0] = nrm * (zmul(mtrx[0], Y0) + zmul(mtrx[1], qubit[1]));
        qubit[1] = nrm * (zmul(mtrx[2], Y0) + zmul(mtrx[3], qubit[1]));

        stateVec[i + offset1] = qubit[0];
        stateVec[i + offset2] = qubit[1];
        nrm1 = dot(qubit[0], qubit[0]);
        nrm2 = dot(qubit[1], qubit[1]);
        if (nrm1 < min_norm) {
            nrm1 = 0.0;
        }
        if (nrm2 >= min_norm) {
            nrm1 += nrm2;
        }
        nrmParts[ID] += nrm1;

        lcv += Nthreads;
        iHigh = lcv;
        i = 0;
        for (p = 0; p < bitCount; p++) {
            iLow = iHigh % qPowersSorted[p];
            i += iLow;
            iHigh = (iHigh - iLow) << 1;
        }
        i += iHigh;
    }
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

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv % (1 << start);
        j = j | ((lcv ^ j) << len);
        for (k = 0; k < partPower; k++) {
            l = j | (k << start);
            amp = stateVec[l];
            remainderStateProb[lcv] += dot(amp, amp);
        }
        remainderStateAngle[lcv] = arg(amp);
    }

    for (lcv = ID; lcv < partPower; lcv += Nthreads) {
        j = lcv << start;
        for (k = 0; k < remainderPower; k++) {
            l = k % (1 << start);
            l = l | ((k ^ l) << len);
            l = j | l;
            amp = stateVec[l];
            partStateProb[lcv] += dot(amp, amp);
        }
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

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv % (1 << start);
        j = j | ((lcv ^ j) << len);
        for (k = 0; k < partPower; k++) {
            l = j | (k << start);
            amp = stateVec[l];
            remainderStateProb[lcv] += dot(amp, amp);
        }
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

void kernel prob(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, global real1* oneChanceBuffer)
{
    bitCapInt ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0] >> 1;
    bitCapInt qPower = bitCapIntPtr[1];
    bitCapInt qMask = qPower - 1;
    real1 oneChancePart = 0.0;
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
        i = iLow + ((iHigh - iLow) << 1);
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
        i = iLow + ((iHigh - iLow) << 1);
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
        i = iLow + ((iHigh - iLow) << valueLength);

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
        i = iLow + ((iHigh - iLow) << 1);

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
        i = iLow + ((iHigh - iLow) << 1);

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
            amp = (cmplx)(0.0, 0.0);
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
    
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        nrm = dot(amp, amp);
        if (nrm < min_norm) {
            nrm = 0.0;
        }
        norm_ptr[ID] += nrm;
    }
}

void kernel applym(global cmplx* stateVec, constant bitCapInt* bitCapIntPtr, constant real1* args_ptr) {
    bitCapInt ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    bitCapInt maxI = bitCapIntPtr[0];
    bitCapInt qPower = bitCapIntPtr[1];
    bitCapInt qMask = qPower - 1;
    bitCapInt savePower = bitCapIntPtr[2];
    bitCapInt discardPower = qPower ^ savePower;
    cmplx nrm = (cmplx)(args_ptr[0], args_ptr[1]);
    bitCapInt i, iLow, iHigh, j;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & qMask;
        i = iLow + ((iHigh - iLow) << 1);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (cmplx)(0.0, 0.0);
    }
}
