#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

void kernel apply2x2(global cmplx* stateVec, constant cmplx* cmplxPtr, constant ulong* ulongPtr)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant cmplx* mtrx = cmplxPtr;

    real1 nrm = cmplxPtr[4].x;
    ulong bitCount = ulongPtr[0];
    ulong maxI = ulongPtr[1];
    ulong offset1 = ulongPtr[2];
    ulong offset2 = ulongPtr[3];
    constant ulong* qPowersSorted = (ulongPtr + 4);

    cmplx Y0;
    ulong i, iLow, iHigh;
    cmplx qubit[2];
    unsigned char p;
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

void kernel apply2x2norm(global cmplx* stateVec, constant cmplx* cmplxPtr, constant ulong* ulongPtr, global real1* nrmParts)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant cmplx* mtrx = cmplxPtr;

    real1 nrm = cmplxPtr[4].x;
    ulong bitCount = ulongPtr[0];
    ulong maxI = ulongPtr[1];
    ulong offset1 = ulongPtr[2];
    ulong offset2 = ulongPtr[3];
    constant ulong* qPowersSorted = (ulongPtr + 4);

    cmplx Y0;
    ulong i, iLow, iHigh;
    cmplx qubit[2];
    unsigned char p;
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
        nrmParts[ID] += dot(qubit[0], qubit[0]) + dot(qubit[1], qubit[1]);

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

void kernel cohere(global cmplx* stateVec1, global cmplx* stateVec2, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong nMaxQPower = ulongPtr[0];
    ulong startMask = ulongPtr[1];
    ulong endMask = ulongPtr[2];
    ulong qubitCount = ulongPtr[3];
    for (lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & startMask], stateVec2[(lcv & endMask) >> qubitCount]);
    }
}

void kernel decohereprob(global cmplx* stateVec, constant ulong* ulongPtr, global real1* partStateProb, global real1* partStateAngle, global real1* remainderStateProb, global real1* remainderStateAngle)
{
    ulong ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong partPower = ulongPtr[0];
    ulong remainderPower = ulongPtr[1];
    ulong start = ulongPtr[2];
    ulong len = ulongPtr[3];
    ulong j, k, l;
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

void kernel decohereamp(global real1* stateProb, global real1* stateAngle, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;
    
    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxQPower = ulongPtr[0];
    real1 angle;
    for (lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((cmplx)(angle + SineShift, angle));
    }
}

void kernel x(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong regMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        nStateVec[lcv] = stateVec[(lcv & otherMask) | ((~lcv) & regMask)];
    }
}

void kernel swap(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong reg1Mask = ulongPtr[1];
    ulong reg2Mask = ulongPtr[2];
    ulong otherMask = ulongPtr[3];
    ulong start1 = ulongPtr[4];
    ulong start2 = ulongPtr[5];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        nStateVec[lcv] = stateVec[ 
                                  (((lcv & reg2Mask) >> start2) << start1) |
                                  (((lcv & reg1Mask) >> start1) << start2) |
                                  (lcv & otherMask)
                                 ];
    }
}

void kernel rol(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong regMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong start = ulongPtr[4];
    ulong shift = ulongPtr[5];
    ulong length = ulongPtr[6];
    ulong otherRes, regRes, regInt, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        inInt = ((regInt >> shift) | (regInt << (length - shift))) & lengthMask;
        nStateVec[lcv] = stateVec[(inInt << start) | otherRes];
    }
}

void kernel ror(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong regMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong start = ulongPtr[4];
    ulong shift = ulongPtr[5];
    ulong length = ulongPtr[6];
    ulong otherRes, regRes, regInt, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        inInt = ((regInt >> (length - shift)) | (regInt << shift)) & lengthMask;
        nStateVec[lcv] = stateVec[(inInt << start) | otherRes];
    }
}

void kernel inc(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, i;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong inOutStart = ulongPtr[4];
    ulong toAdd = ulongPtr[5];
    ulong otherRes, inRes;
    for (i = ID; i < maxI; i += Nthreads) {
        otherRes = (i & otherMask);
        inRes = (i & inOutMask);
        inRes = (((lengthMask + 1 + (inRes >> inOutStart)) - toAdd) & lengthMask) << inOutStart;
        nStateVec[i] = stateVec[inRes | otherRes];
    }
}

void kernel dec(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, i;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong inOutStart = ulongPtr[4];
    ulong toSub = ulongPtr[5];
    ulong otherRes, inRes;
    for (i = ID; i < maxI; i += Nthreads) {
        otherRes = (i & otherMask);
        inRes = (i & inOutMask);
        inRes = (((inRes >> inOutStart) + toSub) & lengthMask) << inOutStart;
        nStateVec[i] = stateVec[inRes | otherRes];
    }
}

void kernel incc(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong carryMask = ulongPtr[4];
    ulong inOutStart = ulongPtr[5];
    ulong toAdd = ulongPtr[6];
    ulong otherRes, inOutRes, outInt, outRes, i, iHigh, iLow;
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

void kernel decc(global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong otherMask = ulongPtr[2];
    ulong lengthMask = ulongPtr[3] - 1;
    ulong carryMask = ulongPtr[4];
    ulong inOutStart = ulongPtr[5];
    ulong toSub = ulongPtr[6];
    ulong otherRes, inOutRes, outInt, outRes, i, iHigh, iLow;
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

void kernel indexedLda(
    global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec, constant unsigned char* values)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inputStart = ulongPtr[1];
    ulong inputMask = ulongPtr[2];
    ulong outputStart = ulongPtr[3];
    ulong valueBytes = ulongPtr[4];
    ulong valueLength = ulongPtr[5];
    ulong lowMask = (1 << outputStart) - 1;
    ulong inputRes, inputInt, outputRes, outputInt;
    ulong i, iLow, iHigh, j;
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
    global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec, constant unsigned char* values)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inputStart = ulongPtr[1];
    ulong inputMask = ulongPtr[2];
    ulong outputStart = ulongPtr[3];
    ulong outputMask = ulongPtr[4];
    ulong otherMask = ulongPtr[5];
    ulong carryIn = ulongPtr[6];
    ulong carryMask = ulongPtr[7];
    ulong lengthPower = ulongPtr[8];
    ulong valueBytes = ulongPtr[9];
    ulong otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    ulong i, iLow, iHigh, j;
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
    global cmplx* stateVec, constant ulong* ulongPtr, global cmplx* nStateVec, constant unsigned char* values)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inputStart = ulongPtr[1];
    ulong inputMask = ulongPtr[2];
    ulong outputStart = ulongPtr[3];
    ulong outputMask = ulongPtr[4];
    ulong otherMask = ulongPtr[5];
    ulong carryIn = ulongPtr[6];
    ulong carryMask = ulongPtr[7];
    ulong lengthPower = ulongPtr[8];
    ulong valueBytes = ulongPtr[9];
    ulong otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    ulong i, iLow, iHigh, j;
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
