#pragma OPENCL EXTENSION cl_khr_fp64 : enable

inline double2 zmul(const double2 lhs, const double2 rhs)
{
    return (lhs * (double2)(rhs.y, -(rhs.y))) + (rhs.x * (double2)(lhs.y, lhs.x));
}

void kernel apply2x2(global double2* stateVec, constant double2* cmplxPtr, constant ulong* ulongPtr)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant double2* mtrx = cmplxPtr;

    double2 nrm = cmplxPtr[4];
    ulong bitCount = ulongPtr[0];
    ulong maxI = ulongPtr[1];
    ulong offset1 = ulongPtr[2];
    ulong offset2 = ulongPtr[3];
    constant ulong* qPowersSorted = (ulongPtr + 4);

    double2 Y0;
    ulong i, iLow, iHigh;
    double2 qubit[2];
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
        qubit[0] = stateVec[i + offset1];
        qubit[1] = stateVec[i + offset2];

        Y0 = qubit[0];
        qubit[0] = zmul(nrm, (zmul(mtrx[0], Y0) + zmul(mtrx[1], qubit[1])));
        qubit[1] = zmul(nrm, (zmul(mtrx[2], Y0) + zmul(mtrx[3], qubit[1])));

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

void kernel apply2x2norm(global double2* stateVec, constant double2* cmplxPtr, constant ulong* ulongPtr, global double* nrmParts)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    constant double2* mtrx = cmplxPtr;

    double2 nrm = cmplxPtr[4];
    ulong bitCount = ulongPtr[0];
    ulong maxI = ulongPtr[1];
    ulong offset1 = ulongPtr[2];
    ulong offset2 = ulongPtr[3];
    constant ulong* qPowersSorted = (ulongPtr + 4);

    double2 Y0;
    ulong i, iLow, iHigh;
    double2 qubit[2];
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
        qubit[0] = stateVec[i + offset1];
        qubit[1] = stateVec[i + offset2];

        Y0 = qubit[0];
        qubit[0] = zmul(nrm, (zmul(mtrx[0], Y0) + zmul(mtrx[1], qubit[1])));
        qubit[1] = zmul(nrm, (zmul(mtrx[2], Y0) + zmul(mtrx[3], qubit[1])));

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

void kernel x(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel swap(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel rol(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel ror(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel inc(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel dec(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel incc(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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

void kernel decc(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
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
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, constant unsigned char* values)
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
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, constant unsigned char* values)
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
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, constant unsigned char* values)
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
