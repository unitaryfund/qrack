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
    ulong otherRes, regRes, regInt, outInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        outInt = ((regInt >> (length - shift)) | (regInt << shift)) & lengthMask;
        nStateVec[(outInt << start) | otherRes] = stateVec[lcv];
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
    ulong otherRes, regRes, regInt, outInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        outInt = ((regInt >> shift) | (regInt << (length - shift))) & lengthMask;
        nStateVec[(outInt << start) | otherRes] = stateVec[lcv];
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
        outRes = carryMask;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = 0;
        }
        outRes |= outInt << inOutStart;
        nStateVec[outRes | otherRes] = stateVec[i];
    }
}

void kernel superposeReg8(
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, global unsigned char* values)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inputStart = ulongPtr[1];
    ulong inputMask = ulongPtr[2];
    ulong outputStart = ulongPtr[3];
    ulong outputPower = 1 << outputStart;
    ulong inputRes, inputInt, outputRes, outputInt;
    ulong i, iLow, iHigh;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (outputPower - 1);
        i = iLow + ((iHigh - iLow) << 8);

        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputInt = values[inputInt];
        outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

void kernel adcReg8(
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, global unsigned char* values)
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
    ulong otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    ulong i, iLow, iHigh;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow + ((iHigh - iLow) << 1);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = (outputRes >> outputStart) + values[inputInt] + carryIn;

        carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[i];
    }
}

void kernel sbcReg8(
    global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec, global unsigned char* values)
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
    ulong otherRes, inputRes, inputInt, outputRes, outputInt, carryRes;
    ulong i, iLow, iHigh;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & (carryMask - 1);
        i = iLow + ((iHigh - iLow) << 1);

        otherRes = i & otherMask;
        inputRes = i & inputMask;
        inputInt = inputRes >> inputStart;
        outputRes = i & outputMask;
        outputInt = (outputRes >> outputStart) + lengthPower - (values[inputInt] + carryIn);

        carryRes = carryMask;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = 0;
        }

        outputRes = outputInt << outputStart;
        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[i];
    }
}
