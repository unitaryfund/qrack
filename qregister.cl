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
        i = 0;
        iLow = iHigh & (carryMask - 1);
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
        i += iHigh;
        otherRes = (i & otherMask);
        inOutRes = (i & inOutMask);
        outInt = (inOutRes >> inOutStart) + toAdd;
        outRes = outInt << inOutStart;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = (outInt << inOutStart) | carryMask;
	}
        nStateVec[(outInt << inOutStart) | otherRes] = stateVec[i];
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
    ulong otherRes, inOutRes, outInt, i, iHigh, iLow;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0;
        iLow = iHigh & (carryMask - 1);
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
        i += iHigh;
        otherRes = (i & otherMask);
        outInt = (lengthMask + (inOutRes >> inOutStart)) - toSub;
        outRes = (outInt << inOutStart) | carryMask;
        if (outInt > lengthMask) {
            outInt &= lengthMask;
            outRes = (outInt << inOutStart);
	}
        nStateVec[(outInt << inOutStart) | otherRes] = stateVec[i];
    }
}

void kernel add(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong inMask = ulongPtr[2];
    ulong otherMask = ulongPtr[3];
    ulong lengthMask = ulongPtr[4] - 1;
    ulong inOutStart = ulongPtr[5];
    ulong inStart = ulongPtr[6];
    ulong otherRes, inOutRes, inOutInt, inRes, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        inOutRes = (lcv & inOutMask);
        inOutInt = inOutRes >> inOutStart;
        inRes = (lcv & inMask);
        inInt = inRes >> inStart;
        nStateVec[(((inOutInt + inInt) & lengthMask) << inOutStart) | otherRes | inRes] = stateVec[lcv];
    }
}

void kernel sub(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxI = ulongPtr[0];
    ulong inOutMask = ulongPtr[1];
    ulong inMask = ulongPtr[2];
    ulong otherMask = ulongPtr[3];
    ulong lengthPower = ulongPtr[4];
    ulong inOutStart = ulongPtr[5];
    ulong inStart = ulongPtr[6];
    ulong otherRes, inOutRes, inOutInt, inRes, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        inOutRes = (lcv & inOutMask);
        inOutInt = inOutRes >> inOutStart;
        inRes = (lcv & inMask);
        inInt = inRes >> inStart;
        nStateVec[(((inOutInt - inInt + lengthPower) & (lengthPower - 1)) << inOutStart) | otherRes | inRes]
            = stateVec[lcv];
    }
}

void kernel addc(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxQPower = ulongPtr[0];
    ulong maxI = ulongPtr[0] >> 1;
    ulong inOutMask = ulongPtr[1];
    ulong inMask = ulongPtr[2];
    ulong carryMask = ulongPtr[3];
    ulong otherMask = ulongPtr[4];
    ulong lengthPower = ulongPtr[5];
    ulong inOutStart = ulongPtr[6];
    ulong inStart = ulongPtr[7];
    ulong carryIndex = ulongPtr[8];
    ulong otherRes, inOutRes, inOutInt, inRes, carryInt, inInt, outInt, outRes;
    ulong iHigh, iLow, i, j;
    double2 tempX, temp1, temp2, tempY;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0;
        iLow = iHigh & (carryMask - 1);
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
        i += iHigh;
        otherRes = (i & otherMask);
        if (otherRes == i) {
            nStateVec[i] = stateVec[i];
        } else {
            inOutRes = (i & inOutMask);
            inOutInt = inOutRes >> inOutStart;
            inRes = (i & inMask);
            inInt = inRes >> inStart;
            outInt = (inOutInt + inInt);
            j = inOutInt - 1 + lengthPower;
            j %= lengthPower;
            j = (j << inOutStart) | (i ^ inOutRes) | carryMask;
            outRes = 0;
            if (outInt >= lengthPower) {
                outRes = carryMask;
                outInt ^= lengthPower;
            }
            outRes |= (outInt << inOutStart) | otherRes | inRes;
            temp1 = stateVec[i] * stateVec[i];
            temp2 = stateVec[j] * stateVec[j];
            tempX = temp1 + temp2;
            if ((temp1.x + temp1.y) > 0.0)
                temp1 = atan2(stateVec[i].x, stateVec[i].y);
            if ((temp2.x + temp2.y) > 0.0)
                temp2 = atan2(stateVec[j].x, stateVec[j].y);
            tempY = temp1 + temp2;
            nStateVec[outRes] = (double2)(tempX.x + tempX.y, tempY.x + tempY.y);
        }
    }
}

void kernel subc(global double2* stateVec, constant ulong* ulongPtr, global double2* nStateVec)
{
    ulong ID, Nthreads, lcv;

    ID = get_global_id(0);
    Nthreads = get_global_size(0);
    ulong maxQPower = ulongPtr[0];
    ulong maxI = ulongPtr[0] >> 1;
    ulong inOutMask = ulongPtr[1];
    ulong inMask = ulongPtr[2];
    ulong carryMask = ulongPtr[3];
    ulong otherMask = ulongPtr[4];
    ulong lengthPower = ulongPtr[5];
    ulong inOutStart = ulongPtr[6];
    ulong inStart = ulongPtr[7];
    ulong carryIndex = ulongPtr[8];
    ulong otherRes, inOutRes, inOutInt, inRes, carryInt, inInt, outInt, outRes;
    ulong iHigh, iLow, i, j;
    double2 tempX, temp1, temp2, tempY;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0;
        iLow = iHigh & (carryMask - 1);
        i += iLow;
        iHigh = (iHigh - iLow) << 1;
        i += iHigh;
        otherRes = (i & otherMask);
        if (otherRes == i) {
            nStateVec[i] = stateVec[i];
        } else {
            inOutRes = (i & inOutMask);
            inOutInt = inOutRes >> inOutStart;
            inRes = (i & inMask);
            inInt = inRes >> inStart;
            outInt = (inOutInt - inInt) + lengthPower;
            j = inOutInt + 1;
            j %= lengthPower;
            j = (j << inOutStart) | (i ^ inOutRes) | carryMask;
            outRes = 0;
            if (outInt >= lengthPower) {
                outRes = carryMask;
                outInt ^= lengthPower;
            }
            outRes |= (outInt << inOutStart) | otherRes | inRes;
            temp1 = stateVec[i] * stateVec[i];
            temp2 = stateVec[j] * stateVec[j];
            tempX = temp1 + temp2;
            if ((temp1.x + temp1.y) > 0.0)
                temp1 = atan2(stateVec[i].x, stateVec[i].y);
            if ((temp2.x + temp2.y) > 0.0)
                temp2 = atan2(stateVec[j].x, stateVec[j].y);
            tempY = temp1 + temp2;
            nStateVec[outRes] = (double2)(tempX.x + tempX.y, tempY.x + tempY.y);
        }
    }
}
