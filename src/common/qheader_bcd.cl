void kernel incbcd(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[3];
    bitCapIntOcl toAdd = bitCapIntOclPtr[4];
    int nibbleCount = bitCapIntOclPtr[5];
    bitCapIntOcl otherRes, partToAdd, inOutRes, inOutInt, outInt, outRes;
    int test1, test2;
    int j;
    // For 64 qubits, we would have 16 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    int nibbles[16];
    bool isValid;
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        partToAdd = toAdd;
        inOutRes = lcv & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = inOutInt & 15U;
            inOutInt >>= 4U;
            test2 = partToAdd % 10;
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        amp = stateVec[lcv];
        if (isValid) {
            outInt = 0;
            outRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= ((bitCapIntOcl)nibbles[j]) << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes;
            nStateVec[outRes] = amp;
        } else {
            nStateVec[lcv] = amp;
        }
    }
}

void kernel incdecbcdc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl carryMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[4];
    bitCapIntOcl toAdd = bitCapIntOclPtr[5];
    int nibbleCount = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, partToAdd, inOutRes, inOutInt, outInt, outRes, carryRes, i;
    int test1, test2;
    int j;
    // For 64 qubits, we would have 16 nibbles. For now, there's no reason not overallocate in
    // fast private memory.
    int nibbles[16];
    bool isValid;
    cmplx amp1, amp2;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & (carryMask - ONE_BCI);
        i |= (lcv ^ i) << ONE_BCI;

        otherRes = i & otherMask;
        partToAdd = toAdd;
        inOutRes = i & inOutMask;
        inOutInt = inOutRes >> inOutStart;
        isValid = true;

        test1 = inOutInt & 15U;
        inOutInt >>= 4U;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        amp1 = stateVec[i];
        amp2 = stateVec[i | carryMask];
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
                outInt |= ((bitCapIntOcl)nibbles[j]) << (j * 4);
            }
            outRes = (outInt << inOutStart) | otherRes | carryRes;
            nStateVec[outRes] = amp1;
            outRes ^= carryMask;
            nStateVec[outRes] = amp2;
        } else {
            nStateVec[i] = amp1;
            nStateVec[i | carryMask] = amp2;
        }
    }
}
