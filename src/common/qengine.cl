//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline cmplx2 zmatrixmul(const real1 nrm, const cmplx4 lhs, const cmplx2 rhs)
{
    return nrm * ((cmplx2)(
        (lhs.lo.x * rhs.x) - (lhs.lo.y * rhs.y) + (lhs.lo.z * rhs.z) - (lhs.lo.w * rhs.w),
        (lhs.lo.x * rhs.y) + (lhs.lo.y * rhs.x) + (lhs.lo.z * rhs.w) + (lhs.lo.w * rhs.z),
        (lhs.hi.x * rhs.x) - (lhs.hi.y * rhs.y) + (lhs.hi.z * rhs.z) - (lhs.hi.w * rhs.w),
        (lhs.hi.x * rhs.y) + (lhs.hi.y * rhs.x) + (lhs.hi.z * rhs.w) + (lhs.hi.w * rhs.z)
    ));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return atan2(cmp.y, cmp.x);
}

#define OFFSET2_ARG bitCapIntOclPtr[0]
#define OFFSET1_ARG bitCapIntOclPtr[1]
#define MAXI_ARG bitCapIntOclPtr[2]
#define BITCOUNT_ARG bitCapIntOclPtr[3]
#define ID get_global_id(0)

#define PREP_2X2()                                                                   \
    bitCapIntOcl lcv, i;                                                                \
    bitCapIntOcl Nthreads = get_global_size(0);                                         \
                                                                                     \
    cmplx4 mtrx = vload8(0, cmplxPtr);                                               \
    real1 nrm = cmplxPtr[8];                                                         \
                                                                                     \
    cmplx2 mulRes;

#define PREP_2X2_NORM()                                                              \
    real1 norm_thresh = cmplxPtr[9];                                                 \
    real1 dotMulRes;

#define PREP_SPECIAL_2X2()                                                           \
    bitCapIntOcl lcv, i;                                                                \
    bitCapIntOcl Nthreads = get_global_size(0);                                         \
    cmplx Y0;

#define PREP_Z_2X2()                                                                 \
    bitCapIntOcl lcv, i;                                                                \
    bitCapIntOcl Nthreads = get_global_size(0);

#define PUSH_APART_GEN()                                                             \
    iHigh = lcv;                                                                     \
    i = 0U;                                                                          \
    for (p = 0U; p < BITCOUNT_ARG; p++) {                                            \
        iLow = iHigh & (qPowersSorted[p] - ONE_BCI);                                 \
        i |= iLow;                                                                   \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                           \
    }                                                                                \
    i |= iHigh;

#define PUSH_APART_1()                                                               \
    i = lcv & qMask;                                                                 \
    i |= (lcv ^ i) << ONE_BCI;

#define PUSH_APART_2()                                                               \
    i = lcv & qMask1;                                                                \
    iHigh = (lcv ^ i) << ONE_BCI;                                                    \
    iLow = iHigh & qMask2;                                                           \
    i |= iLow | ((iHigh ^ iLow) << ONE_BCI);

#define APPLY_AND_OUT()                                                              \
    mulRes.lo = stateVec[i | OFFSET1_ARG];                                           \
    mulRes.hi = stateVec[i | OFFSET2_ARG];                                           \
                                                                                     \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                          \
                                                                                     \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                           \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

#define APPLY_X()                                                                    \
    Y0 = stateVec[i];                                                                \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                         \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z()                                                                    \
    stateVec[i | OFFSET2_ARG] = -stateVec[i | OFFSET2_ARG];

#define SUM_2X2()                                                                    \
    locID = get_local_id(0);                                                         \
    locNthreads = get_local_size(0);                                                 \
    lProbBuffer[locID] = partNrm;                                                    \
                                                                                     \
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {                \
        barrier(CLK_LOCAL_MEM_FENCE);                                                \
        if (locID < lcv) {                                                           \
            lProbBuffer[locID] += lProbBuffer[locID + lcv];                          \
        }                                                                            \
    }                                                                                \
                                                                                     \
    if (locID == 0U) {                                                               \
        nrmParts[get_group_id(0)] = lProbBuffer[0];                                  \
    }

#define NORM_BODY_2X2()                                                              \
    mulRes.lo = stateVec[i | OFFSET1_ARG];                                           \
    mulRes.hi = stateVec[i | OFFSET2_ARG];                                           \
                                                                                     \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                          \
                                                                                     \
    dotMulRes = dot(mulRes.lo, mulRes.lo);                                           \
    if (dotMulRes < norm_thresh) {                                                   \
        mulRes.lo = (cmplx)(ZERO_R1, ZERO_R1);                                       \
    } else {                                                                         \
        partNrm += dotMulRes;                                                        \
    }                                                                                \
                                                                                     \
    dotMulRes = dot(mulRes.hi, mulRes.hi);                                           \
    if (dotMulRes < norm_thresh) {                                                   \
        mulRes.hi = (cmplx)(ZERO_R1, ZERO_R1);                                       \
    } else {                                                                         \
        partNrm += dotMulRes;                                                        \
    }                                                                                \
                                                                                     \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                           \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

void kernel apply2x2(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr, constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2();
    
    bitCapIntOcl iLow, iHigh;
    bitLenInt p;
    
    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_GEN();
        APPLY_AND_OUT();
    }
}

void kernel apply2x2single(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_AND_OUT();
    }
}

void kernel apply2x2double(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2();

    bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    bitCapIntOcl iLow, iHigh;

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_2();
        APPLY_AND_OUT();
    }
}

void kernel apply2x2wide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr, constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2();
    
    bitCapIntOcl iLow, iHigh;
    bitLenInt p;
    
    lcv = ID;
    PUSH_APART_GEN();
    APPLY_AND_OUT();
}

void kernel apply2x2singlewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_AND_OUT();
}

void kernel apply2x2doublewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2();

    bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    bitCapIntOcl iLow, iHigh;

    lcv = ID;
    PUSH_APART_2();
    APPLY_AND_OUT();
}

void kernel apply2x2normsingle(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr, global real1* nrmParts, local real1* lProbBuffer)
{
    PREP_2X2();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    bitCapIntOcl locID, locNthreads;
    cmplx YT;
    real1 partNrm = ZERO_R1;

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        NORM_BODY_2X2();
    }

    SUM_2X2();
}

void kernel apply2x2normsinglewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr, global real1* nrmParts, local real1* lProbBuffer)
{
    PREP_2X2();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    bitCapIntOcl locID, locNthreads;
    cmplx YT;
    real1 partNrm = ZERO_R1;

    lcv = ID;
    PUSH_APART_1();
    NORM_BODY_2X2();

    SUM_2X2();
}

void kernel xsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_SPECIAL_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_X();
    }
}

void kernel xsinglewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_SPECIAL_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_X();
}

void kernel zsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_Z_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_Z();
    }
}

void kernel zsinglewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_Z_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_Z();
}

void kernel uniformlycontrolled(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant bitCapIntOcl* qPowers, constant cmplx4* mtrxs, constant real1* nrmIn, global real1* nrmParts, local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl targetPower = bitCapIntOclPtr[1];
    bitCapIntOcl targetMask = targetPower - ONE_BCI;
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl mtrxSkipLen = bitCapIntOclPtr[3];
    bitCapIntOcl mtrxSkipValueMask = bitCapIntOclPtr[4];

    real1 nrm = nrmIn[0];

    real1 partNrm = ZERO_R1;

    cmplx2 qubit;
    cmplx Y0;

    bitCapIntOcl i, offset;
    bitCapIntOcl j, jHigh, jLow, p;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & targetMask;
        i |= (lcv ^ i) << ONE_BCI;

        offset = 0;
        for (p = 0; p < controlLen; p++) {
            if (i & qPowers[p]) {
                offset |= ONE_BCI << p;
            }
        }

        jHigh = offset;
        j = 0;
        for (p = 0; p < mtrxSkipLen; p++) {
            jLow = jHigh & (qPowers[controlLen + p] - ONE_BCI);
            j |= jLow;
            jHigh = (jHigh ^ jLow) << ONE_BCI;
        }
        j |= jHigh;
        offset = j | mtrxSkipValueMask;

        qubit.lo = stateVec[i];
        qubit.hi = stateVec[i | targetPower];

        qubit = zmatrixmul(nrm, mtrxs[offset], qubit);

        partNrm += dot(qubit, qubit);

        stateVec[i] = qubit.lo;
        stateVec[i | targetPower] = qubit.hi;
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lProbBuffer[locID] = partNrm;
    
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        } 
    }

    if (locID == 0U) {
        nrmParts[get_group_id(0)] = lProbBuffer[0];
    }
}

void kernel compose(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    //bitCapIntOcl nMaxQPower = args.x;
    //bitCapIntOcl qubitCount = args.y;
    //bitCapIntOcl startMask = args.z;
    //bitCapIntOcl endMask = args.w;

    for (lcv = ID; lcv < args.x; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
    }
}

void kernel composewide(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl lcv = ID;
    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    //bitCapIntOcl nMaxQPower = args.x;
    //bitCapIntOcl qubitCount = args.y;
    //bitCapIntOcl startMask = args.z;
    //bitCapIntOcl endMask = args.w;

    nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
}

void kernel composemid(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl nMaxQPower = args.x;
    bitCapIntOcl qubitCount = args.y;
    bitCapIntOcl oQubitCount = args.z;
    bitCapIntOcl startMask = args.w;
    bitCapIntOcl midMask = bitCapIntOclPtr[4];
    bitCapIntOcl endMask = bitCapIntOclPtr[5];    
    bitCapIntOcl start = bitCapIntOclPtr[6];

    for (lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)], stateVec2[(lcv & midMask) >> start]);
    }
}

void kernel decomposeprob(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* remainderStateProb, global real1* remainderStateAngle, global real1* partStateProb, global real1* partStateAngle)
{
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl partPower = args.x;
    bitCapIntOcl remainderPower = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;

    bitCapIntOcl j, k, l;
    cmplx amp;
    real1 partProb, nrm, firstAngle, currentAngle;

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv & ((ONE_BCI << start) - ONE_BCI);
        j |= (lcv ^ j) << len;

        partProb = ZERO_R1;
        firstAngle = -16 * PI_R1;

        for (k = 0U; k < partPower; k++) {
            l = j | (k << start);

            amp = stateVec[l];
            nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm > min_norm) {
                currentAngle = arg(amp);
                if (firstAngle < (-8 * PI_R1)) {
                    firstAngle = currentAngle;
                }
                partStateAngle[k] = currentAngle - firstAngle;
            }
        }

        remainderStateProb[lcv] = partProb;
    }

    for (lcv = ID; lcv < partPower; lcv += Nthreads) {
        j = lcv << start;

        partProb = ZERO_R1;
        firstAngle = -16 * PI_R1;

        for (k = 0U; k < remainderPower; k++) {
            l = k & ((ONE_BCI << start) - ONE_BCI);
            l |= (k ^ l) << len;
            l = j | l;
            
            amp = stateVec[l];
            nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm > min_norm) {
                currentAngle = arg(stateVec[l]);
                if (firstAngle < (-8 * PI_R1)) {
                    firstAngle = currentAngle;
                }
                remainderStateAngle[k] = currentAngle - firstAngle;
            }
        }

        partStateProb[lcv] = partProb;
    }
}

void kernel decomposeamp(global real1* stateProb, global real1* stateAngle, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    real1 angle;
    for (lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((cmplx)(angle + SineShift, angle));
    }
}

void kernel dispose(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl remainderPower = bitCapIntOclPtr[0];
    bitCapIntOcl len = bitCapIntOclPtr[1];
    bitCapIntOcl skipMask = bitCapIntOclPtr[2];
    bitCapIntOcl disposedRes = bitCapIntOclPtr[3];
    bitCapIntOcl i, iLow, iHigh;
    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | ((iHigh ^ iLow) << (bitCapIntOcl)len) | disposedRes;
        nStateVec[lcv] = stateVec[i];
    }
}

void kernel prob(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer, local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl qPower = args.y;
    bitCapIntOcl qMask = qPower - ONE_BCI;

    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapIntOcl i;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << ONE_BCI) | qPower;
        amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lProbBuffer[locID] = oneChancePart;
    
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        } 
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

void kernel probreg(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer, local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl perm = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;
    bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapIntOcl i;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << len);
        amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lProbBuffer[locID] = oneChancePart;
    
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        } 
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

void kernel probregall(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer)
{
    bitCapIntOcl Nthreads, lcv1, lcv2;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl maxJ = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;
    bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    real1 oneChancePart;
    cmplx amp;
    bitCapIntOcl perm;
    bitCapIntOcl i;

    for (lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        perm = lcv1 << start;
        oneChancePart = ZERO_R1;
        for (lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            i = lcv2 & qMask;
            i |= ((lcv2 ^ i) << len);
            amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        oneChanceBuffer[lcv1] = oneChancePart;
    }
}

void kernel probmask(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer, constant bitCapIntOcl* qPowers, local real1* lProbBuffer )
{
    bitCapIntOcl Nthreads, locID, locNthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl mask = args.y;
    bitCapIntOcl perm = args.z;
    bitCapIntOcl len = args.w;

    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapIntOcl i, iHigh, iLow, p;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0U;
        for (p = 0U; p < len; p++) {
            iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lProbBuffer[locID] = oneChancePart;
    
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        } 
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

void kernel probmaskall(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer, constant bitCapIntOcl* qPowersMask, constant bitCapIntOcl* qPowersSkip)
{
    bitCapIntOcl Nthreads, lcv1, lcv2;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl maxJ = args.y;
    bitCapIntOcl maskLen = args.z;
    bitCapIntOcl skipLen = args.w;

    real1 oneChancePart;
    cmplx amp;
    bitCapIntOcl perm;
    bitCapIntOcl i, iHigh, iLow, p;

    for (lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        iHigh = lcv1;
        perm = 0U;
        for (p = 0U; p < skipLen; p++) {
            iLow = iHigh & (qPowersSkip[p] - ONE_BCI);
            perm |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        perm |= iHigh;

        oneChancePart = ZERO_R1;
        for (lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            iHigh = lcv2;
            i = 0U;
            for (p = 0U; p < maskLen; p++) {
                iLow = iHigh & (qPowersMask[p] - ONE_BCI);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << ONE_BCI;
            }
            i |= iHigh;

            amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        oneChanceBuffer[lcv1] = oneChancePart;
    }
}

void kernel rol(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl regMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    bitCapIntOcl start = bitCapIntOclPtr[4];
    bitCapIntOcl shift = bitCapIntOclPtr[5];
    bitCapIntOcl length = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, regRes, regInt, inInt;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = (lcv & otherMask);
        regRes = (lcv & regMask);
        regInt = regRes >> start;
        inInt = ((regInt >> shift) | (regInt << (length - shift))) & lengthMask;
        nStateVec[lcv] = stateVec[(inInt << start) | otherRes];
    }
}

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
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | (i & otherMask)] = stateVec[i];
    }
}

void kernel cinc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
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
        nStateVec[(((((i & inOutMask) >> inOutStart) + toAdd) & lengthMask) << inOutStart) | otherRes | controlMask] = stateVec[i | controlMask];
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
        amp = stateVec[i];
        if (isOverflow) {
            amp = -amp;
        }
        nStateVec[outRes] = amp;
    }
}

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
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

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

void kernel mul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toMul = bitCapIntOclPtr[1];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    bitCapIntOcl carryMask = bitCapIntOclPtr[3];
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
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes] = stateVec[i];
    }
}

void kernel div(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[2];
    bitCapIntOcl carryMask = bitCapIntOclPtr[3];
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
        nStateVec[i] = stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes];
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define MODNOUT(indexIn, indexOut)                                                                       \
    bitCapIntOcl Nthreads, lcv;                                                                             \
                                                                                                         \
    Nthreads = get_global_size(0);                                                                       \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                    \
    bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                   \
    bitCapIntOcl inMask = bitCapIntOclPtr[2];                                                                  \
    bitCapIntOcl outMask = bitCapIntOclPtr[3];                                                                 \
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];                                                               \
    bitCapIntOcl len = bitCapIntOclPtr[5];                                                                     \
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;                                                      \
    bitCapIntOcl inStart = bitCapIntOclPtr[6];                                                                 \
    bitCapIntOcl outStart = bitCapIntOclPtr[7];                                                                \
    bitCapIntOcl skipMask = bitCapIntOclPtr[8];                                                                \
    bitCapIntOcl modN = bitCapIntOclPtr[9];                                                                    \
    bitCapIntOcl otherRes, inRes, outRes;                                                                   \
    bitCapIntOcl i, iHigh, iLow;                                                                            \
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {                                                        \
        iHigh = lcv;                                                                                     \
        iLow = iHigh & skipMask;                                                                         \
        i = iLow | (iHigh ^ iLow) << len;                                                                \
                                                                                                         \
        otherRes = i & otherMask;                                                                        \
        inRes = i & inMask;                                                                              \
        outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                      \
        nStateVec[indexOut] = stateVec[indexIn];                                                         \
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
    bitCapIntOcl outMask = bitCapIntOclPtr[3];
    bitCapIntOcl otherMask = bitCapIntOclPtr[4];
    bitCapIntOcl len = bitCapIntOclPtr[5];
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
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

    bitCapIntOcl i, iLow, iHigh;

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

    bitCapIntOcl i, iLow, iHigh;

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

#define CMOD_START()                                                                 \
    iHigh = lcv;                                                                     \
    i = 0U;                                                                          \
    for (p = 0U; p < (controlLen + len); p++) {                                      \
        iLow = iHigh & (controlPowers[p] - ONE_BCI);                                 \
        i |= iLow;                                                                   \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                           \
    }                                                                                \
    i |= iHigh;                                                                      \

#define CMOD_FINISH()                                                                \
    nStateVec[i] = stateVec[i];                                                      \
    for (j = ONE_BCI; j < ((ONE_BCI << controlLen) - ONE_BCI); j++) {                \
        partControlMask = 0U;                                                        \
        for (k = 0U; k < controlLen; k++) {                                          \
            if (j & (ONE_BCI << k)) {                                                \
                partControlMask |= controlPowers[controlLen + len + k];              \
            }                                                                        \
        }                                                                            \
        nStateVec[i | partControlMask] = stateVec[i | partControlMask];              \
    }

void kernel cmul(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toMul = bitCapIntOclPtr[1];
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    bitCapIntOcl carryMask = bitCapIntOclPtr[5];
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
        nStateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask] = stateVec[i | controlMask];

        CMOD_FINISH();
    }
}

void kernel cdiv(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl toDiv = bitCapIntOclPtr[1];
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[4];
    bitCapIntOcl carryMask = bitCapIntOclPtr[5];
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
        nStateVec[i | controlMask] = stateVec[((outInt & lowMask) << inOutStart) | (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask];

        CMOD_FINISH();
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define CMODNOUT(indexIn, indexOut)                                                                      \
    bitCapIntOcl Nthreads, lcv;                                                                             \
                                                                                                         \
    Nthreads = get_global_size(0);                                                                       \
    bitCapIntOcl maxI = bitCapIntOclPtr[0];                                                                    \
    bitCapIntOcl toMul = bitCapIntOclPtr[1];                                                                   \
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];                                                              \
    bitCapIntOcl controlMask = bitCapIntOclPtr[3];                                                             \
    bitCapIntOcl inMask = bitCapIntOclPtr[4];                                                                  \
    bitCapIntOcl outMask = bitCapIntOclPtr[5];                                                                 \
    bitCapIntOcl modN = bitCapIntOclPtr[6];                                                                    \
    bitCapIntOcl len = bitCapIntOclPtr[7];                                                                     \
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;                                                      \
    bitCapIntOcl inStart = bitCapIntOclPtr[8];                                                                 \
    bitCapIntOcl outStart = bitCapIntOclPtr[9];                                                                \
                                                                                                         \
    bitCapIntOcl otherMask = (maxI - ONE_BCI) ^ (inMask | outMask | controlMask);                           \
    maxI >>= (controlLen + len);                                                                         \
                                                                                                         \
    bitCapIntOcl otherRes, outRes, inRes;                                                                   \
    bitCapIntOcl i, iHigh, iLow, j;                                                                         \
    bitLenInt p, k;                                                                                      \
    bitCapIntOcl partControlMask;                                                                           \
                                                                                                         \
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {                                                        \
        CMOD_START();                                                                                    \
                                                                                                         \
        otherRes = i & otherMask;                                                                        \
        inRes = i & inMask;                                                                              \
        outRes = (((inRes >> inStart) * toMul) % modN) << outStart;                                      \
        nStateVec[indexOut] = stateVec[indexIn];                                                         \
                                                                                                         \
        CMOD_FINISH();                                                                                   \
    }

void kernel cmulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
{
    CMODNOUT((i | controlMask), (inRes | outRes | otherRes | controlMask));
}

void kernel cimulmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
{
    CMODNOUT((inRes | outRes | otherRes | controlMask), (i | controlMask));
}

void kernel cpowmodnout(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitCapIntOcl* controlPowers)
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
    bitCapIntOcl lowMask = (ONE_BCI << len) - ONE_BCI;
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

void kernel indexedLda(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitLenInt* values)
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
        for (j = 0U; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
        }
        outputRes = outputInt << outputStart;
        nStateVec[outputRes | i] = stateVec[i];
    }
}

void kernel indexedAdc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitLenInt* values)
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
        for (j = 0U; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
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

void kernel indexedSbc(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitLenInt* values)
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
        for (j = 0U; j < valueBytes; j++) {
            outputInt |= values[inputInt * valueBytes + j] << (8U * j);
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

void kernel hash(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec, constant bitLenInt* values)
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
        for (j = 0U; j < bytes; j++) {
            outputInt |= values[inputInt * bytes + j] << (8U * j);
        }
        outputRes = outputInt << start;
        nStateVec[outputRes | (lcv & ~inputRes)] = stateVec[lcv];
    }
}

void kernel nrmlze(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr) {
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    real1 norm_thresh = args_ptr[0];
    real1 nrm = args_ptr[1];
    cmplx amp;
    
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        if (dot(amp, amp) < norm_thresh) {
            amp = (cmplx)(ZERO_R1, ZERO_R1);
        }
        stateVec[lcv] = nrm * amp;
    }
}

void kernel nrmlzewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr) {
    bitCapIntOcl lcv = ID;
    real1 norm_thresh = args_ptr[0];
    real1 nrm = args_ptr[1];
    cmplx amp;

    amp = stateVec[lcv];
    if (dot(amp, amp) < norm_thresh) {
        amp = (cmplx)(ZERO_R1, ZERO_R1);
    }
    stateVec[lcv] = nrm * amp;
}

void kernel updatenorm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr, global real1* norm_ptr, local real1* lProbBuffer) {
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    real1 norm_thresh = args_ptr[0];
    cmplx amp;
    real1 nrm;
    real1 partNrm = ZERO_R1;


    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        nrm = dot(amp, amp);
        if (nrm < norm_thresh) {
            nrm = ZERO_R1;
        }
        partNrm += nrm;
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lProbBuffer[locID] = partNrm;
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        } 
    }

    if (locID == 0U) {
        norm_ptr[get_group_id(0)] = lProbBuffer[0];
    }
}

void kernel approxcompare(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global real1* norm_ptr, local real1* lProbBuffer) {
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    cmplx amp;
    real1 partNrm = ZERO_R1;

    // Hopefully, since this is identical redundant work by all elements, the break hits for all at the same time.
    cmplx basePhaseFac1;
    real1 nrm;
    bitCapIntOcl basePerm = 0;
    do {
        amp = stateVec1[basePerm];
        nrm = dot(amp, amp);
        basePerm++;
    } while (nrm < min_norm);

    basePerm--;
    amp = stateVec1[basePerm];
    nrm = dot(amp, amp);

    // If the amplitude we sample for global phase offset correction doesn't match, we're done.
    if (nrm > min_norm) {
        basePhaseFac1 = (ONE_R1 / sqrt(nrm)) * amp;

        amp = stateVec2[basePerm];
        cmplx basePhaseFac2 = (ONE_R1 / sqrt(dot(amp, amp))) * amp;

        for (lcv = ID; lcv < maxI; lcv += Nthreads) {
            amp = zmul(basePhaseFac2, stateVec1[lcv]) - zmul(basePhaseFac1, stateVec2[lcv]);
            partNrm += dot(amp, amp);
        }

        locID = get_local_id(0);
        locNthreads = get_local_size(0);
        lProbBuffer[locID] = partNrm;
    
        for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if (locID < lcv) {
                lProbBuffer[locID] += lProbBuffer[locID + lcv];
            }
        }

        if (locID == 0U) {
            norm_ptr[get_group_id(0)] = lProbBuffer[0];
        }
    } else {
        norm_ptr[get_group_id(0)] = 10;
    }
}

void kernel applym(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr) {
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qPower = bitCapIntOclPtr[1];
    bitCapIntOcl qMask = qPower - ONE_BCI;
    bitCapIntOcl savePower = bitCapIntOclPtr[2];
    bitCapIntOcl discardPower = qPower ^ savePower;
    cmplx nrm = cmplx_ptr[0];
    bitCapIntOcl i, iLow, iHigh, j;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & qMask;
        i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel applymreg(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr) {
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl mask = bitCapIntOclPtr[1];
    bitCapIntOcl result = bitCapIntOclPtr[2];
    cmplx nrm = cmplx_ptr[0];

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = ((lcv & mask) == result) ? zmul(nrm, stateVec[lcv]) : (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel phaseflip(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = -stateVec[lcv];
    }
}

void kernel zerophaseflip(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;
    bitCapIntOcl i, iLow, iHigh;
    
    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl skipMask = bitCapIntOclPtr[1] - ONE_BCI;
    bitCapIntOcl skipLength = bitCapIntOclPtr[2];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & skipMask;
        i = iLow | ((iHigh ^ iLow) << skipLength);

        stateVec[i] = -stateVec[i];
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
    cmplx amp;
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
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    }
}
