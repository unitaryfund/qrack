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

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline cmplx2 zmatrixmul(const real1 nrm, const cmplx4 lhs, const cmplx2 rhs)
{
    return nrm *
        ((cmplx2)((lhs.lo.x * rhs.x) - (lhs.lo.y * rhs.y) + (lhs.lo.z * rhs.z) - (lhs.lo.w * rhs.w),
            (lhs.lo.x * rhs.y) + (lhs.lo.y * rhs.x) + (lhs.lo.z * rhs.w) + (lhs.lo.w * rhs.z),
            (lhs.hi.x * rhs.x) - (lhs.hi.y * rhs.y) + (lhs.hi.z * rhs.z) - (lhs.hi.w * rhs.w),
            (lhs.hi.x * rhs.y) + (lhs.hi.y * rhs.x) + (lhs.hi.z * rhs.w) + (lhs.hi.w * rhs.z)));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return (real1)atan2((real1_f)cmp.y, (real1_f)cmp.x);
}

inline cmplx conj(const cmplx cmp)
{
    return (cmplx)(cmp.x, -cmp.y);
}

#define OFFSET2_ARG bitCapIntOclPtr[0]
#define OFFSET1_ARG bitCapIntOclPtr[1]
#define MAXI_ARG bitCapIntOclPtr[2]
#define BITCOUNT_ARG bitCapIntOclPtr[3]
#define ID get_global_id(0)

#define PREP_2X2()                                                                                                     \
    bitCapIntOcl lcv, i;                                                                                               \
    bitCapIntOcl Nthreads = get_global_size(0);                                                                        \
                                                                                                                       \
    cmplx4 mtrx = *((constant cmplx4*)cmplxPtr);                                                                                 \
    real1 nrm = cmplxPtr[8];                                                                                           \
                                                                                                                       \
    cmplx2 mulRes;

#define PREP_2X2_WIDE()                                                                                                \
    bitCapIntOcl lcv, i;                                                                                               \
                                                                                                                       \
    cmplx4 mtrx = *((constant cmplx4*)cmplxPtr);                                                                                 \
    real1 nrm = cmplxPtr[8];                                                                                           \
                                                                                                                       \
    cmplx2 mulRes;

#define PREP_2X2_NORM()                                                                                                \
    real1 norm_thresh = cmplxPtr[9];                                                                                   \
    real1 dotMulRes;

#define PREP_SPECIAL_2X2()                                                                                             \
    bitCapIntOcl lcv, i;

#define PUSH_APART_GEN()                                                                                               \
    i = 0U;                                                                                                            \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    for (bitLenInt p = 0U; p < BITCOUNT_ARG; p++) {                                                                    \
        bitCapIntOcl iLow = iHigh & (qPowersSorted[p] - ONE_BCI);                                                      \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define PUSH_APART_1()                                                                                                 \
    i = lcv & qMask;                                                                                                   \
    i |= (lcv ^ i) << ONE_BCI;

#define PUSH_APART_2()                                                                                                 \
    i = lcv & qMask1;                                                                                                  \
    bitCapIntOcl iHigh = (lcv ^ i) << ONE_BCI;                                                                         \
    bitCapIntOcl iLow = iHigh & qMask2;                                                                                \
    i |= iLow | ((iHigh ^ iLow) << ONE_BCI);

#define APPLY_AND_OUT()                                                                                                \
    mulRes.lo = stateVec[i | OFFSET1_ARG];                                                                             \
    mulRes.hi = stateVec[i | OFFSET2_ARG];                                                                             \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

#define APPLY_X()                                                                                                      \
    cmplx Y0 = stateVec[i];                                                                                            \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                                                           \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z() stateVec[i | OFFSET2_ARG] = -stateVec[i | OFFSET2_ARG];

#define APPLY_PHASE()                                                                                                  \
    stateVec[i] = zmul(topLeft, stateVec[i]);                                                                          \
    stateVec[i | OFFSET2_ARG] = zmul(bottomRight, stateVec[i | OFFSET2_ARG]);

#define APPLY_INVERT()                                                                                                 \
    cmplx Y0 = stateVec[i];                                                                                            \
    stateVec[i] = zmul(topRight, stateVec[i | OFFSET2_ARG]);                                                           \
    stateVec[i | OFFSET2_ARG] = zmul(bottomLeft, Y0);

#define SUM_2X2()                                                                                                      \
    locID = get_local_id(0);                                                                                           \
    locNthreads = get_local_size(0);                                                                                   \
    lProbBuffer[locID] = partNrm;                                                                                      \
                                                                                                                       \
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {                                                  \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \
        if (locID < lcv) {                                                                                             \
            lProbBuffer[locID] += lProbBuffer[locID + lcv];                                                            \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    if (locID == 0U) {                                                                                                 \
        nrmParts[get_group_id(0)] = lProbBuffer[0];                                                                    \
    }

#define NORM_BODY_2X2()                                                                                                \
    mulRes.lo = stateVec[i | OFFSET1_ARG];                                                                             \
    mulRes.hi = stateVec[i | OFFSET2_ARG];                                                                             \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    dotMulRes = dot(mulRes.lo, mulRes.lo);                                                                             \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.lo = (cmplx)(ZERO_R1, ZERO_R1);                                                                         \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    dotMulRes = dot(mulRes.hi, mulRes.hi);                                                                             \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.hi = (cmplx)(ZERO_R1, ZERO_R1);                                                                         \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

void kernel apply2x2(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2();

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

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_2();
        APPLY_AND_OUT();
    }
}

void kernel apply2x2wide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowersSorted)
{
    PREP_2X2_WIDE();

    lcv = ID;
    PUSH_APART_GEN();
    APPLY_AND_OUT();
}

void kernel apply2x2singlewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_AND_OUT();
}

void kernel apply2x2doublewide(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE();

    bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    bitCapIntOcl qMask2 = bitCapIntOclPtr[4];

    lcv = ID;
    PUSH_APART_2();
    APPLY_AND_OUT();
}

void kernel apply2x2normsingle(global cmplx* stateVec, constant real1* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr,
    global real1* nrmParts, local real1* lProbBuffer)
{
    PREP_2X2();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    bitCapIntOcl locID, locNthreads;
    real1 partNrm = ZERO_R1;

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        NORM_BODY_2X2();
    }

    SUM_2X2();
}

void kernel apply2x2normsinglewide(global cmplx* stateVec, constant real1* cmplxPtr,
    constant bitCapIntOcl* bitCapIntOclPtr, global real1* nrmParts, local real1* lProbBuffer)
{
    PREP_2X2_WIDE();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    bitCapIntOcl locID, locNthreads;
    real1 partNrm = ZERO_R1;

    lcv = ID;
    PUSH_APART_1();
    NORM_BODY_2X2();

    SUM_2X2();
}

void kernel xsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_SPECIAL_2X2();
    bitCapIntOcl Nthreads = get_global_size(0);

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

void kernel xmask(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl lcv, otherRes, setInt, resetInt;

    bitCapIntOcl Nthreads = get_global_size(0);

    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl mask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        setInt = lcv & mask;
        resetInt = setInt ^ mask;

        if (setInt < resetInt) {
            continue;
        }

        setInt |= otherRes;
        resetInt |= otherRes;

        cmplx Y0 = stateVec[resetInt];
        stateVec[resetInt] = stateVec[setInt];
        stateVec[setInt] = Y0;
    }
}

void kernel phaseparity(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplxPtr)
{
    bitCapIntOcl lcv, otherRes, setInt, v;
    bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    bitCapIntOcl paritySize;

    bitCapIntOcl Nthreads = get_global_size(0);

    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl mask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];

    cmplx phaseFac = cmplxPtr[0];
    cmplx iPhaseFac = cmplxPtr[1];

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        otherRes = lcv & otherMask;
        setInt = lcv & mask;

        v = setInt;
        for (paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        setInt |= otherRes;

        stateVec[setInt] = zmul(v ? phaseFac : iPhaseFac, stateVec[setInt]);
    }
}

void kernel zsingle(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl lcv, i;
    bitCapIntOcl Nthreads = get_global_size(0);

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_Z();
    }
}

void kernel zsinglewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl i;
    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    bitCapIntOcl lcv = ID;
    PUSH_APART_1();
    APPLY_Z();
}

void kernel phasesingle(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl lcv, i;
    bitCapIntOcl Nthreads = get_global_size(0);

    bitCapIntOcl qMask = bitCapIntOclPtr[3];
    cmplx topLeft = cmplxPtr[0];
    cmplx bottomRight = cmplxPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_PHASE();
    }
}

void kernel phasesinglewide(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl i;
    
    bitCapIntOcl qMask = bitCapIntOclPtr[2];
    cmplx topLeft = cmplxPtr[0];
    cmplx bottomRight = cmplxPtr[3];

    bitCapIntOcl lcv = ID;
    PUSH_APART_1();
    APPLY_PHASE();
}

void kernel invertsingle(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl lcv, i;
    bitCapIntOcl Nthreads = get_global_size(0);

    bitCapIntOcl qMask = bitCapIntOclPtr[3];
    cmplx topRight = cmplxPtr[1];
    cmplx bottomLeft = cmplxPtr[2];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_INVERT();
    }
}

void kernel invertsinglewide(global cmplx* stateVec, constant cmplx* cmplxPtr, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl i;
    
    bitCapIntOcl qMask = bitCapIntOclPtr[2];
    cmplx topRight = cmplxPtr[1];
    cmplx bottomLeft = cmplxPtr[2];

    bitCapIntOcl lcv = ID;
    PUSH_APART_1();
    APPLY_INVERT();
}


void kernel uniformlycontrolled(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr,
    constant bitCapIntOcl* qPowers, global cmplx4* mtrxs, constant real1* nrmIn, global real1* nrmParts,
    local real1* lProbBuffer)
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

void kernel uniformparityrz(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qMask = bitCapIntOclPtr[1];
    cmplx phaseFac = cmplx_ptr[0];
    cmplx phaseFacAdj = cmplx_ptr[1];
    bitCapIntOcl perm;
    bitLenInt c;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        perm = lcv & qMask;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

void kernel uniformparityrznorm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qMask = bitCapIntOclPtr[1];
    cmplx phaseFac = cmplx_ptr[0];
    cmplx phaseFacAdj = cmplx_ptr[1];
    cmplx nrm = cmplx_ptr[2];
    bitCapIntOcl perm;
    bitLenInt c;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        perm = lcv & qMask;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(nrm, zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj)));
    }
}

void kernel cuniformparityrz(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr, constant bitCapIntOcl* qPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qMask = bitCapIntOclPtr[1];
    bitCapIntOcl cMask = bitCapIntOclPtr[2];
    bitCapIntOcl cLen = bitCapIntOclPtr[3];
    cmplx phaseFac = cmplx_ptr[0];
    cmplx phaseFacAdj = cmplx_ptr[1];
    bitCapIntOcl perm, i, iLow, iHigh, p;
    bitLenInt c;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        i = 0U;
        for (p = 0U; p < cLen; p++) {
            iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh | cMask;
        
        perm = i & qMask;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[i] = zmul(stateVec[i], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

void kernel compose(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    for (lcv = ID; lcv < args.x; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
    }
}

void kernel composewide(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl lcv = ID;
    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
}

void kernel composemid(
    global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    bitCapIntOcl oQubitCount = args.z;
    bitCapIntOcl startMask = args.w;
    bitCapIntOcl midMask = bitCapIntOclPtr[4];
    bitCapIntOcl endMask = bitCapIntOclPtr[5];
    bitCapIntOcl start = bitCapIntOclPtr[6];

    for (lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] =
            zmul(stateVec1[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)], stateVec2[(lcv & midMask) >> start]);
    }
}

void kernel decomposeprob(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr,
    global real1* remainderStateProb, global real1* remainderStateAngle, global real1* partStateProb,
    global real1* partStateAngle)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl partPower = args.x;
    bitCapIntOcl remainderPower = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    bitCapIntOcl len = args.w;

    bitCapIntOcl j, k, l;
    cmplx amp;
    real1 partProb, nrm;

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv & startMask;
        j |= (lcv ^ j) << len;

        partProb = ZERO_R1;

        for (k = 0U; k < partPower; k++) {
            l = j | (k << start);

            amp = stateVec[l];
            nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm >= REAL1_EPSILON) {
                partStateAngle[k] = arg(amp);
            }
        }

        remainderStateProb[lcv] = partProb;
    }

    for (lcv = ID; lcv < partPower; lcv += Nthreads) {
        j = lcv << start;

        partProb = ZERO_R1;

        for (k = 0U; k < remainderPower; k++) {
            l = k & startMask;
            l |= (k ^ l) << len;
            l = j | l;

            amp = stateVec[l];
            nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm >= REAL1_EPSILON) {
                remainderStateAngle[k] = arg(amp);
            }
        }

        partStateProb[lcv] = partProb;
    }
}

void kernel decomposeamp(
    global real1* stateProb, global real1* stateAngle, constant bitCapIntOcl* bitCapIntOclPtr, global cmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    real1 angle, prob;
    for (lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        angle = stateAngle[lcv];
        prob = stateProb[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((cmplx)(angle + SineShift, angle));
    }
}

void kernel disposeprob(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr,
    global real1* remainderStateProb, global real1* remainderStateAngle)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl partPower = args.x;
    bitCapIntOcl remainderPower = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    bitCapIntOcl len = args.w;

    bitCapIntOcl j, k, l;
    cmplx amp;
    real1 partProb, nrm, firstAngle, currentAngle;

    const real1 angleThresh = -8 * PI_R1;
    const real1 initAngle = -16 * PI_R1;

    for (lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        j = lcv & startMask;
        j |= (lcv ^ j) << len;

        partProb = ZERO_R1;

        for (k = 0U; k < partPower; k++) {
            l = j | (k << start);

            amp = stateVec[l];
            nrm = dot(amp, amp);
            partProb += nrm;
        }

        remainderStateProb[lcv] = partProb;
    }

    for (lcv = ID; lcv < partPower; lcv += Nthreads) {
        j = lcv << start;

        firstAngle = initAngle;

        for (k = 0U; k < remainderPower; k++) {
            l = k & startMask;
            l |= (k ^ l) << len;
            l = j | l;

            amp = stateVec[l];
            nrm = dot(amp, amp);

            if (nrm >= REAL1_EPSILON) {
                currentAngle = arg(amp);
                if (firstAngle < angleThresh) {
                    firstAngle = currentAngle;
                }
                remainderStateAngle[k] = currentAngle - firstAngle;
            }
        }
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

void kernel prob(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    local real1* lProbBuffer)
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

void kernel probreg(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    local real1* lProbBuffer)
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

void kernel probmask(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    constant bitCapIntOcl* qPowers, local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, locID, locNthreads, lcv;

    Nthreads = get_global_size(0);

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    //bitCapIntOcl mask = args.y;
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

void kernel probmaskall(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    constant bitCapIntOcl* qPowersMask, constant bitCapIntOcl* qPowersSkip)
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

void kernel probparity(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl mask = args.y;

    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapIntOcl v;
    bool parity;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        parity = false;
        v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity) {
            amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        }
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

void kernel forcemparity(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, global real1* oneChanceBuffer,
    local real1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl mask = args.y;
    bool result = (bitCapIntOclPtr[2] == ONE_BCI);

    real1 oneChancePart = ZERO_R1;
    cmplx amp;
    bitCapIntOcl v;
    bool parity;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        parity = false;
        v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity == result) {
            amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        } else {
            stateVec[lcv] = (cmplx)(ZERO_R1, ZERO_R1);
        }
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

void kernel expperm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant bitCapIntOcl* bitPowers, global real1* expBuffer,
    local real1* lExpBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);

    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl len = bitCapIntOclPtr[1];
    bitCapIntOcl offset = bitCapIntOclPtr[2];

    real1 expectation = 0;
    bitCapIntOcl retIndex;
    bitLenInt p;
    cmplx amp;
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        retIndex = 0;
        for (p = 0; p < len; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= (ONE_BCI << p);
            }
        }
        expectation += (offset + retIndex) * dot(amp, amp);
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lExpBuffer[locID] = expectation;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lExpBuffer[locID] += lExpBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        expBuffer[get_group_id(0)] = lExpBuffer[0];
    }
}

void kernel nrmlze(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr)
{
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

void kernel nrmlzewide(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr)
{
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

void kernel updatenorm(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant real1* args_ptr,
    global real1* norm_ptr, local real1* lProbBuffer)
{
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

void kernel approxcompare(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr,
    global cmplx* part_inner_ptr, local cmplx* lInnerBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    cmplx partInner = (cmplx)(ZERO_R1, ZERO_R1);

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        partInner += zmul(conj(stateVec1[lcv]), stateVec2[lcv]);
    }

    locID = get_local_id(0);
    locNthreads = get_local_size(0);
    lInnerBuffer[locID] = partInner;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (locID < lcv) {
            lInnerBuffer[locID] += lInnerBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        part_inner_ptr[get_group_id(0)] = lInnerBuffer[0];
    }
}

void kernel applym(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qPower = bitCapIntOclPtr[1];
    bitCapIntOcl qMask = qPower - ONE_BCI;
    bitCapIntOcl savePower = bitCapIntOclPtr[2];
    bitCapIntOcl discardPower = qPower ^ savePower;
    cmplx nrm = cmplx_ptr[0];
    bitCapIntOcl i, iLow, iHigh;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & qMask;
        i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (cmplx)(ZERO_R1, ZERO_R1);
    }
}

void kernel applymreg(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr, constant cmplx* cmplx_ptr)
{
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

void kernel clearbuffer(global cmplx* stateVec, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads = get_global_size(0);
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl offset = bitCapIntOclPtr[1];
    maxI += offset;
    const cmplx amp0 = (cmplx)(ZERO_R1, ZERO_R1);
    for (bitCapIntOcl lcv = (ID + offset); lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = amp0;
    }
}

void kernel shufflebuffers(global cmplx* stateVec1, global cmplx* stateVec2, constant bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads = get_global_size(0);
    bitCapIntOcl halfMaxI = bitCapIntOclPtr[0];
    cmplx amp0;
    for (bitCapIntOcl lcv = ID; lcv < halfMaxI; lcv += Nthreads) {
        amp0 = stateVec1[lcv + halfMaxI];
        stateVec1[lcv + halfMaxI] = stateVec2[lcv];
        stateVec2[lcv] = amp0;
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
