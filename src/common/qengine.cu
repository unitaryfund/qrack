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

#include "common/cuda_kernels.cuh"

namespace Qrack {

__device__ inline qCudaCmplx zmul(const qCudaCmplx lhs, const qCudaCmplx rhs)
{
    return make_qCudaCmplx((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

__device__ inline qCudaCmplx2 zmatrixmul(const qCudaReal1 nrm, const qCudaReal1* lhs, const qCudaCmplx2 rhs)
{
    return 
        (make_qCudaCmplx2(nrm * ((lhs[0] * rhs.x) - (lhs[1] * rhs.y) + (lhs[2] * rhs.z) - (lhs[3] * rhs.w)),
            nrm * ((lhs[0] * rhs.y) + (lhs[1] * rhs.x) + (lhs[2] * rhs.w) + (lhs[3] * rhs.z)),
            nrm * ((lhs[4] * rhs.x) - (lhs[5] * rhs.y) + (lhs[6] * rhs.z) - (lhs[7] * rhs.w)),
            nrm * ((lhs[4] * rhs.y) + (lhs[5] * rhs.x) + (lhs[6] * rhs.w) + (lhs[7] * rhs.z))));
}

__device__ inline qCudaReal1 arg(const qCudaCmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return atan2(cmp.y, cmp.x);
}

#define OFFSET2_ARG bitCapIntOclPtr[0]
#define OFFSET1_ARG bitCapIntOclPtr[1]
#define MAXI_ARG bitCapIntOclPtr[2]
#define BITCOUNT_ARG bitCapIntOclPtr[3]
#define ID blockIdx.x* blockDim.x + threadIdx.x

#define PREP_2X2()                                                                                                     \
    bitCapIntOcl lcv, i;                                                                                               \
    bitCapIntOcl Nthreads = gridDim.x * blockDim.x;                                                                    \
                                                                                                                       \
    qCudaReal1* mtrx = qCudaCmplxPtr;                                                                                  \
    qCudaReal1 nrm = qCudaCmplxPtr[8];                                                                                 \
                                                                                                                       \
    qCudaCmplx2 mulRes;
    
#define PREP_2X2_WIDE()                                                                                                \
    bitCapIntOcl lcv, i;                                                                                               \
                                                                                                                       \
    qCudaReal1* mtrx = qCudaCmplxPtr;                                                                                  \
    qCudaReal1 nrm = qCudaCmplxPtr[8];                                                                                 \
                                                                                                                       \
    qCudaCmplx2 mulRes;

#define PREP_2X2_NORM()                                                                                                \
    qCudaReal1 norm_thresh = qCudaCmplxPtr[9];                                                                         \
    qCudaReal1 dotMulRes;

#define PREP_SPECIAL_2X2()                                                                                             \
    bitCapIntOcl lcv, i;                                                                                               \
    qCudaCmplx Y0;

#define PREP_Z_2X2()                                                                                                   \
    bitCapIntOcl lcv, i;                                                                                               \
    bitCapIntOcl Nthreads = gridDim.x * blockDim.x;

#define PUSH_APART_GEN()                                                                                               \
    iHigh = lcv;                                                                                                       \
    i = 0U;                                                                                                            \
    for (p = 0U; p < BITCOUNT_ARG; p++) {                                                                              \
        iLow = iHigh & (qPowersSorted[p] - ONE_BCI);                                                                   \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define PUSH_APART_1()                                                                                                 \
    i = lcv & qMask;                                                                                                   \
    i |= (lcv ^ i) << ONE_BCI;

#define PUSH_APART_2()                                                                                                 \
    i = lcv & qMask1;                                                                                                  \
    iHigh = (lcv ^ i) << ONE_BCI;                                                                                      \
    iLow = iHigh & qMask2;                                                                                             \
    i |= iLow | ((iHigh ^ iLow) << ONE_BCI);

#define APPLY_AND_OUT()                                                                                                \
    mulRes = make_qCudaCmplx2(                                                                                         \
        stateVec[i | OFFSET1_ARG].x, stateVec[i | OFFSET1_ARG].y,                                                      \
        stateVec[i | OFFSET2_ARG].x, stateVec[i | OFFSET2_ARG].y);                                                     \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = make_qCudaCmplx(mulRes.x, mulRes.y);                                                   \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(mulRes.z, mulRes.w);

#define APPLY_X()                                                                                                      \
    Y0 = stateVec[i];                                                                                                  \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                                                           \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z() stateVec[i | OFFSET2_ARG] = -stateVec[i | OFFSET2_ARG];

#define SUM_2X2()                                                                                                      \
    locID = threadIdx.x;                                                                                               \
    locNthreads = blockDim.x;                                                                                          \
    lProbBuffer[locID] = partNrm;                                                                                      \
                                                                                                                       \
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {                                                  \
        __syncthreads();                                                                                               \
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
        mulRes.lo = (qCudaCmplx)(ZERO_R1, ZERO_R1);                                                                    \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    dotMulRes = dot(mulRes.hi, mulRes.hi);                                                                             \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.hi = (qCudaCmplx)(ZERO_R1, ZERO_R1);                                                                    \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = mulRes.lo;                                                                             \
    stateVec[i | OFFSET2_ARG] = mulRes.hi;

__global__ void apply2x2(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted)
{
    PREP_2X2();

    bitCapIntOcl iLow, iHigh;
    bitLenInt p;

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_GEN();
        APPLY_AND_OUT();
    }
}

__global__ void apply2x2single(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_AND_OUT();
    }
}

__global__ void apply2x2double(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
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

__global__ void apply2x2wide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* qPowersSorted)
{
    PREP_2X2_WIDE();

    bitCapIntOcl iLow, iHigh;
    bitLenInt p;

    lcv = ID;
    PUSH_APART_GEN();
    APPLY_AND_OUT();
}

__global__ void apply2x2singlewide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_AND_OUT();
}

__global__ void apply2x2doublewide(
    qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE();

    bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    bitCapIntOcl iLow, iHigh;

    lcv = ID;
    PUSH_APART_2();
    APPLY_AND_OUT();
}

__global__ void apply2x2normsingle(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* nrmParts, qCudaReal1* lProbBuffer)
{
    PREP_2X2();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    bitCapIntOcl locID, locNthreads;
    qCudaReal1 partNrm = ZERO_R1;

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        NORM_BODY_2X2();
    }

    SUM_2X2();
}

__global__ void apply2x2normsinglewide(qCudaCmplx* stateVec, qCudaReal1* qCudaCmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* nrmParts, qCudaReal1* lProbBuffer)
{
    PREP_2X2_WIDE();
    PREP_2X2_NORM();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    bitCapIntOcl locID, locNthreads;
    qCudaReal1 partNrm = ZERO_R1;

    lcv = ID;
    PUSH_APART_1();
    NORM_BODY_2X2();

    SUM_2X2();
}

__global__ void xsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_SPECIAL_2X2();
    bitCapIntOcl Nthreads = gridDim.x * blockDim.x; 

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_X();
    }
}

__global__ void xsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_SPECIAL_2X2();

    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    lcv = ID;
    PUSH_APART_1();
    APPLY_X();
}

__global__ void zsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl lcv, i;
    bitCapIntOcl Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1();
        APPLY_Z();
    }
}

__global__ void zsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl i;
    bitCapIntOcl qMask = bitCapIntOclPtr[2];

    bitCapIntOcl lcv = ID;
    PUSH_APART_1();
    APPLY_Z();
}

__global__ void uniformlycontrolled(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    bitCapIntOcl* qPowers, qCudaCmplx* mtrxs, qCudaReal1* nrmIn,
    qCudaReal1* nrmParts, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl targetPower = bitCapIntOclPtr[1];
    bitCapIntOcl targetMask = targetPower - ONE_BCI;
    bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    bitCapIntOcl mtrxSkipLen = bitCapIntOclPtr[3];
    bitCapIntOcl mtrxSkipValueMask = bitCapIntOclPtr[4];

    qCudaReal1 nrm = nrmIn[0];

    qCudaReal1 partNrm = ZERO_R1;

    qCudaCmplx2 qubit;

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

        qubit = zmatrixmul(nrm, mtrxs[offset * 4U], qubit);

        partNrm += dot(qubit, qubit);

        stateVec[i] = qubit.lo;
        stateVec[i | targetPower] = qubit.hi;
    }

    locID = threadIdx.x;
    locNthreads = blockDim.x;
    lProbBuffer[locID] = partNrm;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        __syncthreads();
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        nrmParts[get_group_id(0)] = lProbBuffer[0];
    }
}

__global__ void compose(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;

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

__global__ void composewide(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
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

__global__ void composemid(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;

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

__global__ void decomposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* remainderStateProb, qCudaReal1* remainderStateAngle, qCudaReal1* partStateProb,
    qCudaReal1* partStateAngle)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl partPower = args.x;
    bitCapIntOcl remainderPower = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;

    bitCapIntOcl j, k, l;
    qCudaCmplx amp;
    qCudaReal1 partProb, nrm, firstAngle, currentAngle;

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

__global__ void decomposeamp(qCudaReal1* stateProb, qCudaReal1* stateAngle,
    bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    qCudaReal1 angle;
    for (lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin((qCudaCmplx)(angle + SineShift, angle));
    }
}

__global__ void dispose(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void prob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl qPower = args.y;
    bitCapIntOcl qMask = qPower - ONE_BCI;

    qCudaReal1 oneChancePart = ZERO_R1;
    qCudaCmplx amp;
    bitCapIntOcl i;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << ONE_BCI) | qPower;
        amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    locID = threadIdx.x;
    locNthreads = blockDim.x;
    lProbBuffer[locID] = oneChancePart;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        __syncthreads();
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

__global__ void probreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl perm = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;
    bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    qCudaReal1 oneChancePart = ZERO_R1;
    qCudaCmplx amp;
    bitCapIntOcl i;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        i = lcv & qMask;
        i |= ((lcv ^ i) << len);
        amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    locID = threadIdx.x;
    locNthreads = blockDim.x;
    lProbBuffer[locID] = oneChancePart;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        __syncthreads();
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

__global__ void probregall(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* oneChanceBuffer)
{
    bitCapIntOcl Nthreads, lcv1, lcv2;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl maxJ = args.y;
    bitCapIntOcl start = args.z;
    bitCapIntOcl len = args.w;
    bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    qCudaReal1 oneChancePart;
    qCudaCmplx amp;
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

__global__ void probmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, bitCapIntOcl* qPowers, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, locID, locNthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl mask = args.y;
    bitCapIntOcl perm = args.z;
    bitCapIntOcl len = args.w;

    qCudaReal1 oneChancePart = ZERO_R1;
    qCudaCmplx amp;
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

    locID = threadIdx.x;
    locNthreads = blockDim.x;
    lProbBuffer[locID] = oneChancePart;

    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        __syncthreads();
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        oneChanceBuffer[get_group_id(0)] = lProbBuffer[0];
    }
}

__global__ void probmaskall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* oneChanceBuffer, bitCapIntOcl* qPowersMask, bitCapIntOcl* qPowersSkip)
{
    bitCapIntOcl Nthreads, lcv1, lcv2;

    Nthreads = gridDim.x * blockDim.x;

    bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    bitCapIntOcl maxI = args.x;
    bitCapIntOcl maxJ = args.y;
    bitCapIntOcl maskLen = args.z;
    bitCapIntOcl skipLen = args.w;

    qCudaReal1 oneChancePart;
    qCudaCmplx amp;
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

__global__ void rol(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void inc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, i;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void cinc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, i, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void incdecc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void incs(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    bitCapIntOcl overflowMask = bitCapIntOclPtr[4];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[5];
    bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, inOutInt, inOutRes, inInt, outInt, outRes;
    qCudaCmplx amp;
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

__global__ void incdecsc1(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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
    qCudaCmplx amp;
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

__global__ void incdecsc2(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl inOutMask = bitCapIntOclPtr[1];
    bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    bitCapIntOcl lengthPower = bitCapIntOclPtr[3];
    bitCapIntOcl signMask = lengthPower >> ONE_BCI;
    bitCapIntOcl carryMask = bitCapIntOclPtr[4];
    bitCapIntOcl inOutStart = bitCapIntOclPtr[5];
    bitCapIntOcl toAdd = bitCapIntOclPtr[6];
    bitCapIntOcl otherRes, inOutInt, inOutRes, inInt, outInt, outRes, i;
    qCudaCmplx amp;
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

__global__ void incbcd(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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
    qCudaCmplx amp;
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

__global__ void incdecbcdc(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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
    qCudaCmplx amp1, amp2;
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

__global__ void mul(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void div(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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
    Nthreads = gridDim.x * blockDim.x;                                                                                 \
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

__global__ void mulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    MODNOUT(i, (inRes | outRes | otherRes));
}

__global__ void imulmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    MODNOUT((inRes | outRes | otherRes), i);
}

__global__ void powmodnout(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void fulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

    qCudaCmplx ins0c0, ins0c1, ins1c0, ins1c1;
    qCudaCmplx outs0c0, outs0c1, outs1c0, outs1c1;

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

__global__ void ifulladd(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

    qCudaCmplx ins0c0, ins0c1, ins1c0, ins1c1;
    qCudaCmplx outs0c0, outs0c1, outs1c0, outs1c1;

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

__global__ void cmul(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void cdiv(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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
        nStateVec[i | controlMask] = stateVec[((outInt & lowMask) << inOutStart) |
            (((outInt & highMask) >> len) << carryStart) | otherRes | controlMask];

        CMOD_FINISH();
    }
}

// The conditional in the body of kernel loop would majorly hurt performance:
#define CMODNOUT(indexIn, indexOut)                                                                                    \
    bitCapIntOcl Nthreads, lcv;                                                                                        \
                                                                                                                       \
    Nthreads = gridDim.x * blockDim.x;                                                                                 \
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

__global__ void cmulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    CMODNOUT((i | controlMask), (inRes | outRes | otherRes | controlMask));
}

__global__ void cimulmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    CMODNOUT((inRes | outRes | otherRes | controlMask), (i | controlMask));
}

__global__ void cpowmodnout(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, bitCapIntOcl* controlPowers)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void indexedLda(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, __constant__ bitLenInt* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void indexedAdc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, __constant__ bitLenInt* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void indexedSbc(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, __constant__ bitLenInt* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void hash(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* nStateVec, __constant__ bitLenInt* values)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void nrmlze(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    qCudaReal1 norm_thresh = args_ptr[0];
    qCudaReal1 nrm = args_ptr[1];
    qCudaCmplx amp;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        if (dot(amp, amp) < norm_thresh) {
            amp = (qCudaCmplx)(ZERO_R1, ZERO_R1);
        }
        stateVec[lcv] = nrm * amp;
    }
}

__global__ void nrmlzewide(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* args_ptr)
{
    bitCapIntOcl lcv = ID;
    qCudaReal1 norm_thresh = args_ptr[0];
    qCudaReal1 nrm = args_ptr[1];
    qCudaCmplx amp;

    amp = stateVec[lcv];
    if (dot(amp, amp) < norm_thresh) {
        amp = (qCudaCmplx)(ZERO_R1, ZERO_R1);
    }
    stateVec[lcv] = nrm * amp;
}

__global__ void updatenorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* args_ptr, qCudaReal1* norm_ptr, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    qCudaReal1 norm_thresh = args_ptr[0];
    qCudaCmplx amp;
    qCudaReal1 nrm;
    qCudaReal1 partNrm = ZERO_R1;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        amp = stateVec[lcv];
        nrm = dot(amp, amp);
        if (nrm < norm_thresh) {
            nrm = ZERO_R1;
        }
        partNrm += nrm;
    }

    locID = threadIdx.x;
    locNthreads = blockDim.x;
    lProbBuffer[locID] = partNrm;
    for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
        __syncthreads();
        if (locID < lcv) {
            lProbBuffer[locID] += lProbBuffer[locID + lcv];
        }
    }

    if (locID == 0U) {
        norm_ptr[get_group_id(0)] = lProbBuffer[0];
    }
}

__global__ void approxcompare(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* norm_ptr, qCudaReal1* lProbBuffer)
{
    bitCapIntOcl Nthreads, lcv, locID, locNthreads;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    qCudaCmplx amp;
    qCudaReal1 partNrm = ZERO_R1;

    // Hopefully, since this is identical redundant work by all elements, the break hits for all at the same time.
    qCudaCmplx basePhaseFac1;
    qCudaReal1 nrm;
    bitCapIntOcl basePerm = 0;
    do {
        amp = stateVec1[basePerm];
        nrm = dot(amp, amp);
        basePerm++;
    } while (nrm < min_norm);

    basePerm--;
    amp = stateVec1[basePerm];
    nrm = dot(amp, amp);

    // If the amplitude we sample for __global__ phase offset correction doesn't match, we're done.
    if (nrm > min_norm) {
        basePhaseFac1 = (ONE_R1 / sqrt(nrm)) * amp;

        amp = stateVec2[basePerm];
        qCudaCmplx basePhaseFac2 = (ONE_R1 / sqrt(dot(amp, amp))) * amp;

        for (lcv = ID; lcv < maxI; lcv += Nthreads) {
            amp = zmul(basePhaseFac2, stateVec1[lcv]) - zmul(basePhaseFac1, stateVec2[lcv]);
            partNrm += dot(amp, amp);
        }

        locID = threadIdx.x;
        locNthreads = blockDim.x;
        lProbBuffer[locID] = partNrm;

        for (lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {
            __syncthreads();
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

__global__ void applym(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl qPower = bitCapIntOclPtr[1];
    bitCapIntOcl qMask = qPower - ONE_BCI;
    bitCapIntOcl savePower = bitCapIntOclPtr[2];
    bitCapIntOcl discardPower = qPower ^ savePower;
    qCudaCmplx nrm = qCudaCmplx_ptr[0];
    bitCapIntOcl i, iLow, iHigh;

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        iHigh = lcv;
        iLow = iHigh & qMask;
        i = iLow | ((iHigh ^ iLow) << ONE_BCI);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = (qCudaCmplx)(ZERO_R1, ZERO_R1);
    }
}

__global__ void applymreg(
    qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* qCudaCmplx_ptr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl mask = bitCapIntOclPtr[1];
    bitCapIntOcl result = bitCapIntOclPtr[2];
    qCudaCmplx nrm = qCudaCmplx_ptr[0];

    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = ((lcv & mask) == result) ? zmul(nrm, stateVec[lcv]) : (qCudaCmplx)(ZERO_R1, ZERO_R1);
    }
}

__global__ void phaseflip(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = -stateVec[lcv];
    }
}

__global__ void zerophaseflip(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;
    bitCapIntOcl i, iLow, iHigh;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void cphaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;
    bitCapIntOcl i, iLow, iHigh;

    Nthreads = gridDim.x * blockDim.x;
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

__global__ void phaseflipifless(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    bitCapIntOcl Nthreads, lcv;

    Nthreads = gridDim.x * blockDim.x;
    bitCapIntOcl maxI = bitCapIntOclPtr[0];
    bitCapIntOcl regMask = bitCapIntOclPtr[1];
    bitCapIntOcl greaterPerm = bitCapIntOclPtr[2];
    bitCapIntOcl start = bitCapIntOclPtr[3];
    for (lcv = ID; lcv < maxI; lcv += Nthreads) {
        if (((lcv & regMask) >> start) < greaterPerm)
            stateVec[lcv] = -stateVec[lcv];
    }
}

} // namespace Qrack
