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

__device__ inline qCudaReal1 qCudaArg(const qCudaCmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return atan2(cmp.y, cmp.x);
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal2 a, qCudaReal2 b)
{
  return a.x*b.x + a.y*b.y;
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal4 a, qCudaReal4 b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
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
    qCudaReal1 norm_thresh = qCudaCmplxPtr[9];

#define PUSH_APART_GEN()                                                                                               \
    bitCapIntOcl i = 0U;                                                                                               \
    bitCapIntOcl iHigh = lcv;                                                                                          \
    for (bitLenInt p = 0U; p < BITCOUNT_ARG; p++) {                                                                    \
        bitCapIntOcl iLow = iHigh & (qPowersSorted[p] - ONE_BCI);                                                      \
        i |= iLow;                                                                                                     \
        iHigh = (iHigh ^ iLow) << ONE_BCI;                                                                             \
    }                                                                                                                  \
    i |= iHigh;

#define PUSH_APART_1()                                                                                                 \
    bitCapIntOcl i = lcv & qMask;                                                                                      \
    i |= (lcv ^ i) << ONE_BCI;

#define PUSH_APART_2()                                                                                                 \
    bitCapIntOcl i = lcv & qMask1;                                                                                     \
    bitCapIntOcl iHigh = (lcv ^ i) << ONE_BCI;                                                                         \
    bitCapIntOcl iLow = iHigh & qMask2;                                                                                \
    i |= iLow | ((iHigh ^ iLow) << ONE_BCI);

#define APPLY_AND_OUT()                                                                                                \
    qudaCmplx2 mulRes = make_qCudaCmplx2(                                                                              \
        stateVec[i | OFFSET1_ARG].x, stateVec[i | OFFSET1_ARG].y,                                                      \
        stateVec[i | OFFSET2_ARG].x, stateVec[i | OFFSET2_ARG].y);                                                     \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = make_qCudaCmplx(mulRes.x, mulRes.y);                                                   \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(mulRes.z, mulRes.w);

#define APPLY_X()                                                                                                      \
    const qCudaCmplx Y0 = stateVec[i];                                                                                 \
    stateVec[i] = stateVec[i | OFFSET2_ARG];                                                                           \
    stateVec[i | OFFSET2_ARG] = Y0;

#define APPLY_Z()                                                                                                      \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(-stateVec[i | OFFSET2_ARG].x, -stateVec[i | OFFSET2_ARG].y);

#define APPLY_PHASE()                                                                                                  \
    stateVec[i] = zmul(topLeft, stateVec[i]);                                                                          \
    stateVec[i | OFFSET2_ARG] = zmul(bottomRight, stateVec[i | OFFSET2_ARG]);

#define APPLY_INVERT()                                                                                                 \
    const qCudaCmplx Y0 = stateVec[i];                                                                                 \
    stateVec[i] = zmul(topRight, stateVec[i | OFFSET2_ARG]);                                                           \
    stateVec[i | OFFSET2_ARG] = zmul(bottomLeft, Y0);

#define NORM_BODY_2X2()                                                                                                \
    qCudaCmplx mulResLo = stateVec[i | OFFSET1_ARG];                                                                   \
    qCudaCmplx mulResHi = stateVec[i | OFFSET2_ARG];                                                                   \
    mulRes = make_qCudaCmplx2(mulResLo.x, mulResLo.y, mulResHi.x, mulResHi.y);                                         \
                                                                                                                       \
    mulRes = zmatrixmul(nrm, mtrx, mulRes);                                                                            \
                                                                                                                       \
    qCudaCmplx mulResPart = make_qCudaCmplx(mulRes.x, mulRes.y);                                                       \
                                                                                                                       \
    dotMulRes = qCudaDot(mulResPart, mulResPart);                                                                      \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.x = ZERO_R1;                                                                                            \
        mulRes.y = ZERO_R1;                                                                                            \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    mulResPart = make_qCudaCmplx(mulRes.z, mulRes.w);                                                                  \
                                                                                                                       \
    dotMulRes = qCudaDot(mulResPart, mulResPart);                                                                      \
    if (dotMulRes < norm_thresh) {                                                                                     \
        mulRes.z = ZERO_R1;                                                                                            \
        mulRes.w = ZERO_R1;                                                                                            \
    } else {                                                                                                           \
        partNrm += dotMulRes;                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    stateVec[i | OFFSET1_ARG] = make_qCudaCmplx(mulRes.x, mulRes.y);                                                   \
    stateVec[i | OFFSET2_ARG] = make_qCudaCmplx(mulRes.z, mulRes.w);

#define SUM_LOCAL(part)                                                                                                \
    const bitCapIntOcl locID = get_local_id(0);                                                                        \
    const bitCapIntOcl locNthreads = get_local_size(0);                                                                \
    lBuffer[locID] = part;                                                                                             \
                                                                                                                       \
    for (bitCapIntOcl lcv = (locNthreads >> ONE_BCI); lcv > 0U; lcv >>= ONE_BCI) {                                     \
        __syncthreads();                                                                                               \
        if (locID < lcv) {                                                                                             \
            lBuffer[locID] += lBuffer[locID + lcv];                                                                    \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    if (locID == 0U) {                                                                                                 \
        sumBuffer[blockIdx.x] = lBuffer[0];                                                                            \
    }

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
        nrmParts[blockIdx.x] = lProbBuffer[0];                                                                         \
    }

__global__ void apply2x2(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr,
    bitCapIntOcl* qPowersSorted)
{
    PREP_2X2()

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_GEN()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2single(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2double(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_2()
        APPLY_AND_OUT()
    }
}

__global__ void apply2x2wide(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr,
    bitCapIntOcl* qPowersSorted)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl lcv = ID;

    PUSH_APART_GEN()
    APPLY_AND_OUT()
}

__global__ void apply2x2singlewide(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_1()
    APPLY_AND_OUT()
}

__global__ void apply2x2doublewide(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    PREP_2X2_WIDE()

    const bitCapIntOcl qMask1 = bitCapIntOclPtr[3];
    const bitCapIntOcl qMask2 = bitCapIntOclPtr[4];
    const bitCapIntOcl lcv = ID;

    PUSH_APART_2()
    APPLY_AND_OUT()
}

__global__ void apply2x2normsingle(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* sumBuffer, qCudaReal1* lBuffer)
{
    PREP_2X2()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    real1 partNrm = ZERO_R1;
    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        NORM_BODY_2X2()
    }

    SUM_LOCAL(partNrm)
}

__global__ void apply2x2normsinglewide(qCudaCmplx* stateVec, qCudaReal1*cmplxPtr,
    bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer, qCudaReal1* lBuffer)
{
    PREP_2X2_WIDE()
    PREP_2X2_NORM()

    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;

    real1 partNrm = ZERO_R1;
    PUSH_APART_1()
    NORM_BODY_2X2()

    SUM_LOCAL(partNrm)
}

__global__ void xsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_X()
    }
}

__global__ void xsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_X()
}

__global__ void xmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{

    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl otherRes = lcv & otherMask;
        bitCapIntOcl setInt = lcv & mask;
        bitCapIntOcl resetInt = setInt ^ mask;

        if (setInt < resetInt) {
            continue;
        }

        setInt |= otherRes;
        resetInt |= otherRes;

        const qCudaCmplx Y0 = stateVec[resetInt];
        stateVec[resetInt] = stateVec[setInt];
        stateVec[setInt] = Y0;
    }
}

__global__ void phaseparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplxPtr)
{
    const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const qCudaCmplx phaseFac = cmplxPtr[0];
    const qCudaCmplx iPhaseFac = cmplxPtr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl setInt = lcv & mask;

        bitCapIntOcl v = setInt;
        for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
            v ^= v >> paritySize;
        }
        v &= 1U;

        setInt |= lcv & otherMask;

        stateVec[setInt] = zmul(v ? phaseFac : iPhaseFac, stateVec[setInt]);
    }
}

__global__ void zsingle(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_Z()
    }
}

__global__ void zsinglewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_Z()
}

__global__ void phasesingle(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const qCudaCmplx topLeft = cmplxPtr[0];
    const qCudaCmplx bottomRight = cmplxPtr[3];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_PHASE()
    }
}

__global__ void phasesinglewide(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const qCudaCmplx topLeft = cmplxPtr[0];
    const qCudaCmplx bottomRight = cmplxPtr[3];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_PHASE()
}

__global__ void invertsingle(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl qMask = bitCapIntOclPtr[3];
    const qCudaCmplx topRight = cmplxPtr[1];
    const qCudaCmplx bottomLeft = cmplxPtr[2];

    for (bitCapIntOcl lcv = ID; lcv < MAXI_ARG; lcv += Nthreads) {
        PUSH_APART_1()
        APPLY_INVERT()
    }
}

__global__ void invertsinglewide(qCudaCmplx* stateVec, qCudaCmplx* cmplxPtr, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl qMask = bitCapIntOclPtr[2];
    const qCudaCmplx topRight = cmplxPtr[1];
    const qCudaCmplx bottomLeft = cmplxPtr[2];

    const bitCapIntOcl lcv = ID;
    PUSH_APART_1()
    APPLY_INVERT()
}


__global__ void uniformlycontrolled(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    bitCapIntOcl* qPowers, global cmplx4* mtrxs, qCudaReal1*nrmIn, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl targetPower = bitCapIntOclPtr[1];
    const bitCapIntOcl targetMask = targetPower - ONE_BCI;
    const bitCapIntOcl controlLen = bitCapIntOclPtr[2];
    const bitCapIntOcl mtrxSkipLen = bitCapIntOclPtr[3];
    const bitCapIntOcl mtrxSkipValueMask = bitCapIntOclPtr[4];
    const qCudaReal1 nrm = nrmIn[0];

    real1 partNrm = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & targetMask;
        i |= (lcv ^ i) << ONE_BCI;

        bitCapIntOcl offset = 0;
        for (bitLenInt p = 0; p < controlLen; p++) {
            if (i & qPowers[p]) {
                offset |= ONE_BCI << p;
            }
        }

        bitCapIntOcl jHigh = offset;
        bitCapIntOcl j = 0;
        for (bitLenInt p = 0; p < mtrxSkipLen; p++) {
            bitCapIntOcl jLow = jHigh & (qPowers[controlLen + p] - ONE_BCI);
            j |= jLow;
            jHigh = (jHigh ^ jLow) << ONE_BCI;
        }
        j |= jHigh;
        offset = j | mtrxSkipValueMask;

        qudaCmplx2 qubit = make_qCudaCmplx2(stateVec[i], stateVec[i | targetPower]);

        qubit = zmatrixmul(nrm, mtrxs[offset], qubit);

        partNrm += dot(qubit, qubit);

        stateVec[i] = qubit.lo;
        stateVec[i | targetPower] = qubit.hi;
    }

    SUM_LOCAL(partNrm)
}

__global__ void uniformparityrz(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const qCudaCmplx phaseFac = cmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = cmplx_ptr[1];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

__global__ void uniformparityrznorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const qCudaCmplx phaseFac = cmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = cmplx_ptr[1];
    const qCudaCmplx nrm = cmplx_ptr[2];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl perm = lcv & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[lcv] = zmul(nrm, zmul(stateVec[lcv], ((c & 1U) ? phaseFac : phaseFacAdj)));
    }
}

__global__ void cuniformparityrz(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr, bitCapIntOcl* qPowers)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qMask = bitCapIntOclPtr[1];
    const bitCapIntOcl cMask = bitCapIntOclPtr[2];
    const bitLenInt cLen = (bitLenInt)bitCapIntOclPtr[3];
    const qCudaCmplx phaseFac = cmplx_ptr[0];
    const qCudaCmplx phaseFacAdj = cmplx_ptr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < cLen; p++) {
            bitCapIntOcl iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh | cMask;

        bitCapIntOcl perm = i & qMask;
        bitLenInt c;
        for (c = 0; perm; c++) {
            // clear the least significant bit set
            perm &= perm - ONE_BCI;
        }
        stateVec[i] = zmul(stateVec[i], ((c & 1U) ? phaseFac : phaseFacAdj));
    }
}

__global__ void compose(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    for (bitCapIntOcl lcv = ID; lcv < args.x; lcv += Nthreads) {
        nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
    }
}

__global__ void composewide(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl lcv = ID;
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    // For reference:
    // bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    // bitCapIntOcl startMask = args.z;
    // bitCapIntOcl endMask = args.w;

    nStateVec[lcv] = zmul(stateVec1[lcv & args.z], stateVec2[(lcv & args.w) >> args.y]);
}

__global__ void composemid(
    qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl nMaxQPower = args.x;
    // bitCapIntOcl qubitCount = args.y;
    const bitCapIntOcl oQubitCount = args.z;
    const bitCapIntOcl startMask = args.w;
    const bitCapIntOcl midMask = bitCapIntOclPtr[4];
    const bitCapIntOcl endMask = bitCapIntOclPtr[5];
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[6];

    for (bitCapIntOcl lcv = ID; lcv < nMaxQPower; lcv += Nthreads) {
        nStateVec[lcv] =
            zmul(stateVec1[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)], stateVec2[(lcv & midMask) >> start]);
    }
}

__global__ void decomposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* remainderStateProb, qCudaReal1* remainderStateAngle, qCudaReal1* partStateProb,
    qCudaReal1* partStateAngle)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl partPower = args.x;
    const bitCapIntOcl remainderPower = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    const bitLenInt len = (bitLenInt)args.w;

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < partPower; k++) {
            bitCapIntOcl l = j | (k << start);

            cmplx amp = stateVec[l];
            real1 nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm >= REAL1_EPSILON) {
                partStateAngle[k] = arg(amp);
            }
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < remainderPower; k++) {
            bitCapIntOcl l = k & startMask;
            l |= (k ^ l) << len;
            l = j | l;

            cmplx amp = stateVec[l];
            real1 nrm = dot(amp, amp);
            partProb += nrm;

            if (nrm >= REAL1_EPSILON) {
                remainderStateAngle[k] = arg(amp);
            }
        }

        partStateProb[lcv] = partProb;
    }
}

__global__ void decomposeamp(
    qCudaReal1* stateProb, qCudaReal1* stateAngle, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxQPower = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < maxQPower; lcv += Nthreads) {
        const qCudaReal1 angle = stateAngle[lcv];
        nStateVec[lcv] = sqrt(stateProb[lcv]) * sin(make_qCudaCmplx(angle + SineShift, angle));
    }
}

__global__ void disposeprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr,
    qCudaReal1* remainderStateProb, qCudaReal1* remainderStateAngle)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl partPower = args.x;
    const bitCapIntOcl remainderPower = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitCapIntOcl startMask = (ONE_BCI << start) - ONE_BCI;
    const bitLenInt len = args.w;
    const qCudaReal1 angleThresh = -8 * PI_R1;
    const qCudaReal1 initAngle = -16 * PI_R1;

    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        bitCapIntOcl j = lcv & startMask;
        j |= (lcv ^ j) << len;

        real1 partProb = ZERO_R1;

        for (bitCapIntOcl k = 0U; k < partPower; k++) {
            bitCapIntOcl l = j | (k << start);

            cmplx amp = stateVec[l];
            real1 nrm = dot(amp, amp);
            partProb += nrm;
        }

        remainderStateProb[lcv] = partProb;
    }

    for (bitCapIntOcl lcv = ID; lcv < partPower; lcv += Nthreads) {
        const bitCapIntOcl j = lcv << start;

        real1 firstAngle = initAngle;

        for (bitCapIntOcl k = 0U; k < remainderPower; k++) {
            bitCapIntOcl l = k & startMask;
            l |= (k ^ l) << len;
            l = j | l;

            cmplx amp = stateVec[l];
            real1 nrm = dot(amp, amp);

            if (nrm >= REAL1_EPSILON) {
                real1 currentAngle = arg(amp);
                if (firstAngle < angleThresh) {
                    firstAngle = currentAngle;
                }
                remainderStateAngle[k] = currentAngle - firstAngle;
            }
        }
    }
}

__global__ void dispose(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl remainderPower = bitCapIntOclPtr[0];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl skipMask = bitCapIntOclPtr[2];
    const bitCapIntOcl disposedRes = bitCapIntOclPtr[3];
    for (bitCapIntOcl lcv = ID; lcv < remainderPower; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & skipMask;
        bitCapIntOcl i = iLow | ((lcv ^ iLow) << (bitCapIntOcl)len) | disposedRes;
        nStateVec[lcv] = stateVec[i];
    }
}

__global__ void prob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qPower = args.y;
    const bitCapIntOcl qMask = qPower - ONE_BCI;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << ONE_BCI) | qPower;
        const qCudaCmplx amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void cprob(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    qudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl qPower = args.y;
    const bitCapIntOcl qControlPower = args.z;
    const bitCapIntOcl qControlMask = args.w;
    bitCapIntOcl qMask1, qMask2;
    if (qPower < qControlPower) {
        qMask1 = qPower - ONE_BCI;
        qMask2 = qControlPower - ONE_BCI;
    } else {
        qMask1 = qControlPower - ONE_BCI;
        qMask2 = qPower - ONE_BCI;
    }

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        PUSH_APART_2()
        i |= qPower | qControlMask;
        const qCudaCmplx amp = stateVec[i];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl perm = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitLenInt len = (bitLenInt)args.w;
    const bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl i = lcv & qMask;
        i |= ((lcv ^ i) << len);
        const qCudaCmplx amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probregall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl maxJ = args.y;
    const bitLenInt start = (bitLenInt)args.z;
    const bitLenInt len = (bitLenInt)args.w;
    const bitCapIntOcl qMask = (ONE_BCI << start) - ONE_BCI;

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        const bitCapIntOcl perm = lcv1 << start;
        real1 oneChancePart = ZERO_R1;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            bitCapIntOcl i = lcv2 & qMask;
            i |= ((lcv2 ^ i) << len);
            cmplx amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

__global__ void probmask(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    bitCapIntOcl* qPowers, qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    //bitCapIntOcl mask = args.y;
    const bitCapIntOcl perm = args.z;
    const bitLenInt len = (bitLenInt)args.w;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl iHigh = lcv;
        bitCapIntOcl i = 0U;
        for (bitLenInt p = 0U; p < len; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        const qCudaCmplx amp = stateVec[i | perm];
        oneChancePart += dot(amp, amp);
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void probmaskall(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    bitCapIntOcl* qPowersMask, bitCapIntOcl* qPowersSkip)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl4 args = vload4(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl maxJ = args.y;
    const bitLenInt maskLen = (bitLenInt)args.z;
    const bitLenInt skipLen = (bitLenInt)args.w;

    for (bitCapIntOcl lcv1 = ID; lcv1 < maxI; lcv1 += Nthreads) {
        bitCapIntOcl iHigh = lcv1;
        bitCapIntOcl perm = 0U;
        for (bitLenInt p = 0U; p < skipLen; p++) {
            const bitCapIntOcl iLow = iHigh & (qPowersSkip[p] - ONE_BCI);
            perm |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        perm |= iHigh;

        real1 oneChancePart = ZERO_R1;
        for (bitCapIntOcl lcv2 = 0U; lcv2 < maxJ; lcv2++) {
            iHigh = lcv2;
            bitCapIntOcl i = 0U;
            for (bitLenInt p = 0U; p < maskLen; p++) {
                bitCapIntOcl iLow = iHigh & (qPowersMask[p] - ONE_BCI);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << ONE_BCI;
            }
            i |= iHigh;

            const qCudaCmplx amp = stateVec[i | perm];
            oneChancePart += dot(amp, amp);
        }
        sumBuffer[lcv1] = oneChancePart;
    }
}

__global__ void probparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{

    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity) {
            const qCudaCmplx amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        }
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void forcemparity(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl2 args = vload2(0, bitCapIntOclPtr);
    const bitCapIntOcl maxI = args.x;
    const bitCapIntOcl mask = args.y;
    const bool result = (bitCapIntOclPtr[2] == ONE_BCI);

    real1 oneChancePart = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bool parity = false;
        bitCapIntOcl v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity == result) {
            const qCudaCmplx amp = stateVec[lcv];
            oneChancePart += dot(amp, amp);
        } else {
            stateVec[lcv] = make_qCudaCmplx(ZERO_R1, ZERO_R1);
        }
    }

    SUM_LOCAL(oneChancePart)
}

__global__ void expperm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, bitCapIntOcl* bitPowers, qCudaReal1* sumBuffer,
    qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitLenInt len = (bitLenInt)bitCapIntOclPtr[1];
    const bitCapIntOcl offset = bitCapIntOclPtr[2];

    real1 expectation = 0;
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        bitCapIntOcl retIndex = 0;
        for (bitLenInt p = 0; p < len; p++) {
            if (lcv & bitPowers[p]) {
                retIndex |= (ONE_BCI << p);
            }
        }
        const qCudaCmplx amp = stateVec[lcv];
        expectation += (offset + retIndex) * dot(amp, amp);
    }

    SUM_LOCAL(expectation)
}

__global__ void nrmlze(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const qCudaReal1 norm_thresh = args_ptr[0].x;
    const qCudaCmplx nrm = args_ptr[1];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        cmplx amp = stateVec[lcv];
        if (dot(amp, amp) < norm_thresh) {
            amp = make_qCudaCmplx(ZERO_R1, ZERO_R1);
        }
        stateVec[lcv] = zmul(nrm, amp);
    }
}

__global__ void nrmlzewide(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* args_ptr)
{
    const bitCapIntOcl lcv = ID;
    const qCudaReal1 norm_thresh = args_ptr[0].x;
    const qCudaCmplx nrm = args_ptr[1];

    cmplx amp = stateVec[lcv];
    if (dot(amp, amp) < norm_thresh) {
        amp = make_qCudaCmplx(ZERO_R1, ZERO_R1);
    }
    stateVec[lcv] = zmul(nrm, amp);
}

__global__ void updatenorm(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaReal1*args_ptr,
    qCudaReal1* sumBuffer, qCudaReal1* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const qCudaReal1 norm_thresh = args_ptr[0];
    real1 partNrm = ZERO_R1;

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const qCudaCmplx amp = stateVec[lcv];
        real1 nrm = dot(amp, amp);
        if (nrm < norm_thresh) {
            nrm = ZERO_R1;
        }
        partNrm += nrm;
    }

    SUM_LOCAL(partNrm)
}

__global__ void approxcompare(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr,
    qCudaCmplx* sumBuffer, qCudaCmplx* lBuffer)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    qCudaCmplx partInner = make_qCudaCmplx(ZERO_R1, ZERO_R1);

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        partInner += zmul(conj(stateVec1[lcv]), stateVec2[lcv]);
    }

    SUM_LOCAL(partInner)
}

__global__ void applym(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl qPower = bitCapIntOclPtr[1];
    const bitCapIntOcl qMask = qPower - ONE_BCI;
    const bitCapIntOcl savePower = bitCapIntOclPtr[2];
    const bitCapIntOcl discardPower = qPower ^ savePower;
    const qCudaCmplx nrm = cmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl iLow = lcv & qMask;
        const bitCapIntOcl i = iLow | ((lcv ^ iLow) << ONE_BCI);

        stateVec[i | savePower] = zmul(nrm, stateVec[i | savePower]);
        stateVec[i | discardPower] = make_qCudaCmplx(ZERO_R1, ZERO_R1);
    }
}

__global__ void applymreg(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* cmplx_ptr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl mask = bitCapIntOclPtr[1];
    const bitCapIntOcl result = bitCapIntOclPtr[2];
    const qCudaCmplx nrm = cmplx_ptr[0];

    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = ((lcv & mask) == result) ? zmul(nrm, stateVec[lcv]) : make_qCudaCmplx(ZERO_R1, ZERO_R1);
    }
}

__global__ void clearbuffer(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0] + bitCapIntOclPtr[1];
    const bitCapIntOcl offset = bitCapIntOclPtr[1];
    const qCudaCmplx amp0 = make_qCudaCmplx(ZERO_R1, ZERO_R1);
    for (bitCapIntOcl lcv = (ID + offset); lcv < maxI; lcv += Nthreads) {
        stateVec[lcv] = amp0;
    }
}

__global__ void shufflebuffers(qCudaCmplx* stateVec1, qCudaCmplx* stateVec2, bitCapIntOcl* bitCapIntOclPtr)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl halfMaxI = bitCapIntOclPtr[0];
    for (bitCapIntOcl lcv = ID; lcv < halfMaxI; lcv += Nthreads) {
        const qCudaCmplx amp0 = stateVec1[lcv + halfMaxI];
        stateVec1[lcv + halfMaxI] = stateVec2[lcv];
        stateVec2[lcv] = amp0;
    }
}

__global__ void rol(qCudaCmplx* stateVec, bitCapIntOcl* bitCapIntOclPtr, qCudaCmplx* nStateVec)
{
    const bitCapIntOcl Nthreads = get_global_size(0);
    const bitCapIntOcl maxI = bitCapIntOclPtr[0];
    const bitCapIntOcl regMask = bitCapIntOclPtr[1];
    const bitCapIntOcl otherMask = bitCapIntOclPtr[2];
    const bitCapIntOcl lengthMask = bitCapIntOclPtr[3] - ONE_BCI;
    const bitLenInt start = (bitLenInt)bitCapIntOclPtr[4];
    const bitLenInt shift = (bitLenInt)bitCapIntOclPtr[5];
    const bitLenInt length = (bitLenInt)bitCapIntOclPtr[6];
    for (bitCapIntOcl lcv = ID; lcv < maxI; lcv += Nthreads) {
        const bitCapIntOcl regInt = (lcv & regMask) >> start;
        nStateVec[lcv] = stateVec[((((regInt >> shift) | (regInt << (length - shift))) & lengthMask) << start) | (lcv & otherMask)];
    }
}
} //namespace Qrack
