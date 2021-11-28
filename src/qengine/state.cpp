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

#include <chrono>
#include <thread>

#include "qengine_cpu.hpp"

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#define complex2 Complex8x2Simd
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#define complex2 Complex16x2Simd
#endif
#endif

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateVec) {                                                                                                   \
        return;                                                                                                        \
    }

namespace Qrack {

/**
 * Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with
 * a shared random number generator, with a specific phase.
 *
 * (Note that "useHostMem" is required as a parameter to normalize constructors for use with the
 * CreateQuantumInterface() factory, but it serves no function in QEngineCPU.)
 *
 * \warning Overall phase is generally arbitrary and unknowable. Setting two QEngineCPU instances to the same
 * phase usually makes sense only if they are initialized at the same time.
 */
QEngineCPU::QEngineCPU(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG, bool useSparseStateVec,
    real1_f norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, true, useHardwareRNG, norm_thresh)
    , isSparse(useSparseStateVec)
{
#if ENABLE_ENV_VARS
    pStridePow =
        (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
    pStridePow = PSTRIDEPOW;
#endif

    SetConcurrency(std::thread::hardware_concurrency());

    stateVec = AllocStateVec(maxQPowerOcl);
    stateVec->clear();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        stateVec->write((bitCapIntOcl)initState, GetNonunitaryPhase());
    } else {
        stateVec->write((bitCapIntOcl)initState, phaseFac);
    }
}

complex QEngineCPU::GetAmplitude(bitCapInt perm)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_CMPLX;
    }

    return stateVec->read((bitCapIntOcl)perm);
}

void QEngineCPU::SetAmplitude(bitCapInt perm, complex amp)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec && !norm(amp)) {
        return;
    }

    runningNorm = REAL1_DEFAULT_ARG;

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
        stateVec->clear();
    }

    stateVec->write((bitCapIntOcl)perm, amp);
}

void QEngineCPU::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Dump();

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
    }

    stateVec->clear();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        complex phase;
        if (randGlobalPhase) {
            real1_f angle = Rand() * 2 * PI_R1;
            phase = complex((real1)cos(angle), (real1)sin(angle));
        } else {
            phase = ONE_CMPLX;
        }
        stateVec->write((bitCapIntOcl)perm, phase);
    } else {
        real1 nrm = abs(phaseFac);
        stateVec->write((bitCapIntOcl)perm, phaseFac / nrm);
    }

    runningNorm = ONE_R1;
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCPU::SetQuantumState(const complex* inputState)
{
    Dump();

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
    }

    stateVec->copy_in(inputState);
    runningNorm = REAL1_DEFAULT_ARG;
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineCPU::GetQuantumState(complex* outputState)
{
    if (!stateVec) {
        std::fill(outputState, outputState + maxQPowerOcl, ZERO_CMPLX);
        return;
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->copy_out(outputState);
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineCPU::GetProbs(real1* outputProbs)
{
    if (!stateVec) {
        std::fill(outputProbs, outputProbs + maxQPowerOcl, ZERO_R1);
        return;
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->get_probs(outputProbs);
}

/**
 * Apply a 2x2 matrix to the state vector
 *
 * A fundamental operation used by almost all gates.
 */

#if ENABLE_COMPLEX_X2

union ComplexUnion {
    complex2 cmplx2;
    complex cmplx[2];

    inline ComplexUnion(){};
    inline ComplexUnion(const complex& cmplx0, const complex& cmplx1)
    {
        cmplx[0] = cmplx0;
        cmplx[1] = cmplx1;
    }
};

#define NORM_THRESH_KERNEL(fn)                                                                                         \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));                              \
        qubit.cmplx2 = fn;                                                                                             \
                                                                                                                       \
        real1 dotMulRes = norm(qubit.cmplx[0]);                                                                        \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit.cmplx[0] = ZERO_CMPLX;                                                                               \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        dotMulRes = norm(qubit.cmplx[1]);                                                                              \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit.cmplx[1] = ZERO_CMPLX;                                                                               \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);                                \
    }

#define NORM_CALC_KERNEL(fn)                                                                                           \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));                              \
        qubit.cmplx2 = fn;                                                                                             \
        rngNrm[cpu] += norm(qubit.cmplx2);                                                                             \
        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);                                \
    };

void QEngineCPU::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapIntOcl* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    std::shared_ptr<complex> mtrxS(new complex[4], std::default_delete<complex[]>());
    std::copy(matrix, matrix + 4, mtrxS.get());

    std::shared_ptr<bitCapIntOcl> qPowersSortedS(new bitCapIntOcl[bitCount], std::default_delete<bitCapIntOcl[]>());
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSortedS.get());

    bool doApplyNorm = doNormalize && (bitCount == 1) && (runningNorm > ZERO_R1);
    doCalcNorm = doCalcNorm && (doApplyNorm || (runningNorm <= ZERO_R1));

    real1 nrm = doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch(maxQPower >> bitCount,
        [this, mtrxS, qPowersSortedS, offset1, offset2, bitCount, doCalcNorm, doApplyNorm, nrm, nrm_thresh] {
            complex* mtrx = mtrxS.get();
            bitCapIntOcl* qPowersSorted = qPowersSortedS.get();

            real1_f norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : nrm_thresh;
            unsigned numCores = GetConcurrencyLevel();

            ComplexUnion mtrxCol1(mtrx[0], mtrx[2]);
            ComplexUnion mtrxCol2(mtrx[1], mtrx[3]);

            std::unique_ptr<real1[]> rngNrm(new real1[numCores]());
            ParallelFunc fn;
            if (!doCalcNorm) {
                fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                    ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));
                    qubit.cmplx2 = matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);
                    stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
                };
            } else if (norm_thresh > ZERO_R1) {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    fn = NORM_THRESH_KERNEL(matrixMul(nrm, mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2));
                } else {
                    fn = NORM_THRESH_KERNEL(matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2));
                }
            } else {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    fn = NORM_CALC_KERNEL(matrixMul(nrm, mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2));
                } else {
                    fn = NORM_CALC_KERNEL(matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2));
                }
            }

            if (stateVec->is_sparse()) {
                bitCapIntOcl setMask = offset1 ^ offset2;
                bitCapIntOcl filterMask = 0;
                for (bitLenInt i = 0; i < bitCount; i++) {
                    filterMask |= (qPowersSorted[i] & ~setMask);
                }
                bitCapIntOcl filterValues = filterMask & offset1 & offset2;
                par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
            } else {
                par_for_mask(0, maxQPowerOcl, qPowersSorted, bitCount, fn);
            }

            if (doApplyNorm) {
                runningNorm = ONE_R1;
            }

            if (!doCalcNorm) {
                return;
            }

            real1 rNrm = ZERO_R1;
            for (unsigned i = 0; i < numCores; i++) {
                rNrm += rngNrm[i];
            }
            rngNrm.reset();
            runningNorm = rNrm;

            if (runningNorm == ZERO_R1) {
                ZeroAmplitudes();
            }
        });
}
#else

#define NORM_THRESH_KERNEL(fn1, fn2)                                                                                   \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex qubit[2];                                                                                              \
                                                                                                                       \
        complex Y0 = stateVec->read(lcv + offset1);                                                                    \
        qubit[1] = stateVec->read(lcv + offset2);                                                                      \
                                                                                                                       \
        qubit[0] = fn1;                                                                                                \
        qubit[1] = fn2;                                                                                                \
                                                                                                                       \
        real1 dotMulRes = norm(qubit[0]);                                                                              \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit[0] = ZERO_CMPLX;                                                                                     \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        dotMulRes = norm(qubit[1]);                                                                                    \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit[1] = ZERO_CMPLX;                                                                                     \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);                                            \
    }

#define NORM_CALC_KERNEL(fn1, fn2)                                                                                     \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex qubit[2];                                                                                              \
                                                                                                                       \
        complex Y0 = stateVec->read(lcv + offset1);                                                                    \
        qubit[1] = stateVec->read(lcv + offset2);                                                                      \
                                                                                                                       \
        qubit[0] = fn1;                                                                                                \
        qubit[1] = fn2;                                                                                                \
                                                                                                                       \
        rngNrm[cpu] = norm(qubit[0]) + norm(qubit[1]);                                                                 \
                                                                                                                       \
        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);                                            \
    };

void QEngineCPU::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapIntOcl* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    std::shared_ptr<complex> mtrxS(new complex[4], std::default_delete<complex[]>());
    std::copy(matrix, matrix + 4, mtrxS.get());

    std::shared_ptr<bitCapIntOcl> qPowersSortedS(new bitCapIntOcl[bitCount], std::default_delete<bitCapIntOcl[]>());
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSortedS.get());

    bool doApplyNorm = doNormalize && (bitCount == 1) && (runningNorm > ZERO_R1);
    doCalcNorm = doCalcNorm && (doApplyNorm || (runningNorm <= ZERO_R1));

    real1 nrm = doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch(maxQPower >> bitCount,
        [this, mtrxS, qPowersSortedS, offset1, offset2, bitCount, doCalcNorm, doApplyNorm, nrm, nrm_thresh] {
            complex* mtrx = mtrxS.get();
            bitCapIntOcl* qPowersSorted = qPowersSortedS.get();

            real1_f norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : nrm_thresh;
            unsigned numCores = GetConcurrencyLevel();

            std::unique_ptr<real1[]> rngNrm(new real1[numCores]());
            ParallelFunc fn;
            if (!doCalcNorm) {
                fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                    complex Y0 = stateVec->read(lcv + offset1);
                    complex Y1 = stateVec->read(lcv + offset2);
                    stateVec->write2(
                        lcv + offset1, (mtrx[0] * Y0) + (mtrx[1] * Y1), lcv + offset2, (mtrx[2] * Y0) + (mtrx[3] * Y1));
                };
            } else if (norm_thresh > ZERO_R1) {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    fn = NORM_THRESH_KERNEL(
                        nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1])), nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1])));
                } else {
                    fn = NORM_THRESH_KERNEL(
                        (mtrx[0] * Y0) + (mtrx[1] * qubit[1]), (mtrx[2] * Y0) + (mtrx[3] * qubit[1]));
                }
            } else {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    fn = NORM_CALC_KERNEL(
                        nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1])), nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1])));
                } else {
                    fn = NORM_CALC_KERNEL((mtrx[0] * Y0) + (mtrx[1] * qubit[1]), (mtrx[2] * Y0) + (mtrx[3] * qubit[1]));
                }
            }

            if (stateVec->is_sparse()) {
                bitCapIntOcl setMask = offset1 ^ offset2;
                bitCapIntOcl filterMask = 0;
                for (bitLenInt i = 0; i < bitCount; i++) {
                    filterMask |= (qPowersSorted[i] & ~setMask);
                }
                bitCapIntOcl filterValues = filterMask & offset1 & offset2;
                par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
            } else {
                par_for_mask(0, maxQPowerOcl, qPowersSorted, bitCount, fn);
            }

            if (doApplyNorm) {
                runningNorm = ONE_R1;
            }

            if (!doCalcNorm) {
                return;
            }

            real1 rNrm = ZERO_R1;
            for (unsigned i = 0; i < numCores; i++) {
                rNrm += rngNrm[i];
            }
            rngNrm.reset();
            runningNorm = rNrm;

            if (runningNorm == ZERO_R1) {
                ZeroAmplitudes();
            }
        });
}
#endif

void QEngineCPU::XMask(bitCapInt mask)
{
    CHECK_ZERO_SKIP();

    if (!mask) {
        return;
    }

    if (!(mask & (mask - ONE_BCI))) {
        X(log2(mask));
        return;
    }

    if (stateVec->is_sparse()) {
        QInterface::XMask(mask);
        return;
    }

    Dispatch(maxQPower, [this, mask] {
        bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
        bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ maskOcl;
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl otherRes = lcv & otherMask;
            bitCapIntOcl setInt = lcv & maskOcl;
            bitCapIntOcl resetInt = setInt ^ maskOcl;

            if (setInt < resetInt) {
                return;
            }

            setInt |= otherRes;
            resetInt |= otherRes;

            complex Y0 = stateVec->read(resetInt);
            stateVec->write(resetInt, stateVec->read(setInt));
            stateVec->write(setInt, Y0);
        };

        par_for(0, maxQPowerOcl, fn);
    });
}

void QEngineCPU::PhaseParity(real1_f radians, bitCapInt mask)
{
    CHECK_ZERO_SKIP();

    if (!mask) {
        return;
    }

    if (!(mask & (mask - ONE_BCI))) {
        complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    if (stateVec->is_sparse()) {
        QInterface::PhaseParity(radians, mask);
        return;
    }

    Dispatch(maxQPower, [this, mask, radians] {
        bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
        complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
        complex iPhaseFac = ONE_CMPLX / phaseFac;
        bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
        bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ maskOcl;
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl otherRes = lcv & otherMask;
            bitCapIntOcl setInt = lcv & maskOcl;

            bitCapIntOcl v = setInt;
            for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
                v ^= v >> paritySize;
            }
            v &= 1U;

            setInt |= otherRes;

            stateVec->write(setInt, (v ? phaseFac : iPhaseFac) * stateVec->read(setInt));
        };

        par_for(0, maxQPowerOcl, fn);
    });
}

void QEngineCPU::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
    bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        Mtrx(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    bitCapIntOcl targetPower = pow2Ocl(qubitIndex);

    real1 nrm = (runningNorm > ZERO_R1) ? ONE_R1 / (real1)sqrt(runningNorm) : ONE_R1;

    std::unique_ptr<bitCapIntOcl[]> qPowers(new bitCapIntOcl[controlLen]);
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = pow2Ocl(controls[i]);
    }

    std::unique_ptr<bitCapIntOcl[]> mtrxSkipPowersOcl(new bitCapIntOcl[mtrxSkipLen]);
    for (bitLenInt i = 0; i < mtrxSkipLen; i++) {
        mtrxSkipPowersOcl[i] = (bitCapIntOcl)mtrxSkipPowers[i];
    }

    bitCapIntOcl mtrxSkipValueMaskOcl = (bitCapIntOcl)mtrxSkipValueMask;

    unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> rngNrm(new real1[numCores]());

    Finish();

    par_for_skip(0, maxQPowerOcl, targetPower, 1, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        bitCapIntOcl offset = 0;
        for (bitLenInt j = 0; j < controlLen; j++) {
            if (lcv & qPowers[j]) {
                offset |= pow2Ocl(j);
            }
        }

        bitCapIntOcl i, iHigh;
        iHigh = offset;
        i = 0;
        for (bitCapIntOcl p = 0; p < mtrxSkipLen; p++) {
            bitCapIntOcl iLow = iHigh & (mtrxSkipPowersOcl[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        offset = i | mtrxSkipValueMaskOcl;

        // Offset is permutation * 4, for the components of 2x2 matrices. (Note that this sacrifices 2 qubits of
        // capacity for the unsigned bitCapInt.)
        offset *= 4;

        complex qubit[2];

        complex Y0 = stateVec->read(lcv);
        qubit[1] = stateVec->read(lcv | targetPower);

        qubit[0] = nrm * ((mtrxs[0 + offset] * Y0) + (mtrxs[1 + offset] * qubit[1]));
        qubit[1] = nrm * ((mtrxs[2 + offset] * Y0) + (mtrxs[3 + offset] * qubit[1]));

        rngNrm[cpu] += norm(qubit[0]) + norm(qubit[1]);

        stateVec->write2(lcv, qubit[0], lcv | targetPower, qubit[1]);
    });

    runningNorm = ZERO_R1;
    for (unsigned i = 0; i < numCores; i++) {
        runningNorm += rngNrm[i];
    }
}

void QEngineCPU::UniformParityRZ(const bitCapInt& mask, const real1_f& angle)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, mask, angle] {
        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac(cosine, sine);
        complex phaseFacAdj(cosine, -sine);
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapInt perm = lcv & mask;
            // From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
            // c accumulates the total bits set in v
            bitLenInt c;
            for (c = 0; perm; c++) {
                // clear the least significant bit set
                perm &= perm - ONE_BCI;
            }
            stateVec->write(lcv, stateVec->read(lcv) * ((c & 1U) ? phaseFac : phaseFacAdj));
        };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0, maxQPowerOcl, fn);
        }
    });
}

void QEngineCPU::CUniformParityRZ(
    const bitLenInt* cControls, const bitLenInt& controlLen, const bitCapInt& mask, const real1_f& angle)
{
    if (!controlLen) {
        return UniformParityRZ(mask, angle);
    }

    CHECK_ZERO_SKIP();

    std::vector<bitLenInt> controls(cControls, cControls + controlLen);
    std::sort(controls.begin(), controls.end());

    Dispatch(maxQPower >> controlLen, [this, controls, mask, angle] {
        bitCapIntOcl controlMask = 0;
        std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controls.size()]);
        for (bitLenInt i = 0; i < (bitLenInt)controls.size(); i++) {
            controlPowers[i] = pow2Ocl(controls[i]);
            controlMask |= controlPowers[i];
        }

        real1 cosine = (real1)cos(angle);
        real1 sine = (real1)sin(angle);
        complex phaseFac(cosine, sine);
        complex phaseFacAdj(cosine, -sine);

        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapInt perm = lcv & mask;
            // From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
            // c accumulates the total bits set in v
            bitLenInt c;
            for (c = 0; perm; c++) {
                // clear the least significant bit set
                perm &= perm - ONE_BCI;
            }
            stateVec->write(controlMask | lcv, stateVec->read(controlMask | lcv) * ((c & 1U) ? phaseFac : phaseFacAdj));
        };

        par_for_mask(0, maxQPowerOcl, controlPowers.get(), controls.size(), fn);
    });
}

/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old unit that was added.
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy)
{
    bitLenInt result = qubitCount;
    bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    if (!stateVec || !toCopy->stateVec) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(nQubitCount);
        return result;
    }

    bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    bitCapIntOcl startMask = maxQPowerOcl - ONE_BCI;
    bitCapIntOcl endMask = (toCopy->maxQPowerOcl - ONE_BCI) << qubitCount;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        nStateVec->write(lcv, stateVec->read(lcv & startMask) * toCopy->stateVec->read((lcv & endMask) >> qubitCount));
    };

    if ((toCopy->doNormalize) && (toCopy->runningNorm != ONE_R1)) {
        toCopy->NormalizeState();
    }
    toCopy->Finish();

    if (stateVec->is_sparse() || toCopy->stateVec->is_sparse()) {
        par_for_sparse_compose(
            CastStateVecSparse()->iterable(), toCopy->CastStateVecSparse()->iterable(), qubitCount, fn);
    } else {
        par_for(0, nMaxQPower, fn);
    }

    SetQubitCount(nQubitCount);

    ResetStateVec(nStateVec);

    return result;
}

/**
 * Combine (a copy of) another QEngineCPU with this one, inserted at the "start" index. (If the programmer doesn't want
 * to "cheat," it is left up to them to delete the old unit that was added.
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy, bitLenInt start)
{
    bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    if (!stateVec || !toCopy->stateVec) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(nQubitCount);
        return start;
    }

    bitLenInt oQubitCount = toCopy->qubitCount;
    bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    bitCapIntOcl startMask = pow2MaskOcl(start);
    bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }
    toCopy->Finish();

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    par_for(0, nMaxQPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        nStateVec->write(lcv,
            stateVec->read((lcv & startMask) | ((lcv & endMask) >> oQubitCount)) *
                toCopy->stateVec->read((lcv & midMask) >> start));
    });

    SetQubitCount(nQubitCount);

    ResetStateVec(nStateVec);

    return start;
}

/**
 * Combine (copies) each QEngineCPU in the vector with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old unit that was added.
 *
 * Returns a mapping of the index into the new QEngine that each old one was mapped to.
 */
std::map<QInterfacePtr, bitLenInt> QEngineCPU::Compose(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;
    bitLenInt nQubitCount = qubitCount;

    bitLenInt toComposeCount = toCopy.size();

    std::vector<bitLenInt> offset(toComposeCount);
    std::vector<bitCapIntOcl> mask(toComposeCount);

    bitCapIntOcl startMask = maxQPowerOcl - ONE_BCI;
    bitCapIntOcl nMaxQPower;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    for (bitLenInt i = 0; i < toComposeCount; i++) {
        QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[i]);
        if (src->doNormalize) {
            src->NormalizeState();
        }
        src->Finish();
        mask[i] = (src->maxQPowerOcl - ONE_BCI) << (bitCapIntOcl)nQubitCount;
        offset[i] = nQubitCount;
        ret[toCopy[i]] = nQubitCount;
        nQubitCount += src->GetQubitCount();
    }

    nMaxQPower = pow2Ocl(nQubitCount);

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    par_for(0, nMaxQPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        nStateVec->write(lcv, stateVec->read(lcv & startMask));

        for (bitLenInt j = 0; j < toComposeCount; j++) {
            QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[j]);
            nStateVec->write(lcv, nStateVec->read(lcv) * src->stateVec->read((lcv & mask[j]) >> offset[j]));
        }
    });

    SetQubitCount(nQubitCount);

    ResetStateVec(nStateVec);

    return ret;
}

/**
 * Minimally decompose a set of contigious bits from the separable unit. The
 * length of this separable unit is reduced by the length of bits decomposed, and
 * the bits removed are output in the destination QEngineCPU pointer. The
 * destination object must be initialized to the correct number of bits, in 0
 * permutation state.
 */
void QEngineCPU::DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr destination)
{
    if (length == 0) {
        return;
    }

    bitLenInt nLength = qubitCount - length;

    if (!stateVec) {
        if (nLength == 0) {
            SetQubitCount(1);
        } else {
            SetQubitCount(nLength);
        }
        if (destination) {
            destination->ZeroAmplitudes();
        }
        return;
    }

    if (destination && !destination->stateVec) {
        // Reinitialize stateVec RAM
        destination->SetPermutation(0);
    }

    bitCapIntOcl partPower = pow2Ocl(length);
    bitCapIntOcl remainderPower = pow2Ocl(nLength);

    std::unique_ptr<real1[]> remainderStateProb(new real1[remainderPower]());
    std::unique_ptr<real1[]> remainderStateAngle(new real1[remainderPower]());
    std::unique_ptr<real1[]> partStateProb;
    std::unique_ptr<real1[]> partStateAngle;
    if (destination) {
        partStateProb = std::unique_ptr<real1[]>(new real1[partPower]());
        partStateAngle = std::unique_ptr<real1[]>(new real1[partPower]());
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (destination) {
        par_for(0, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0; k < partPower; k++) {
                bitCapIntOcl l = j | (k << start);

                complex amp = stateVec->read(l);
                real1 nrm = norm(amp);
                remainderStateProb[lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    partStateAngle[k] = arg(amp);
                }
            }
        });

        par_for(0, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv << start;

            for (bitCapIntOcl k = 0; k < remainderPower; k++) {
                bitCapIntOcl l = k & pow2MaskOcl(start);
                l |= (k ^ l) << length;
                l = j | l;

                complex amp = stateVec->read(l);
                real1 nrm = norm(amp);
                partStateProb[lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    remainderStateAngle[k] = arg(amp);
                }
            }
        });
    } else {
        par_for(0, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0; k < partPower; k++) {
                bitCapIntOcl l = j | (k << start);

                remainderStateProb[lcv] += norm(stateVec->read(l));
            }
        });

        par_for(0, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv << start;

            for (bitCapIntOcl k = 0; k < remainderPower; k++) {
                bitCapIntOcl l = k & pow2MaskOcl(start);
                l |= (k ^ l) << length;
                l = j | l;

                complex amp = stateVec->read(l);

                if (norm(amp) > amplitudeFloor) {
                    remainderStateAngle[k] = arg(amp);
                }
            }
        });
    }

    if (destination) {
        destination->Dump();

        par_for(0, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            destination->stateVec->write(lcv,
                (real1)(std::sqrt(partStateProb[lcv])) * complex(cos(partStateAngle[lcv]), sin(partStateAngle[lcv])));
        });

        partStateProb.reset();
        partStateAngle.reset();
    }

    if (nLength == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(nLength);
    }
    ResetStateVec(AllocStateVec(maxQPowerOcl));

    par_for(0, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        stateVec->write(lcv,
            (real1)(std::sqrt(remainderStateProb[lcv])) *
                complex(cos(remainderStateAngle[lcv]), sin(remainderStateAngle[lcv])));
    });
}

void QEngineCPU::Decompose(bitLenInt start, QInterfacePtr destination)
{
    DecomposeDispose(start, destination->GetQubitCount(), std::dynamic_pointer_cast<QEngineCPU>(destination));
}

void QEngineCPU::Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QEngineCPUPtr)NULL); }

void QEngineCPU::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (length == 0) {
        return;
    }

    bitLenInt nLength = qubitCount - length;

    if (!stateVec) {
        if (nLength == 0) {
            SetQubitCount(1);
        } else {
            SetQubitCount(nLength);
        }
        return;
    }

    bitCapIntOcl disposedPermOcl = (bitCapIntOcl)disposedPerm;
    bitCapIntOcl remainderPower = pow2Ocl(nLength);
    bitCapIntOcl skipMask = pow2Ocl(start) - ONE_BCI;
    bitCapIntOcl disposedRes = disposedPermOcl << (bitCapIntOcl)start;
    bitCapIntOcl saveMask = ~((pow2Ocl(start + length) - ONE_BCI) ^ skipMask);

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    StateVectorPtr nStateVec = AllocStateVec(remainderPower);
    stateVec->isReadLocked = false;

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl i, iLow, iHigh;
            iHigh = lcv & saveMask;
            iLow = iHigh & skipMask;
            i = iLow | ((iHigh ^ iLow) >> (bitCapIntOcl)length);
            nStateVec->write(i, stateVec->read(lcv));
        });
    } else {
        par_for(0, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl i, iLow, iHigh;
            iHigh = lcv;
            iLow = iHigh & skipMask;
            i = iLow | ((iHigh ^ iLow) << (bitCapIntOcl)length) | disposedRes;
            nStateVec->write(lcv, stateVec->read(i));
        });
    }

    if (nLength == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(nLength);
    }

    ResetStateVec(nStateVec);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1_f QEngineCPU::Prob(bitLenInt qubit)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1;
    }

    bitCapIntOcl qPower = pow2Ocl(qubit);
    real1 oneChance = ZERO_R1;

    unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        oneChanceBuff[cpu] += norm(stateVec->read(lcv | qPower));
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(qPower, qPower, qPower), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, qPower, 1U, fn);
    }
    stateVec->isReadLocked = true;

    for (unsigned i = 0; i < numCores; i++) {
        oneChance += oneChanceBuff[i];
    }

    return clampProb(oneChance);
}

/// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
real1_f QEngineCPU::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1;
    }

    return norm(stateVec->read((bitCapIntOcl)fullRegister));
}

// Returns probability of permutation of the register
real1_f QEngineCPU::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1;
    }

    int num_threads = GetConcurrencyLevel();
    std::unique_ptr<real1[]> probs(new real1[num_threads]());

    bitCapIntOcl perm = ((bitCapIntOcl)permutation) << ((bitCapIntOcl)start);

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        probs[cpu] += norm(stateVec->read(lcv | perm));
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, bitRegMaskOcl(start, length), perm), fn);
    } else {
        par_for_skip(0, maxQPowerOcl, pow2Ocl(start), length, fn);
    }
    stateVec->isReadLocked = true;

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    return clampProb(prob);
}

// Returns probability of permutation of the mask
real1_f QEngineCPU::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1;
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (length = 0; v; length++) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[length]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());

    int num_threads = GetConcurrencyLevel();
    std::unique_ptr<real1[]> probs(new real1[num_threads]());

    bitCapIntOcl permutationOcl = (bitCapIntOcl)permutation;

    stateVec->isReadLocked = false;
    par_for_mask(
        0, maxQPowerOcl, skipPowers.get(), skipPowersVec.size(), [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            probs[cpu] += norm(stateVec->read(lcv | permutationOcl));
        });
    stateVec->isReadLocked = true;

    skipPowers.reset();

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    return clampProb(prob);
}

real1_f QEngineCPU::ProbParity(const bitCapInt& mask)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec || !mask) {
        return ZERO_R1;
    }

    real1 oddChance = ZERO_R1;

    unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oddChanceBuff(new real1[numCores]());

    bitCapIntOcl maskOcl = (bitCapIntOcl)mask;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        bool parity = false;
        bitCapIntOcl v = lcv & maskOcl;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity) {
            oddChanceBuff[cpu] += norm(stateVec->read(lcv));
        }
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }
    stateVec->isReadLocked = true;

    for (unsigned i = 0; i < numCores; i++) {
        oddChance += oddChanceBuff[i];
    }

    return clampProb(oddChance);
}

bool QEngineCPU::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    if (!stateVec || !mask) {
        return false;
    }

    if (!doForce) {
        result = (Rand() <= ProbParity(mask));
    }

    real1 oddChance = ZERO_R1;

    unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oddChanceBuff(new real1[numCores]());

    bitCapIntOcl maskOcl = (bitCapIntOcl)mask;

    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        bool parity = false;
        bitCapIntOcl v = lcv & maskOcl;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity == result) {
            oddChanceBuff[cpu] += norm(stateVec->read(lcv));
        } else {
            stateVec->write(lcv, ZERO_CMPLX);
        }
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPowerOcl, fn);
    }
    stateVec->isReadLocked = true;

    for (unsigned i = 0; i < numCores; i++) {
        oddChance += oddChanceBuff[i];
    }

    oddChanceBuff.reset();

    runningNorm = oddChance;

    if (!doNormalize) {
        NormalizeState();
    }

    return result;
}

real1_f QEngineCPU::SumSqrDiff(QEngineCPUPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }
    toCompare->Finish();

    if (!stateVec && !toCompare->stateVec) {
        return ZERO_R1;
    }

    if (!stateVec) {
        toCompare->UpdateRunningNorm();
        return toCompare->runningNorm;
    }

    if (!toCompare->stateVec) {
        UpdateRunningNorm();
        return runningNorm;
    }

    stateVec->isReadLocked = false;
    toCompare->stateVec->isReadLocked = false;

    unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> partInner(new complex[numCores]());

    par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        partInner[cpu] += conj(stateVec->read(lcv)) * toCompare->stateVec->read(lcv);
    });

    stateVec->isReadLocked = true;
    toCompare->stateVec->isReadLocked = true;

    complex totInner = ZERO_CMPLX;
    for (unsigned i = 0; i < numCores; i++) {
        totInner += partInner[i];
    }

    return ONE_R1 - clampProb(norm(totInner));
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, greaterPerm, start, length, flagIndex] {
        bitCapIntOcl regMask = bitRegMaskOcl(start, length);
        bitCapIntOcl flagMask = pow2Ocl(flagIndex);

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if ((((lcv & regMask) >> start) < greaterPerm) & ((lcv & flagMask) == flagMask))
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, greaterPerm, start, length] {
        bitCapIntOcl regMask = bitRegMaskOcl(start, length);

        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            if (((lcv & regMask) >> start) < greaterPerm)
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

void QEngineCPU::ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPower, [this, regMask, result, nrm] {
        ParallelFunc fn = [&](const bitCapIntOcl& i, const unsigned& cpu) {
            if ((i & regMask) == result) {
                stateVec->write(i, nrm * stateVec->read(i));
            } else {
                stateVec->write(i, ZERO_CMPLX);
            }
        };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0, maxQPowerOcl, fn);
        }

        runningNorm = ONE_R1;
    });
}

void QEngineCPU::NormalizeState(real1_f nrm_f, real1_f norm_thresh_f)
{
    CHECK_ZERO_SKIP();

    real1 nrm = (real1)nrm_f;
    real1 norm_thresh = (real1)norm_thresh_f;

    Finish();

    if (nrm < ZERO_R1) {
        nrm = runningNorm;
    }
    if ((nrm <= ZERO_R1) || (nrm == ONE_R1)) {
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }

    nrm = ONE_R1 / std::sqrt(nrm);

    if (norm_thresh <= ZERO_R1) {
        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            complex amp = stateVec->read(lcv) * nrm;
            stateVec->write(lcv, amp);
        });
    } else {
        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            complex amp = stateVec->read(lcv);
            if (norm(amp) < norm_thresh) {
                amp = ZERO_CMPLX;
            }
            stateVec->write(lcv, nrm * amp);
        });
    }

    runningNorm = ONE_R1;
}

void QEngineCPU::UpdateRunningNorm(real1_f norm_thresh)
{
    Finish();

    if (!stateVec) {
        runningNorm = ZERO_R1;
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }
    runningNorm = par_norm(maxQPowerOcl, stateVec, norm_thresh);

    if (runningNorm == ZERO_R1) {
        ZeroAmplitudes();
    }
}

StateVectorPtr QEngineCPU::AllocStateVec(bitCapIntOcl elemCount)
{
    if (isSparse) {
        return std::make_shared<StateVectorSparse>(elemCount);
    } else {
        return std::make_shared<StateVectorArray>(elemCount);
    }
}
} // namespace Qrack
