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
    real1_f norm_thresh, std::vector<int> devList, bitLenInt qubitThreshold)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, true, useHardwareRNG, norm_thresh)
    , isSparse(useSparseStateVec)
{
    SetConcurrency(std::thread::hardware_concurrency());

    stateVec = AllocStateVec(maxQPower);
    stateVec->clear();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        stateVec->write(initState, GetNonunitaryPhase());
    } else {
        stateVec->write(initState, phaseFac);
    }
}

complex QEngineCPU::GetAmplitude(bitCapInt perm)
{
    if (!stateVec) {
        return ZERO_CMPLX;
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    return stateVec->read(perm);
}

void QEngineCPU::SetAmplitude(bitCapInt perm, complex amp)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    runningNorm -= norm(stateVec->read(perm));
    runningNorm += norm(amp);

    if (runningNorm <= amplitudeFloor) {
        ZeroAmplitudes();
        return;
    }

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPower));
        stateVec->clear();
    }

    stateVec->write(perm, amp);
}

void QEngineCPU::SetPermutation(bitCapInt perm, complex phaseFac)
{
    Dump();

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPower));
    }

    stateVec->clear();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        complex phase;
        if (randGlobalPhase) {
            real1 angle = Rand() * 2 * PI_R1;
            phase = complex(cos(angle), sin(angle));
        } else {
            phase = complex(ONE_R1, ZERO_R1);
        }
        stateVec->write(perm, phase);
    } else {
        real1 nrm = abs(phaseFac);
        stateVec->write(perm, phaseFac / nrm);
    }

    runningNorm = ONE_R1;
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCPU::SetQuantumState(const complex* inputState)
{
    Dump();

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPower));
    }

    stateVec->copy_in(inputState);
    runningNorm = ONE_R1;

    UpdateRunningNorm();
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineCPU::GetQuantumState(complex* outputState)
{
    if (!stateVec) {
        std::fill(outputState, outputState + (bitCapIntOcl)maxQPower, ZERO_CMPLX);
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
        std::fill(outputProbs, outputProbs + (bitCapIntOcl)maxQPower, ZERO_R1);
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

void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapInt* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    complex* mtrx = new complex[4];
    std::copy(matrix, matrix + 4, mtrx);

    bitCapInt* qPowersSorted = new bitCapInt[bitCount];
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSorted);

    doCalcNorm = (doCalcNorm || (runningNorm != ONE_R1)) && doNormalize && (bitCount == 1);

    real1 nrm = doNormalize && (runningNorm != REAL1_DEFAULT_ARG) ? (ONE_R1 / std::sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch([this, mtrx, qPowersSorted, offset1, offset2, bitCount, doCalcNorm, nrm, nrm_thresh] {
        real1 norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : nrm_thresh;
        int numCores = GetConcurrencyLevel();

        ComplexUnion mtrxCol1(mtrx[0], mtrx[2]);
        ComplexUnion mtrxCol2(mtrx[1], mtrx[3]);

        real1* rngNrm = NULL;
        ParallelFunc fn;
        if (doCalcNorm) {
            rngNrm = new real1[numCores]();
            if (nrm != ONE_R1) {
                if (norm_thresh > ZERO_R1) {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));

                        qubit.cmplx2 = matrixMul(nrm, mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);

                        real1 dotMulRes = norm(qubit.cmplx[0]);
                        if (dotMulRes < norm_thresh) {
                            qubit.cmplx[0] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        dotMulRes = norm(qubit.cmplx[1]);
                        if (dotMulRes < norm_thresh) {
                            qubit.cmplx[1] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

#if FPPOW < 6
                        stateVec->write(lcv + offset1, qubit.cmplx[0]);
                        stateVec->write(lcv + offset2, qubit.cmplx[1]);
#else
                        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
#endif
                    };
                } else {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));

                        qubit.cmplx2 = matrixMul(nrm, mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);

                        rngNrm[cpu] += norm(qubit.cmplx[0]) + norm(qubit.cmplx[1]);

#if FPPOW < 6
                        stateVec->write(lcv + offset1, qubit.cmplx[0]);
                        stateVec->write(lcv + offset2, qubit.cmplx[1]);
#else
                        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
#endif
                    };
                }
            } else {
                if (norm_thresh > ZERO_R1) {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));

                        qubit.cmplx2 = matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);

                        real1 dotMulRes = norm(qubit.cmplx[0]);
                        if (dotMulRes < norm_thresh) {
                            qubit.cmplx[0] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        dotMulRes = norm(qubit.cmplx[1]);
                        if (dotMulRes < norm_thresh) {
                            qubit.cmplx[1] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

#if FPPOW < 6
                        stateVec->write(lcv + offset1, qubit.cmplx[0]);
                        stateVec->write(lcv + offset2, qubit.cmplx[1]);
#else
                        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
#endif
                    };
                } else {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));

                        qubit.cmplx2 = matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);

                        rngNrm[cpu] += norm(qubit.cmplx[0]) + norm(qubit.cmplx[1]);

#if FPPOW < 6
                        stateVec->write(lcv + offset1, qubit.cmplx[0]);
                        stateVec->write(lcv + offset2, qubit.cmplx[1]);
#else
                        stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
#endif
                    };
                }
            }
        } else {
            fn = [&](const bitCapInt& lcv, const int& cpu) {
                ComplexUnion qubit(stateVec->read(lcv + offset1), stateVec->read(lcv + offset2));

                qubit.cmplx2 = matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);
#if FPPOW < 6
                stateVec->write(lcv + offset1, qubit.cmplx[0]);
                stateVec->write(lcv + offset2, qubit.cmplx[1]);
#else
                stateVec->write2(lcv + offset1, qubit.cmplx[0], lcv + offset2, qubit.cmplx[1]);
#endif
            };
        }

        if (stateVec->is_sparse()) {
            bitCapInt setMask = offset1 ^ offset2;
            bitCapInt filterMask = 0;
            for (bitLenInt i = 0; i < bitCount; i++) {
                filterMask |= (qPowersSorted[i] & ~setMask);
            }
            bitCapInt filterValues = filterMask & offset1 & offset2;
            par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
        } else {
            par_for_mask(0, maxQPower, qPowersSorted, bitCount, fn);
        }

        delete[] mtrx;
        delete[] qPowersSorted;

        if (doCalcNorm) {
            real1 rNrm = ZERO_R1;
            for (int i = 0; i < numCores; i++) {
                rNrm += rngNrm[i];
            }
            runningNorm = rNrm;
            delete[] rngNrm;

            if (runningNorm == ZERO_R1) {
                ZeroAmplitudes();
            }
        }
    });
}
#else
void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapInt* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    complex* mtrx = new complex[4];
    std::copy(matrix, matrix + 4, mtrx);

    bitCapInt* qPowersSorted = new bitCapInt[bitCount];
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSorted);

    doCalcNorm = (doCalcNorm || (runningNorm != ONE_R1)) && doNormalize && (bitCount == 1);

    real1 nrm = doNormalize ? (ONE_R1 / std::sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch([this, mtrx, qPowersSorted, offset1, offset2, bitCount, doCalcNorm, nrm, nrm_thresh] {
        real1 norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : nrm_thresh;
        int numCores = GetConcurrencyLevel();

        real1* rngNrm = NULL;
        ParallelFunc fn;
        if (doCalcNorm) {
            rngNrm = new real1[numCores]();

            if (nrm != ONE_R1) {
                if (norm_thresh > ZERO_R1) {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        complex qubit[2];

                        complex Y0 = stateVec->read(lcv + offset1);
                        qubit[1] = stateVec->read(lcv + offset2);

                        qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
                        qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

                        real1 dotMulRes = norm(qubit[0]);
                        if (dotMulRes < norm_thresh) {
                            qubit[0] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        dotMulRes = norm(qubit[1]);
                        if (dotMulRes < norm_thresh) {
                            qubit[1] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);
                    };
                } else {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        complex qubit[2];

                        complex Y0 = stateVec->read(lcv + offset1);
                        qubit[1] = stateVec->read(lcv + offset2);

                        qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
                        qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

                        rngNrm[cpu] = norm(qubit[0]) + norm(qubit[1]);

                        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);
                    };
                }
            } else {
                if (norm_thresh > ZERO_R1) {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        complex qubit[2];

                        complex Y0 = stateVec->read(lcv + offset1);
                        qubit[1] = stateVec->read(lcv + offset2);

                        qubit[0] = (mtrx[0] * Y0) + (mtrx[1] * qubit[1]);
                        qubit[1] = (mtrx[2] * Y0) + (mtrx[3] * qubit[1]);

                        real1 dotMulRes = norm(qubit[0]);
                        if (dotMulRes < norm_thresh) {
                            qubit[0] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        dotMulRes = norm(qubit[1]);
                        if (dotMulRes < norm_thresh) {
                            qubit[1] = ZERO_CMPLX;
                        } else {
                            rngNrm[cpu] += dotMulRes;
                        }

                        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);
                    };
                } else {
                    fn = [&](const bitCapInt& lcv, const int& cpu) {
                        complex qubit[2];

                        complex Y0 = stateVec->read(lcv + offset1);
                        qubit[1] = stateVec->read(lcv + offset2);

                        qubit[0] = (mtrx[0] * Y0) + (mtrx[1] * qubit[1]);
                        qubit[1] = (mtrx[2] * Y0) + (mtrx[3] * qubit[1]);

                        rngNrm[cpu] = norm(qubit[0]) + norm(qubit[1]);

                        stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);
                    };
                }
            }
        } else {
            fn = [&](const bitCapInt& lcv, const int& cpu) {
                complex qubit[2];

                complex Y0 = stateVec->read(lcv + offset1);
                qubit[1] = stateVec->read(lcv + offset2);

                qubit[0] = (mtrx[0] * Y0) + (mtrx[1] * qubit[1]);
                qubit[1] = (mtrx[2] * Y0) + (mtrx[3] * qubit[1]);

                stateVec->write2(lcv + offset1, qubit[0], lcv + offset2, qubit[1]);
            };
        }

        if (stateVec->is_sparse()) {
            bitCapInt setMask = offset1 ^ offset2;
            bitCapInt filterMask = 0;
            for (bitLenInt i = 0; i < bitCount; i++) {
                filterMask |= (qPowersSorted[i] & ~setMask);
            }
            bitCapInt filterValues = filterMask & offset1 & offset2;
            par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
        } else {
            par_for_mask(0, maxQPower, qPowersSorted, bitCount, fn);
        }

        delete[] mtrx;
        delete[] qPowersSorted;

        if (doCalcNorm) {
            real1 rNrm = ZERO_R1;
            for (int i = 0; i < numCores; i++) {
                rNrm += rngNrm[i];
            }
            runningNorm = rNrm;
            delete[] rngNrm;
        }
    });
}
#endif

void QEngineCPU::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
    bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    bitCapInt targetPower = pow2(qubitIndex);

    real1 nrm = ONE_R1 / std::sqrt(runningNorm);

    bitCapInt* qPowers = new bitCapInt[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = pow2(controls[i]);
    }

    int numCores = GetConcurrencyLevel();
    real1* rngNrm = new real1[numCores];
    std::fill(rngNrm, rngNrm + numCores, ZERO_R1);

    Finish();

    par_for_skip(0, maxQPower, targetPower, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapIntOcl offset = 0;
        for (bitLenInt j = 0; j < controlLen; j++) {
            if (lcv & qPowers[j]) {
                offset |= pow2Ocl(j);
            }
        }

        bitCapInt i, iHigh, iLow;
        bitCapIntOcl p;
        iHigh = offset;
        i = 0;
        for (p = 0; p < mtrxSkipLen; p++) {
            iLow = iHigh & (mtrxSkipPowers[p] - ONE_BCI);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << ONE_BCI;
        }
        i |= iHigh;

        offset = (bitCapIntOcl)(i | mtrxSkipValueMask);

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
    for (int i = 0; i < numCores; i++) {
        runningNorm += rngNrm[i];
    }

    delete[] rngNrm;
    delete[] qPowers;
}

void QEngineCPU::UniformParityRZ(const bitCapInt& mask, const real1_f& angle)
{
    CHECK_ZERO_SKIP();

    Dispatch([this, mask, angle] {
        real1 cosine = cos(angle);
        real1 sine = sin(angle);
        complex phaseFac(cosine, sine);
        complex phaseFacAdj(cosine, -sine);
        ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
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
            par_for(0, maxQPower, fn);
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

    Dispatch([this, controls, mask, angle] {
        bitCapInt controlMask = 0;
        bitCapInt* controlPowers = new bitCapInt[controls.size()];
        for (bitLenInt i = 0; i < controls.size(); i++) {
            controlPowers[i] = pow2(controls[i]);
            controlMask |= controlPowers[i];
        }

        real1 cosine = cos(angle);
        real1 sine = sin(angle);
        complex phaseFac(cosine, sine);
        complex phaseFacAdj(cosine, -sine);

        ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
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

        par_for_mask(0, maxQPower, controlPowers, controls.size(), fn);

        delete[] controlPowers;
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
    bitCapInt nMaxQPower = pow2(nQubitCount);
    bitCapInt startMask = maxQPower - ONE_BCI;
    bitCapInt endMask = (toCopy->maxQPower - ONE_BCI) << qubitCount;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
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
    bitLenInt oQubitCount = toCopy->qubitCount;
    bitLenInt nQubitCount = qubitCount + oQubitCount;
    bitCapInt nMaxQPower = pow2(nQubitCount);
    bitCapInt startMask = pow2Mask(start);
    bitCapInt midMask = bitRegMask(start, oQubitCount);
    bitCapInt endMask = pow2Mask(qubitCount + oQubitCount) & ~(startMask | midMask);

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

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
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

    bitLenInt i;
    bitLenInt toComposeCount = toCopy.size();

    std::vector<bitLenInt> offset(toComposeCount);
    std::vector<bitCapInt> mask(toComposeCount);

    bitCapInt startMask = maxQPower - ONE_BCI;
    bitLenInt nQubitCount = qubitCount;
    bitCapInt nMaxQPower;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    for (i = 0; i < toComposeCount; i++) {
        QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[i]);
        if (src->doNormalize) {
            src->NormalizeState();
        }
        src->Finish();
        mask[i] = (src->GetMaxQPower() - ONE_BCI) << (bitCapIntOcl)nQubitCount;
        offset[i] = nQubitCount;
        ret[toCopy[i]] = nQubitCount;
        nQubitCount += src->GetQubitCount();
    }

    nMaxQPower = pow2(nQubitCount);

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
        nStateVec->write(lcv, stateVec->read(lcv & startMask));

        for (bitLenInt j = 0; j < toComposeCount; j++) {
            QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[j]);
            nStateVec->write(lcv, nStateVec->read(lcv) * src->stateVec->read((lcv & mask[j]) >> offset[j]));
        }
    });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

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

    if (!stateVec) {
        SetQubitCount(qubitCount - length);
        return;
    }

    bitLenInt nLength = qubitCount - length;

    bitCapIntOcl partPower = pow2Ocl(length);
    bitCapIntOcl remainderPower = pow2Ocl(nLength);

    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower]();
    real1* partStateAngle;
    real1* partStateProb;
    if (destination) {
        partStateAngle = new real1[partPower]();
        partStateProb = new real1[partPower]();
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (destination) {
        par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt j, l;
            bitCapIntOcl k;
            j = lcv & pow2Mask(start);
            j |= (lcv ^ j) << length;

            real1 firstAngle = -16 * M_PI;
            real1 currentAngle;
            real1 nrm;

            for (k = 0; k < partPower; k++) {
                l = j | (k << start);

                nrm = norm(stateVec->read(l));
                remainderStateProb[(bitCapIntOcl)lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    currentAngle = arg(stateVec->read(l));
                    if (firstAngle < (-8 * M_PI)) {
                        firstAngle = currentAngle;
                    }
                    partStateAngle[k] = currentAngle - firstAngle;
                }
            }
        });

        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt j, l;
            bitCapIntOcl k;
            j = lcv << start;

            real1 firstAngle = -16 * M_PI;
            real1 currentAngle;
            real1 nrm;

            for (k = 0; k < remainderPower; k++) {
                l = k & pow2Mask(start);
                l |= (k ^ l) << length;
                l = j | l;

                nrm = norm(stateVec->read(l));
                partStateProb[(bitCapIntOcl)lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    currentAngle = arg(stateVec->read(l));
                    if (firstAngle < (-8 * M_PI)) {
                        firstAngle = currentAngle;
                    }
                    remainderStateAngle[k] = currentAngle - firstAngle;
                }
            }
        });
    } else {
        par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt j, l;
            bitCapIntOcl k;
            j = lcv & pow2Mask(start);
            j |= (lcv ^ j) << length;

            for (k = 0; k < partPower; k++) {
                l = j | (k << start);

                remainderStateProb[(bitCapIntOcl)lcv] += norm(stateVec->read(l));
            }
        });

        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt j, l;
            bitCapIntOcl k;
            j = lcv << start;

            real1 firstAngle = -16 * M_PI;
            real1 currentAngle;

            for (k = 0; k < remainderPower; k++) {
                l = k & pow2Mask(start);
                l |= (k ^ l) << length;
                l = j | l;

                if (norm(stateVec->read(l)) > amplitudeFloor) {
                    currentAngle = arg(stateVec->read(l));
                    if (firstAngle < (-8 * M_PI)) {
                        firstAngle = currentAngle;
                    }
                    remainderStateAngle[k] = currentAngle - firstAngle;
                }
            }
        });
    }

    if (destination != nullptr) {
        destination->Dump();

        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            destination->stateVec->write(lcv,
                (real1)(std::sqrt(partStateProb[(bitCapIntOcl)lcv])) *
                    complex(cos(partStateAngle[(bitCapIntOcl)lcv]), sin(partStateAngle[(bitCapIntOcl)lcv])));
        });
    }

    if (nLength == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(nLength);
    }
    ResetStateVec(AllocStateVec(maxQPower));

    par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec->write(lcv,
            (real1)(std::sqrt(remainderStateProb[(bitCapIntOcl)lcv])) *
                complex(cos(remainderStateAngle[(bitCapIntOcl)lcv]), sin(remainderStateAngle[(bitCapIntOcl)lcv])));
    });

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
    if (destination) {
        delete[] partStateProb;
        delete[] partStateAngle;
    }
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

    if (!stateVec) {
        SetQubitCount(qubitCount - length);
        return;
    }

    bitLenInt nLength = qubitCount - length;
    bitCapInt remainderPower = pow2(nLength);
    bitCapInt skipMask = pow2(start) - ONE_BCI;
    bitCapInt disposedRes = disposedPerm << (bitCapIntOcl)start;
    bitCapInt saveMask = ~((pow2(start + length) - ONE_BCI) ^ skipMask);

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    StateVectorPtr nStateVec = AllocStateVec(remainderPower);
    stateVec->isReadLocked = false;

    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), [&](const bitCapInt lcv, const int cpu) {
            bitCapInt i, iLow, iHigh;
            iHigh = lcv & saveMask;
            iLow = iHigh & skipMask;
            i = iLow | ((iHigh ^ iLow) >> (bitCapIntOcl)length);
            nStateVec->write(i, stateVec->read(lcv));
        });
    } else {
        par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt i, iLow, iHigh;
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
    if (!stateVec) {
        return ZERO_R1;
    }

    bitCapInt qPower = pow2(qubit);
    real1 oneChance = 0;

    int numCores = GetConcurrencyLevel();
    real1* oneChanceBuff = new real1[numCores]();

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        oneChanceBuff[cpu] += norm(stateVec->read(lcv | qPower));
    };

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(qPower, qPower, qPower), fn);
    } else {
        par_for_skip(0, maxQPower, qPower, 1U, fn);
    }
    stateVec->isReadLocked = true;

    for (int i = 0; i < numCores; i++) {
        oneChance += oneChanceBuff[i];
    }

    delete[] oneChanceBuff;

    return clampProb(oneChance);
}

/// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
real1_f QEngineCPU::ProbAll(bitCapInt fullRegister)
{
    if (!stateVec) {
        return ZERO_R1;
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    return norm(stateVec->read(fullRegister));
}

// Returns probability of permutation of the register
real1_f QEngineCPU::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    if (!stateVec) {
        return ZERO_R1;
    }

    int num_threads = GetConcurrencyLevel();
    real1* probs = new real1[num_threads]();

    bitCapInt perm = permutation << start;

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) { probs[cpu] += norm(stateVec->read(lcv | perm)); };

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, bitRegMask(start, length), perm), fn);
    } else {
        par_for_skip(0, maxQPower, pow2(start), length, fn);
    }
    stateVec->isReadLocked = true;

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    delete[] probs;

    return clampProb(prob);
}

// Returns probability of permutation of the mask
real1_f QEngineCPU::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    if (!stateVec) {
        return ZERO_R1;
    }

    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapInt* skipPowers = new bitCapInt[length];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    int num_threads = GetConcurrencyLevel();
    real1* probs = new real1[num_threads]();

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->isReadLocked = false;
    par_for_mask(0, maxQPower, skipPowers, skipPowersVec.size(),
        [&](const bitCapInt lcv, const int cpu) { probs[cpu] += norm(stateVec->read(lcv | permutation)); });
    stateVec->isReadLocked = true;

    delete[] skipPowers;

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    delete[] probs;

    return clampProb(prob);
}

real1_f QEngineCPU::ProbParity(const bitCapInt& mask)
{
    if (!stateVec || !mask) {
        return ZERO_R1;
    }

    real1 oddChance = 0;

    int numCores = GetConcurrencyLevel();
    real1* oddChanceBuff = new real1[numCores]();

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bool parity = false;
        bitCapInt v = lcv & mask;
        while (v) {
            parity = !parity;
            v = v & (v - ONE_BCI);
        }

        if (parity) {
            oddChanceBuff[cpu] += norm(stateVec->read(lcv));
        }
    };

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0, maxQPower, fn);
    }
    stateVec->isReadLocked = true;

    for (int i = 0; i < numCores; i++) {
        oddChance += oddChanceBuff[i];
    }

    delete[] oddChanceBuff;

    return clampProb(oddChance);
}

bool QEngineCPU::ForceMParity(const bitCapInt& mask, bool result, bool doForce)
{
    if (!stateVec || !mask) {
        return ZERO_R1;
    }

    if (!doForce) {
        result = (Rand() <= ProbParity(mask));
    }

    real1 oddChance = 0;

    int numCores = GetConcurrencyLevel();
    real1* oddChanceBuff = new real1[numCores]();

    ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) {
        bool parity = false;
        bitCapInt v = lcv & mask;
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
        par_for(0, maxQPower, fn);
    }
    stateVec->isReadLocked = true;

    for (int i = 0; i < numCores; i++) {
        oddChance += oddChanceBuff[i];
    }

    delete[] oddChanceBuff;

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
        return 4.0f;
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

    stateVec->isReadLocked = false;
    toCompare->stateVec->isReadLocked = false;

    int numCores = GetConcurrencyLevel();
    real1* partError = new real1[numCores]();

    complex basePhaseFac1;
    real1 nrm = 0;
    bitCapInt basePerm;
    for (basePerm = 0; basePerm < maxQPower; basePerm++) {
        nrm = norm(stateVec->read(basePerm));
        if (nrm > amplitudeFloor) {
            basePhaseFac1 = (real1)(ONE_R1 / (real1)sqrt(nrm)) * (complex)stateVec->read(basePerm);
            break;
        }
    }

    if (basePerm == maxQPower) {
        // Max square difference:
        return 4.0f;
    }

    nrm = norm(toCompare->stateVec->read(basePerm));

    complex basePhaseFac2 = (real1)(ONE_R1 / (real1)sqrt(nrm)) * (complex)toCompare->stateVec->read(basePerm);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        real1 elemError = norm(
            basePhaseFac2 * (complex)stateVec->read(lcv) - basePhaseFac1 * (complex)toCompare->stateVec->read(lcv));
        partError[cpu] += elemError;
    });

    stateVec->isReadLocked = true;
    toCompare->stateVec->isReadLocked = true;

    real1 totError = ZERO_R1;
    for (int i = 0; i < numCores; i++) {
        totError += partError[i];
    }

    delete[] partError;

    return totError;
}

/// Phase flip always - equivalent to Z X Z X on any bit in the QEngineCPU
void QEngineCPU::PhaseFlip()
{
    CHECK_ZERO_SKIP();

    // This gate has no physical consequence. We only enable it for "book-keeping," if the engine is not using global
    // phase offsets.
    if (randGlobalPhase) {
        return;
    }

    Dispatch([this] {
        ParallelFunc fn = [&](const bitCapInt lcv, const int cpu) { stateVec->write(lcv, -stateVec->read(lcv)); };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0, maxQPower, fn);
        }
    });
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineCPU::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    Dispatch([this, start, length] {
        par_for_skip(0, maxQPower, pow2(start), length,
            [&](const bitCapInt lcv, const int cpu) { stateVec->write(lcv, -stateVec->read(lcv)); });
    });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void QEngineCPU::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    CHECK_ZERO_SKIP();

    Dispatch([this, greaterPerm, start, length, flagIndex] {
        bitCapInt regMask = bitRegMask(start, length);
        bitCapInt flagMask = pow2(flagIndex);

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            if ((((lcv & regMask) >> start) < greaterPerm) & ((lcv & flagMask) == flagMask))
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

/// This is an expedient for an adaptive Grover's search for a function's global minimum.
void QEngineCPU::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    Dispatch([this, greaterPerm, start, length] {
        bitCapInt regMask = bitRegMask(start, length);

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            if (((lcv & regMask) >> start) < greaterPerm)
                stateVec->write(lcv, -stateVec->read(lcv));
        });
    });
}

void QEngineCPU::ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
{
    CHECK_ZERO_SKIP();

    Dispatch([this, regMask, result, nrm] {
        ParallelFunc fn = [&](const bitCapInt i, const int cpu) {
            if ((i & regMask) == result) {
                stateVec->write(i, nrm * stateVec->read(i));
            } else {
                stateVec->write(i, complex(ZERO_R1, ZERO_R1));
            }
        };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0, maxQPower, fn);
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
        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            complex amp = stateVec->read(lcv) * nrm;
            stateVec->write(lcv, amp);
        });
    } else {
        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
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
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }
    runningNorm = par_norm(maxQPower, stateVec, norm_thresh);

    if (runningNorm <= amplitudeFloor) {
        ZeroAmplitudes();
    }
}

StateVectorPtr QEngineCPU::AllocStateVec(bitCapInt elemCount)
{
    if (isSparse) {
        return std::make_shared<StateVectorSparse>(elemCount);
    } else {
        return std::make_shared<StateVectorArray>(elemCount);
    }
}

void QEngineCPU::ResetStateVec(StateVectorPtr sv)
{
    // Removing this first line would not be a leak, but it's good to have the internal interface:
    FreeStateVec();
    stateVec = sv;
}
} // namespace Qrack
