//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine_cpu.hpp"

#include <chrono>
#include <thread>

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
    bool randomGlobalPhase, bool useHostMem, int64_t deviceID, bool useHardwareRNG, bool useSparseStateVec,
    real1_f norm_thresh, std::vector<int64_t> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, true, useHardwareRNG, norm_thresh)
    , isSparse(useSparseStateVec)
    , maxQubits(-1)
{
#if ENABLE_ENV_VARS
    if (getenv("QRACK_MAX_CPU_QB")) {
        maxQubits = std::stoi(std::string(getenv("QRACK_MAX_CPU_QB")));
    }
#endif

    if (qBitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
    }

    if (!qubitCount) {
        ZeroAmplitudes();
        return;
    }
    stateVec = AllocStateVec(maxQPowerOcl);
    stateVec->clear();

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        stateVec->write((bitCapIntOcl)initState, GetNonunitaryPhase());
    } else {
        stateVec->write((bitCapIntOcl)initState, phaseFac);
    }
}

void QEngineCPU::GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCPU::GetAmplitudePage range is out-of-bounds!");
    }

    Finish();

    if (stateVec) {
        stateVec->copy_out(pagePtr, offset, length);
    } else {
        par_for(0, length, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { pagePtr[lcv] = ZERO_CMPLX; });
    }
}
void QEngineCPU::SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCPU::SetAmplitudePage range is out-of-bounds!");
    }

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
        stateVec->clear();
    }

    Finish();

    stateVec->copy_in(pagePtr, offset, length);

    if (doNormalize) {
        runningNorm = REAL1_DEFAULT_ARG;
    }
}
void QEngineCPU::SetAmplitudePage(
    QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
{
    if (isBadPermRange(dstOffset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCPU::SetAmplitudePage source range is out-of-bounds!");
    }

    QEngineCPUPtr pageEngineCpuPtr = std::dynamic_pointer_cast<QEngineCPU>(pageEnginePtr);

    if (isBadPermRange(srcOffset, length, pageEngineCpuPtr->maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCPU::SetAmplitudePage source range is out-of-bounds!");
    }

    StateVectorPtr oStateVec = pageEngineCpuPtr->stateVec;

    if (!stateVec && !oStateVec) {
        return;
    }

    if (!oStateVec && (length == maxQPowerOcl)) {
        ZeroAmplitudes();
        return;
    }

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
        stateVec->clear();
    }

    Finish();
    pageEngineCpuPtr->Finish();

    stateVec->copy_in(oStateVec, srcOffset, dstOffset, length);

    runningNorm = REAL1_DEFAULT_ARG;
}
void QEngineCPU::ShuffleBuffers(QEnginePtr engine)
{
    if (qubitCount != engine->GetQubitCount()) {
        throw std::invalid_argument("QEngineCPU::ShuffleBuffers argument size differs from this!");
    }

    QEngineCPUPtr engineCpu = std::dynamic_pointer_cast<QEngineCPU>(engine);

    if (!stateVec && !(engineCpu->stateVec)) {
        return;
    }

    if (!stateVec) {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
        stateVec->clear();
    }

    if (!(engineCpu->stateVec)) {
        engineCpu->ResetStateVec(engineCpu->AllocStateVec(maxQPowerOcl));
        engineCpu->stateVec->clear();
    }

    Finish();
    engineCpu->Finish();

    stateVec->shuffle(engineCpu->stateVec);

    runningNorm = REAL1_DEFAULT_ARG;
    engineCpu->runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineCPU::CopyStateVec(QEnginePtr src)
{
    if (qubitCount != src->GetQubitCount()) {
        throw std::invalid_argument("QEngineCPU::CopyStateVec argument size differs from this!");
    }

    if (src->IsZeroAmplitude()) {
        ZeroAmplitudes();
        return;
    }

    if (stateVec) {
        Dump();
    } else {
        ResetStateVec(AllocStateVec(maxQPowerOcl));
    }

    if (isSparse) {
        std::unique_ptr<complex[]> sv(new complex[maxQPowerOcl]);
        src->GetQuantumState(sv.get());
        SetQuantumState(sv.get());
    } else {
        src->GetQuantumState(std::dynamic_pointer_cast<StateVectorArray>(stateVec)->amplitudes.get());
    }

    runningNorm = src->GetRunningNorm();
}

complex QEngineCPU::GetAmplitude(bitCapInt perm)
{
    if (bi_compare(perm, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::GetAmplitude argument out-of-bounds!");
    }

    // WARNING: Does not normalize!
    Finish();

    if (!stateVec) {
        return ZERO_CMPLX;
    }

    return stateVec->read((bitCapIntOcl)perm);
}

void QEngineCPU::SetAmplitude(bitCapInt perm, complex amp)
{
    if (bi_compare(perm, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::SetAmplitude argument out-of-bounds!");
    }

    // WARNING: Does not normalize!
    Finish();

    if (!stateVec && !norm(amp)) {
        return;
    }

    if (runningNorm != REAL1_DEFAULT_ARG) {
        runningNorm += norm(amp) - norm(stateVec->read((bitCapIntOcl)perm));
    }

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
            real1_f angle = Rand() * 2 * (real1_f)PI_R1;
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
        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { outputState[lcv] = ZERO_CMPLX; });
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
        par_for(0, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) { outputProbs[lcv] = ZERO_R1; });
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

#define NORM_THRESH_KERNEL(o1, o2, fn)                                                                                 \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex2 qubit = stateVec->read2(lcv + o1, lcv + o2);                                                          \
        qubit = fn;                                                                                                    \
                                                                                                                       \
        real1 dotMulRes = norm(qubit.c(0U));                                                                           \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit.f[0U] = ZERO_R1;                                                                                     \
            qubit.f[1U] = ZERO_R1;                                                                                     \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        dotMulRes = norm(qubit.c(1U));                                                                                 \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit.f[2U] = ZERO_R1;                                                                                     \
            qubit.f[3U] = ZERO_R1;                                                                                     \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
        stateVec->write2(lcv + offset1, qubit.c(0U), lcv + offset2, qubit.c(1U));                                      \
    }

#define NORM_CALC_KERNEL(o1, o2, fn)                                                                                   \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex2 qubit = stateVec->read2(lcv + o1, lcv + o2);                                                          \
        qubit = fn;                                                                                                    \
        rngNrm[cpu] += norm(qubit);                                                                                    \
        stateVec->write2(lcv + offset1, qubit.c(0U), lcv + offset2, qubit.c(1U));                                      \
    };

void QEngineCPU::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapIntOcl* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    if ((offset1 >= maxQPowerOcl) || (offset2 >= maxQPowerOcl)) {
        throw std::invalid_argument(
            "QEngineCPU::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
    }

    for (bitLenInt i = 0U; i < bitCount; ++i) {
        if (qPowsSorted[i] >= maxQPowerOcl) {
            throw std::invalid_argument(
                "QEngineCPU::Apply2x2 parameter qPowsSorted array values must be within allocated qubit bounds!");
        }
        if (i && (qPowsSorted[i - 1U] == qPowsSorted[i])) {
            throw std::invalid_argument("QEngineCPU::Apply2x2 parameter qPowSorted array values cannot be "
                                        "duplicated (for control and target qubits)!");
        }
    }

    std::shared_ptr<complex> mtrxS(new complex[4U], std::default_delete<complex[]>());
    std::copy(matrix, matrix + 4U, mtrxS.get());

    std::vector<bitCapIntOcl> qPowersSorted(bitCount);
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSorted.begin());

    const bool doApplyNorm = doNormalize && (bitCount == 1U) && (runningNorm > ZERO_R1);
    doCalcNorm &= doApplyNorm || (runningNorm <= ZERO_R1);

    const real1 nrm = doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch(maxQPowerOcl >> bitCount,
        [this, mtrxS, qPowersSorted, offset1, offset2, bitCount, doCalcNorm, doApplyNorm, nrm, nrm_thresh] {
            complex* mtrx = mtrxS.get();

            const real1_f norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : nrm_thresh;
            const unsigned numCores = GetConcurrencyLevel();

            const complex2 mtrxCol1(mtrx[0U], mtrx[2U]);
            const complex2 mtrxCol2(mtrx[1U], mtrx[3U]);

            const complex2 mtrxCol1Shuff = mtrxColShuff(mtrxCol1);
            const complex2 mtrxCol2Shuff = mtrxColShuff(mtrxCol2);

            const complex2 mtrxPhase = (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) ? complex2(mtrx[0U], mtrx[3U])
                                                                                    : complex2(mtrx[1U], mtrx[2U]);

            std::unique_ptr<real1[]> rngNrm(new real1[numCores]());
            ParallelFunc fn;
            if (!doCalcNorm) {
                if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        complex2 qubit = stateVec->read2(lcv + offset1, lcv + offset2);
                        qubit = mtrxPhase * qubit;
                        stateVec->write2(lcv + offset1, qubit.c(0U), lcv + offset2, qubit.c(1U));
                    };
                } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        complex2 qubit = stateVec->read2(lcv + offset2, lcv + offset1);
                        qubit = mtrxPhase * qubit;
                        stateVec->write2(lcv + offset1, qubit.c(0U), lcv + offset2, qubit.c(1U));
                    };
                } else {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        complex2 qubit = stateVec->read2(lcv + offset1, lcv + offset2);
                        qubit = matrixMul(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit);
                        stateVec->write2(lcv + offset1, qubit.c(0U), lcv + offset2, qubit.c(1U));
                    };
                }
            } else if (norm_thresh > ZERO_R1) {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
                        fn = NORM_THRESH_KERNEL(offset1, offset2, nrm * mtrxPhase * qubit);
                    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
                        fn = NORM_THRESH_KERNEL(offset2, offset1, nrm * mtrxPhase * qubit);
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            offset1, offset2, matrixMul(nrm, mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit));
                    }
                } else {
                    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
                        fn = NORM_THRESH_KERNEL(offset1, offset2, mtrxPhase * qubit);
                    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
                        fn = NORM_THRESH_KERNEL(offset2, offset1, nrm * mtrxPhase * qubit);
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            offset1, offset2, matrixMul(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit));
                    }
                }
            } else {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
                        fn = NORM_CALC_KERNEL(offset1, offset2, nrm * mtrxPhase * qubit);
                    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
                        fn = NORM_CALC_KERNEL(offset2, offset1, nrm * mtrxPhase * qubit);
                    } else {
                        fn = NORM_CALC_KERNEL(
                            offset1, offset2, matrixMul(nrm, mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit));
                    }
                } else {
                    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
                        fn = NORM_CALC_KERNEL(offset1, offset2, mtrxPhase * qubit);
                    } else if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
                        fn = NORM_CALC_KERNEL(offset2, offset1, mtrxPhase * qubit);
                    } else {
                        fn = NORM_CALC_KERNEL(
                            offset1, offset2, matrixMul(mtrxCol1, mtrxCol2, mtrxCol1Shuff, mtrxCol2Shuff, qubit));
                    }
                }
            }

            if (stateVec->is_sparse()) {
                const bitCapIntOcl setMask = offset1 ^ offset2;
                bitCapIntOcl filterMask = 0U;
                for (bitLenInt i = 0U; i < bitCount; ++i) {
                    filterMask |= qPowersSorted[i];
                }
                filterMask &= ~setMask;
                const bitCapIntOcl filterValues = filterMask & offset1 & offset2;
                par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
            } else {
                par_for_mask(0U, maxQPowerOcl, qPowersSorted, fn);
            }

            if (doApplyNorm) {
                runningNorm = ONE_R1;
            }

            if (!doCalcNorm) {
                return;
            }

            real1 rNrm = ZERO_R1;
            for (unsigned i = 0U; i < numCores; ++i) {
                rNrm += rngNrm[i];
            }
            rngNrm.reset();
            runningNorm = rNrm;

            if (runningNorm <= FP_NORM_EPSILON) {
                ZeroAmplitudes();
            }
        });
}
#else

#define NORM_THRESH_KERNEL(fn1, fn2)                                                                                   \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex qubit[2U];                                                                                             \
                                                                                                                       \
        const complex Y0 = stateVec->read(lcv + offset1);                                                              \
        qubit[1U] = stateVec->read(lcv + offset2);                                                                     \
                                                                                                                       \
        qubit[0U] = fn1;                                                                                               \
        qubit[1U] = fn2;                                                                                               \
                                                                                                                       \
        real1 dotMulRes = norm(qubit[0U]);                                                                             \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit[0U] = ZERO_CMPLX;                                                                                    \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
                                                                                                                       \
        dotMulRes = norm(qubit[1U]);                                                                                   \
        if (dotMulRes < norm_thresh) {                                                                                 \
            qubit[1U] = ZERO_CMPLX;                                                                                    \
        } else {                                                                                                       \
            rngNrm[cpu] += dotMulRes;                                                                                  \
        }                                                                                                              \
        stateVec->write2(lcv + offset1, qubit[0U], lcv + offset2, qubit[1U]);                                          \
    }

#define NORM_CALC_KERNEL(fn1, fn2)                                                                                     \
    [&](const bitCapIntOcl& lcv, const unsigned& cpu) {                                                                \
        complex qubit[2U];                                                                                             \
                                                                                                                       \
        const complex Y0 = stateVec->read(lcv + offset1);                                                              \
        qubit[1U] = stateVec->read(lcv + offset2);                                                                     \
                                                                                                                       \
        qubit[0U] = fn1;                                                                                               \
        qubit[1U] = fn2;                                                                                               \
                                                                                                                       \
        rngNrm[cpu] = norm(qubit[0U]) + norm(qubit[1U]);                                                               \
                                                                                                                       \
        stateVec->write2(lcv + offset1, qubit[0U], lcv + offset2, qubit[1U]);                                          \
    };

void QEngineCPU::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* matrix, const bitLenInt bitCount,
    const bitCapIntOcl* qPowsSorted, bool doCalcNorm, real1_f nrm_thresh)
{
    CHECK_ZERO_SKIP();

    if ((offset1 >= maxQPowerOcl) || (offset2 >= maxQPowerOcl)) {
        throw std::invalid_argument(
            "QEngineCPU::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
    }

    for (bitLenInt i = 0U; i < bitCount; ++i) {
        if (qPowsSorted[i] >= maxQPowerOcl) {
            throw std::invalid_argument(
                "QEngineCPU::Apply2x2 parameter qPowsSorted array values must be within allocated qubit bounds!");
        }
        if (i && (qPowsSorted[i - 1U] == qPowsSorted[i])) {
            throw std::invalid_argument("QEngineCPU::Apply2x2 parameter qPowsSorted array values cannot be "
                                        "duplicated (for control and target qubits)!");
        }
    }

    std::shared_ptr<complex> mtrxS(new complex[4U], std::default_delete<complex[]>());
    std::copy(matrix, matrix + 4U, mtrxS.get());

    std::vector<bitCapIntOcl> qPowersSorted(bitCount);
    std::copy(qPowsSorted, qPowsSorted + bitCount, qPowersSorted.begin());

    const bool doApplyNorm = doNormalize && (bitCount == 1U) && (runningNorm > ZERO_R1);
    doCalcNorm &= doApplyNorm || (runningNorm <= ZERO_R1);

    const real1 nrm = doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm) {
        runningNorm = ONE_R1;
    }

    Dispatch(maxQPowerOcl >> bitCount,
        [this, mtrxS, qPowersSorted, offset1, offset2, bitCount, doCalcNorm, doApplyNorm, nrm, nrm_thresh] {
            complex* mtrx = mtrxS.get();
            const complex mtrx0 = mtrx[0U];
            const complex mtrx1 = mtrx[1U];
            const complex mtrx2 = mtrx[2U];
            const complex mtrx3 = mtrx[3U];

            const real1 norm_thresh = (nrm_thresh < ZERO_R1) ? amplitudeFloor : (real1)nrm_thresh;
            const unsigned numCores = GetConcurrencyLevel();

            std::unique_ptr<real1[]> rngNrm(new real1[numCores]());
            ParallelFunc fn;
            if (!doCalcNorm) {
                if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        stateVec->write2(lcv + offset1, mtrx0 * stateVec->read(lcv + offset1), lcv + offset2,
                            mtrx3 * stateVec->read(lcv + offset2));
                    };
                } else if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        stateVec->write2(lcv + offset1, mtrx1 * stateVec->read(lcv + offset2), lcv + offset2,
                            mtrx2 * stateVec->read(lcv + offset1));
                    };
                } else {
                    fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                        complex Y0 = stateVec->read(lcv + offset1);
                        complex Y1 = stateVec->read(lcv + offset2);
                        stateVec->write2(
                            lcv + offset1, (mtrx0 * Y0) + (mtrx1 * Y1), lcv + offset2, (mtrx2 * Y0) + (mtrx3 * Y1));
                    };
                }
            } else if (norm_thresh > ZERO_R1) {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
                        fn = NORM_THRESH_KERNEL(nrm * (mtrx0 * Y0), nrm * (mtrx3 * qubit[1U]));
                    } else if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
                        fn = NORM_THRESH_KERNEL(nrm * (mtrx1 * qubit[1U]), nrm * (mtrx2 * Y0));
                    } else {
                        fn = NORM_THRESH_KERNEL(
                            nrm * ((mtrx0 * Y0) + (mtrx1 * qubit[1U])), nrm * ((mtrx2 * Y0) + (mtrx3 * qubit[1U])));
                    }
                } else {
                    if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
                        fn = NORM_THRESH_KERNEL(mtrx0 * Y0, mtrx3 * qubit[1U]);
                    } else if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
                        fn = NORM_THRESH_KERNEL(mtrx1 * qubit[1U], mtrx2 * Y0);
                    } else {
                        fn = NORM_THRESH_KERNEL((mtrx0 * Y0) + (mtrx1 * qubit[1U]), (mtrx2 * Y0) + (mtrx3 * qubit[1U]));
                    }
                }
            } else {
                if (abs(ONE_R1 - nrm) > REAL1_EPSILON) {
                    if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
                        fn = NORM_CALC_KERNEL(nrm * (mtrx0 * Y0), nrm * (mtrx3 * qubit[1U]));
                    } else if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
                        fn = NORM_CALC_KERNEL(nrm * (mtrx1 * qubit[1U]), nrm * (mtrx2 * Y0));
                    } else {
                        fn = NORM_CALC_KERNEL(
                            nrm * ((mtrx0 * Y0) + (mtrx1 * qubit[1U])), nrm * ((mtrx2 * Y0) + (mtrx3 * qubit[1U])));
                    }
                } else {
                    if (IS_NORM_0(mtrx1) && IS_NORM_0(mtrx2)) {
                        fn = NORM_CALC_KERNEL(mtrx0 * Y0, mtrx3 * qubit[1U]);
                    } else if (IS_NORM_0(mtrx0) && IS_NORM_0(mtrx3)) {
                        fn = NORM_CALC_KERNEL(mtrx1 * qubit[1U], mtrx2 * Y0);
                    } else {
                        fn = NORM_CALC_KERNEL((mtrx0 * Y0) + (mtrx1 * qubit[1U]), (mtrx2 * Y0) + (mtrx3 * qubit[1U]));
                    }
                }
            }

            if (stateVec->is_sparse()) {
                const bitCapIntOcl setMask = offset1 ^ offset2;
                bitCapIntOcl filterMask = 0U;
                for (bitLenInt i = 0U; i < bitCount; ++i) {
                    filterMask |= qPowersSorted[i];
                }
                filterMask &= ~setMask;
                const bitCapIntOcl filterValues = filterMask & offset1 & offset2;
                par_for_set(CastStateVecSparse()->iterable(setMask, filterMask, filterValues), fn);
            } else {
                par_for_mask(0U, maxQPowerOcl, qPowersSorted, fn);
            }

            if (doApplyNorm) {
                runningNorm = ONE_R1;
            }

            if (!doCalcNorm) {
                return;
            }

            real1 rNrm = ZERO_R1;
            for (unsigned i = 0U; i < numCores; ++i) {
                rNrm += rngNrm[i];
            }
            rngNrm.reset();
            runningNorm = rNrm;

            if (runningNorm <= FP_NORM_EPSILON) {
                ZeroAmplitudes();
            }
        });
}
#endif

void QEngineCPU::XMask(bitCapInt mask)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::XMask mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (bi_compare_0(mask) == 0) {
        return;
    }

    if (isPowerOfTwo(mask)) {
        X(log2(mask));
        return;
    }

    if (stateVec->is_sparse()) {
        QInterface::XMask(mask);
        return;
    }

    Dispatch(maxQPowerOcl, [this, mask] {
        const bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
        const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ maskOcl;
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl otherRes = lcv & otherMask;
            bitCapIntOcl setInt = lcv & maskOcl;
            bitCapIntOcl resetInt = setInt ^ maskOcl;

            if (setInt < resetInt) {
                return;
            }

            setInt |= otherRes;
            resetInt |= otherRes;

            const complex Y0 = stateVec->read(resetInt);
            stateVec->write(resetInt, stateVec->read(setInt));
            stateVec->write(setInt, Y0);
        };

        par_for(0U, maxQPowerOcl, fn);
    });
}

void QEngineCPU::PhaseParity(real1_f radians, bitCapInt mask)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::PhaseParity mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    if (bi_compare_0(mask) == 0) {
        return;
    }

    if (isPowerOfTwo(mask)) {
        const complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
        Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
        return;
    }

    if (stateVec->is_sparse()) {
        QInterface::PhaseParity(radians, mask);
        return;
    }

    Dispatch(maxQPowerOcl, [this, mask, radians] {
        const bitCapIntOcl parityStartSize = 4U * sizeof(bitCapIntOcl);
        const complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
        const complex iPhaseFac = ONE_CMPLX / phaseFac;
        const bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
        const bitCapIntOcl otherMask = (maxQPowerOcl - 1U) ^ maskOcl;
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl otherRes = lcv & otherMask;
            bitCapIntOcl setInt = lcv & maskOcl;

            bitCapIntOcl v = setInt;
            for (bitCapIntOcl paritySize = parityStartSize; paritySize > 0U; paritySize >>= 1U) {
                v ^= v >> paritySize;
            }
            v &= 1U;

            setInt |= otherRes;

            stateVec->write(setInt, (v ? phaseFac : iPhaseFac) * stateVec->read(setInt));
        };

        par_for(0U, maxQPowerOcl, fn);
    });
}

void QEngineCPU::UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
    const complex* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (!controls.size()) {
        Mtrx(mtrxs + ((bitCapIntOcl)mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    if (qubitIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, qubitCount, "QEngineCPU::UniformlyControlledSingleBit control is out-of-bounds!");

    const bitCapIntOcl targetPower = pow2Ocl(qubitIndex);

    std::vector<bitCapIntOcl> qPowers(controls.size());
    std::transform(controls.begin(), controls.end(), qPowers.begin(), pow2Ocl);

    std::vector<bitCapIntOcl> mtrxSkipPowersOcl(mtrxSkipPowers.size());
    std::transform(mtrxSkipPowers.begin(), mtrxSkipPowers.end(), mtrxSkipPowersOcl.begin(),
        [](bitCapInt i) { return (bitCapIntOcl)i; });

    const bitCapIntOcl mtrxSkipValueMaskOcl = (bitCapIntOcl)mtrxSkipValueMask;

    const real1 nrm = (runningNorm > ZERO_R1) ? ONE_R1 / (real1)sqrt(runningNorm) : ONE_R1;

    ParallelFunc fn;
    if (doNormalize && ((ONE_R1 - nrm) > FP_NORM_EPSILON)) {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl offset = 0U;
            for (size_t j = 0U; j < controls.size(); ++j) {
                if (lcv & qPowers[j]) {
                    offset |= pow2Ocl(j);
                }
            }

            bitCapIntOcl i = 0U;
            bitCapIntOcl iHigh = offset;
            for (bitCapIntOcl p = 0U; p < mtrxSkipPowers.size(); ++p) {
                bitCapIntOcl iLow = iHigh & (mtrxSkipPowersOcl[p] - 1U);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << 1U;
            }
            i |= iHigh;

            // Offset is permutation * 4, for the components of 2x2 matrices. (Note that this sacrifices 2 qubits of
            // capacity for the unsigned bitCapInt.)
            offset = (i | mtrxSkipValueMaskOcl) * 4U;

            complex qubit[2U];

            const complex Y0 = stateVec->read(lcv);
            qubit[1U] = stateVec->read(lcv | targetPower);

            qubit[0U] = nrm * ((mtrxs[0U + offset] * Y0) + (mtrxs[1U + offset] * qubit[1U]));
            qubit[1U] = nrm * ((mtrxs[2U + offset] * Y0) + (mtrxs[3U + offset] * qubit[1U]));

            stateVec->write2(lcv, qubit[0U], lcv | targetPower, qubit[1U]);
        };
    } else {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl offset = 0U;
            for (size_t j = 0U; j < controls.size(); ++j) {
                if (lcv & qPowers[j]) {
                    offset |= pow2Ocl(j);
                }
            }

            bitCapIntOcl i = 0U;
            bitCapIntOcl iHigh = offset;
            for (bitCapIntOcl p = 0U; p < mtrxSkipPowers.size(); ++p) {
                bitCapIntOcl iLow = iHigh & (mtrxSkipPowersOcl[p] - 1U);
                i |= iLow;
                iHigh = (iHigh ^ iLow) << 1U;
            }
            i |= iHigh;

            // Offset is permutation * 4, for the components of 2x2 matrices. (Note that this sacrifices 2 qubits of
            // capacity for the unsigned bitCapInt.)
            offset = (i | mtrxSkipValueMaskOcl) * 4U;

            complex qubit[2U];

            const complex Y0 = stateVec->read(lcv);
            qubit[1U] = stateVec->read(lcv | targetPower);

            qubit[0U] = (mtrxs[0U + offset] * Y0) + (mtrxs[1U + offset] * qubit[1U]);
            qubit[1U] = (mtrxs[2U + offset] * Y0) + (mtrxs[3U + offset] * qubit[1U]);

            stateVec->write2(lcv, qubit[0U], lcv | targetPower, qubit[1U]);
        };
    }

    Finish();

    par_for_skip(0U, maxQPowerOcl, targetPower, 1U, fn);

    if (doNormalize) {
        runningNorm = ONE_R1;
    }
}

void QEngineCPU::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::UniformParityRZ mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    Dispatch(maxQPowerOcl, [this, mask, angle] {
        const real1 cosine = (real1)cos(angle);
        const real1 sine = (real1)sin(angle);
        const complex phaseFac(cosine, sine);
        const complex phaseFacAdj(cosine, -sine);
        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl perm = lcv & (bitCapIntOcl)mask;
            // From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
            // c accumulates the total bits set in v
            bitLenInt c;
            for (c = 0U; perm; ++c) {
                // clear the least significant bit set
                perm &= perm - 1U;
            }
            stateVec->write(lcv, stateVec->read(lcv) * ((c & 1U) ? phaseFac : phaseFacAdj));
        };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0U, maxQPowerOcl, fn);
        }
    });
}

void QEngineCPU::CUniformParityRZ(const std::vector<bitLenInt>& cControls, bitCapInt mask, real1_f angle)
{
    if (!cControls.size()) {
        return UniformParityRZ(mask, angle);
    }

    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::CUniformParityRZ mask out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(cControls, qubitCount, "QEngineCPU::CUniformParityRZ control is out-of-bounds!");

    CHECK_ZERO_SKIP();

    std::vector<bitLenInt> controls(cControls.begin(), cControls.end());
    std::sort(controls.begin(), controls.end());

    Dispatch(maxQPowerOcl >> cControls.size(), [this, controls, mask, angle] {
        bitCapIntOcl controlMask = 0U;
        std::vector<bitCapIntOcl> controlPowers(controls.size());
        for (size_t i = 0U; i < controls.size(); ++i) {
            controlPowers[i] = pow2Ocl(controls[i]);
            controlMask |= controlPowers[i];
        }

        const real1 cosine = (real1)cos(angle);
        const real1 sine = (real1)sin(angle);
        const complex phaseFac(cosine, sine);
        const complex phaseFacAdj(cosine, -sine);

        ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl perm = lcv & (bitCapIntOcl)mask;
            // From https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetNaive
            // c accumulates the total bits set in v
            bitLenInt c;
            for (c = 0U; perm; ++c) {
                // clear the least significant bit set
                perm &= perm - 1U;
            }
            stateVec->write(controlMask | lcv, stateVec->read(controlMask | lcv) * ((c & 1U) ? phaseFac : phaseFacAdj));
        };

        par_for_mask(0U, maxQPowerOcl, controlPowers, fn);
    });
}

/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old unit that was added.
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy)
{
    const bitLenInt result = qubitCount;

    if (!toCopy->qubitCount) {
        return result;
    }

    const bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    if (nQubitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
    }

    if (!qubitCount) {
        Finish();
        SetQubitCount(toCopy->qubitCount);
        toCopy->Finish();
        runningNorm = toCopy->runningNorm;
        if (toCopy->stateVec) {
            stateVec = AllocStateVec(toCopy->maxQPowerOcl);
            stateVec->copy(toCopy->stateVec);
        }

        return 0U;
    }

    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    if (!stateVec || !toCopy->stateVec) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(nQubitCount);
        return result;
    }

    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = maxQPowerOcl - 1U;
    const bitCapIntOcl endMask = (toCopy->maxQPowerOcl - 1U) << qubitCount;

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
        par_for(0U, nMaxQPower, fn);
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
    if (start > qubitCount) {
        throw std::invalid_argument("QEngineCPU::Compose start index is out-of-bounds!");
    }

    if (!qubitCount) {
        Compose(toCopy);
        return 0U;
    }

    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    const bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    if (nQubitCount > maxQubits) {
        throw std::invalid_argument(
            "Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
    }

    if (!stateVec || !toCopy->stateVec) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(nQubitCount);
        return start;
    }

    const bitLenInt oQubitCount = toCopy->qubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = pow2MaskOcl(start);
    const bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    const bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);

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

    par_for(0U, nMaxQPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
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
    const bitLenInt toComposeCount = toCopy.size();
    bitLenInt nQubitCount = qubitCount;
    std::map<QInterfacePtr, bitLenInt> ret;
    std::vector<bitLenInt> offset(toComposeCount);
    std::vector<bitCapIntOcl> mask(toComposeCount);

    const bitCapIntOcl startMask = maxQPowerOcl - 1U;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    for (bitLenInt i = 0U; i < toComposeCount; ++i) {
        QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[i]);
        if (src->doNormalize) {
            src->NormalizeState();
        }
        src->Finish();
        mask[i] = (src->maxQPowerOcl - 1U) << (bitCapIntOcl)nQubitCount;
        offset[i] = nQubitCount;
        ret[toCopy[i]] = nQubitCount;
        nQubitCount += src->GetQubitCount();
    }

    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);
    stateVec->isReadLocked = false;

    par_for(0U, nMaxQPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        nStateVec->write(lcv, stateVec->read(lcv & startMask));

        for (bitLenInt j = 0U; j < toComposeCount; ++j) {
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
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nLength = qubitCount - length;

    if (!stateVec) {
        SetQubitCount(nLength);
        if (destination) {
            destination->ZeroAmplitudes();
        }
        return;
    }

    if (!nLength) {
        if (destination) {
            destination->stateVec = stateVec;
        }
        stateVec = NULL;
        SetQubitCount(0U);

        return;
    }

    if (destination && !destination->stateVec) {
        // Reinitialize stateVec RAM
        destination->SetPermutation(ZERO_BCI);
    }

    const bitCapIntOcl partPower = pow2Ocl(length);
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);

    std::unique_ptr<real1[]> remainderStateProb(new real1[remainderPower]());
    std::unique_ptr<real1[]> remainderStateAngle(new real1[remainderPower]());
    std::unique_ptr<real1[]> partStateProb;
    std::unique_ptr<real1[]> partStateAngle;
    if (destination) {
        // Note that the extra parentheses mean to init as 0:
        partStateProb = std::unique_ptr<real1[]>(new real1[partPower]());
        partStateAngle = std::unique_ptr<real1[]>(new real1[partPower]());
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (destination) {
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0U; k < partPower; ++k) {
                bitCapIntOcl l = j | (k << start);

                const complex amp = stateVec->read(l);
                const real1 nrm = norm(amp);
                remainderStateProb[lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    partStateAngle[k] = arg(amp);
                }
            }
        });

        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv << start;

            for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
                bitCapIntOcl l = k & pow2MaskOcl(start);
                l |= j | ((k ^ l) << length);

                const complex amp = stateVec->read(l);
                const real1 nrm = norm(amp);
                partStateProb[lcv] += nrm;

                if (nrm > amplitudeFloor) {
                    remainderStateAngle[k] = arg(amp);
                }
            }
        });
    } else {
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0U; k < partPower; ++k) {
                remainderStateProb[lcv] += norm(stateVec->read(j | (k << start)));
            }
        });

        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j;
            j = lcv << start;

            for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
                bitCapIntOcl l = k & pow2MaskOcl(start);
                l |= j | ((k ^ l) << length);

                const complex amp = stateVec->read(l);

                if (norm(amp) > amplitudeFloor) {
                    remainderStateAngle[k] = arg(amp);
                }
            }
        });
    }

    if (destination) {
        destination->Dump();

        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            destination->stateVec->write(lcv,
                (real1)(std::sqrt((real1_s)partStateProb[lcv])) *
                    complex(cos(partStateAngle[lcv]), sin(partStateAngle[lcv])));
        });

        partStateProb.reset();
        partStateAngle.reset();
    }

    SetQubitCount(nLength);

    ResetStateVec(AllocStateVec(maxQPowerOcl));

    par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        stateVec->write(lcv,
            (real1)(std::sqrt((real1_s)remainderStateProb[lcv])) *
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
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::Dispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nLength = qubitCount - length;

    if (!stateVec) {
        SetQubitCount(nLength);
        return;
    }

    const bitCapIntOcl disposedPermOcl = (bitCapIntOcl)disposedPerm;
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);
    const bitCapIntOcl skipMask = pow2Ocl(start) - 1U;
    const bitCapIntOcl disposedRes = disposedPermOcl << (bitCapIntOcl)start;

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    StateVectorPtr nStateVec = AllocStateVec(remainderPower);
    stateVec->isReadLocked = false;

    par_for(0U, remainderPower, [&](const bitCapIntOcl& iHigh, const unsigned& cpu) {
        const bitCapIntOcl iLow = iHigh & skipMask;
        nStateVec->write(iHigh, stateVec->read(iLow | ((iHigh ^ iLow) << (bitCapIntOcl)length) | disposedRes));
    });

    if (!nLength) {
        SetQubitCount(1U);
    } else {
        SetQubitCount(nLength);
    }

    ResetStateVec(nStateVec);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1_f QEngineCPU::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QEngineCPU::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1_F;
    }

    if (qubitCount == 1U) {
        return norm(stateVec->read(1U));
    }

    const bitCapIntOcl qPower = pow2Ocl(qubit);
    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    ParallelFunc fn;
    if (isSparse) {
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            oneChanceBuff[cpu] += norm(stateVec->read(lcv | qPower));
        };
    } else {
#if ENABLE_COMPLEX_X2
        if (qPower == 1U) {
            fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                oneChanceBuff[cpu] += norm(stateVec->read2((lcv << 2U) | 1U, (lcv << 2U) | 3U));
            };
        } else {
            fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
                oneChanceBuff[cpu] += norm(stateVec->read2((lcv << 1U) | qPower, (lcv << 1U) | 1U | qPower));
            };
        }
#else
        fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            oneChanceBuff[cpu] += norm(stateVec->read(lcv | qPower));
        };
#endif
    }

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(qPower, qPower, qPower), fn);
    } else {
#if ENABLE_COMPLEX_X2
        if (qPower == 1U) {
            par_for(0U, maxQPowerOcl >> 2U, fn);
        } else {
            par_for_skip(0U, maxQPowerOcl >> 1U, qPower >> 1U, 1U, fn);
        }
#else
        par_for_skip(0U, maxQPowerOcl, qPower, 1U, fn);
#endif
    }
    stateVec->isReadLocked = true;

    real1 oneChance = ZERO_R1;
    for (unsigned i = 0U; i < numCores; ++i) {
        oneChance += oneChanceBuff[i];
    }

    return clampProb((real1_f)oneChance);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state, if control is in |0>/|1>, false/true,
/// "controlState".
real1_f QEngineCPU::CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target)
{
    if (!stateVec) {
        return ZERO_R1_F;
    }

    real1_f controlProb = Prob(control);
    if (!controlState) {
        controlProb = ONE_R1 - controlProb;
    }
    if (controlProb <= FP_NORM_EPSILON) {
        return ZERO_R1;
    }
    if ((ONE_R1 - controlProb) <= FP_NORM_EPSILON) {
        return Prob(target);
    }

    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QEngineCPU::CtrlOrAntiProb target index parameter must be within allocated qubit bounds!");
    }

    const bitCapIntOcl qControlPower = pow2Ocl(control);
    const bitCapIntOcl qControlMask = controlState ? qControlPower : 0U;
    const bitCapIntOcl qPower = pow2Ocl(target);
    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oneChanceBuff(new real1[numCores]());

    ParallelFunc fn = (ParallelFunc)([&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        if ((lcv & qControlPower) == qControlMask) {
            oneChanceBuff[cpu] += norm(stateVec->read(lcv | qPower));
        }
    });

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(qPower, qPower, qPower), fn);
    } else {
        par_for_skip(0U, maxQPowerOcl, qPower, 1U, fn);
    }
    stateVec->isReadLocked = true;

    real1 oneChance = ZERO_R1;
    for (unsigned i = 0U; i < numCores; ++i) {
        oneChance += oneChanceBuff[i];
    }
    oneChance /= controlProb;

    return clampProb((real1_f)oneChance);
}

// Returns probability of permutation of the register
real1_f QEngineCPU::ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
{
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1_F;
    }

    const unsigned num_threads = GetConcurrencyLevel();
    std::unique_ptr<real1[]> probs(new real1[num_threads]());

    const bitCapIntOcl perm = (bitCapIntOcl)permutation << ((bitCapIntOcl)start);
    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        probs[cpu] += norm(stateVec->read(lcv | perm));
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(0, bitRegMaskOcl(start, length), perm), fn);
    } else {
        par_for_skip(0U, maxQPowerOcl, pow2Ocl(start), length, fn);
    }
    stateVec->isReadLocked = true;

    real1 prob = ZERO_R1;
    for (unsigned thrd = 0; thrd < num_threads; ++thrd) {
        prob += probs[thrd];
    }

    return clampProb((real1_f)prob);
}

// Returns probability of permutation of the mask
real1_f QEngineCPU::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::ProbMask mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec) {
        return ZERO_R1_F;
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    while (v) {
        bitCapIntOcl oldV = v;
        v &= v - 1U; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    const unsigned num_threads = GetConcurrencyLevel();
    std::unique_ptr<real1[]> probs(new real1[num_threads]());

    const bitCapIntOcl permutationOcl = (bitCapIntOcl)permutation;
    stateVec->isReadLocked = false;
    par_for_mask(0U, maxQPowerOcl, skipPowersVec, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        probs[cpu] += norm(stateVec->read(lcv | permutationOcl));
    });
    stateVec->isReadLocked = true;

    real1 prob = ZERO_R1;
    for (unsigned thrd = 0; thrd < num_threads; ++thrd) {
        prob += probs[thrd];
    }

    return clampProb((real1_f)prob);
}

real1_f QEngineCPU::ProbParity(bitCapInt mask)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::ProbParity mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (!stateVec || (bi_compare_0(mask) == 0)) {
        return ZERO_R1_F;
    }

    real1 oddChance = ZERO_R1;

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oddChanceBuff(new real1[numCores]());

    const bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        bool parity = false;
        bitCapIntOcl v = lcv & maskOcl;
        while (v) {
            parity = !parity;
            v = v & (v - 1U);
        }

        if (parity) {
            oddChanceBuff[cpu] += norm(stateVec->read(lcv));
        }
    };

    stateVec->isReadLocked = false;
    if (stateVec->is_sparse()) {
        par_for_set(CastStateVecSparse()->iterable(), fn);
    } else {
        par_for(0U, maxQPowerOcl, fn);
    }
    stateVec->isReadLocked = true;

    for (unsigned i = 0U; i < numCores; ++i) {
        oddChance += oddChanceBuff[i];
    }

    return clampProb((real1_f)oddChance);
}

bitCapInt QEngineCPU::MAll()
{
    const real1_f rnd = Rand();
    real1_f totProb = ZERO_R1_F;
    bitCapInt lastNonzero = maxQPower;
    bi_decrement(&lastNonzero, 1U);
    bitCapInt perm = ZERO_BCI;
    while (perm < maxQPower) {
        const real1_f partProb = ProbAll(perm);
        if (partProb > REAL1_EPSILON) {
            totProb += partProb;
            if ((totProb > rnd) || ((ONE_R1_F - totProb) <= FP_NORM_EPSILON)) {
                SetPermutation(perm);
                return perm;
            }
            lastNonzero = perm;
        }
        bi_increment(&perm, 1U);
    }

    SetPermutation(lastNonzero);
    return lastNonzero;
}

bool QEngineCPU::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    if (bi_compare(mask, maxQPower) >= 0) {
        throw std::invalid_argument("QEngineCPU::ForceMParity mask out-of-bounds!");
    }

    if (!stateVec || (bi_compare_0(mask) == 0)) {
        return false;
    }

    if (!doForce) {
        result = (Rand() <= ProbParity(mask));
    }

    real1 oddChance = ZERO_R1;

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<real1[]> oddChanceBuff(new real1[numCores]());

    const bitCapIntOcl maskOcl = (bitCapIntOcl)mask;
    ParallelFunc fn = [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        bool parity = false;
        bitCapIntOcl v = lcv & maskOcl;
        while (v) {
            parity = !parity;
            v = v & (v - 1U);
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
        par_for(0U, maxQPowerOcl, fn);
    }
    stateVec->isReadLocked = true;

    for (unsigned i = 0U; i < numCores; ++i) {
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
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
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
        return ZERO_R1_F;
    }

    if (!stateVec) {
        toCompare->UpdateRunningNorm();
        return (real1_f)(toCompare->runningNorm);
    }

    if (!toCompare->stateVec) {
        UpdateRunningNorm();
        return (real1_f)runningNorm;
    }

    stateVec->isReadLocked = false;
    toCompare->stateVec->isReadLocked = false;

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> partInner(new complex[numCores]());

    par_for(0U, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        partInner[cpu] += conj(stateVec->read(lcv)) * toCompare->stateVec->read(lcv);
    });

    stateVec->isReadLocked = true;
    toCompare->stateVec->isReadLocked = true;

    complex totInner = ZERO_CMPLX;
    for (unsigned i = 0U; i < numCores; ++i) {
        totInner += partInner[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(totInner));
}

void QEngineCPU::ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
{
    CHECK_ZERO_SKIP();

    Dispatch(maxQPowerOcl, [this, regMask, result, nrm] {
        const bitCapIntOcl regMaskOcl = (bitCapIntOcl)regMask;
        const bitCapIntOcl resultOcl = (bitCapIntOcl)result;
        ParallelFunc fn = [&](const bitCapIntOcl& i, const unsigned& cpu) {
            if ((i & regMaskOcl) == resultOcl) {
                stateVec->write(i, nrm * stateVec->read(i));
            } else {
                stateVec->write(i, ZERO_CMPLX);
            }
        };

        if (stateVec->is_sparse()) {
            par_for_set(CastStateVecSparse()->iterable(), fn);
        } else {
            par_for(0U, maxQPowerOcl, fn);
        }

        runningNorm = ONE_R1;
    });
}

void QEngineCPU::NormalizeState(real1_f nrm_f, real1_f norm_thresh_f, real1_f phaseArg)
{
    CHECK_ZERO_SKIP();

    if ((runningNorm == REAL1_DEFAULT_ARG) && (nrm_f == REAL1_DEFAULT_ARG)) {
        UpdateRunningNorm();
    }

    real1 nrm = (real1)nrm_f;
    real1 norm_thresh = (real1)norm_thresh_f;

    if (nrm < ZERO_R1) {
        // runningNorm can be set by OpenCL queue pop, so finish first.
        Finish();
        nrm = runningNorm;
    }
    // We might avoid the clFinish().
    if (nrm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
        return;
    }
    if ((abs(ONE_R1 - nrm) <= FP_NORM_EPSILON) && ((phaseArg * phaseArg) <= FP_NORM_EPSILON)) {
        return;
    }
    // We might have async execution of gates still happening.
    Finish();

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }
    nrm = ONE_R1 / std::sqrt((real1_s)nrm);
    complex cNrm = std::polar(nrm, (real1)phaseArg);

    if (norm_thresh <= ZERO_R1) {
        par_for(0U, maxQPowerOcl,
            [&](const bitCapIntOcl& lcv, const unsigned& cpu) { stateVec->write(lcv, cNrm * stateVec->read(lcv)); });
    } else {
        par_for(0U, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            complex amp = stateVec->read(lcv);
            if (norm(amp) < norm_thresh) {
                amp = ZERO_CMPLX;
            }
            stateVec->write(lcv, cNrm * amp);
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
        norm_thresh = (real1_f)amplitudeFloor;
    }
    runningNorm = par_norm(maxQPowerOcl, stateVec, norm_thresh);

    if (runningNorm <= FP_NORM_EPSILON) {
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
