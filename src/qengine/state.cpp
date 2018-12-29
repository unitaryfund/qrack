//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <thread>

#include <iostream>

#include "qengine_cpu.hpp"

#if ENABLE_COMPLEX_X2
#if ENABLE_COMPLEX8
#include "common/complex8x2simd.hpp"
#define complex2 Complex8x2Simd
#else
#include "common/complex16x2simd.hpp"
#define complex2 Complex16x2Simd
#endif
#endif

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
QEngineCPU::QEngineCPU(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp,
    complex phaseFac, bool doNorm, bool randomGlobalPhase, bool useHostMem)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem)
    , stateVec(NULL)
{
    SetConcurrencyLevel(std::thread::hardware_concurrency());
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    runningNorm = ONE_R1;
    SetQubitCount(qBitCount);

    stateVec = AllocStateVec(maxQPower);
    std::fill(stateVec, stateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    if (phaseFac == complex(-999.0, -999.0)) {
        complex phase;
        if (randGlobalPhase) {
            real1 angle = Rand() * 2.0 * PI_R1;
            phase = complex(cos(angle), sin(angle));
        } else {
            phase = complex(ONE_R1, ZERO_R1);
        }
        stateVec[initState] = phase;
    } else {
        stateVec[initState] = phaseFac;
    }
}

QEngineCPU::QEngineCPU(QEngineCPUPtr toCopy)
    : QEngine(
          toCopy->qubitCount, toCopy->rand_generator, toCopy->doNormalize, toCopy->randGlobalPhase, toCopy->useHostRam)
    , stateVec(NULL)
{
    SetConcurrencyLevel(std::thread::hardware_concurrency());
    stateVec = AllocStateVec(maxQPower);
    CopyState(toCopy);
}

complex* QEngineCPU::GetStateVector() { return stateVec; }

complex QEngineCPU::GetAmplitude(bitCapInt perm)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }
    return stateVec[perm];
}

void QEngineCPU::SetPermutation(bitCapInt perm, complex phaseFac)
{
    std::fill(stateVec, stateVec + maxQPower, complex(ZERO_R1, ZERO_R1));

    if (phaseFac == complex(-999.0, -999.0)) {
        complex phase;
        if (randGlobalPhase) {
            real1 angle = Rand() * 2.0 * PI_R1;
            phase = complex(cos(angle), sin(angle));
        } else {
            phase = complex(ONE_R1, ZERO_R1);
        }
        stateVec[perm] = phase;
    } else {
        real1 nrm = abs(phaseFac);
        stateVec[perm] = phaseFac / nrm;
    }

    runningNorm = ONE_R1;
}

void QEngineCPU::CopyState(QInterfacePtr orig)
{
    /* Set the size and reset the stateVec to the correct size. */
    SetQubitCount(orig->GetQubitCount());
    ResetStateVec(AllocStateVec(maxQPower));

    QEngineCPUPtr src = std::dynamic_pointer_cast<QEngineCPU>(orig);
    std::copy(src->stateVec, src->stateVec + src->maxQPower, stateVec);
}

void QEngineCPU::ResetStateVec(complex* nStateVec)
{
    free(stateVec);
    stateVec = nStateVec;
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCPU::SetQuantumState(complex* inputState)
{
    std::copy(inputState, inputState + maxQPower, stateVec);
    runningNorm = ONE_R1;
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineCPU::GetQuantumState(complex* outputState)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    std::copy(stateVec, stateVec + maxQPower, outputState);
}

    /**
     * Apply a 2x2 matrix to the state vector
     *
     * A fundamental operation used by almost all gates.
     */

#if ENABLE_COMPLEX_X2

#if ENABLE_COMPLEX8
union ComplexUnion {
    complex2 cmplx2;
    float comp[4];

    inline ComplexUnion(){};
    inline ComplexUnion(const complex& cmplx0, const complex& cmplx1)
    {
        cmplx2 = complex2(real(cmplx0), imag(cmplx0), real(cmplx1), imag(cmplx1));
    }
};
#else
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
#endif

void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    int numCores = GetConcurrencyLevel();
    real1 nrm = doNormalize ? (ONE_R1 / std::sqrt(runningNorm)) : ONE_R1;
    ComplexUnion mtrxCol1(mtrx[0], mtrx[2]);
    ComplexUnion mtrxCol2(mtrx[1], mtrx[3]);

    if (doCalcNorm && (bitCount == 1)) {
        real1* rngNrm = new real1[numCores];
        std::fill(rngNrm, rngNrm + numCores, ZERO_R1);
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
            ComplexUnion qubit(stateVec[lcv + offset1], stateVec[lcv + offset2]);

            qubit.cmplx2 = matrixMul(nrm, mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);
#if ENABLE_COMPLEX8
            stateVec[lcv + offset1] = complex(qubit.comp[0], qubit.comp[1]);
            stateVec[lcv + offset2] = complex(qubit.comp[2], qubit.comp[3]);
            rngNrm[cpu] += norm(qubit.cmplx2);
#else
            stateVec[lcv + offset1] = qubit.cmplx[0];
            stateVec[lcv + offset2] = qubit.cmplx[1];
            rngNrm[cpu] += norm(qubit.cmplx[0]) + norm(qubit.cmplx[1]);
#endif
        });
        runningNorm = ZERO_R1;
        for (int i = 0; i < numCores; i++) {
            runningNorm += rngNrm[i];
        }
        delete[] rngNrm;
    } else {
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
            ComplexUnion qubit(stateVec[lcv + offset1], stateVec[lcv + offset2]);

            qubit.cmplx2 = matrixMul(mtrxCol1.cmplx2, mtrxCol2.cmplx2, qubit.cmplx2);
#if ENABLE_COMPLEX8
            stateVec[lcv + offset1] = complex(qubit.comp[0], qubit.comp[1]);
            stateVec[lcv + offset2] = complex(qubit.comp[2], qubit.comp[3]);
#else
            stateVec[lcv + offset1] = qubit.cmplx[0];
            stateVec[lcv + offset2] = qubit.cmplx[1];
#endif
        });
        if (doNormalize && doCalcNorm) {
            UpdateRunningNorm();
        }
    }
}
#else
void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    int numCores = GetConcurrencyLevel();
    real1 nrm = doNormalize ? (ONE_R1 / std::sqrt(runningNorm)) : ONE_R1;

    if (doCalcNorm && (bitCount == 1)) {
        real1* rngNrm = new real1[numCores];
        std::fill(rngNrm, rngNrm + numCores, ZERO_R1);
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
            complex qubit[2];

            complex Y0 = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
            qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));
            rngNrm[cpu] += norm(qubit[0]) + norm(qubit[1]);

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
        });
        runningNorm = ZERO_R1;
        for (int i = 0; i < numCores; i++) {
            runningNorm += rngNrm[i];
        }
        delete[] rngNrm;
    } else {
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
            complex qubit[2];

            complex Y0 = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            qubit[0] = (mtrx[0] * Y0) + (mtrx[1] * qubit[1]);
            qubit[1] = (mtrx[2] * Y0) + (mtrx[3] * qubit[1]);

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
        });
        if (doNormalize && doCalcNorm) {
            UpdateRunningNorm();
        }
    }
}
#endif

/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
bitLenInt QEngineCPU::Cohere(QEngineCPUPtr toCopy)
{
    bitLenInt result = qubitCount;

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    if ((toCopy->doNormalize) && (toCopy->runningNorm != ONE_R1)) {
        toCopy->NormalizeState();
    }

    bitCapInt nQubitCount = qubitCount + toCopy->qubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy->qubitCount)) - 1) << qubitCount;

    complex* nStateVec = AllocStateVec(nMaxQPower);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
        nStateVec[lcv] = stateVec[lcv & startMask] * toCopy->stateVec[(lcv & endMask) >> qubitCount];
    });

    SetQubitCount(nQubitCount);

    ResetStateVec(nStateVec);

    return result;
}

/**
 * Combine (copies) each QEngineCPU in the vector with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 *
 * Returns a mapping of the index into the new QEngine that each old one was mapped to.
 */
std::map<QInterfacePtr, bitLenInt> QEngineCPU::Cohere(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    bitLenInt i;
    bitLenInt toCohereCount = toCopy.size();

    std::vector<bitLenInt> offset(toCohereCount);
    std::vector<bitCapInt> mask(toCohereCount);

    bitCapInt startMask = maxQPower - 1;
    bitCapInt nQubitCount = qubitCount;
    bitCapInt nMaxQPower;

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    for (i = 0; i < toCohereCount; i++) {
        QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[i]);
        if ((src->doNormalize) && (src->runningNorm != ONE_R1)) {
            src->NormalizeState();
        }
        mask[i] = ((1 << src->GetQubitCount()) - 1) << nQubitCount;
        offset[i] = nQubitCount;
        ret[toCopy[i]] = nQubitCount;
        nQubitCount += src->GetQubitCount();
    }

    nMaxQPower = 1 << nQubitCount;

    complex* nStateVec = AllocStateVec(nMaxQPower);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
        nStateVec[lcv] = stateVec[lcv & startMask];

        for (bitLenInt j = 0; j < toCohereCount; j++) {
            QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[j]);
            nStateVec[lcv] *= src->stateVec[(lcv & mask[j]) >> offset[j]];
        }
    });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

    ResetStateVec(nStateVec);

    return ret;
}

/**
 * Minimally decohere a set of contigious bits from the full coherent unit. The
 * length of this coherent unit is reduced by the length of bits decohered, and
 * the bits removed are output in the destination QEngineCPU pointer. The
 * destination object must be initialized to the correct number of bits, in 0
 * permutation state.
 */
void QEngineCPU::DecohereDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr destination)
{
    if (length == 0) {
        return;
    }

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);

    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower];

    par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt j, k, l;
        j = lcv % (1 << start);
        j = j | ((lcv ^ j) << length);

        real1 angle = -2 * M_PI;
        real1 nrm;

        for (k = 0; k < partPower; k++) {
            l = j | (k << start);

            nrm = norm(stateVec[l]);
            remainderStateProb[lcv] += nrm;

            if ((angle < -M_PI) && (nrm > min_norm)) {
                angle = arg(stateVec[l]);
            }
        }
        remainderStateAngle[lcv] = angle;
    });

    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    if (destination != nullptr) {
        real1* partStateProb = new real1[partPower]();
        real1* partStateAngle = new real1[partPower];

        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt j, k, l;
            j = lcv << start;

            real1 angle = -2 * M_PI;
            real1 nrm;

            for (k = 0; k < remainderPower; k++) {
                l = k % (1 << start);
                l = l | ((k ^ l) << length);
                l = j | l;

                nrm = norm(stateVec[l]);
                partStateProb[lcv] += nrm;

                if ((angle < -M_PI) && (nrm > min_norm)) {
                    angle = arg(stateVec[l]);
                }
            }
            partStateAngle[lcv] = angle;
        });

        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            destination->stateVec[lcv] =
                (real1)(std::sqrt(partStateProb[lcv])) * complex(cos(partStateAngle[lcv]), sin(partStateAngle[lcv]));
        });

        delete[] partStateProb;
        delete[] partStateAngle;
    }

    ResetStateVec(AllocStateVec(maxQPower));

    par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec[lcv] = (real1)(std::sqrt(remainderStateProb[lcv])) *
            complex(cos(remainderStateAngle[lcv]), sin(remainderStateAngle[lcv]));
    });

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
}

void QEngineCPU::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    DecohereDispose(start, length, std::dynamic_pointer_cast<QEngineCPU>(destination));
}

void QEngineCPU::Dispose(bitLenInt start, bitLenInt length) { DecohereDispose(start, length, (QEngineCPUPtr) nullptr); }

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1 QEngineCPU::Prob(bitLenInt qubit)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt qPower = (1 << qubit);
    bitCapInt qMask = qPower - 1;
    real1 oneChance = 0;

    int numCores = GetConcurrencyLevel();
    real1* oneChanceBuff = new real1[numCores]();

    par_for(0, maxQPower >> 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt i = lcv & qMask;
        i |= ((lcv ^ i) << 1) | qPower;
        oneChanceBuff[cpu] += norm(stateVec[i]);
    });

    for (int i = 0; i < numCores; i++) {
        oneChance += oneChanceBuff[i];
    }

    delete[] oneChanceBuff;

    return oneChance;
}

/// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
real1 QEngineCPU::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    return norm(stateVec[fullRegister]);
}

// Returns probability of permutation of the register
real1 QEngineCPU::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    int num_threads = GetConcurrencyLevel();
    real1* probs = new real1[num_threads]();

    bitCapInt perm = permutation << start;

    par_for_skip(0, maxQPower, (1U << start), length,
        [&](const bitCapInt lcv, const int cpu) { probs[cpu] += norm(stateVec[lcv | perm]); });

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    delete[] probs;

    return prob;
}

// Returns probability of permutation of the mask
real1 QEngineCPU::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapInt* skipPowers = new bitCapInt[skipPowersVec.size()];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    int num_threads = GetConcurrencyLevel();
    real1* probs = new real1[num_threads]();

    par_for_mask(0, maxQPower, skipPowers, skipPowersVec.size(),
        [&](const bitCapInt lcv, const int cpu) { probs[cpu] += norm(stateVec[lcv | permutation]); });

    delete[] skipPowers;

    real1 prob = ZERO_R1;
    for (int thrd = 0; thrd < num_threads; thrd++) {
        prob += probs[thrd];
    }

    delete[] probs;

    return prob;
}

bool QEngineCPU::ApproxCompare(QEngineCPUPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    // Make sure both engines are normalized
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }
    if (toCompare->doNormalize && (toCompare->runningNorm != ONE_R1)) {
        toCompare->NormalizeState();
    }

    int numCores = GetConcurrencyLevel();
    real1* partError = new real1[numCores]();

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        real1 elemError = norm(stateVec[lcv] - toCompare->stateVec[lcv]);
        partError[cpu] += elemError;
    });

    real1 totError = ZERO_R1;
    for (int i = 0; i < numCores; i++) {
        totError += partError[i];
    }

    delete[] partError;

    return totError < (maxQPower * min_norm);
}

void QEngineCPU::NormalizeState(real1 nrm)
{
    if (nrm < ZERO_R1) {
        nrm = runningNorm;
    }
    if ((nrm <= ZERO_R1) || (nrm == ONE_R1)) {
        return;
    }

    nrm = std::sqrt(nrm);

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec[lcv] /= nrm;
        //"min_norm" is defined in qinterface.hpp
        if (norm(stateVec[lcv]) < min_norm) {
            stateVec[lcv] = complex(ZERO_R1, ZERO_R1);
        }
    });

    runningNorm = ONE_R1;
}

void QEngineCPU::UpdateRunningNorm() { runningNorm = par_norm(maxQPower, stateVec); }

complex* QEngineCPU::AllocStateVec(bitCapInt elemCount, bool ovrride)
{
// elemCount is always a power of two, but might be smaller than ALIGN_SIZE
#ifdef __APPLE__
    void* toRet;
    posix_memalign(
        &toRet, ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
    return (complex*)toRet;
#else
    return (complex*)aligned_alloc(
        ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
#endif
}
} // namespace Qrack
