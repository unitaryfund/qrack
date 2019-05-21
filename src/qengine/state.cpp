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
QEngineCPU::QEngineCPU(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int deviceID, bool useHardwareRNG)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, true, useHardwareRNG)
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
    : QEngine(toCopy->qubitCount, toCopy->rand_generator, toCopy->doNormalize, toCopy->randGlobalPhase, true,
          toCopy->hardware_rand_generator != NULL)
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
    FreeStateVec();
    stateVec = nStateVec;
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCPU::SetQuantumState(const complex* inputState)
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

/// Get all probabilities, in unsigned int permutation basis
void QEngineCPU::GetProbs(real1* outputProbs)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    std::transform(stateVec, stateVec + maxQPower, outputProbs, normHelper);
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

void QEngineCPU::UniformlyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
{
    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(mtrxs, true, qubitIndex);
        return;
    }

    bitCapInt targetPower = 1 << qubitIndex;

    real1 nrm = ONE_R1 / std::sqrt(runningNorm);

    bitCapInt* qPowers = new bitCapInt[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
    }

    int numCores = GetConcurrencyLevel();
    real1* rngNrm = new real1[numCores];
    std::fill(rngNrm, rngNrm + numCores, ZERO_R1);

    par_for_skip(0, maxQPower, targetPower, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt offset = 0;
        for (bitLenInt j = 0; j < controlLen; j++) {
            if (lcv & qPowers[j]) {
                offset |= 1 << j;
            }
        }

        // Offset is permutation * 4, for the components of 2x2 matrices. (Note that this sacrifices 2 qubits of
        // capacity for the unsigned bitCapInt.)
        offset *= 4;

        complex qubit[2];

        complex Y0 = stateVec[lcv];
        qubit[1] = stateVec[lcv | targetPower];

        qubit[0] = nrm * ((mtrxs[0 + offset] * Y0) + (mtrxs[1 + offset] * qubit[1]));
        qubit[1] = nrm * ((mtrxs[2 + offset] * Y0) + (mtrxs[3 + offset] * qubit[1]));

        rngNrm[cpu] += norm(qubit[0]) + norm(qubit[1]);

        stateVec[lcv] = qubit[0];
        stateVec[lcv | targetPower] = qubit[1];
    });

    runningNorm = ZERO_R1;
    for (int i = 0; i < numCores; i++) {
        runningNorm += rngNrm[i];
    }

    delete[] rngNrm;
    delete[] qPowers;
}

/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old unit that was added.
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy)
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
 * Combine (a copy of) another QEngineCPU with this one, inserted at the "start" index. (If the programmer doesn't want
 * to "cheat," it is left up to them to delete the old unit that was added.
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy, bitLenInt start)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    if ((toCopy->doNormalize) && (toCopy->runningNorm != ONE_R1)) {
        toCopy->NormalizeState();
    }

    bitCapInt oQubitCount = toCopy->qubitCount;
    bitCapInt nQubitCount = qubitCount + oQubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << start) - 1;
    bitCapInt midMask = ((1 << oQubitCount) - 1) << start;
    bitCapInt endMask = ((1 << (qubitCount + oQubitCount)) - 1) & ~(startMask | midMask);

    complex* nStateVec = AllocStateVec(nMaxQPower);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
        nStateVec[lcv] =
            stateVec[(lcv & startMask) | ((lcv & endMask) >> oQubitCount)] * toCopy->stateVec[(lcv & midMask) >> start];
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

    bitCapInt startMask = maxQPower - 1;
    bitCapInt nQubitCount = qubitCount;
    bitCapInt nMaxQPower;

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    for (i = 0; i < toComposeCount; i++) {
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

        for (bitLenInt j = 0; j < toComposeCount; j++) {
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

    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);

    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower]();
    real1* partStateAngle = new real1[partPower]();
    real1* partStateProb = new real1[partPower]();

    par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt j, k, l;
        j = lcv & ((1U << start) - 1);
        j |= (lcv ^ j) << length;

        real1 firstAngle = -16 * M_PI;
        real1 currentAngle;
        real1 nrm;

        for (k = 0; k < partPower; k++) {
            l = j | (k << start);

            nrm = norm(stateVec[l]);
            remainderStateProb[lcv] += nrm;

            if (nrm > min_norm) {
                currentAngle = arg(stateVec[l]);
                if (firstAngle < (-8 * M_PI)) {
                    firstAngle = currentAngle;
                }
                partStateAngle[k] = currentAngle - firstAngle;
            }
        }
    });

    par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt j, k, l;
        j = lcv << start;

        real1 firstAngle = -16 * M_PI;
        real1 currentAngle;
        real1 nrm;

        for (k = 0; k < remainderPower; k++) {
            l = k & ((1 << start) - 1);
            l |= (k ^ l) << length;
            l = j | l;

            nrm = norm(stateVec[l]);
            partStateProb[lcv] += nrm;

            if (nrm > min_norm) {
                currentAngle = arg(stateVec[l]);
                if (firstAngle < (-8 * M_PI)) {
                    firstAngle = currentAngle;
                }
                remainderStateAngle[k] = currentAngle - firstAngle;
            }
        }
    });

    bitCapInt i, j, k;
    i = 0;
    j = 0;
    k = 0;
    while (remainderStateProb[i] < min_norm) {
        i++;
    }
    k = i & ((1U << start) - 1);
    k |= (i ^ k) << length;

    while (partStateProb[j] < min_norm) {
        j++;
    }
    k |= j << start;

    real1 refAngle = arg(stateVec[k]);
    real1 angleOffset = refAngle - (remainderStateAngle[i] + partStateAngle[j]);

    for (bitCapInt l = 0; l < partPower; l++) {
        partStateAngle[l] += angleOffset;
    }

    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    if (destination != nullptr) {
        par_for(0, partPower, [&](const bitCapInt lcv, const int cpu) {
            destination->stateVec[lcv] =
                (real1)(std::sqrt(partStateProb[lcv])) * complex(cos(partStateAngle[lcv]), sin(partStateAngle[lcv]));
        });
    }

    ResetStateVec(AllocStateVec(maxQPower));

    par_for(0, remainderPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec[lcv] = (real1)(std::sqrt(remainderStateProb[lcv])) *
            complex(cos(remainderStateAngle[lcv]), sin(remainderStateAngle[lcv]));
    });

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
    delete[] partStateProb;
    delete[] partStateAngle;
}

void QEngineCPU::Decompose(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    DecomposeDispose(start, length, std::dynamic_pointer_cast<QEngineCPU>(destination));
}

void QEngineCPU::Dispose(bitLenInt start, bitLenInt length)
{
    DecomposeDispose(start, length, (QEngineCPUPtr) nullptr);
}

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

    return clampProb(oneChance);
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

    return clampProb(prob);
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

    return clampProb(prob);
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

    return totError < approxcompare_error;
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
        stateVec[lcv] *= nrm;
    });

    runningNorm = ONE_R1;
}

void QEngineCPU::UpdateRunningNorm() { runningNorm = par_norm(maxQPower, stateVec); }

complex* QEngineCPU::AllocStateVec(bitCapInt elemCount, bool ovrride)
{
// elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE,
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
    return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (complex*)_aligned_malloc(
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount,
        QRACK_ALIGN_SIZE);
#else
    return (complex*)aligned_alloc(QRACK_ALIGN_SIZE,
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
#endif
}
} // namespace Qrack
