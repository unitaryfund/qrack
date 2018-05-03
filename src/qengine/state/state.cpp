//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <thread>

#include "qengine_cpu.hpp"

namespace Qrack {

/**
 * Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with
 * a shared random number generator, with a specific phase.
 *
 * \warning Overall phase is generally arbitrary and unknowable. Setting two QEngineCPU instances to the same
 * phase usually makes sense only if they are initialized at the same time.
 */
QEngineCPU::QEngineCPU(
    bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, complex phaseFac)
    : QInterface(qBitCount),
      stateVec(NULL),
      rand_distribution(0.0, 1.0)
{
    SetConcurrencyLevel(std::thread::hardware_concurrency());
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    if (rgp == NULL) {
        rand_generator = std::make_shared<std::default_random_engine>();
        randomSeed = std::time(0);
        SetRandomSeed(randomSeed);
    } else {
        rand_generator = rgp;
    }

    runningNorm = 1.0;
    SetQubitCount(qBitCount);

    stateVec = AllocStateVec(maxQPower);
    std::fill(stateVec, stateVec + maxQPower, complex(0.0, 0.0));
    if (phaseFac == complex(-999.0, -999.0)) {
        double angle = Rand() * 2.0 * M_PI;
        stateVec[initState] = complex(cos(angle), sin(angle));
    } else {
        stateVec[initState] = phaseFac;
    }
}

complex* QEngineCPU::GetState()
{
    return stateVec;
}

void QEngineCPU::CopyState(QInterfacePtr orig)
{
    /* Set the size and reset the stateVec to the correct size. */
    SetQubitCount(orig->GetQubitCount());
    ResetStateVec(AllocStateVec(maxQPower));

    QEngineCPUPtr src = std::dynamic_pointer_cast<QEngineCPU>(orig);
    std::copy(src->GetState(), src->GetState() + (1 << (src->GetQubitCount())), stateVec);
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
}

/**
 * Apply a 2x2 matrix to the state vector
 *
 * A fundamental operation used by almost all gates.
 */
void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    int numCores = GetConcurrencyLevel();
#if ENABLE_AVX
    complex2 nrm = complex2((bitCount == 1) ? (1.0 / runningNorm) : 1.0, 0.0, (bitCount == 1) ? (1.0 / runningNorm) : 1.0, 0.0);    
    ComplexUnion mtrxCol1;
    ComplexUnion mtrxCol2;
    mtrxCol1.cmplx[0] = mtrx[0];
    mtrxCol1.cmplx[1] = mtrx[2];
    mtrxCol2.cmplx[0] = mtrx[1];
    mtrxCol2.cmplx[1] = mtrx[3];
#else
    complex nrm = complex((bitCount == 1) ? (1.0 / runningNorm) : 1.0, 0.0);
#endif

    if (doCalcNorm && (bitCount == 1)) {
        double* rngNrm = new double[numCores]; 
        std::fill(rngNrm, rngNrm + numCores, 0.0);
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
#if ENABLE_AVX
            ComplexUnion qubit(stateVec[lcv + offset1], stateVec[lcv + offset2]);

            qubit.cmplx2 = nrm * ((mtrxCol1.cmplx2 * dupeLo(qubit.cmplx2)) + (mtrxCol2.cmplx2 * dupeHi(qubit.cmplx2)));
            rngNrm[cpu] += norm(qubit.cmplx[0]) + norm(qubit.cmplx[1]);

            stateVec[lcv + offset1] = qubit.cmplx[0];
            stateVec[lcv + offset2] = qubit.cmplx[1];
#else
            complex qubit[2];

            qubit[0] = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            complex Y0 = qubit[0];
            qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
            qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));
            rngNrm[cpu] += norm(qubit[0]) + norm(qubit[1]);

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
#endif
        });
        runningNorm = 0.0;
        for (int i = 0; i < numCores; i++) {
            runningNorm += rngNrm[i];
        }
        delete[] rngNrm;
        runningNorm = sqrt(runningNorm);
    }
    else {
        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv, const int cpu) {
#if ENABLE_AVX
            ComplexUnion qubit;
            qubit.cmplx[0] = stateVec[lcv + offset1];
            qubit.cmplx[1] = stateVec[lcv + offset2];

            qubit.cmplx2 = nrm * ((mtrxCol1.cmplx2 * dupeLo(qubit.cmplx2)) + (mtrxCol2.cmplx2 * dupeHi(qubit.cmplx2)));

            stateVec[lcv + offset1] = qubit.cmplx[0];
            stateVec[lcv + offset2] = qubit.cmplx[1];
#else
            complex qubit[2];

            qubit[0] = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            complex Y0 = qubit[0];
            qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
            qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
#endif
        });
        if (doCalcNorm) {
            UpdateRunningNorm();
        }
        else {
            runningNorm = 1.0;
        }
    }
}

/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
bitLenInt QEngineCPU::Cohere(QEngineCPUPtr toCopy)
{
    bitLenInt result = qubitCount;

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    if (toCopy->runningNorm != 1.0) {
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
    UpdateRunningNorm();

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

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    for (i = 0; i < toCohereCount; i++) {
        QEngineCPUPtr src = std::dynamic_pointer_cast<Qrack::QEngineCPU>(toCopy[i]);
        if (src->runningNorm != 1.0) {
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
    UpdateRunningNorm();

    return ret;
}

/**
 * Minimally decohere a set of contigious bits from the full coherent unit. The
 * length of this coherent unit is reduced by the length of bits decohered, and
 * the bits removed are output in the destination QEngineCPU pointer. The
 * destination object must be initialized to the correct number of bits, in 0
 * permutation state.
 */
void QEngineCPU::Decohere(bitLenInt start, bitLenInt length, QEngineCPUPtr destination)
{
    if (length == 0) {
        return;
    }

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);
    bitCapInt mask = (partPower - 1) << start;
    bitCapInt startMask = (1 << start) - 1;
    bitCapInt endMask = (maxQPower - 1) ^ (mask | startMask);
    bitCapInt i;

    double* partStateProb = new double[partPower]();
    double* remainderStateProb = new double[remainderPower]();
    double* partStateAngle = new double[partPower];
    double* remainderStateAngle = new double[remainderPower];
    double prob, angle;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        angle = arg(stateVec[i]);
        partStateProb[(i & mask) >> start] += prob;
        partStateAngle[(i & mask) >> start] = angle;
        remainderStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
        remainderStateAngle[(i & startMask) | ((i & endMask) >> length)] = angle;
    }

    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    ResetStateVec(AllocStateVec(maxQPower));

    for (i = 0; i < partPower; i++) {
        destination->stateVec[i] = sqrt(partStateProb[i]) * complex(cos(partStateAngle[i]), sin(partStateAngle[i]));
    }

    delete []partStateProb;
    delete []partStateAngle;

    for (i = 0; i < remainderPower; i++) {
        stateVec[i] = sqrt(remainderStateProb[i]) * complex(cos(remainderStateAngle[i]), sin(remainderStateAngle[i]));
    }

    delete []remainderStateProb;
    delete []remainderStateAngle;

    UpdateRunningNorm();
    destination->UpdateRunningNorm();
}

void QEngineCPU::Dispose(bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt mask = (partPower - 1) << start;
    bitCapInt startMask = (1 << start) - 1;
    bitCapInt endMask = (maxQPower - 1) ^ (mask | startMask);
    bitCapInt i;

    /* Disposing of the entire object. */
    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);       // Leave as a single bit for safety.
        ResetStateVec(AllocStateVec(maxQPower));

        return;
    }


    double *partStateProb = new double[1<<(qubitCount - length)]();
    double *partStateAngle = new double[1<<(qubitCount - length)];
    double prob, angle;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        angle = arg(stateVec[i]);
        partStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
        partStateAngle[(i & startMask) | ((i & endMask) >> length)] = angle;
    }

    SetQubitCount(qubitCount - length);

    ResetStateVec(AllocStateVec(maxQPower));

    for (i = 0; i < maxQPower; i++) {
        stateVec[i] = sqrt(partStateProb[i]) * complex(cos(partStateAngle[i]), sin(partStateAngle[i]));
    }

    delete []partStateProb;
    delete []partStateAngle;

    UpdateRunningNorm();
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
double QEngineCPU::Prob(bitLenInt qubit)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt qPower = 1 << qubit;
    double oneChance = 0;
    bitCapInt lcv;

    for (lcv = 0; lcv < maxQPower; lcv++) {
        if ((lcv & qPower) == qPower) {
            oneChance += norm(stateVec[lcv]);
        }
    }

    return oneChance;
}

/// PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
double QEngineCPU::ProbAll(bitCapInt fullRegister)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    return norm(stateVec[fullRegister]);
}

void QEngineCPU::NormalizeState()
{
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec[lcv] /= runningNorm;
        if (norm(stateVec[lcv]) < 1e-15) {
            stateVec[lcv] = complex(0.0, 0.0);
        }
    });
    runningNorm = 1.0;
}

void QEngineCPU::UpdateRunningNorm() { runningNorm = par_norm(maxQPower, stateVec); }

complex* QEngineCPU::AllocStateVec(bitCapInt elemCount)
{
    // elemCount is always a power of two, but might be smaller than ALIGN_SIZE
    return (complex*)aligned_alloc(ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
}

} // namespace Qrack
