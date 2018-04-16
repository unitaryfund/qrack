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
    bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp, Complex16 phaseFac)
    : QInterface(qBitCount),
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
    qubitCount = qBitCount;
    maxQPower = 1 << qBitCount;
    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
    stateVec.reset();
    stateVec = std::move(sv);
    std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    if (phaseFac == Complex16(-999.0, -999.0)) {
        double angle = Rand() * 2.0 * M_PI;
        stateVec[initState] = Complex16(cos(angle), sin(angle));
    } else {
        stateVec[initState] = phaseFac;
    }
}

void QEngineCPU::ResetStateVec(std::unique_ptr<Complex16[]> nStateVec)
{
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCPU::SetQuantumState(Complex16* inputState)
{
    std::copy(&(inputState[0]), &(inputState[0]) + maxQPower, &(stateVec[0]));
}

/**
 * Apply a 2x2 matrix to the state vector
 *
 * A fundamental operation used by almost all gates.
 */
void QEngineCPU::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm)
{
    Complex16 nrm = Complex16(doApplyNorm ? (1.0 / runningNorm) : 1.0, 0.0);

    par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv) {
        Complex16 qubit[2];

        qubit[0] = stateVec[lcv + offset1];
        qubit[1] = stateVec[lcv + offset2];

        Complex16 Y0 = qubit[0];
        qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
        qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));

        stateVec[lcv + offset1] = qubit[0];
        stateVec[lcv + offset2] = qubit[1];
    });

    if (doCalcNorm) {
        UpdateRunningNorm();
    } else {
        runningNorm = 1.0;
    }
}/**
 * Combine (a copy of) another QEngineCPU with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
void QEngineCPU::Cohere(QEngineCPUPtr toCopy)
{
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

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv) {
        nStateVec[lcv] = stateVec[lcv & startMask] * toCopy->stateVec[(lcv & endMask) >> qubitCount];
    });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

    ResetStateVec(std::move(nStateVec));
    UpdateRunningNorm();
}

#if 0
/**
 * Combine (copies) each QEngineCPU in the vector with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
void QEngineCPU::Cohere(std::vector<QEngineCPUPtr> toCopy)
{
    bitLenInt i;
    bitLenInt toCohereCount = toCopy.size();

    std::vector<bitLenInt> offset(toCohereCount);
    std::vector<bitCapInt> mask(toCohereCount);

    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt nQubitCount = qubitCount;
    bitCapInt nMaxQPower;

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    for (i = 0; i < toCohereCount; i++) {
        if (toCopy[i]->runningNorm != 1.0) {
            toCopy[i]->NormalizeState();
        }
        mask[i] = ((1 << toCopy[i]->GetQubitCount()) - 1) << nQubitCount;
        offset[i] = nQubitCount;
        nQubitCount += toCopy[i]->GetQubitCount();
    }

    nMaxQPower = 1 << nQubitCount;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv) {
        nStateVec[lcv] = stateVec[lcv & startMask];
        for (bitLenInt j = 0; j < toCohereCount; j++) {
            nStateVec[lcv] *= toCopy[j]->stateVec[(lcv & mask[j]) >> offset[j]];
        }
    });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

    ResetStateVec(std::move(nStateVec));
    UpdateRunningNorm();
}
#endif

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

    std::unique_ptr<double[]> partStateProb(new double[partPower]());
    std::unique_ptr<double[]> remainderStateProb(new double[remainderPower]());
    std::unique_ptr<double[]> partStateAngle(new double[partPower]());
    std::unique_ptr<double[]> remainderStateAngle(new double[remainderPower]());
    double prob, angle;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        angle = arg(stateVec[i]);
        partStateProb[(i & mask) >> start] += prob;
        partStateAngle[(i & mask) >> start] = angle;
        remainderStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
        remainderStateAngle[(i & startMask) | ((i & endMask) >> length)] = angle;
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;

    std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
    ResetStateVec(std::move(sv));

    for (i = 0; i < partPower; i++) {
        destination->stateVec[i] = sqrt(partStateProb[i]) * Complex16(cos(partStateAngle[i]), sin(partStateAngle[i]));
    }

    for (i = 0; i < remainderPower; i++) {
        stateVec[i] = sqrt(remainderStateProb[i]) * Complex16(cos(remainderStateAngle[i]), sin(remainderStateAngle[i]));
    }

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

    std::unique_ptr<double[]> partStateProb(new double[maxQPower - partPower]());
    std::unique_ptr<double[]> partStateAngle(new double[maxQPower - partPower]());
    double prob, angle;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        angle = arg(stateVec[i]);
        partStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
        partStateAngle[(i & startMask) | ((i & endMask) >> length)] = angle;
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;

    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]());
    ResetStateVec(std::move(sv));

    for (i = 0; i < maxQPower; i++) {
        stateVec[i] = sqrt(partStateProb[i]) * Complex16(cos(partStateAngle[i]), sin(partStateAngle[i]));
    }

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

/// PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
void QEngineCPU::ProbArray(double* probArray)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt lcv;
    for (lcv = 0; lcv < maxQPower; lcv++) {
        probArray[lcv] = norm(stateVec[lcv]);
    }
}

void QEngineCPU::NormalizeState()
{
    par_for(0, maxQPower, [&](const bitCapInt lcv) {
        stateVec[lcv] /= runningNorm;
        if (norm(stateVec[lcv]) < 1e-15) {
            stateVec[lcv] = Complex16(0.0, 0.0);
        }
    });
    runningNorm = 1.0;
}

void QEngineCPU::UpdateRunningNorm() { runningNorm = par_norm(maxQPower, &(stateVec[0])); }

} // namespace Qrack
