//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and phase, to
// leverage what advantages classical emulation of qubits
// can have.
//
// See the register-wise "CoherentUnit::X" gate implementation for inline
// documentation on the general algorithm by which basically all register-wise gates
// operate.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <atomic>
#include <bitset>
#include <future>
#include <iostream>
#include <thread>

#include "qregister.hpp"

namespace Qrack {

template <class BidirectionalIterator>
void reverse(BidirectionalIterator first, BidirectionalIterator last, bitCapInt stride)
{
    while ((first < last) && (first < (last - stride))) {
        last -= stride;
        std::iter_swap(first, last);
        first += stride;
    }
}

template <class BidirectionalIterator>
void rotate(BidirectionalIterator first, BidirectionalIterator middle, BidirectionalIterator last, bitCapInt stride)
{
    reverse(first, middle, stride);
    reverse(middle, last, stride);
    reverse(first, last, stride);
}

const Complex16 CMPLX_I_2X2[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(1.0, 0.0) };

/** Protected constructor for SeparatedUnit */
CoherentUnit::CoherentUnit()
    : rand_distribution(0.0, 1.0)
{
    // This method body left intentionally empty
    randomSeed = std::time(0);
}

/**
 * Initialize a coherent unit with qBitCount number pf bits, to initState
 * unsigned integer permutation state.  The `initState` parameter is,
 * effectively, the initial pattern of |0> and |1>'s that the qubits should be
 * initialized to.
 *
 * For example, in a two qubit system, there are the following values:
 *
 *    |00>
 *    |01>
 *    |10>
 *    |11>
 *
 * If the desired initial state is |10>, then the index value of 2
 * will be passed in to initState.  The constructor will then,
 * using a random \f$ \theta \f$, initialize that state to
 * Complex16Simd(\f$cos(\theta)\f$, \f$sin(\theta)\f$).  It's worth
 * noting that this is still a unit vector:
 *
 * \f$
 *   cos(\theta)^2 + sin(\theta)^2 = 1
 * \f$
 *
 * Broadly speaking, a non-random \f$\theta\f$ could be used, but doing so
 * replicates the unknowable initial phase of a physical QM system, and has
 * impacts on subsequent operations accordingly.
 */
CoherentUnit::CoherentUnit(bitLenInt qBitCount, bitCapInt initState)
    : CoherentUnit(qBitCount, initState, Complex16(-999.0, -999.0), NULL)
{
}

/** Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with a
 * shared random number generator */
CoherentUnit::CoherentUnit(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, initState, Complex16(-999.0, -999.0), rgp)
{
}

/**
 * Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with
 * a shared random number generator, with a specific phase.
 *
 * \warning Overall phase is generally arbitrary and unknowable. Setting two CoherentUnit instances to the same
 * phase usually makes sense only if they are initialized at the same time.
 */
CoherentUnit::CoherentUnit(
    bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp)
    : gateQueue(qBitCount)
    , isQueued(qBitCount)
    , rand_distribution(0.0, 1.0)
    , numCores(std::thread::hardware_concurrency())
{
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    if (rgp == NULL) {
        rand_generator_ptr = std::make_shared<std::default_random_engine>();
        randomSeed = std::time(0);
        SetRandomSeed(randomSeed);
    } else {
        rand_generator_ptr = rgp;
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

    for (bitLenInt i = 0; i < qBitCount; i++) {
        isQueued[i] = false;
        gateQueue[i] = std::move(std::unique_ptr<Complex16[]>(new Complex16[4]));
        gateQueue[i][0] = Complex16(1.0, 0.0);
        gateQueue[i][1] = Complex16(0.0, 0.0);
        gateQueue[i][2] = Complex16(0.0, 0.0);
        gateQueue[i][3] = Complex16(1.0, 0.0);
    }
}

/**
 * Initialize a coherent unit with qBitCount number of bits, to initState
 * unsigned integer permutation state, with a specific phase.
 *
 * \warning Overall phase is generally arbitrary and unknowable. Setting two CoherentUnit instances to the same
 * phase usually makes sense only if they are initialized at the same time.
 */
CoherentUnit::CoherentUnit(bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac)
    : CoherentUnit(qBitCount, initState, phaseFac, NULL)
{
}

/** Initialize a coherent unit with qBitCount number of bits, all to |0> state. */
CoherentUnit::CoherentUnit(bitLenInt qBitCount)
    : CoherentUnit(qBitCount, 0, Complex16(-999.0, -999.0), NULL)
{
}

CoherentUnit::CoherentUnit(bitLenInt qBitCount, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, 0, Complex16(-999.0, -999.0), rgp)
{
}

/** Initialize a coherent unit with qBitCount number of bits, all to |0> state. */
CoherentUnit::CoherentUnit(bitLenInt qBitCount, Complex16 phaseFac)
    : CoherentUnit(qBitCount, 0, phaseFac, NULL)
{
}

/** Initialize a coherent unit with qBitCount number of bits, all to |0> state. */
CoherentUnit::CoherentUnit(bitLenInt qBitCount, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, 0, phaseFac, NULL)
{
}

/// PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
CoherentUnit::CoherentUnit(const CoherentUnit& pqs)
    : gateQueue(pqs.qubitCount)
    , isQueued(pqs.qubitCount)
    , rand_distribution(0.0, 1.0)
    , numCores(std::thread::hardware_concurrency())
{
    rand_generator_ptr = pqs.rand_generator_ptr;
    randomSeed = std::time(0);
    SetRandomSeed(randomSeed);

    runningNorm = pqs.runningNorm;
    qubitCount = pqs.qubitCount;
    maxQPower = pqs.maxQPower;

    std::copy((pqs.isQueued).begin(), (pqs.isQueued).begin() + pqs.qubitCount, isQueued.begin());
    for (bitLenInt i = 0; i < pqs.qubitCount; i++) {
        gateQueue[i] = std::move(std::unique_ptr<Complex16[]>(new Complex16[4]));
        std::copy(&(pqs.gateQueue[i][0]), &(pqs.gateQueue[i][0]) + 4, &(gateQueue[i][0]));
    }

    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
    stateVec.reset();
    stateVec = std::move(sv);
    SetQuantumState(&pqs.stateVec[0]);
}

/// Set the random seed (primarily used for testing)
void CoherentUnit::SetRandomSeed(uint32_t seed)
{
    randomSeed = seed;
    rand_generator_ptr->seed(seed);
}

/// PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
void CoherentUnit::CloneRawState(Complex16* output)
{
    FlushQueue(0, qubitCount);
    if (runningNorm != 1.0) {
        NormalizeState();
    }
    std::copy(&(stateVec[0]), &(stateVec[0]) + maxQPower, &(output[0]));
}

/// Generate a random double from 0 to 1
double CoherentUnit::Rand() { return rand_distribution(*rand_generator_ptr); }

void CoherentUnit::ResetStateVec(std::unique_ptr<Complex16[]> nStateVec)
{
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
void CoherentUnit::SetPermutation(bitCapInt perm) { SetReg(0, qubitCount, perm); }

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void CoherentUnit::SetQuantumState(Complex16* inputState)
{
    ResetQueue(0, qubitCount);
    std::copy(&(inputState[0]), &(inputState[0]) + maxQPower, &(stateVec[0]));
}

/**
 * Combine (a copy of) another CoherentUnit with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
void CoherentUnit::Cohere(CoherentUnit& toCopy)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    if (toCopy.runningNorm != 1.0) {
        toCopy.NormalizeState();
    }

    bitLenInt i;
    bitCapInt nQubitCount = qubitCount + toCopy.qubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy.qubitCount)) - 1) << qubitCount;

    std::vector<std::unique_ptr<Complex16[]>> nGateQueue(nQubitCount);
    std::vector<bool> nIsQueued(nQubitCount);
    std::copy(isQueued.begin(), isQueued.begin() + qubitCount, nIsQueued.begin());
    for (i = 0; i < qubitCount; i++) {
        nGateQueue[i] = std::move(gateQueue[i]);
    }
    for (i = 0; i < toCopy.qubitCount; i++) {
        nIsQueued[i + qubitCount] = toCopy.isQueued[i];
        nGateQueue[i + qubitCount] = std::move(std::unique_ptr<Complex16[]>(new Complex16[4]));
        std::copy(&(toCopy.gateQueue[i][0]), &(toCopy.gateQueue[i][0]) + 4, &(nGateQueue[i + qubitCount][0]));
    }
    gateQueue = std::move(nGateQueue);
    isQueued = nIsQueued;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
        nStateVec[lcv] = stateVec[lcv & startMask] * toCopy.stateVec[(lcv & endMask) >> qubitCount];
    });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

    ResetStateVec(std::move(nStateVec));
    UpdateRunningNorm();
}

/**
 * Combine (copies) each CoherentUnit in the vector with this one, after the last bit
 * index of this one. (If the programmer doesn't want to "cheat," it is left up
 * to them to delete the old coherent unit that was added.
 */
void CoherentUnit::Cohere(std::vector<std::shared_ptr<CoherentUnit>> toCopy)
{
    bitLenInt i, j, k;
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

    std::vector<std::unique_ptr<Complex16[]>> nGateQueue(nQubitCount);
    std::vector<bool> nIsQueued(nQubitCount);
    std::copy(isQueued.begin(), isQueued.begin() + qubitCount, nIsQueued.begin());
    for (i = 0; i < qubitCount; i++) {
        nGateQueue[i] = std::move(gateQueue[i]);
    }
    k = 0;
    for (i = 0; i < toCohereCount; i++) {
        for (j = 0; j < toCopy[i]->GetQubitCount(); j++) {
            nIsQueued[k + qubitCount] = toCopy[i]->isQueued[i];
            nGateQueue[k + qubitCount] = std::move(std::unique_ptr<Complex16[]>(new Complex16[4]));
            std::copy(&(toCopy[i]->gateQueue[j][0]), &(toCopy[i]->gateQueue[j][0]) + 4, &(nGateQueue[k + qubitCount][0]));
            k++;
        }
    }
    gateQueue = std::move(nGateQueue);
    isQueued = nIsQueued;

    nMaxQPower = 1 << nQubitCount;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);

    par_for(0, nMaxQPower, [&](const bitCapInt lcv, const int cpu) {
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

/**
 * Minimally decohere a set of contigious bits from the full coherent unit. The
 * length of this coherent unit is reduced by the length of bits decohered, and
 * the bits removed are output in the destination CoherentUnit pointer. The
 * destination object must be initialized to the correct number of bits, in 0
 * permutation state.
 */
void CoherentUnit::Decohere(bitLenInt start, bitLenInt length, CoherentUnit& destination)
{
    if (length == 0) {
        return;
    }

    FlushQueue(start, length);
    gateQueue.erase(gateQueue.begin() + start, gateQueue.begin() + start + length);
    isQueued.erase(isQueued.begin() + start, isQueued.begin() + start + length);

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
        destination.stateVec[i] = sqrt(partStateProb[i]) * Complex16(cos(partStateAngle[i]), sin(partStateAngle[i]));
    }

    for (i = 0; i < remainderPower; i++) {
        stateVec[i] = sqrt(remainderStateProb[i]) * Complex16(cos(remainderStateAngle[i]), sin(remainderStateAngle[i]));
    }

    UpdateRunningNorm();
    destination.UpdateRunningNorm();
}

void CoherentUnit::Dispose(bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    gateQueue.erase(gateQueue.begin() + start, gateQueue.begin() + start + length);
    isQueued.erase(isQueued.begin() + start, isQueued.begin() + start + length);

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

// Logic Gates:

/// "AND" compare two bits in CoherentUnit, and store result in outputBit
void CoherentUnit::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, false);
        if (inputBit1 == inputBit2) {
            CNOT(inputBit1, outputBit);
        } else {
            CCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid AND arguments.");
    }
}

/// "AND" compare a qubit in CoherentUnit with a classical bit, and store result in outputBit
void CoherentUnit::CLAND(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (!inputClassicalBit) {
        SetBit(outputBit, false);
    } else if (inputQBit != outputBit) {
        SetBit(outputBit, false);
        CNOT(inputQBit, outputBit);
    }
}

/// "OR" compare two bits in CoherentUnit, and store result in outputBit
void CoherentUnit::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    /* Same bit, no action necessary. */
    if ((inputBit1 == inputBit2) && (inputBit2 == outputBit)) {
        return;
    }

    if ((inputBit1 != outputBit) && (inputBit2 != outputBit)) {
        SetBit(outputBit, true);
        if (inputBit1 == inputBit2) {
            AntiCNOT(inputBit1, outputBit);
        } else {
            AntiCCNOT(inputBit1, inputBit2, outputBit);
        }
    } else {
        throw std::invalid_argument("Invalid OR arguments.");
    }
}

/// "OR" compare a qubit in CoherentUnit with a classical bit, and store result in outputBit
void CoherentUnit::CLOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputClassicalBit) {
        SetBit(outputBit, true);
    } else if (inputQBit != outputBit) {
        SetBit(outputBit, false);
        CNOT(inputQBit, outputBit);
    }
}

/// "XOR" compare two bits in CoherentUnit, and store result in outputBit
void CoherentUnit::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    if (((inputBit1 == inputBit2) && (inputBit2 == outputBit))) {
        SetBit(outputBit, false);
        return;
    }

    if (inputBit1 == outputBit) {
        CNOT(inputBit2, outputBit);
    } else if (inputBit2 == outputBit) {
        CNOT(inputBit1, outputBit);
    } else {
        SetBit(outputBit, false);
        CNOT(inputBit1, outputBit);
        CNOT(inputBit2, outputBit);
    }
}

/// "XOR" compare a qubit in CoherentUnit with a classical bit, and store result in outputBit
void CoherentUnit::CLXOR(bitLenInt inputQBit, bool inputClassicalBit, bitLenInt outputBit)
{
    if (inputQBit != outputBit) {
        SetBit(outputBit, inputClassicalBit);
        CNOT(inputQBit, outputBit);
    } else if (inputClassicalBit) {
        X(outputBit);
    }
}

/// Doubly-controlled not
void CoherentUnit::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }

    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };

    bitCapInt qPowers[4];
    bitCapInt qPowersSorted[3];
    qPowers[1] = 1 << control1;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << control2;
    qPowersSorted[1] = qPowers[2];
    qPowers[3] = 1 << target;
    qPowersSorted[2] = qPowers[3];
    qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(qPowers[0], qPowers[1] + qPowers[2], pauliX, 3, qPowersSorted, false);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void CoherentUnit::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    // if ((control1 >= qubitCount) || (control2 >= qubitCount))
    //	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
    if (control1 == control2) {
        throw std::invalid_argument("CCNOT control bits cannot be same bit.");
    }
    if (control1 == target || control2 == target) {
        throw std::invalid_argument("CCNOT control bits cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };

    bitCapInt qPowers[4];
    bitCapInt qPowersSorted[3];
    qPowers[1] = 1 << control1;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << control2;
    qPowersSorted[1] = qPowers[2];
    qPowers[3] = 1 << target;
    qPowersSorted[2] = qPowers[3];
    qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(0, qPowers[3], pauliX, 3, qPowersSorted, false);
}

/// Controlled not
void CoherentUnit::CNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliX);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void CoherentUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("CNOT control bit cannot also be target.");
    }

    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplyAntiControlled2x2(control, target, pauliX);
}

/// Hadamard gate
void CoherentUnit::H(bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount) throw std::invalid_argument("operation on bit index greater than total
    // bits.");
    const Complex16 had[4] = { Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
        Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0) };
    ApplySingleBit(qubitIndex, had, true);
}

/// Measurement gate
bool CoherentUnit::M(bitLenInt qubitIndex)
{
    FlushQueue(qubitIndex);

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bool result;
    double prob = Rand();
    double angle = Rand() * 2.0 * M_PI;
    double cosine = cos(angle);
    double sine = sin(angle);
    Complex16 nrm;

    bitCapInt qPowers = 1 << qubitIndex;
    double oneChance = Prob(qubitIndex);

    result = (prob < oneChance) && oneChance > 0.0;
    double nrmlzr = 1.0;
    if (result) {
        if (oneChance > 0.0) {
            nrmlzr = oneChance;
        }

        nrm = Complex16(cosine, sine) / nrmlzr;

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            if ((lcv & qPowers) == 0) {
                stateVec[lcv] = Complex16(0.0, 0.0);
            } else {
                stateVec[lcv] = nrm * stateVec[lcv];
            }
        });
    } else {
        if (oneChance < 1.0) {
            nrmlzr = sqrt(1.0 - oneChance);
        }

        nrm = Complex16(cosine, sine) / nrmlzr;

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            if ((lcv & qPowers) == 0) {
                stateVec[lcv] = nrm * stateVec[lcv];
            } else {
                stateVec[lcv] = Complex16(0.0, 0.0);
            }
        });
    }

    UpdateRunningNorm();

    return result;
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
double CoherentUnit::Prob(bitLenInt qubitIndex)
{
    FlushQueue(qubitIndex);

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt qPower = 1 << qubitIndex;
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
double CoherentUnit::ProbAll(bitCapInt fullRegister)
{
    FlushQueue(0, qubitCount);

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    return norm(stateVec[fullRegister]);
}

/// PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
void CoherentUnit::ProbArray(double* probArray)
{
    FlushQueue(0, qubitCount);

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt lcv;
    for (lcv = 0; lcv < maxQPower; lcv++) {
        probArray[lcv] = norm(stateVec[lcv]);
    }
}

/// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
void CoherentUnit::RT(double radians, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total // bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 mtrx[4] = { Complex16(1.0, 0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(cosine, sine) };
    ApplySingleBit(qubitIndex, mtrx, true);
}

/**
 * Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around |1> state.
 *
 * NOTE THAT * DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void CoherentUnit::RTDyad(int numerator, int denominator, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RT((M_PI * numerator * 2) / denominator, qubitIndex);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void CoherentUnit::RX(double radians, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    // throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRX[4] = { Complex16(cosine, 0.0), Complex16(0.0, -sine), Complex16(0.0, -sine),
        Complex16(cosine, 0.0) };
    ApplySingleBit(qubitIndex, pauliRX, true);
}

/**
 * Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void CoherentUnit::RXDyad(int numerator, int denominator, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RX((-M_PI * numerator * 2) / denominator, qubitIndex);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void CoherentUnit::RY(double radians, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRY[4] = { Complex16(cosine, 0.0), Complex16(-sine, 0.0), Complex16(sine, 0.0),
        Complex16(cosine, 0.0) };
    ApplySingleBit(qubitIndex, pauliRY, true);
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) /
 * denominator) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void CoherentUnit::RYDyad(int numerator, int denominator, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RY((-M_PI * numerator * 2) / denominator, qubitIndex);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void CoherentUnit::RZ(double radians, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 pauliRZ[4] = { Complex16(cosine, -sine), Complex16(0.0, 0.0), Complex16(0.0, 0.0),
        Complex16(cosine, sine) };
    ApplySingleBit(qubitIndex, pauliRZ, true);
}

/**
 * Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS AND LACKS DIVISION BY A FACTOR OF TWO.
 */
void CoherentUnit::RZDyad(int numerator, int denominator, bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    RZ((-M_PI * numerator * 2) / denominator, qubitIndex);
}

/// Set individual bit to pure |0> (false) or |1> (true) state
void CoherentUnit::SetBit(bitLenInt qubitIndex1, bool value)
{
    if (value != M(qubitIndex1)) {
        X(qubitIndex1);
    }
}

/// Swap values of two bits in register
void CoherentUnit::Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2)
{
    // if ((qubitIndex1 >= qubitCount) || (qubitIndex2 >= qubitCount))
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (qubitIndex1 != qubitIndex2) {
        // Does not necessarily commute with single bit gates;
        FlushQueue(qubitIndex1);
        FlushQueue(qubitIndex2);

        const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0),
            Complex16(0.0, 0.0) };

        bitCapInt qPowers[3];
        bitCapInt qPowersSorted[2];
        qPowers[1] = 1 << qubitIndex1;
        qPowersSorted[0] = qPowers[1];
        qPowers[2] = 1 << qubitIndex2;
        qPowersSorted[1] = qPowers[2];
        qPowers[0] = qPowers[1] + qPowers[2];
        std::sort(qPowersSorted, qPowersSorted + 2);
        Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false);
    }
}

/// NOT gate, which is also Pauli x matrix
void CoherentUnit::X(bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubitIndex, pauliX, false);
}

/// Apply Pauli Y matrix to bit
void CoherentUnit::Y(bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplySingleBit(qubitIndex, pauliY, false);
}

/// Apply Pauli Z matrix to bit
void CoherentUnit::Z(bitLenInt qubitIndex)
{
    // if (qubitIndex >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplySingleBit(qubitIndex, pauliZ, false);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void CoherentUnit::CRT(double radians, bitLenInt control, bitLenInt target)
{
    // if ((control >= qubitCount) || (target >= qubitCount))
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target) {
        throw std::invalid_argument("control bit cannot also be target.");
    }

    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 mtrx[4] = { Complex16(1.0, 0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(cosine, sine) };
    ApplyControlled2x2(control, target, mtrx);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void CoherentUnit::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    // if (target >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRTDyad control bit cannot also be target.");
    CRT((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void CoherentUnit::CRX(double radians, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRX control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRX[4] = { Complex16(cosine, 0.0), Complex16(0.0, -sine), Complex16(0.0, -sine),
        Complex16(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRX);
}

/**
 * Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI *
 * numerator) / denominator) around Pauli x axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void CoherentUnit::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRXDyad control bit cannot also be target.");
    CRX((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void CoherentUnit::CRY(double radians, bitLenInt control, bitLenInt target)
{
    // if (control >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    if (control == target)
        throw std::invalid_argument("CRY control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    Complex16 pauliRY[4] = { Complex16(cosine, 0.0), Complex16(-sine, 0.0), Complex16(sine, 0.0),
        Complex16(cosine, 0.0) };
    ApplyControlled2x2(control, target, pauliRY);
}

/**
 * Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void CoherentUnit::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRYDyad control bit cannot also be target.");
    CRY((-M_PI * numerator * 2) / denominator, control, target);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void CoherentUnit::CRZ(double radians, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZ control bit cannot also be target.");
    double cosine = cos(radians / 2.0);
    double sine = sin(radians / 2.0);
    const Complex16 pauliRZ[4] = { Complex16(cosine, -sine), Complex16(0.0, 0.0), Complex16(0.0, 0.0),
        Complex16(cosine, sine) };
    ApplyControlled2x2(control, target, pauliRZ);
}

/**
 * Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION
 * OPERATORS.
 */
void CoherentUnit::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    if (control == target)
        throw std::invalid_argument("CRZDyad control bit cannot also be target.");
    CRZ((-M_PI * numerator * 2) / denominator, control, target);
}

/// Apply controlled Pauli Y matrix to bit
void CoherentUnit::CY(bitLenInt control, bitLenInt target)
{
    // if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CY control bit cannot also be target.");
    const Complex16 pauliY[4] = { Complex16(0.0, 0.0), Complex16(0.0, -1.0), Complex16(0.0, 1.0), Complex16(0.0, 0.0) };
    ApplyControlled2x2(control, target, pauliY);
}

/// Apply controlled Pauli Z matrix to bit
void CoherentUnit::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplyControlled2x2(control, target, pauliZ);
}

// Single register instructions:

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void CoherentUnit::X(bitLenInt start, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        X(start);
        return;
    }

    // If we have depth optimizations queued, keep pushing on that queue.
    if (CheckQueued(start, length)) {
        for (bitLenInt i = 0; i < length; i++) {
            X(start + i);
        }
        return;
    }

    // Otherwise, use an optimized register-wise gate.

    // As a fundamental gate, the register-wise X could proceed like so:
    // for (bitLenInt lcv = 0; lcv < length; lcv++) {
    //    X(start + lcv);
    //}

    // Basically ALL register-wise gates proceed by essentially the same
    // algorithm as this simple X gate.

    // We first form bit masks for those qubits involved in the operation, and
    // those not involved in the operation. We might have more than one
    // register involved in the operation in general, but we only have one, in
    // this case.
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ inOutMask;

    // Sometimes we transform the state in place. Alternatively, we often
    // allocate a new permutation state vector to transfer old probabilities
    // and phases into.
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);

    // This function call is a parallel "for" loop. We have several variants of
    // the parallel for loop. Some skip certain permutations in order to
    // optimize. Some take a new permutation state vector for output, and some
    // just transform the permutation state vector in place.
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        // Set nStateVec, indexed by the loop control variable (lcv) with
        // the X'ed bits inverted, with the value of stateVec indexed by
        // lcv.

        // This is the body of the parallel "for" loop. We iterate over
        // permutations of bits.  We're going to transform from input
        // permutation state to output permutation state, and transfer the
        // probability and phase of the input permutation to the output
        // permutation.  These are the bits that aren't involved in the
        // operation.
        bitCapInt otherRes = (lcv & otherMask);

        // These are the bits in the register that is being operated on. In
        // all permutation states, the bits acted on by the gate should be
        // transformed in the logically appropriate way from input
        // permutation to output permutation. Since this is an X gate, we
        // take the involved bits and bitwise NOT them.
        bitCapInt inOutRes = ((~lcv) & inOutMask);

        // Now, we just transfer the untransformed input state's phase and
        // probability to the transformed output state.
        nStateVec[inOutRes | otherRes] = stateVec[lcv];

        // For other operations, like the quantum equivalent of a logical
        // "AND," we might have two input registers and one output
        // register. The transformation would be that we use bit masks to
        // bitwise "AND" the input values in every permutation and place
        // this logical result into the output register with another bit
        // mask, for every possible permutation state. Basically all the
        // register-wise operations in Qrack proceed this same way.
    });
    // We replace our old permutation state vector with the new one we just
    // filled, at the end.
    ResetStateVec(std::move(nStateVec));
}

/// Apply Pauli Y matrix to each bit
void CoherentUnit::Y(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Y(start + lcv);
    }
}

/// Apply Pauli Z matrix to each bit
void CoherentUnit::Z(bitLenInt start, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        Z(start);
        return;
    }

    if (CheckQueued(start, length)) {
        for (bitLenInt i = 0; i < length; i++) {
            Z(start + i);
        }
    } else {
        bitCapInt inOutMask = ((1 << length) - 1) << start;
        bitCapInt otherMask = ((1 << qubitCount) - 1) ^ inOutMask;
        std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = lcv & otherMask;
            bitCapInt inOutRes = lcv & inOutMask;
            bitCapInt inOutInt = inOutRes >> start;
            bitLenInt bitCount;
            for (bitCount = 0; inOutInt; bitCount++) {
                inOutInt &= inOutInt - 1;  
            }
            nStateVec[inOutRes | otherRes] = (bitCount & 1) ? -stateVec[lcv] : stateVec[lcv];
        });
        ResetStateVec(std::move(nStateVec));
    }
}

/// Bitwise swap
void CoherentUnit::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        Swap(start1, start2);
        return;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(start1, length);
    FlushQueue(start2, length);

    int distance = start1 - start2;
    if (distance < 0) {
        distance *= -1;
    }
    if (distance < length) {
        bitLenInt i;
        for (i = 0; i < length; i++) {
            Swap(start1 + i, start2 + i);
        }
    } else {
        bitCapInt reg1Mask = ((1 << length) - 1) << start1;
        bitCapInt reg2Mask = ((1 << length) - 1) << start2;
        bitCapInt otherMask = maxQPower - 1;
        otherMask ^= reg1Mask | reg2Mask;
        std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = (lcv & otherMask);
            bitCapInt reg1Res = ((lcv & reg1Mask) >> (start1)) << (start2);
            bitCapInt reg2Res = ((lcv & reg2Mask) >> (start2)) << (start1);
            nStateVec[reg1Res | reg2Res | otherRes] = stateVec[lcv];
        });
        // We replace our old permutation state vector with the new one we just filled, at the end.
        ResetStateVec(std::move(nStateVec));
    }
}

/// Apply Hadamard gate to each bit in "length," starting from bit index "start"
void CoherentUnit::H(bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        H(start + lcv);
    }
}

///"Phase shift gate" - Rotates each bit as e^(-i*\theta/2) around |1> state
void CoherentUnit::RT(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RT(radians, start + lcv);
    }
}

/**
 * Dyadic fraction "phase shift gate" - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around |1> state.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR OF
 * TWO.
 */
void CoherentUnit::RTDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RTDyad(numerator, denominator, start + lcv);
    }
}

/// x axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli x axis
void CoherentUnit::RX(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RX(radians, start + lcv);
    }
}

/**
 * Dyadic fraction x axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli x
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void CoherentUnit::RXDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RXDyad(numerator, denominator, start + lcv);
    }
}

/// y axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli y axis
void CoherentUnit::RY(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RY(radians, start + lcv);
    }
}

/**
 * Dyadic fraction y axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void CoherentUnit::RYDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RYDyad(numerator, denominator, start + lcv);
    }
}

/// z axis rotation gate - Rotates each bit as e^(-i*\theta/2) around Pauli z axis
void CoherentUnit::RZ(double radians, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RZ(radians, start + lcv);
    }
}

/**
 * Dyadic fraction z axis rotation gate - Rotates each bit as e^(i*(M_PI * numerator) / denominator) around Pauli y
 * axis.
 *
 * NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS AND LACKS DIVISION BY A FACTOR
 * OF TWO.
 */
void CoherentUnit::RZDyad(int numerator, int denominator, bitLenInt start, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        RZDyad(numerator, denominator, start + lcv);
    }
}

/// Controlled "phase shift gate"
void CoherentUnit::CRT(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRT(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction "phase shift gate"
void CoherentUnit::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRTDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled x axis rotation
void CoherentUnit::CRX(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRX(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction x axis rotation gate - for each bit, if control bit is true, rotates target bit as as
/// e^(i*(M_PI * numerator) / denominator) around Pauli x axis
void CoherentUnit::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRXDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled y axis rotation
void CoherentUnit::CRY(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRY(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction y axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli y axis
void CoherentUnit::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRYDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Controlled z axis rotation
void CoherentUnit::CRZ(double radians, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZ(radians, control + lcv, target + lcv);
    }
}

/// Controlled dyadic fraction z axis rotation gate - for each bit, if control bit is true, rotates target bit as
/// e^(i*(M_PI * numerator) / denominator) around Pauli z axis
void CoherentUnit::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CRZDyad(numerator, denominator, control + lcv, target + lcv);
    }
}

/// Apply controlled Pauli Y matrix to each bit
void CoherentUnit::CY(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CY(control + lcv, target + lcv);
    }
}

/// Apply controlled Pauli Z matrix to each bit
void CoherentUnit::CZ(bitLenInt control, bitLenInt target, bitLenInt length)
{
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        CZ(control + lcv, target + lcv);
    }
}

/// Bit-parallel "CNOT" two bit ranges in CoherentUnit, and store result in range starting at output
void CoherentUnit::CNOT(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt length)
{
    if (inputStart1 != inputStart2) {
        for (bitLenInt i = 0; i < length; i++) {
            CNOT(inputStart1 + i, inputStart2 + i);
        }
    }
}

/// "AND" compare two bit ranges in CoherentUnit, and store result in range starting at output
void CoherentUnit::AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            AND(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "AND" compare a bit range in CoherentUnit with a classical unsigned integer, and store result in range starting at
/// output
void CoherentUnit::CLAND(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLAND(qInputStart + i, cBit, outputStart + i);
    }
}

/// "OR" compare two bit ranges in CoherentUnit, and store result in range starting at output
void CoherentUnit::OR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            OR(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "OR" compare a bit range in CoherentUnit with a classical unsigned integer, and store result in range starting at
/// output
void CoherentUnit::CLOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// "XOR" compare two bit ranges in CoherentUnit, and store result in range starting at output
void CoherentUnit::XOR(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length)
{
    if (!((inputStart1 == inputStart2) && (inputStart2 == outputStart))) {
        for (bitLenInt i = 0; i < length; i++) {
            XOR(inputStart1 + i, inputStart2 + i, outputStart + i);
        }
    }
}

/// "XOR" compare a bit range in CoherentUnit with a classical unsigned integer, and store result in range starting at
/// output
void CoherentUnit::CLXOR(bitLenInt qInputStart, bitCapInt classicalInput, bitLenInt outputStart, bitLenInt length)
{
    bool cBit;
    for (bitLenInt i = 0; i < length; i++) {
        cBit = (1 << i) & classicalInput;
        CLXOR(qInputStart + i, cBit, outputStart + i);
    }
}

/// Arithmetic shift left, with last 2 bits as sign and carry
void CoherentUnit::ASL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROL(shift, start, length);
            Swap(end - 1, end - 2);
            SetReg(start, shift, 0);
        }
    }
}

/// Arithmetic shift right, with last 2 bits as sign and carry
void CoherentUnit::ASR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Swap(end - 1, end - 2);
            ROR(shift, start, length);
            Swap(end - 1, end - 2);
            SetReg(end - shift, shift, 0);
        }
    }
}

/// Logical shift left, filling the extra bits with |0>
void CoherentUnit::LSL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROL(shift, start, length);
            SetReg(start, shift, 0);
        }
    }
}

/// Logical shift right, filling the extra bits with |0>
void CoherentUnit::LSR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            ROR(shift, start, length);
            SetReg(end - shift, shift, 0);
        }
    }
}

/// Add integer (without sign)
void CoherentUnit::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(start, length);

    bitCapInt lengthPower = 1 << length;
    toAdd %= lengthPower;
    if ((length > 0) && (toAdd > 0)) {
        bitCapInt otherMask = (1 << qubitCount) - 1;
        bitCapInt inOutMask = (lengthPower - 1) << start;
        otherMask ^= inOutMask;
        std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
        std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = (lcv & otherMask);
            bitCapInt inOutRes = (lcv & inOutMask);
            bitCapInt inOutInt = inOutRes >> start;
            bitCapInt outInt = inOutInt + toAdd;
            bitCapInt outRes;
            if (outInt < lengthPower) {
                outRes = (outInt << start) | otherRes;
            } else {
                outRes = ((outInt - lengthPower) << start) | otherRes;
            }
            nStateVec[outRes] = stateVec[lcv];
        });
        ResetStateVec(std::move(nStateVec));
    }
}

/// Add BCD integer (without sign)
void CoherentUnit::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);

    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = (partToAdd % 10);
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            nStateVec[(outInt << (inOutStart)) | otherRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(std::move(nStateVec));
}

/// Add BCD integer (without sign, with carry)
void CoherentUnit::INCBCDC(
    bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);
    FlushQueue(carryIndex);

    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;

    otherMask ^= inOutMask | carryMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToAdd = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15;
        test2 = partToAdd % 10;
        partToAdd /= 10;
        nibbles[0] = test1 + test2;
        if ((test1 > 9) || (test2 > 9)) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = partToAdd % 10;
            partToAdd /= 10;
            nibbles[j] = test1 + test2;
            if ((test1 > 9) || (test2 > 9)) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            bitCapInt outRes = 0;
            bitCapInt carryRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] > 9) {
                    nibbles[j] -= 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]++;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Add an integer to the register, with sign and without carry. Because the
 * register length is an arbitrary number of bits, the sign bit position on the
 * integer to add is variable. Hence, the integer to add is specified as cast
 * to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void CoherentUnit::INCS(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);
    FlushQueue(overflowIndex);

    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << (inOutStart)) | otherRes;
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
                isOverflow = true;
        }
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Add an integer to the register, with sign and with carry. If the overflow is set, flip phase on overflow. Because the
 * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
 * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
 * position before the cast.
 */
void CoherentUnit::INCSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        FlushQueue(carryIndex);
        toAdd++;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);

    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask | carryMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
                isOverflow = true;
        }
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is an
 * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void CoherentUnit::INCSC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);
    FlushQueue(carryIndex);

    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toAdd;
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        bool isOverflow = false;
        // Both negative:
        if (inOutInt & inInt & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > (signMask))
                isOverflow = true;
        }
        // Both positive:
        else if ((~inOutInt) & (~inInt) & (signMask)) {
            if ((inOutInt + inInt) >= (signMask))
                isOverflow = true;
        }
        if (isOverflow) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/// Subtract integer (without sign)
void CoherentUnit::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    toSub %= lengthPower;
    if ((length > 0) && (toSub > 0)) {
        //Does not necessarily commute with single bit queues
        FlushQueue(start, length);

        bitCapInt otherMask = (1 << qubitCount) - 1;
        bitCapInt inOutMask = (lengthPower - 1) << start;
        otherMask ^= inOutMask;
        std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
        std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

        par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
            bitCapInt otherRes = (lcv & otherMask);
            bitCapInt inOutRes = (lcv & inOutMask);
            bitCapInt inOutInt = inOutRes >> start;
            bitCapInt outInt = inOutInt - toSub + lengthPower;
            bitCapInt outRes;
            if (outInt < lengthPower) {
                outRes = (outInt << start) | otherRes;
            } else {
                outRes = ((outInt - lengthPower) << start) | otherRes;
            }
            nStateVec[outRes] = stateVec[lcv];
        });
        ResetStateVec(std::move(nStateVec));
    }
}

/// Subtract BCD integer (without sign)
void CoherentUnit::DECBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);

    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToSub = toAdd;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;
        for (j = 0; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = (partToSub % 10);
            partToSub /= 10;
            nibbles[j] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            nStateVec[(outInt << (inOutStart)) | otherRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void CoherentUnit::DECS(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);
    FlushQueue(overflowIndex);

    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = overflowMask;
        bitCapInt outInt = inOutInt - toSub + lengthPower;
        bitCapInt outRes;
        if (outInt < lengthPower) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - lengthPower) << (inOutStart)) | otherRes;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Subtract an integer from the register, with sign and with carry. If the overflow is set, flip phase on overflow.
 * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is variable.
 * Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void CoherentUnit::DECSC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        FlushQueue(carryIndex);
    } else {
        toSub++;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);

    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toSub;
        bitCapInt outInt = (inOutInt - toSub) + (lengthPower);
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | carryMask;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        if (isOverflow && ((outRes & overflowMask) == overflowMask)) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/**
 * Subtract an integer from the register, with sign and with carry. Flip phase on overflow. Because the register length
 * is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void CoherentUnit::DECSC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        FlushQueue(carryIndex);
        toSub++;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);


    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;

    otherMask ^= inOutMask | carryMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, carryMask, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt inInt = toSub;
        bitCapInt outInt = (inOutInt - toSub) + (lengthPower);
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes | (carryMask);
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes;
        }
        bool isOverflow = false;
        // First negative:
        if (inOutInt & (~inInt) & (signMask)) {
            inOutInt = ((~inOutInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) > signMask)
                isOverflow = true;
        }
        // First positive:
        else if (inOutInt & (~inInt) & (signMask)) {
            inInt = ((~inInt) & (lengthPower - 1)) + 1;
            if ((inOutInt + inInt) >= signMask)
                isOverflow = true;
        }
        if (isOverflow) {
            nStateVec[outRes] = -stateVec[lcv];
        } else {
            nStateVec[outRes] = stateVec[lcv];
        }
    });
    ResetStateVec(std::move(nStateVec));
}

/// Subtract BCD integer (without sign, with carry)
void CoherentUnit::DECBCDC(
    bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        FlushQueue(carryIndex);
        toSub++;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length); 

    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt carryMask = 1 << carryIndex;
    otherMask ^= inOutMask | carryMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv, const int cpu) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt partToSub = toSub;
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        char test1, test2;
        unsigned char j;
        char* nibbles = new char[nibbleCount];
        bool isValid = true;

        test1 = inOutInt & 15;
        test2 = partToSub % 10;
        partToSub /= 10;
        nibbles[0] = test1 - test2;
        if (test1 > 9) {
            isValid = false;
        }

        for (j = 1; j < nibbleCount; j++) {
            test1 = (inOutInt & (15 << (j * 4))) >> (j * 4);
            test2 = partToSub % 10;
            partToSub /= 10;
            nibbles[j] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }
        }
        if (isValid) {
            bitCapInt outInt = 0;
            bitCapInt outRes = 0;
            bitCapInt carryRes = 0;
            for (j = 0; j < nibbleCount; j++) {
                if (nibbles[j] < 0) {
                    nibbles[j] += 10;
                    if ((unsigned char)(j + 1) < nibbleCount) {
                        nibbles[j + 1]--;
                    } else {
                        carryRes = carryMask;
                    }
                }
                outInt |= nibbles[j] << (j * 4);
            }
            outRes = (outInt << (inOutStart)) | otherRes | carryRes;
            nStateVec[outRes] = stateVec[lcv];
        } else {
            nStateVec[lcv] = stateVec[lcv];
        }
        delete[] nibbles;
    });
    ResetStateVec(std::move(nStateVec));
}

/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
void CoherentUnit::QFT(bitLenInt start, bitLenInt length)
{
    //Uses single bit gates
    if (length > 0) {
        bitLenInt end = start + length;
        bitLenInt i, j;
        for (i = start; i < end; i++) {
            H(i);
            for (j = 1; j < (end - i); j++) {
                CRTDyad(1, 1 << j, i + j, i);
            }
        }
    }
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void CoherentUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(start, length);

    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((lcv & (~(regMask))) == lcv)
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void CoherentUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(start, length);
    FlushQueue(flagIndex);

    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt flagMask = 1 << flagIndex;

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((((lcv & regMask) >> (start)) < greaterPerm) & ((lcv & flagMask) == flagMask))
            stateVec[lcv] = -stateVec[lcv];
    });
}

/// Phase flip always - equivalent to Z X Z X on any bit in the CoherentUnit
void CoherentUnit::PhaseFlip()
{
    //Commutes with single bit queues
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) { stateVec[lcv] = -stateVec[lcv]; });
}

/// Set register bits to given permutation
void CoherentUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    FlushQueue(start, length);

    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (value == 1));
    } else if ((start == 0) && (length == qubitCount)) {
        double angle = Rand() * 2.0 * M_PI;

        runningNorm = 1.0;
        std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0, 0.0));
        stateVec[value] = Complex16(cos(angle), sin(angle));
    } else {
        bool bitVal;
        bitCapInt regVal = MReg(start, length);
        for (bitCapInt i = 0; i < length; i++) {
            bitVal = regVal & (1 << i);
            if ((bitVal && !(value & (1 << i))) || (!bitVal && (value & (1 << i))))
                X(start + i);
        }
    }
}

/// Measure permutation state of a register
bitCapInt CoherentUnit::MReg(bitLenInt start, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        if (M(start)) {
            return 1;
        } else {
            return 0;
        }
    }

    FlushQueue(start, length);

    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bool foundPerm;
    double prob = Rand();
    double angle = Rand() * 2.0 * M_PI;
    double cosine = cos(angle);
    double sine = sin(angle);
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    double probArray[lengthPower];
    double lowerProb, largestProb, nrmlzr;
    bitCapInt lcv, result;

    for (lcv = 0; lcv < lengthPower; lcv++) {
        probArray[lcv] = 0.0;
    }

    for (lcv = 0; lcv < maxQPower; lcv++) {
        probArray[(lcv & regMask) >> start] += norm(stateVec[lcv]);
    }

    lcv = 0;
    foundPerm = false;
    lowerProb = 0.0;
    largestProb = 0.0;
    result = lengthPower - 1;
    while ((!foundPerm) && (lcv < maxQPower)) {
        if ((probArray[lcv] + lowerProb) > prob) {
            foundPerm = true;
            result = lcv;
            nrmlzr = probArray[lcv];
        } else {
            if (largestProb <= probArray[lcv]) {
                largestProb = probArray[lcv];
                result = lcv;
                nrmlzr = largestProb;
            }
            lowerProb += probArray[lcv];
            lcv++;
        }
    }

    bitCapInt resultPtr = result << start;
    Complex16 nrm = Complex16(cosine, sine) / nrmlzr;

    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        if ((lcv & resultPtr) == resultPtr) {
            stateVec[lcv] = nrm * stateVec[lcv];
        } else {
            stateVec[lcv] = Complex16(0.0, 0.0);
        }
    });

    UpdateRunningNorm();

    return result;
}

/// Measure permutation state of an 8 bit register
unsigned char CoherentUnit::MReg8(bitLenInt start) { return MReg(start, 8); }

void CoherentUnit::ApplySingleBit(bitLenInt qubitIndex, const Complex16* mtrx, bool doCalcNorm)
{
    if (qubitCount <= 2) {
        bitCapInt qPowers[1];
        qPowers[0] = 1 << qubitIndex;
        Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
    }
    else {
        isQueued[qubitIndex] = true;
        Mul2x2(mtrx, &(gateQueue[qubitIndex][0]));
    }
}

void CoherentUnit::ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx)
{
    FlushQueue(control);
    FlushQueue(target);

    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << target;
    qPowersSorted[1] = qPowers[2];
    qPowers[0] = qPowers[1] + qPowers[2];
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, false);
}

void CoherentUnit::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx)
{
    FlushQueue(control);
    FlushQueue(target);

    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowersSorted[0] = qPowers[1];
    qPowers[2] = 1 << target;
    qPowersSorted[1] = qPowers[2];
    qPowers[0] = qPowers[1] + qPowers[2];
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(0, qPowers[2], mtrx, 2, qPowersSorted, false);
}

void CoherentUnit::NormalizeState()
{
    par_for(0, maxQPower, [&](const bitCapInt lcv, const int cpu) {
        stateVec[lcv] /= runningNorm;
        if (norm(stateVec[lcv]) < 1e-15) {
            stateVec[lcv] = Complex16(0.0, 0.0);
        }
    });
    runningNorm = 1.0;
}

void CoherentUnit::Reverse(bitLenInt first, bitLenInt last)
{
    while ((first < last) && (first < (last - 1))) {
        last--;
        Swap(first, last);
        first++;
    }
}

void CoherentUnit::UpdateRunningNorm() { runningNorm = par_norm(maxQPower, &(stateVec[0])); }

void CoherentUnit::Mul2x2(const Complex16* leftIn, Complex16* rightOut) {
    Complex16 rightDupe[4];
    std::copy(rightOut, rightOut + 4, rightDupe);
    std::vector<std::future<void>> futures(4);
    futures[0] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[0] = leftIn[0] * rightDupe[0] + leftIn[1] * rightDupe[2];
    });
    futures[1] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[1] = leftIn[0] * rightDupe[1] + leftIn[1] * rightDupe[3];
    });
    futures[2] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[2] = leftIn[2] * rightDupe[0] + leftIn[3] * rightDupe[2];
    });
    futures[3] = std::async(std::launch::async, [rightOut, leftIn, rightDupe]() {
        rightOut[3] = leftIn[2] * rightDupe[1] + leftIn[3] * rightDupe[3];
    });
    for (int i = 0; i < 4; i++) {
        futures[i].get();
    }
}

void CoherentUnit::FlushQueue(bitLenInt index) {
    if (isQueued[index]) {
        isQueued[index] = false;
        bitCapInt qPowers[1];
        qPowers[0] = 1 << index;
        Apply2x2(0, qPowers[0], &(gateQueue[index][0]), 1, qPowers, true);
        std::copy(CMPLX_I_2X2, CMPLX_I_2X2 + 4, &(gateQueue[index][0]));
    }
}

void CoherentUnit::FlushQueue(bitLenInt start, bitLenInt length) {
    for (bitLenInt i = 0; i < length; i++) {
        FlushQueue(start + i);
    }
}

void CoherentUnit::ResetQueue(bitLenInt index) {
    if (isQueued[index]) {
        isQueued[index] = false;
        std::copy(CMPLX_I_2X2, CMPLX_I_2X2 + 4, &(gateQueue[index][0]));
    }
}

void CoherentUnit::ResetQueue(bitLenInt start, bitLenInt length) {
    for (bitLenInt i = 0; i < length; i++) {
        ResetQueue(start + i);
    }
}

bool CoherentUnit::CheckQueued(bitLenInt start, bitLenInt length) {
    for (bitLenInt i = 0; i < length; i++) {
        if (isQueued[start + i]) {
            return true;
        }
    }
    return false;
}

/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void CoherentUnit::par_for_inc(const bitCapInt begin, const bitCapInt end, IncrementFunc inc, ParallelFunc fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;

    std::vector<std::future<void>> futures(numCores);
    bitCapInt pStride = ParStride;

    if ((int)(maxQPower / ParStride) < numCores) {
        bitCapInt j;
        for (bitCapInt i = 0; i < end; i++) {
            j = inc(i, 0);
            if (j >= end) {
                break;
            }
            fn(j, 0);
        }
    }
    else {
        for (int cpu = 0; cpu < numCores; cpu++) {
            futures[cpu] = std::async(std::launch::async, [cpu, &idx, end, inc, fn, pStride]() {
                bitCapInt j, k;
                bitCapInt strideEnd = end / pStride;
                for (bitCapInt i = idx++; i < strideEnd; i = idx++) {
                    for (j = 0; j < pStride; j++) {
                        k = inc(i * pStride + j, cpu);
                        /* Easiest to clamp on end. */
                        if (k >= end) {
                            break;
                        }
                        fn(k, cpu);
                    }
                    if (k >= end) {
                        break;
                    }
                }
            });
        }

        for (int cpu = 0; cpu < numCores; cpu++) {
            futures[cpu].get();
        }
    }
}

void CoherentUnit::par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn)
{
    par_for_inc(begin, end, [](const bitCapInt i, int cpu) { return i; }, fn);
}

void CoherentUnit::par_for_skip(
    const bitCapInt begin, const bitCapInt end, const bitCapInt skipMask, const bitLenInt maskWidth, ParallelFunc fn)
{
    /*
     * Add maskWidth bits by shifting the incrementor up that number of
     * bits, filling with 0's.
     *
     * For example, if the skipMask is 0x8, then the lowMask will be 0x7
     * and the high mask will be ~(0x7 + 0x8) ==> ~0xf, shifted by the
     * number of extra bits to add.
     */
    bitCapInt lowMask = skipMask - 1;
    bitCapInt highMask = (~(lowMask + skipMask)) << (maskWidth - 1);

    IncrementFunc incFn = [lowMask, highMask, maskWidth](
                              bitCapInt i, int cpu) { return ((i << maskWidth) & highMask) | (i & lowMask); };

    par_for_inc(begin, end, incFn, fn);
}

void CoherentUnit::par_for_mask(
    const bitCapInt begin, const bitCapInt end, const bitCapInt* maskArray, const bitLenInt maskLen, ParallelFunc fn)
{
    if (maskLen > qubitCount) {
        throw std::invalid_argument("Too many masks");
    }

    for (int i = 1; i < maskLen; i++) {
        if (maskArray[i] < maskArray[i - 1]) {
            throw std::invalid_argument("Masks must be ordered by size");
        }
    }

    /* Pre-calculate the masks to simplify the increment function later. */
    bitCapInt masks[maskLen][2];

    for (int i = 0; i < maskLen; i++) {
        masks[i][0] = maskArray[i] - 1; // low mask
        masks[i][1] = (~(masks[i][0] + maskArray[i])); // high mask
    }

    IncrementFunc incFn = [&masks, maskLen](bitCapInt i, int cpu) {
        /* Push i apart, one mask at a time. */
        for (int m = 0; m < maskLen; m++) {
            i = ((i << 1) & masks[m][1]) | (i & masks[m][0]);
        }
        return i;
    };

    par_for_inc(begin, end, incFn, fn);
}

double CoherentUnit::par_norm(const bitCapInt maxQPower, const Complex16* stateArray)
{
    // const double* sAD = reinterpret_cast<const double*>(stateArray);
    // double* sSAD = new double[maxQPower * 2];
    // std::partial_sort_copy(sAD, sAD + (maxQPower * 2), sSAD, sSAD + (maxQPower * 2));
    // Complex16* sorted = reinterpret_cast<Complex16*>(sSAD);

    
    double nrmSqr = 0;
    if ((int)(maxQPower / ParStride) < numCores) {
        for (bitCapInt i = 0; i < maxQPower; i++) {
            nrmSqr += norm(stateArray[i]);
        }
    }
    else {
        std::atomic<bitCapInt> idx;
        idx = 0;
        double* nrmPart = new double[numCores];
        std::vector<std::future<void>> futures(numCores);
        bitCapInt pStride = ParStride;
        for (int cpu = 0; cpu != numCores; ++cpu) {
            futures[cpu] = std::async(std::launch::async, [cpu, &idx, maxQPower, stateArray, nrmPart, pStride]() {
                double sqrNorm = 0.0;
                // double smallSqrNorm = 0.0;
                bitCapInt i, j , k;
                for (;;) {
                    i = idx++;
                    for (j = 0; j < pStride; j++) {
                        k = i * pStride + j;
                        if (k >= maxQPower)
                            break;
                        sqrNorm += norm(stateArray[k]);
                    }
                    if (k >= maxQPower)
                        break;
                }
                nrmPart[cpu] = sqrNorm;
            });
        }

        for (int cpu = 0; cpu != numCores; ++cpu) {
            futures[cpu].get();
            nrmSqr += nrmPart[cpu];
        }
        delete[] nrmPart;
    }
    
    return sqrt(nrmSqr);
}

} // namespace Qrack
