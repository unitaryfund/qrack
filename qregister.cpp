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

#include "qregister.hpp"
#include <bitset>
#include <iostream>

#include "par_for.hpp"

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

/**
 * The "Qrack::CoherentUnit" class represents one or more coherent quantum
 * processor registers, including primitive bit logic gates and (abstract)
 * opcodes-like methods.
 */

///Protected constructor for SeparatedUnit
CoherentUnit::CoherentUnit() : rand_distribution(0.0, 1.0) {
    //This method body left intentionally empty
    randomSeed = std::time(0);
}

/// Initialize a coherent unit with qBitCount number pf bits, to initState unsigned integer permutation state
CoherentUnit::CoherentUnit(bitLenInt qBitCount, bitCapInt initState)
    : rand_distribution(0.0, 1.0)
{
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    (*rand_generator_ptr) = std::default_random_engine();
    randomSeed = std::time(0);
    SetRandomSeed(randomSeed);

    double angle = Rand() * 2.0 * M_PI;
    runningNorm = 1.0;
    qubitCount = qBitCount;
    maxQPower = 1 << qBitCount;
    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
    stateVec.reset();
    stateVec = std::move(sv);
    std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    stateVec[initState] = Complex16(cos(angle), sin(angle));
}

/// Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state, with a shared random number generator
CoherentUnit::CoherentUnit(bitLenInt qBitCount, bitCapInt initState, std::default_random_engine rgp[])
    : rand_distribution(0.0, 1.0)
{
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    rand_generator_ptr[0] = rgp[0];
    randomSeed = std::time(0);
    SetRandomSeed(randomSeed);

    double angle = Rand() * 2.0 * M_PI;
    runningNorm = 1.0;
    qubitCount = qBitCount;
    maxQPower = 1 << qBitCount;
    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
    stateVec.reset();
    stateVec = std::move(sv);
    std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    stateVec[initState] = Complex16(cos(angle), sin(angle));
}

/// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
CoherentUnit::CoherentUnit(bitLenInt qBitCount)
    : CoherentUnit(qBitCount, 0)
{
}

/// PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
CoherentUnit::CoherentUnit(const CoherentUnit& pqs)
    : rand_distribution(0.0, 1.0)
{
    (*rand_generator_ptr) = std::default_random_engine();
    randomSeed = std::time(0);
    SetRandomSeed(randomSeed);

    runningNorm = pqs.runningNorm;
    qubitCount = pqs.qubitCount;
    maxQPower = pqs.maxQPower;

    std::unique_ptr<Complex16[]> sv(new Complex16[maxQPower]);
    stateVec.reset();
    stateVec = std::move(sv);
    SetQuantumState(&pqs.stateVec[0]);
}

/// Set the random seed (primarily used for testing)
void CoherentUnit::SetRandomSeed(uint32_t seed) {
    randomSeed = seed;
    rand_generator_ptr->seed(seed);
}

/// PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
void CoherentUnit::CloneRawState(Complex16* output)
{
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
void CoherentUnit::SetPermutation(bitCapInt perm)
{
    double angle = Rand() * 2.0 * M_PI;

    runningNorm = 1.0;
    std::fill(&(stateVec[0]), &(stateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    stateVec[perm] = Complex16(cos(angle), sin(angle));
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void CoherentUnit::SetQuantumState(Complex16* inputState)
{
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

    bitCapInt nQubitCount = qubitCount + toCopy.qubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy.qubitCount)) - 1) << qubitCount;

    double angle = Rand() * 2.0 * M_PI;
    Complex16 phaseFac(cos(angle), sin(angle));
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[nMaxQPower]);
    bitCapInt bciArgs[3] = { startMask, endMask, qubitCount };
    par_for_cohere(0, nMaxQPower, &(stateVec[0]), bciArgs, phaseFac, &(nStateVec[0]), &(toCopy.stateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            const Complex16 phaseFac, Complex16* nStateVec, Complex16* toCopyStateVec) {
            nStateVec[lcv] = phaseFac *
                sqrt(norm(stateVec[(lcv & (bciArgs[0]))]) *
                    norm(toCopyStateVec[((lcv & (bciArgs[1])) >> (bciArgs[2]))]));
        });

    qubitCount = nQubitCount;
    maxQPower = nMaxQPower;

    ResetStateVec(std::move(nStateVec));
    UpdateRunningNorm();
    toCopy.UpdateRunningNorm();
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
    double prob;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        partStateProb[(i & mask) >> start] += prob;
        remainderStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;

    std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
    ResetStateVec(std::move(sv));

    double angle = Rand() * 2.0 * M_PI;
    Complex16 phaseFac(cos(angle), sin(angle));

    for (i = 0; i < partPower; i++) {
        destination.stateVec[i] = sqrt(partStateProb[i]) * phaseFac;
    }

    angle = Rand() * 2.0 * M_PI;
    phaseFac = Complex16(cos(angle), sin(angle));

    for (i = 0; i < remainderPower; i++) {
        stateVec[i] = sqrt(remainderStateProb[i]) * phaseFac;
    }

    UpdateRunningNorm();
    destination.UpdateRunningNorm();
}

void CoherentUnit::Dispose(bitLenInt start, bitLenInt length)
{
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);
    bitCapInt mask = (partPower - 1) << start;
    bitCapInt startMask = (1 << start) - 1;
    bitCapInt endMask = (maxQPower - 1) ^ (mask | startMask);
    bitCapInt i;

    std::unique_ptr<double[]> remainderStateProb(new double[remainderPower]());
    double prob;

    for (i = 0; i < maxQPower; i++) {
        prob = norm(stateVec[i]);
        remainderStateProb[(i & startMask) | ((i & endMask) >> length)] += prob;
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;

    std::unique_ptr<Complex16[]> sv(new Complex16[remainderPower]());
    ResetStateVec(std::move(sv));

    double angle = Rand() * 2.0 * M_PI;
    Complex16 phaseFac(cos(angle), sin(angle));

    for (i = 0; i < remainderPower; i++) {
        stateVec[i] = sqrt(remainderStateProb[i]) * phaseFac;
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
    Apply2x2(qPowers[0], qPowers[1] + qPowers[2], pauliX, 3, qPowersSorted, false, false);
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
    Apply2x2(0, qPowers[3], pauliX, 3, qPowersSorted, false, false);
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
    ApplyControlled2x2(control, target, pauliX, false);
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
    ApplyAntiControlled2x2(control, target, pauliX, false);
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
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    bool result;
    double prob = Rand();
    double angle = Rand() * 2.0 * M_PI;
    double cosine = cos(angle);
    double sine = sin(angle);

    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubitIndex;
    double oneChance = Prob(qubitIndex);

    result = (prob < oneChance) && oneChance > 0.0;
    double nrmlzr = 1.0;
    if (result) {
        if (oneChance > 0.0) {
            nrmlzr = oneChance;
        }

        par_for_all(0, maxQPower, &(stateVec[0]), Complex16(cosine, sine) / nrmlzr, NULL, qPowers,
            [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx,
                const bitCapInt* qPowers) {
                if ((lcv & qPowers[0]) == 0) {
                    stateVec[lcv] = Complex16(0.0, 0.0);
                } else {
                    stateVec[lcv] = nrm * stateVec[lcv];
                }
            });
    } else {
        if (oneChance < 1.0) {
            nrmlzr = sqrt(1.0 - oneChance);
        }

        par_for_all(0, maxQPower, &(stateVec[0]), Complex16(cosine, sine) / nrmlzr, NULL, qPowers,
            [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx,
                const bitCapInt* qPowers) {
                if ((lcv & qPowers[0]) == 0) {
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
    if (runningNorm != 1.0) {
        NormalizeState();
    }

    return norm(stateVec[fullRegister]);
}

/// PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
void CoherentUnit::ProbArray(double* probArray)
{
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
        const Complex16 pauliX[4] = { Complex16(0.0, 0.0), Complex16(1.0, 0.0), Complex16(1.0, 0.0),
            Complex16(0.0, 0.0) };

        bitCapInt qPowers[3];
        bitCapInt qPowersSorted[2];
        qPowers[1] = 1 << qubitIndex1;
        qPowers[2] = 1 << qubitIndex2;
        qPowers[0] = qPowers[1] + qPowers[2];
        if (qubitIndex1 < qubitIndex2) {
            qPowersSorted[0] = qPowers[1];
            qPowersSorted[1] = qPowers[2];
        } else {
            qPowersSorted[0] = qPowers[2];
            qPowersSorted[1] = qPowers[1];
        }

        Apply2x2(qPowers[2], qPowers[1], pauliX, 2, qPowersSorted, false, false);
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
    ApplyControlled2x2(control, target, mtrx, true);
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
    ApplyControlled2x2(control, target, pauliRX, true);
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
    ApplyControlled2x2(control, target, pauliRY, true);
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
    ApplyControlled2x2(control, target, pauliRZ, true);
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
    ApplyControlled2x2(control, target, pauliY, false);
}

/// Apply controlled Pauli Z matrix to bit
void CoherentUnit::CZ(bitLenInt control, bitLenInt target)
{
    // if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total
    // bits.");
    if (control == target)
        throw std::invalid_argument("CZ control bit cannot also be target.");
    const Complex16 pauliZ[4] = { Complex16(1.0, 0.0), Complex16(0.0, 0.0), Complex16(0.0, 0.0), Complex16(-1.0, 0.0) };
    ApplyControlled2x2(control, target, pauliZ, false);
}

// Single register instructions:

/// Apply X ("not") gate to each bit in "length," starting from bit index "start"
void CoherentUnit::X(bitLenInt start, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        X(start);
        return;
    }

    // By fundamental gates, the register-wise X could proceed like so:
    // for (bitLenInt lcv = 0; lcv < length; lcv++) {
    //    X(start + lcv);
    //}

    // Basically ALL register-wise gates proceed by essentially the same algorithm as this simple X gate.

    // We first form bit masks for those qubits involved in the operation, and those not involved in the operation. We
    // might have more than one register involved in the operation in general, but we only have one, in this case.
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    // We want to parallelize where possible, so we pack most of the arguments we need in looping into an array, to be
    // passed into a parallel "for" loop.
    bitCapInt bciArgs[2] = { inOutMask, otherMask };
    // Sometimes we transform the state in place. Alternatively, we often allocate a new permutation state vector to
    // transfer old probabilities and phases into.
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    // This function call is a parallel "for" loop. We have several variants of the parallel for loop. Some skip
    // certain permutations in order to optimize. Some take a new permutation state vector for output, and some just
    // transform the permutation state vector in place.
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            // This is the body of the parallel "for" loop. We iterate over permutations of bits.
            // We're going to transform from input permutation state to output permutation state, and transfer the
            // probability and phase of the input permutation to the output permutation.  These are the bits that aren't
            // involved in the operation.
            bitCapInt otherRes = (lcv & bciArgs[1]);
            // These are the bits in the register that is being operated on. In all permutation states, the bits acted
            // on by the gate should be transformed in the logically appropriate way from input permutation to output
            // permutation. Since this is an X gate, we take the involved bits and bitwise NOT them.
            bitCapInt inOutRes = ((~lcv) & bciArgs[0]);
            // Now, we just transfer the untransformed input state's phase and probability to the transformed output
            // state.
            nStateVec[inOutRes | otherRes] = stateVec[lcv];
            // For other operations, like the quantum equivalent of a logical "AND," we might have two input registers
            // and one output register. The transformation would be that we use bit masks to bitwise "AND" the input
            // values in every permutation and place this logical result into the output register with another bit mask,
            // for every possible permutation state. Basically all the register-wise operations in Qrack proceed this
            // same way.
        });
    // We replace our old permutation state vector with the new one we just filled, at the end.
    ResetStateVec(std::move(nStateVec));
}

/// Bitwise swap
void CoherentUnit::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        Swap(start1, start2);
        return;
    }

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
        bitCapInt bciArgs[5] = { reg1Mask, start1, reg2Mask, start2, otherMask };
        std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
        par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
            [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
                Complex16* nStateVec) {
                bitCapInt otherRes = (lcv & bciArgs[4]);
                bitCapInt reg1Res = ((lcv & bciArgs[0]) >> (bciArgs[1])) << (bciArgs[3]);
                bitCapInt reg2Res = ((lcv & bciArgs[2]) >> (bciArgs[3])) << (bciArgs[1]);
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
    for (bitLenInt lcv = 0; lcv < length; lcv++) {
        Z(start + lcv);
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
            Reverse(start, end);
            Reverse(start, start + shift);
            Reverse(start + shift, end);
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
            Reverse(start + shift, end);
            Reverse(start, start + shift);
            Reverse(start, end);
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
    bitCapInt lengthPower = 1 << length;
    toAdd %= lengthPower;
    if ((length > 0) && (toAdd > 0)) {
        bitCapInt i, j;
        bitLenInt end = start + length;
        bitCapInt startPower = 1 << start;
        bitCapInt endPower = 1 << end;
        bitCapInt iterPower = 1 << (qubitCount - end);
        bitCapInt maxLCV = iterPower * endPower;

        for (i = 0; i < startPower; i++) {
            for (j = 0; j < maxLCV; j += endPower) {
                rotate(&(stateVec[0]) + i + j, &(stateVec[0]) + ((lengthPower - toAdd) * startPower) + i + j,
                    &(stateVec[0]) + endPower + i + j, startPower);
            }
        }
    }
}

/// Add BCD integer (without sign)
void CoherentUnit::INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[5] = { inOutMask, toAdd, otherMask, inOutStart, nibbleCount };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[2]));
            bitCapInt partToAdd = bciArgs[1];
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[3]);
            char test1, test2;
            unsigned char j;
            char* nibbles = new char[bciArgs[4]];
            bool isValid = true;
            for (j = 0; j < bciArgs[4]; j++) {
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
                for (j = 0; j < bciArgs[4]; j++) {
                    if (nibbles[j] > 9) {
                        nibbles[j] -= 10;
                        if ((unsigned char)(j + 1) < bciArgs[4]) {
                            nibbles[j + 1]++;
                        }
                    }
                    outInt |= nibbles[j] << (j * 4);
                }
                nStateVec[(outInt << (bciArgs[3])) | otherRes] = stateVec[lcv];
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
    bitCapInt bciArgs[6] = { inOutMask, toAdd, carryMask, otherMask, inOutStart, nibbleCount };
    par_for_skip(0, maxQPower, 1 << carryIndex, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt partToAdd = bciArgs[1];
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[4]);
            char test1, test2;
            unsigned char j;
            char* nibbles = new char[bciArgs[5]];
            bool isValid = true;

            test1 = inOutInt & 15;
            test2 = partToAdd % 10;
            partToAdd /= 10;
            nibbles[0] = test1 + test2;
            if ((test1 > 9) || (test2 > 9)) {
                isValid = false;
            }

            for (j = 1; j < bciArgs[5]; j++) {
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
                for (j = 0; j < bciArgs[5]; j++) {
                    if (nibbles[j] > 9) {
                        nibbles[j] -= 10;
                        if ((unsigned char)(j + 1) < bciArgs[5]) {
                            nibbles[j + 1]++;
                        } else {
                            carryRes = bciArgs[2];
                        }
                    }
                    outInt |= nibbles[j] << (j * 4);
                }
                outRes = (outInt << (bciArgs[4])) | otherRes | carryRes;
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
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[7] = { inOutMask, toAdd, overflowMask, otherMask, lengthPower, inOutStart, signMask };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[1];
            bitCapInt outInt = inOutInt + bciArgs[1];
            bitCapInt outRes;
            if (outInt < bciArgs[4]) {
                outRes = (outInt << (bciArgs[5])) | otherRes;
            } else {
                outRes = ((outInt - bciArgs[4]) << (bciArgs[5])) | otherRes;
            }
            bool isOverflow = false;
            // Both negative:
            if (inOutInt & inInt & (bciArgs[6])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > (bciArgs[6]))
                    isOverflow = true;
            }
            // Both positive:
            else if ((~inOutInt) & (~inInt) & (bciArgs[6])) {
                if ((inOutInt + inInt) >= (bciArgs[6]))
                    isOverflow = true;
            }
            if (isOverflow && ((outRes & bciArgs[2]) == bciArgs[2])) {
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
        toAdd++;
    }
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    bitCapInt edgeMask = inOutMask | carryMask;
    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[10] = { inOutMask, toAdd, carryMask, otherMask, lengthPower, inOutStart, carryIndex, edgeMask,
        overflowMask, signMask };
    par_for_skip(0, maxQPower, carryMask, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[1];
            bitCapInt outInt = inOutInt + bciArgs[1];
            bitCapInt outRes;
            if (outInt < (bciArgs[4])) {
                outRes = (outInt << (bciArgs[5])) | otherRes;
            } else {
                outRes = ((outInt - (bciArgs[4])) << (bciArgs[5])) | otherRes | (bciArgs[2]);
            }
            bool isOverflow = false;
            // Both negative:
            if (inOutInt & inInt & (bciArgs[9])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > (bciArgs[9]))
                    isOverflow = true;
            }
            // Both positive:
            else if ((~inOutInt) & (~inInt) & (bciArgs[9])) {
                if ((inOutInt + inInt) >= (bciArgs[9]))
                    isOverflow = true;
            }
            if (isOverflow && ((outRes & bciArgs[8]) == bciArgs[8])) {
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
    bitCapInt edgeMask = inOutMask | carryMask;
    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[9] = { inOutMask, toAdd, carryMask, otherMask, lengthPower, inOutStart, carryIndex, edgeMask,
        signMask };
    par_for_skip(0, maxQPower, carryMask, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[1];
            bitCapInt outInt = inOutInt + bciArgs[1];
            bitCapInt outRes;
            if (outInt < (bciArgs[4])) {
                outRes = (outInt << (bciArgs[5])) | otherRes;
            } else {
                outRes = ((outInt - (bciArgs[4])) << (bciArgs[5])) | otherRes | (bciArgs[2]);
            }
            bool isOverflow = false;
            // Both negative:
            if (inOutInt & inInt & (bciArgs[9])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > (bciArgs[9]))
                    isOverflow = true;
            }
            // Both positive:
            else if ((~inOutInt) & (~inInt) & (bciArgs[9])) {
                if ((inOutInt + inInt) >= (bciArgs[9]))
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
        bitCapInt i, j;
        bitLenInt end = start + length;
        bitCapInt startPower = 1 << start;
        bitCapInt endPower = 1 << end;
        bitCapInt iterPower = 1 << (qubitCount - end);
        bitCapInt maxLCV = iterPower * endPower;
        for (i = 0; i < startPower; i++) {
            for (j = 0; j < maxLCV; j += endPower) {
                rotate(&(stateVec[0]) + i + j, &(stateVec[0]) + (toSub * startPower) + i + j,
                    &(stateVec[0]) + endPower + i + j, startPower);
            }
        }
    }
}

/// Subtract BCD integer (without sign)
void CoherentUnit::DECBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[5] = { inOutMask, toAdd, otherMask, inOutStart, nibbleCount };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[2]));
            bitCapInt partToSub = bciArgs[1];
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[3]);
            char test1, test2;
            unsigned char j;
            char* nibbles = new char[bciArgs[4]];
            bool isValid = true;
            for (j = 0; j < bciArgs[4]; j++) {
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
                for (j = 0; j < bciArgs[4]; j++) {
                    if (nibbles[j] < 0) {
                        nibbles[j] += 10;
                        if ((unsigned char)(j + 1) < bciArgs[4]) {
                            nibbles[j + 1]--;
                        }
                    }
                    outInt |= nibbles[j] << (j * 4);
                }
                nStateVec[(outInt << (bciArgs[3])) | otherRes] = stateVec[lcv];
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
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    otherMask ^= inOutMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[7] = { inOutMask, toSub, overflowMask, otherMask, lengthPower, inOutStart, signMask };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[2];
            bitCapInt outInt = inOutInt - bciArgs[1] + bciArgs[4];
            bitCapInt outRes;
            if (outInt < bciArgs[4]) {
                outRes = (outInt << (bciArgs[5])) | otherRes;
            } else {
                outRes = ((outInt - bciArgs[4]) << (bciArgs[5])) | otherRes;
            }
            bool isOverflow = false;
            // First negative:
            if (inOutInt & (~inInt) & (bciArgs[6])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > bciArgs[6])
                    isOverflow = true;
            }
            // First positive:
            else if (inOutInt & (~inInt) & (bciArgs[6])) {
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) >= bciArgs[6])
                    isOverflow = true;
            }
            if (isOverflow && ((outRes & bciArgs[2]) == bciArgs[2])) {
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
        toSub++;
    }
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    bitCapInt edgeMask = inOutMask | carryMask;
    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[10] = { inOutMask, toSub, carryMask, otherMask, lengthPower, inOutStart, carryIndex, edgeMask,
        overflowMask, signMask };
    par_for_skip(0, maxQPower, carryMask, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[1];
            bitCapInt outInt = (inOutInt - bciArgs[1]) + (bciArgs[4]);
            bitCapInt outRes;
            if (outInt < (bciArgs[4])) {
                outRes = (outInt << (bciArgs[5])) | otherRes | (bciArgs[2]);
            } else {
                outRes = ((outInt - (bciArgs[4])) << (bciArgs[5])) | otherRes;
            }
            bool isOverflow = false;
            // First negative:
            if (inOutInt & (~inInt) & (bciArgs[9])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > bciArgs[9])
                    isOverflow = true;
            }
            // First positive:
            else if (inOutInt & (~inInt) & (bciArgs[9])) {
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) >= bciArgs[9])
                    isOverflow = true;
            }
            if (isOverflow && ((outRes & bciArgs[8]) == bciArgs[8])) {
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
        toSub++;
    }
    bitCapInt signMask = 1 << (length - 1);
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    bitCapInt edgeMask = inOutMask | carryMask;
    otherMask ^= inOutMask | carryMask;
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    bitCapInt bciArgs[9] = { inOutMask, toSub, carryMask, otherMask, lengthPower, inOutStart, carryIndex, edgeMask,
        signMask };
    par_for_skip(0, maxQPower, carryMask, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[5]);
            bitCapInt inInt = bciArgs[1];
            bitCapInt outInt = (inOutInt - bciArgs[1]) + (bciArgs[4]);
            bitCapInt outRes;
            if (outInt < (bciArgs[4])) {
                outRes = (outInt << (bciArgs[5])) | otherRes | (bciArgs[2]);
            } else {
                outRes = ((outInt - (bciArgs[4])) << (bciArgs[5])) | otherRes;
            }
            bool isOverflow = false;
            // First negative:
            if (inOutInt & (~inInt) & (bciArgs[9])) {
                inOutInt = ((~inOutInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) > bciArgs[9])
                    isOverflow = true;
            }
            // First positive:
            else if (inOutInt & (~inInt) & (bciArgs[9])) {
                inInt = ((~inInt) & (bciArgs[4] - 1)) + 1;
                if ((inOutInt + inInt) >= bciArgs[9])
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
        toSub++;
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
    bitCapInt bciArgs[8] = { inOutMask, toSub, carryMask, otherMask, inOutStart, nibbleCount };
    par_for_skip(0, maxQPower, 1 << carryIndex, 1, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](const bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs,
            Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[3]));
            bitCapInt partToSub = bciArgs[1];
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[4]);
            char test1, test2;
            unsigned char j;
            char* nibbles = new char[bciArgs[5]];
            bool isValid = true;

            test1 = inOutInt & 15;
            test2 = partToSub % 10;
            partToSub /= 10;
            nibbles[0] = test1 - test2;
            if (test1 > 9) {
                isValid = false;
            }

            for (j = 1; j < bciArgs[5]; j++) {
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
                for (j = 0; j < bciArgs[5]; j++) {
                    if (nibbles[j] < 0) {
                        nibbles[j] += 10;
                        if ((unsigned char)(j + 1) < bciArgs[5]) {
                            nibbles[j + 1]--;
                        } else {
                            carryRes = bciArgs[2];
                        }
                    }
                    outInt |= nibbles[j] << (j * 4);
                }
                outRes = (outInt << (bciArgs[4])) | otherRes | carryRes;
                nStateVec[outRes] = stateVec[lcv];
            } else {
                nStateVec[lcv] = stateVec[lcv];
            }
            delete[] nibbles;
        });
    ResetStateVec(std::move(nStateVec));
}

/// Multiply a quantum register by a classical integer, with sign and without carry.
/*
void CoherentUnit::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt length)
{
    if (toMul == 0) {
        SetPermutation(0);
        return;
    }
    else if (toMul == 1) {
        return;
    }

    bitCapInt origPermCount = maxQPower;
    bitCapInt origQubitCount = qubitCount;
    if ((inOutStart + length) != origQubitCount) {
        Swap(inOutStart, origQubitCount - length, length);
    }
    CoherentUnit carry = CoherentUnit(length, 0);
    Cohere(carry);

    bitCapInt lengthPower = 1 << (length * 2);
    bitCapInt inOutMask = (lengthPower - 1) << (origQubitCount - length);
    bitCapInt otherMask = (origPermCount - 1) ^ ((1 << length) - 1);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + (1 << (origQubitCount + length)), Complex16(0.0, 0.0));
    bitCapInt bciArgs[5] = { inOutMask, toMul, otherMask, lengthPower, (origQubitCount - length) };
    par_for_copy(0, origPermCount, &(stateVec[0]), bciArgs, &(nStateVec[0]),
        [](bitCapInt lcv, const int cpu, const Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            bitCapInt otherRes = (lcv & (bciArgs[2]));
            bitCapInt inOutRes = (lcv & (bciArgs[0]));
            bitCapInt inOutInt = inOutRes >> (bciArgs[4]);
            bitCapInt outInt = (inOutInt * bciArgs[1]) & bciArgs[3];
            bitCapInt outRes = (outInt << (bciArgs[4])) | otherRes;
            nStateVec[outRes] += stateVec[lcv];
            if (norm(stateVec[lcv]) > 0.0) {
                std::cout<<(int)lcv<<", "<<(int)outRes<<std::endl;
            }
        });
    ResetStateVec(std::move(nStateVec));

    MReg(origQubitCount, length);
    Dispose(origQubitCount, length);
    if ((inOutStart + length) != origQubitCount) {
        Swap(inOutStart, origQubitCount - length, length);
    }
}
*/

/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
void CoherentUnit::QFT(bitLenInt start, bitLenInt length)
{
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
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt bciArgs[1] = { regMask };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, NULL,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            if ((lcv & (~(bciArgs[0]))) == lcv)
                stateVec[lcv] = -stateVec[lcv];
        });
}

/// For chips with a sign flag, flip the phase of states where the register is negative.
void CoherentUnit::CPhaseFlip(bitLenInt toTest)
{
    bitCapInt testMask = 1 << toTest;
    bitCapInt bciArgs[1] = { testMask };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, NULL,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            if ((lcv & bciArgs[0]) == bciArgs[0])
                stateVec[lcv] = -stateVec[lcv];
        });
}

/// The 6502 uses its carry flag also as a greater-than/less-than flag, for the CMP operation.
void CoherentUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt flagMask = 1 << flagIndex;
    bitCapInt bciArgs[4] = { regMask, flagMask, start, greaterPerm };
    par_for_copy(0, maxQPower, &(stateVec[0]), bciArgs, NULL,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            if ((((lcv & bciArgs[0]) >> (bciArgs[2])) < bciArgs[3]) & ((lcv & bciArgs[1]) == bciArgs[1]))
                stateVec[lcv] = -stateVec[lcv];
        });
}

/// Phase flip always - equivalent to Z X Z X on any bit in the CoherentUnit
void CoherentUnit::PhaseFlip()
{
    par_for_copy(0, maxQPower, &(stateVec[0]), NULL, NULL,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const bitCapInt* bciArgs, Complex16* nStateVec) {
            stateVec[lcv] = -stateVec[lcv];
        });
}

/// Set register bits to given permutation
void CoherentUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    // First, single bit operations are better optimized for this special case:
    if (length == 1) {
        SetBit(start, (value == 1));
        return;
    }

    bool bitVal;
    bitCapInt regVal = MReg(start, length);
    for (bitLenInt i = 0; i < length; i++) {
        bitVal = regVal & (1 << i);
        if ((bitVal && !(value & (1 << i))) || (!bitVal && (value & (1 << i))))
            X(start + i);
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

    bitCapInt resultPtr[1] = { result << start };
    par_for_all(0, maxQPower, &(stateVec[0]), Complex16(cosine, sine) / nrmlzr, NULL, resultPtr,
        [](const bitCapInt lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx,
            const bitCapInt* resultPtr) {
            if ((lcv & (*resultPtr)) == (*resultPtr)) {
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
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubitIndex;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, true, doCalcNorm);
}

void CoherentUnit::ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(qPowers[0], qPowers[1], mtrx, 2, qPowersSorted, false, doCalcNorm);
}

void CoherentUnit::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx, bool doCalcNorm)
{
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[2];
    qPowers[1] = 1 << control;
    qPowers[2] = 1 << target;
    qPowers[0] = qPowers[1] + qPowers[2];
    if (control < target) {
        qPowersSorted[0] = qPowers[1];
        qPowersSorted[1] = qPowers[2];
    } else {
        qPowersSorted[0] = qPowers[2];
        qPowersSorted[1] = qPowers[1];
    }
    Apply2x2(0, qPowers[2], mtrx, 2, qPowersSorted, false, doCalcNorm);
}

void CoherentUnit::NormalizeState()
{
    par_for_mult(0, maxQPower, runningNorm, &(stateVec[0]),
        [](const bitCapInt lcv, const int cpu, const double runningNorm, Complex16* stateVec) {
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
} // namespace Qrack
