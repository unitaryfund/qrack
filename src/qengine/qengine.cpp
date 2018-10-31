//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qengine.hpp"

namespace Qrack {

bool does2x2PhaseShift(const complex* mtrx)
{
    bool doesShift = false;
    real1 phase = -M_PI * 2;
    for (int i = 0; i < 4; i++) {
        if (norm(mtrx[i]) > ZERO_R1) {
            if (phase < -M_PI) {
                phase = arg(mtrx[i]);
                continue;
            }

            real1 diff = arg(mtrx[i]) - phase;
            if (diff < ZERO_R1) {
                diff = -diff;
            }
            if (diff > M_PI) {
                diff = (2 * M_PI) - diff;
            }
            if (diff > min_norm) {
                doesShift = true;
                break;
            }
        }
    }
    return doesShift;
}

void QEngine::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapInt powerTest = result ? qPower : 0;
    ApplyM(qPower, powerTest, nrm);
}

/// PSEUDO-QUANTUM - Acts like a measurement gate, except with a specified forced result.
bool QEngine::ForceM(bitLenInt qubit, bool result, bool doForce, real1 nrmlzr)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    if (!doForce) {
        real1 prob = Rand();
        real1 oneChance = Prob(qubit);
        result = ((prob < oneChance) && (oneChance > ZERO_R1));
        nrmlzr = ONE_R1;
        if (result) {
            nrmlzr = oneChance;
        } else {
            nrmlzr = ONE_R1 - oneChance;
        }
    }
    if (nrmlzr > min_norm) {
        bitCapInt qPower = 1 << qubit;
        real1 angle = Rand() * 2 * M_PI;
        ApplyM(qPower, result, complex(cos(angle), sin(angle)) / (real1)(sqrt(nrmlzr)));
    } else {
        NormalizeState(ZERO_R1);
    }

    return result;
}

/// Measurement gate
bool QEngine::M(bitLenInt qubit)
{
    // Measurement introduces an overall phase shift. Since it is applied to every state, this will not change the
    // status of our cached knowledge of phase separability. However, measurement could set some amplitudes to zero,
    // meaning the relative amplitude phases might only become separable in the process if they are not already.
    if (knowIsPhaseSeparable && (!isPhaseSeparable)) {
        knowIsPhaseSeparable = false;
    }
    return ForceM(qubit, false, false);
}

/// Measure permutation state of a register
bitCapInt QEngine::M(const bitLenInt* bits, const bitLenInt& length)
{
    // Measurement introduces an overall phase shift. Since it is applied to every state, this will not change the
    // status of our cached knowledge of phase separability. However, measurement could set some amplitudes to zero,
    // meaning the relative amplitude phases might only become separable in the process if they are not already.
    if (knowIsPhaseSeparable && (!isPhaseSeparable)) {
        knowIsPhaseSeparable = false;
    }

    // Single bit operations are better optimized for this special case:
    if (length == 1) {
        if (M(bits[0])) {
            return 1;
        } else {
            return 0;
        }
    }

    if (runningNorm != ONE_R1) {
        NormalizeState();
    }

    bitCapInt i;

    real1 prob = Rand();
    real1 angle = Rand() * 2.0 * M_PI;
    real1 cosine = cos(angle);
    real1 sine = sin(angle);

    bitCapInt* qPowers = new bitCapInt[length];
    bitCapInt regMask = 0;
    for (i = 0; i < length; i++) {
        qPowers[i] = 1 << bits[i];
        regMask |= qPowers[i];
    }
    std::sort(qPowers, qPowers + length);

    bitCapInt lengthPower = 1 << length;
    real1* probArray = new real1[lengthPower]();
    real1 nrmlzr = ONE_R1;
    bitCapInt lcv, result;

    ProbMaskAll(regMask, probArray);

    lcv = 0;
    real1 lowerProb = ZERO_R1;
    real1 largestProb = ZERO_R1;
    result = lengthPower - 1;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    while ((lowerProb < prob) && (lcv < lengthPower)) {
        lowerProb += probArray[lcv];
        if (largestProb <= probArray[lcv]) {
            largestProb = probArray[lcv];
            nrmlzr = largestProb;
            result = lcv;
        }
        lcv++;
    }
    if (lcv < lengthPower) {
        lcv--;
        result = lcv;
        nrmlzr = probArray[lcv];
    }

    delete[] probArray;

    i = 0;
    for (bitLenInt p = 0; p < length; p++) {
        if ((1 << p) & result) {
            i |= qPowers[p];
        }
    }
    result = i;

    delete[] qPowers;

    complex nrm = complex(cosine, sine) / (real1)(sqrt(nrmlzr));

    ApplyM(regMask, result, nrm);

    return result;
}

void QEngine::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QEngine::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyControlled2x2(controls, controlLen, target, mtrx, false);
}

void QEngine::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    ApplyAntiControlled2x2(controls, controlLen, target, mtrx, false);
}

void QEngine::ApplyControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    bitCapInt fullMask = 0;
    bitCapInt controlMask;
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
        fullMask |= qPowers[i];
    }
    controlMask = fullMask;
    qPowers[controlLen] = 1 << target;
    fullMask |= qPowers[controlLen];
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(controlMask, fullMask, mtrx, controlLen + 1, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

void QEngine::ApplyControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowers[0], (qPowers[0]) | (qPowers[1]), mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngine::ApplyAntiControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
    }
    qPowers[controlLen] = 1 << target;
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(0, qPowers[controlLen], mtrx, controlLen + 1, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

void QEngine::ApplyAntiControlled2x2(bitLenInt control, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << control;
    qPowers[1] = 1 << target;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(0, qPowers[1], mtrx, 2, qPowersSorted, doCalcNorm);
}

void QEngine::ApplyDoublyControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[3];
    qPowers[0] = 1 << control1;
    qPowers[1] = 1 << control2;
    qPowers[2] = 1 << target;
    std::copy(qPowers, qPowers + 3, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(
        (qPowers[0]) | (qPowers[1]), (qPowers[0]) | (qPowers[1]) | (qPowers[2]), mtrx, 3, qPowersSorted, doCalcNorm);
}

void QEngine::ApplyDoublyAntiControlled2x2(
    bitLenInt control1, bitLenInt control2, bitLenInt target, const complex* mtrx, bool doCalcNorm)
{
    if (does2x2PhaseShift(mtrx)) {
        knowIsPhaseSeparable = false;
    }
    bitCapInt qPowers[3];
    bitCapInt qPowersSorted[3];
    qPowers[0] = 1 << control1;
    qPowers[1] = 1 << control2;
    qPowers[2] = 1 << target;
    std::copy(qPowers, qPowers + 3, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 3);
    Apply2x2(0, qPowers[2], mtrx, 3, qPowersSorted, doCalcNorm);
}

/// Swap values of two bits in register
void QEngine::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << qubit1;
    qPowers[1] = 1 << qubit2;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], pauliX, 2, qPowersSorted, false);
}

void QEngine::ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray)
{
    bitCapInt lengthPower = 1U << length;
    for (bitCapInt lcv = 0; lcv < lengthPower; lcv++) {
        probsArray[lcv] = ProbReg(start, length, lcv);
    }
}

void QEngine::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length;
    std::vector<bitCapInt> powersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
    }

    v = ~mask; // count the number of bits set in v
    bitCapInt power;
    bitLenInt len; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (len = 0; v; len++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
        power = (v ^ oldV) & oldV;
        if (power < mask) {
            skipPowersVec.push_back(power);
        } else {
            v = 0;
        }
    }

    bitCapInt lengthPower = 1 << length;
    bitCapInt lcv;

    bitLenInt p;
    bitCapInt i, iHigh, iLow;
    for (lcv = 0; lcv < lengthPower; lcv++) {
        iHigh = lcv;
        i = 0;
        for (p = 0; p < (skipPowersVec.size()); p++) {
            iLow = iHigh & (skipPowersVec[p] - 1);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1;
            if (iHigh == 0) {
                break;
            }
        }
        i |= iHigh;
        probsArray[lcv] = ProbMask(mask, i);
    }
}

/// Measure permutation state of a register
bitCapInt QEngine::MReg(bitLenInt start, bitLenInt length)
{
    // Measurement introduces an overall phase shift. Since it is applied to every state, this will not change the
    // status of our cached knowledge of phase separability. However, measurement could set some amplitudes to zero,
    // meaning the relative amplitude phases might only become separable in the process if they are not already.
    if (knowIsPhaseSeparable && (!isPhaseSeparable)) {
        knowIsPhaseSeparable = false;
    }

    // Single bit operations are better optimized for this special case:
    if (length == 1) {
        if (M(start)) {
            return 1;
        } else {
            return 0;
        }
    }

    if (runningNorm != ONE_R1) {
        NormalizeState();
    }

    real1 prob = Rand();
    real1 angle = Rand() * 2.0 * M_PI;
    real1 cosine = cos(angle);
    real1 sine = sin(angle);
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    real1* probArray = new real1[lengthPower]();
    real1 nrmlzr = ONE_R1;
    bitCapInt lcv, result;

    ProbRegAll(start, length, probArray);

    lcv = 0;
    real1 lowerProb = ZERO_R1;
    real1 largestProb = ZERO_R1;
    result = lengthPower - 1;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    while ((lowerProb < prob) && (lcv < lengthPower)) {
        lowerProb += probArray[lcv];
        if (largestProb <= probArray[lcv]) {
            largestProb = probArray[lcv];
            nrmlzr = largestProb;
            result = lcv;
        }
        lcv++;
    }
    if (lcv < lengthPower) {
        lcv--;
        result = lcv;
        nrmlzr = probArray[lcv];
    }

    delete[] probArray;

    bitCapInt resultPtr = result << start;
    complex nrm = complex(cosine, sine) / (real1)(sqrt(nrmlzr));

    ApplyM(regMask, resultPtr, nrm);

    return result;
}

} // namespace Qrack
