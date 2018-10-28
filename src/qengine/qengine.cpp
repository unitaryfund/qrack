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

    bitCapInt skipMask = ~regMask;
    std::vector<bitCapInt> skipPowersVec;
    bitCapInt pwr = 1U;
    for (i = 0; i < (sizeof(bitCapInt) * 8); i++) {
        if (pwr & skipMask) {
            skipPowersVec.push_back(pwr);
        }
        pwr <<= 1U;
    }

    bitCapInt lengthPower = 1 << length;
    real1* probArray = new real1[lengthPower]();
    real1 lowerProb, largestProb;
    real1 nrmlzr = ONE_R1;
    bitCapInt lcv, result;

    bitLenInt p;
    bitCapInt iHigh, iLow;
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
        probArray[lcv] = ProbMask(regMask, i);
    }

    lcv = 0;
    lowerProb = ZERO_R1;
    largestProb = ZERO_R1;
    result = lengthPower - 1;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    while (lcv < lengthPower) {
        if ((probArray[lcv] + lowerProb) > prob) {
            result = lcv;
            nrmlzr = probArray[lcv];
            lcv = lengthPower;
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

    delete[] probArray;

    i = 0;
    for (p = 0; p < length; p++) {
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
} // namespace Qrack
