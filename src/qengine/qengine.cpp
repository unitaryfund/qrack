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

#include "qengine.hpp"

namespace Qrack {

/// PSEUDO-QUANTUM - Acts like a measurement gate, except with a specified forced result.
bool QEngine::ForceM(bitLenInt qubit, bool result, bool doForce, bool doApply)
{
    if (doNormalize) {
        NormalizeState();
    }

    real1 oneChance = Prob(qubit);
    if (!doForce) {
        if (oneChance >= ONE_R1) {
            result = true;
        } else if (oneChance <= ZERO_R1) {
            result = false;
        } else {
            real1 prob = Rand();
            result = (prob <= oneChance);
        }
    }

    real1 nrmlzr;
    if (result) {
        nrmlzr = oneChance;
    } else {
        nrmlzr = ONE_R1 - oneChance;
    }

    if (nrmlzr <= ZERO_R1) {
        throw "ERROR: Forced a measurement result with 0 probability";
    }

    if (doApply && (nrmlzr != ONE_R1)) {
        bitCapInt qPower = pow2(qubit);
        ApplyM(qPower, result, GetNonunitaryPhase() / (real1)(std::sqrt(nrmlzr)));
    }

    return result;
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values, bool doApply)
{
    // Single bit operations are better optimized for this special case:
    if (length == 1U) {
        if (values == NULL) {
            if (M(bits[0])) {
                return (pow2(bits[0]));
            } else {
                return 0U;
            }
        } else {
            if (ForceM(bits[0], values[0])) {
                return (pow2(bits[0]));
            } else {
                return 0U;
            }
        }
    }

    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl i;

    complex phase = GetNonunitaryPhase();

    bitCapInt* qPowers = new bitCapInt[length];
    bitCapInt regMask = 0;
    for (i = 0; i < length; i++) {
        qPowers[i] = pow2(bits[i]);
        regMask |= qPowers[i];
    }
    std::sort(qPowers, qPowers + length);

    bitCapIntOcl lengthPower = pow2Ocl(length);
    real1 nrmlzr = ONE_R1;
    bitCapIntOcl lcv;
    bitCapInt result;
    complex nrm;

    if (values != NULL) {
        result = 0;
        for (bitLenInt j = 0; j < length; j++) {
            result |= values[j] ? pow2(bits[j]) : 0;
        }
        nrmlzr = ProbMask(regMask, result);
        nrm = phase / (real1)(std::sqrt(nrmlzr));
        if (nrmlzr != ONE_R1) {
            ApplyM(regMask, result, nrm);
        }

        // No need to check against probabilities:
        return result;
    }

    real1 prob = Rand();
    real1* probArray = new real1[lengthPower]();

    ProbMaskAll(regMask, probArray);

    lcv = 0;
    real1 lowerProb = ZERO_R1;
    real1 largestProb = ZERO_R1;
    result = lengthPower - ONE_BCI;

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
        if (lcv > 0) {
            lcv--;
        }
        result = lcv;
        nrmlzr = probArray[lcv];
    }

    delete[] probArray;

    i = 0;
    for (bitLenInt p = 0; p < length; p++) {
        if (pow2(p) & result) {
            i |= (bitCapIntOcl)qPowers[p];
        }
    }
    result = i;

    delete[] qPowers;

    nrm = phase / (real1)(std::sqrt(nrmlzr));

    if (doApply && (nrmlzr != ONE_R1)) {
        ApplyM(regMask, result, nrm);
    }

    return result;
}

void QEngine::ApplySingleBit(const complex* mtrx, bitLenInt qubit)
{
    if (IsIdentity(mtrx)) {
        return;
    }

    bool doCalcNorm = doNormalize &&
        !(((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) || ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)));

    bitCapInt qPowers[1];
    qPowers[0] = pow2(qubit);
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QEngine::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IsIdentity(mtrx)) {
        return;
    }

    if (controlLen == 0) {
        ApplySingleBit(mtrx, target);
        return;
    }

    bool doCalcNorm = doNormalize &&
        !(((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) || ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)));

    ApplyControlled2x2(controls, controlLen, target, mtrx);
    if (doCalcNorm) {
        UpdateRunningNorm();
    }
}

void QEngine::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (IsIdentity(mtrx)) {
        return;
    }

    if (controlLen == 0) {
        ApplySingleBit(mtrx, target);
        return;
    }

    bool doCalcNorm = doNormalize &&
        !(((norm(mtrx[1]) == 0) && (norm(mtrx[2]) == 0)) || ((norm(mtrx[0]) == 0) && (norm(mtrx[3]) == 0)));

    ApplyAntiControlled2x2(controls, controlLen, target, mtrx);
    if (doCalcNorm) {
        UpdateRunningNorm();
    }
}

void QEngine::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | pow2(qubit1), skipMask | pow2(qubit2), pauliX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(pow2(qubit1), pow2(qubit2), pauliX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | pow2(qubit1), skipMask | pow2(qubit2), sqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(pow2(qubit1), pow2(qubit2), sqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | pow2(qubit1), skipMask | pow2(qubit2), iSqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
    }
    qPowersSorted[controlLen] = pow2(qubit1);
    qPowersSorted[controlLen + 1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(pow2(qubit1), pow2(qubit2), iSqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::ApplyControlled2x2(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1U];
    bitCapInt fullMask = 0U;
    bitCapInt controlMask;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
        fullMask |= qPowersSorted[i];
    }
    controlMask = fullMask;
    qPowersSorted[controlLen] = pow2(target);
    fullMask |= pow2(target);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1U);
    Apply2x2(controlMask, fullMask, mtrx, controlLen + 1U, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::ApplyAntiControlled2x2(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1U];
    for (bitLenInt i = 0U; i < controlLen; i++) {
        qPowersSorted[i] = pow2(controls[i]);
    }
    qPowersSorted[controlLen] = pow2(target);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1U);
    Apply2x2(0U, pow2(target), mtrx, controlLen + 1U, qPowersSorted, false);
    delete[] qPowersSorted;
}

/// Swap values of two bits in register
void QEngine::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt qPowersSorted[2];
    qPowersSorted[0] = pow2(qubit1);
    qPowersSorted[1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], pauliX, 2, qPowersSorted, false);
}

/// Swap values of two bits in register, applying a phase factor of i if bits are different
void QEngine::ISwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ONE_R1), complex(ZERO_R1, ONE_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt qPowersSorted[2];
    qPowersSorted[0] = pow2(qubit1);
    qPowersSorted[1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], pauliX, 2, qPowersSorted, false);
}

/// Square root of swap gate
void QEngine::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt qPowersSorted[2];
    qPowersSorted[0] = pow2(qubit1);
    qPowersSorted[1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], sqrtX, 2, qPowersSorted, false);
}

/// Inverse of square root of swap gate
void QEngine::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt qPowersSorted[2];
    qPowersSorted[0] = pow2(qubit1);
    qPowersSorted[1] = pow2(qubit2);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], iSqrtX, 2, qPowersSorted, false);
}

/// "fSim" gate, (useful in the simulation of particles with fermionic statistics)
void QEngine::FSim(real1_f theta, real1_f phi, bitLenInt qubit1, bitLenInt qubit2)
{
    real1 cosTheta = cos(theta);
    real1 sinTheta = sin(theta);

    if (cosTheta != ONE_R1) {
        const complex fSimSwap[4] = { complex(cosTheta, ZERO_R1), complex(ZERO_R1, sinTheta),
            complex(ZERO_R1, sinTheta), complex(cosTheta, ZERO_R1) };
        bitCapInt qPowersSorted[2];
        qPowersSorted[0] = pow2(qubit1);
        qPowersSorted[1] = pow2(qubit2);
        std::sort(qPowersSorted, qPowersSorted + 2);
        Apply2x2(qPowersSorted[0], qPowersSorted[1], fSimSwap, 2, qPowersSorted, false);
    }

    if (phi == ZERO_R1) {
        return;
    }

    bitLenInt controls[1] = { qubit1 };
    ApplyControlledSinglePhase(controls, 1, qubit2, ONE_CMPLX, exp(complex(ZERO_R1, phi)));
}

void QEngine::ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray)
{
    bitCapIntOcl lengthPower = pow2Ocl(length);
    for (bitCapIntOcl lcv = 0; lcv < lengthPower; lcv++) {
        probsArray[lcv] = ProbReg(start, length, lcv);
    }
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce, bool doApply)
{
    // Single bit operations are better optimized for this special case:
    if (length == 1U) {
        if (ForceM(start, ((bitCapIntOcl)result) & ONE_BCI, doForce)) {
            return ONE_BCI;
        } else {
            return 0;
        }
    }

    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapInt regMask = (lengthPower - ONE_BCI) << (bitCapIntOcl)start;
    real1 nrmlzr = ONE_BCI;

    if (doForce) {
        nrmlzr = ProbMask(regMask, result << (bitCapIntOcl)start);
    } else {
        bitCapIntOcl lcv = 0;
        real1* probArray = new real1[lengthPower]();
        ProbRegAll(start, length, probArray);

        real1 prob = Rand();
        real1 lowerProb = ZERO_R1;
        real1 largestProb = ZERO_R1;
        result = lengthPower - ONE_BCI;

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
    }

    bitCapInt resultPtr = result << (bitCapIntOcl)start;
    complex nrm = GetNonunitaryPhase() / (real1)(std::sqrt(nrmlzr));

    if (doApply) {
        ApplyM(regMask, resultPtr, nrm);
    }

    return result;
}

/// Add integer (without sign, with carry)
void QEngine::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECC(toAdd, inOutStart, length, carryIndex);
}

/// Subtract integer (without sign, with carry)
void QEngine::DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECC(invToSub, inOutStart, length, carryIndex);
}

/**
 * Add an integer to the register, with sign and with carry. Flip phase on overflow. Because the register length is an
 * arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QEngine::INCSC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and with carry. Flip phase on overflow. Because the register length
 * is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is
 * specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the
 * cast.
 */
void QEngine::DECSC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECSC(invToSub, inOutStart, length, carryIndex);
}

/**
 * Add an integer to the register, with sign and with carry. If the overflow is set, flip phase on overflow. Because the
 * register length is an arbitrary number of bits, the sign bit position on the integer to add is variable. Hence, the
 * integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the appropriate
 * position before the cast.
 */
void QEngine::INCSC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECSC(toAdd, inOutStart, length, overflowIndex, carryIndex);
}

/**
 * Subtract an integer from the register, with sign and with carry. If the overflow is set, flip phase on overflow.
 * Because the register length is an arbitrary number of bits, the sign bit position on the integer to add is variable.
 * Hence, the integer to add is specified as cast to an unsigned format, with the sign bit assumed to be set at the
 * appropriate position before the cast.
 */
void QEngine::DECSC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt invToSub = pow2(length) - toSub;
    INCDECSC(invToSub, inOutStart, length, overflowIndex, carryIndex);
}

/// Add BCD integer (without sign, with carry)
void QEngine::INCBCDC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INCDECBCDC(toAdd, inOutStart, length, carryIndex);
}

/// Subtract BCD integer (without sign, with carry)
void QEngine::DECBCDC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    bitCapInt maxVal = intPow(10U, length / 4U);
    toSub %= maxVal;
    bitCapInt invToSub = maxVal - toSub;
    INCDECBCDC(invToSub, inOutStart, length, carryIndex);
}

} // namespace Qrack
