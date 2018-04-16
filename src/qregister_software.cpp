//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <iostream>

#include "qregister.hpp"

namespace Qrack {

/// "Circular shift left" - shift bits left, and carry last bits.
void CoherentUnit::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(start, length);

    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);

    par_for(0, maxQPower, [&](const bitCapInt lcv) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt regRes = (lcv & (regMask));
        bitCapInt regInt = regRes >> (start);
        bitCapInt outInt = (regInt >> (length - shift)) | ((regInt << (shift)) & (lengthPower - 1));
        nStateVec[(outInt << (start)) + otherRes] = stateVec[lcv];
    });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void CoherentUnit::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(start, length);

    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;

    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);

    par_for(0, maxQPower, [&](const bitCapInt lcv) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt regRes = (lcv & (regMask));
        bitCapInt regInt = regRes >> (start);
        bitCapInt outInt = (regInt >> (shift)) | ((regInt << (length - shift)) & (lengthPower - 1));
        nStateVec[(outInt << (start)) + otherRes] = stateVec[lcv];
    });
    stateVec.reset();
    stateVec = std::move(nStateVec);
}

/// Add integer (without sign, with carry)
void CoherentUnit::INCC(bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        FlushQueue(carryIndex);
        toAdd++;
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(inOutStart, length);

    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;

    otherMask ^= inOutMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt outInt = inOutInt + toAdd;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | (carryMask);
        }
        nStateVec[outRes] = stateVec[lcv];
    });
    ResetStateVec(std::move(nStateVec));
}

/// Subtract integer (without sign, with carry)
void CoherentUnit::DECC(bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
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

    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = ((1 << length) - 1) << inOutStart;
    bitCapInt otherMask = (1 << qubitCount) - 1;

    otherMask ^= inOutMask;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, 1 << carryIndex, 1, [&](const bitCapInt lcv) {
        bitCapInt otherRes = (lcv & (otherMask));
        bitCapInt inOutRes = (lcv & (inOutMask));
        bitCapInt inOutInt = inOutRes >> (inOutStart);
        bitCapInt outInt = (inOutInt + lengthPower) - toSub;
        bitCapInt outRes;
        if (outInt < (lengthPower)) {
            outRes = (outInt << (inOutStart)) | otherRes;
        } else {
            outRes = ((outInt - (lengthPower)) << (inOutStart)) | otherRes | carryMask;
        }
        nStateVec[outRes] = stateVec[lcv];
    });
    ResetStateVec(std::move(nStateVec));
}

/// Set 8 bit register bits based on read from classical memory
unsigned char CoherentUnit::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    //Does not necessarily commute with single bit queues
    FlushQueue(inputStart, 8);

    bitCapInt i, outputInt;
    SetReg(outputStart, 8, 0);

    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt skipPower = 1 << outputStart;

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    par_for_skip(0, maxQPower, skipPower, 8, [&](const bitCapInt lcv) {
        bitCapInt inputRes = lcv & (inputMask);
        bitCapInt inputInt = inputRes >> (inputStart);
        bitCapInt outputInt = values[inputInt];
        bitCapInt outputRes = outputInt << (outputStart);
        nStateVec[outputRes | lcv] = stateVec[lcv];
    });

    double prob, average;

    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> outputStart;
        prob = norm(nStateVec[i]);
        average += prob * outputInt;
    }

    ResetStateVec(std::move(nStateVec));

    return (unsigned char)(average + 0.5);
}

/// Add based on an indexed load from classical memory
unsigned char CoherentUnit::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    // This a quantum/classical interface method, similar to SuperposeReg8.
    // Like SuperposeReg8, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are ADded with Carry (ADC) to values entangled in the "outputStart" register with the "inputStart" register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry has to first to be measured for its input value.
    bitCapInt carryIn = 0;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 1;
        X(carryIndex);
        FlushQueue(carryIndex);
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(outputStart, 8);
    FlushQueue(inputStart, 8);

    // We calloc a new stateVector for output.
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    // We're going to loop over every eigenstate in the vector, (except, we
    // already know the carry is zero).  This bit masks let us quickly
    // distinguish the different values of the input register, output register,
    // carry, and other bits that aren't involved in the operation.
    bitCapInt i, outputInt;
    bitCapInt lengthPower = 1 << 8;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask));
    bitCapInt skipPower = 1 << carryIndex;

    par_for_skip(0, maxQPower, skipPower, 1, [&](const bitCapInt lcv) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & (otherMask);

        // These are bits that index the classical memory we're loading from:
        bitCapInt inputRes = lcv & (inputMask);

        // If we read these as a char type, this is their value as a char:
        bitCapInt inputInt = inputRes >> (inputStart);

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & (outputMask);

        // Maintaining the entanglement, we add the classical input value
        // corresponding with the state of the "inputStart" register to
        // "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = (outputRes >> outputStart) + values[inputInt] + carryIn;

        // If we exceed max char, we subtract 256 and entangle the carry as
        // set.
        bitCapInt carryRes = 0;
        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }
        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << outputStart;

        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    });

    // At the end, just as a convenience, we return the expectation value for
    // the addition result.
    double prob, average;

    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> outputStart;
        prob = norm(nStateVec[i]);
        average += prob * outputInt;
    }

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(std::move(nStateVec));

    // Return the expectation value.
    return (unsigned char)(average + 0.5);
}

/// Subtract based on an indexed load from classical memory
unsigned char CoherentUnit::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    // This a quantum/classical interface method, similar to SuperposeReg8.
    // Like SuperposeReg8, up to a page of classical memory is loaded based on a quantum mechanically coherent offset by
    // the "inputStart" register. Instead of just loading this page superposed into "outputStart," though, its values
    // are SuBtracted with Carry (SBC) from values entangled in the "outputStart" register with the "inputStart"
    // register.

    //"inputStart" and "outputStart" point to the beginning of two quantum registers. The carry qubit is at index
    //"carryIndex." "values" is a page of key-value pairs of classical memory to load based on offset by the
    //"inputStart" register.

    // The carry (or "borrow") has to first to be measured for its input value.
    bitCapInt carryIn = 1;
    if (M(carryIndex)) {
        // If the carry is set, we borrow 1 going in. We always initially clear the carry after testing for borrow in.
        carryIn = 0;
        X(carryIndex);
        FlushQueue(carryIndex);
    }

    //Does not necessarily commute with single bit queues
    FlushQueue(outputStart, 8);
    FlushQueue(inputStart, 8);

    // We calloc a new stateVector for output.
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));

    // We're going to loop over every eigenstate in the vector, (except, we already know the carry is zero).
    // This bit masks let us quickly distinguish the different values of the input register, output register, carry, and
    // other bits that aren't involved in the operation.
    bitCapInt i, outputInt;
    bitCapInt lengthPower = 1 << 8;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask));
    bitCapInt skipPower = 1 << carryIndex;

    par_for_skip(0, maxQPower, skipPower, 1, [&](const bitCapInt lcv) {
        // These are qubits that are not directly involved in the
        // operation. We iterate over all of their possibilities, but their
        // input value matches their output value:
        bitCapInt otherRes = lcv & (otherMask);

        // These are bits that index the classical memory we're loading
        // from:
        bitCapInt inputRes = lcv & (inputMask);

        // If we read these as a char type, this is their value as a char:
        bitCapInt inputInt = inputRes >> (inputStart);

        // This is the initial value that's entangled with the "inputStart"
        // register in "outputStart."
        bitCapInt outputRes = lcv & (outputMask);

        // Maintaining the entanglement, we subtract the classical input
        // value corresponding with the state of the "inputStart" register
        // from "outputStart" register value its entangled with in this
        // iteration of the loop.
        bitCapInt outputInt = ((outputRes >> outputStart) + lengthPower) - (values[inputInt] + carryIn);

        // If our subtractions results in less than 0, we add 256 and
        // entangle the carry as set.  (Since we're using unsigned types,
        // we start by adding 256 with the carry, and then subtract 256 and
        // clear the carry if we don't have a borrow-out.)
        bitCapInt carryRes = 0;

        if (outputInt >= lengthPower) {
            outputInt -= lengthPower;
            carryRes = carryMask;
        }

        // We shift the output integer back to correspondence with its
        // register bits, and entangle it with the input and carry, and
        // shunt the uninvoled "other" bits from input to output.
        outputRes = outputInt << outputStart;

        nStateVec[outputRes | inputRes | otherRes | carryRes] = stateVec[lcv];
    });

    double prob, average;

    // At the end, just as a convenience, we return the expectation value for
    // the subtraction result.
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> outputStart;
        prob = norm(nStateVec[i]);
        average += prob * outputInt;
    }

    // Finally, we dealloc the old state vector and replace it with the one we
    // just calculated.
    ResetStateVec(std::move(nStateVec));

    // Return the expectation value.
    return (unsigned char)(average + 0.5);
}

// Private CoherentUnit methods
void CoherentUnit::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, const bool doCalcNorm)
{
    Complex16 nrm = Complex16((bitCount == 1) ? (1.0 / runningNorm) : 1.0, 0.0);
    if (doCalcNorm) {
        runningNorm = 0.0;

        par_for_mask(0, maxQPower, qPowersSorted, bitCount, [&](const bitCapInt lcv) {
            Complex16 qubit[2];

            qubit[0] = stateVec[lcv + offset1];
            qubit[1] = stateVec[lcv + offset2];

            Complex16 Y0 = qubit[0];
            qubit[0] = nrm * ((mtrx[0] * Y0) + (mtrx[1] * qubit[1]));
            qubit[1] = nrm * ((mtrx[2] * Y0) + (mtrx[3] * qubit[1]));
            runningNorm += norm(qubit[0]) + norm(qubit[1]);

            stateVec[lcv + offset1] = qubit[0];
            stateVec[lcv + offset2] = qubit[1];
        });
    }
    else {
        runningNorm = 1.0;

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
    }
}
} // namespace Qrack
