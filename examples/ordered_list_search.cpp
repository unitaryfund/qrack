//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This example demonstrates Grover's search, applied for the purpose of finding a value in a ordered list. (This relies
// on the IndexedLDA/IndexedADC/IndexedSBC methods of Qrack. IndexedADC and IndexedSBC can be shown to be unitary
// operations, while IndexedLDA is unitary up to the requirement that the "value register" is set to zero before
// applying the operation.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iomanip> // For setw
#include <iostream> // For cout

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

// We align our "classical" cache to 64 bit boundaries, for optimal OpenCL performance.
#define ALIGN_SIZE 64
unsigned char* cl_alloc(size_t ucharCount)
{
#ifdef __APPLE__
    void* toRet;
    posix_memalign(&toRet, ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
    return (unsigned char*)toRet;
#else
    return (unsigned char*)aligned_alloc(ALIGN_SIZE,
        ((sizeof(unsigned char) * ucharCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(unsigned char) * ucharCount));
#endif
}

int main()
{
    // *** (Variant of) Grover's search to find a value in an ordered list. ***
    // The oracle is made with integer subtraction/addition and a doubly controlled phase flip.

    // Grover's algorithm ideally returns a 100% probable correct result in the case of choosing one of four distinct
    // items that match our search criteria. With an ordered list, we can leverage this for a "quaternary" search akin
    // to a classical "binary search." (The complexity order of this search does not outperform that of the binary
    // search, but this program is still an example of a simple quantum program.)

    // At each step of the search, we select the quadrant with bounds that could contain our value. In an ideal
    // noiseless quantum computer, this search should be deterministic.

    int i, j;
    bitLenInt partStart;
    bitLenInt partLength;

    // Both CPU and GPU types share the QInterface API.
#if ENABLE_OPENCL
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_OPENCL, 20, 0);
#else
    QInterfacePtr qReg = CreateQuantumInterface(QINTERFACE_CPU, 20, 0);
#endif

    const bitLenInt indexLength = 6;
    const bitLenInt valueLength = 6;
    const bitLenInt carryIndex = 19;
    const int TARGET_VALUE = 6;
    const int TARGET_KEY = 5;

    bool foundPerm = false;

    unsigned char* toLoad = cl_alloc(1 << indexLength);
    for (i = 0; i < TARGET_KEY; i++) {
        toLoad[i] = 2;
    }
    toLoad[TARGET_KEY] = TARGET_VALUE;
    for (i = (TARGET_KEY + 1); i < (1 << indexLength); i++) {
        toLoad[i] = 7;
    }

    qReg->SetPermutation(0);
    partLength = indexLength;

    for (i = 0; i < (indexLength / 2); i++) {
        // We're in an exact permutation basis state, at this point after every iteration of the loop, unless more than
        // one quadrant contained a match for our search target on the previous iteration. We can check the quadrant
        // boundaries, without disturbing the state. If there was more than one match, we either collapse into a valid
        // state, so that we can continue as expected for one matching quadrant, or we collapse into an identifiably
        // invalid set of bounds that cannot contain our match, which can be identified by checking two values and
        // proceeding with special case logic.

        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt unfixedMask = (1 << unfixedLength) - 1;
        bitCapInt key = (qReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);

        // (We could either manipulate the quantum bits directly to check the bounds, or rely on auxiliary classical
        // computing components, as need and efficiency dictate).
        bitCapInt lowBound = toLoad[key];
        bitCapInt highBound = toLoad[key | unfixedMask];

        if (lowBound == TARGET_VALUE) {
            // We've found our match, and the key register already contains the correct value.
            std::cout << "Is low bound";
            foundPerm = true;
            break;
        } else if (highBound == TARGET_VALUE) {
            // We've found our match, but our key register points to the opposite bound.
            std::cout << "Is high bound";
            qReg->X(2 * valueLength, partLength);
            foundPerm = true;
            break;
        } else if (((lowBound < TARGET_VALUE) && (highBound < TARGET_VALUE)) ||
            ((lowBound > TARGET_VALUE) && (highBound > TARGET_VALUE))) {
            // If we measure the key as a quadrant that doesn't contain our value, then either there is more than one
            // quadrant with bounds that match our target value, or there is no match to our target in the list.
            foundPerm = false;
            break;
        }

        // Prepare partial index superposition, of two most significant qubits that have not yet been fixed:
        partLength = indexLength - ((i + 1) * 2);
        partStart = (2 * valueLength) + partLength;
        qReg->H(partStart, 2);

        // Load lower bound of quadrants:
        qReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);

        if (partLength > 0) {
            // In this branch, our quadrant is "degenerate," (we mean, having more than one key/value pair).

            // Load upper bound of quadrants:
            qReg->X(2 * valueLength, partLength);
            qReg->IndexedADC(2 * valueLength, indexLength, valueLength, valueLength - 1, carryIndex, toLoad);

            // This begins the "oracle." Our "oracle" is true if the target can be within the bounds of this quadrant,
            // and false otherwise: Set value bits to borrow from:
            qReg->X(valueLength - 1);
            qReg->X(2 * valueLength - 1);
            // Subtract from the value registers with the bits to borrow from:
            qReg->DEC(TARGET_VALUE, 0, valueLength);
            qReg->DEC(TARGET_VALUE, valueLength, valueLength);
            // If both are higher, this is not the quadrant, and neither flips the borrow.
            // If both are lower, this is not the quadrant, and both flip the borrow.
            // If one is higher and one is lower, the low register borrow bit is flipped, and high register borrow is
            // not.
            qReg->X(valueLength - 1);
            qReg->CCNOT(valueLength - 1, 2 * valueLength - 1, carryIndex);
            // Flip the phase is the test bit is set:
            qReg->Z(carryIndex);
            // Reverse everything but the phase flip:
            qReg->CCNOT(valueLength - 1, 2 * valueLength - 1, carryIndex);
            qReg->X(valueLength - 1);
            qReg->INC(TARGET_VALUE, valueLength, valueLength);
            qReg->INC(TARGET_VALUE, 0, valueLength);
            qReg->X(2 * valueLength - 1);
            qReg->X(valueLength - 1);
            // This ends the "oracle."
        } else {
            // In this branch, we have one key/value pair in each quadrant, so we can use our usual Grover's oracle.

            // We map from input to output.
            qReg->DEC(TARGET_VALUE, 0, valueLength - 1);
            // Phase flip the target state.
            qReg->ZeroPhaseFlip(0, valueLength - 1);
            // We map back from outputs to inputs.
            qReg->INC(TARGET_VALUE, 0, valueLength - 1);
        }

        // Now, we flip the phase of the input state:

        // Reverse the operations we used to construct the state:
        qReg->X(carryIndex);
        if (partLength > 0) {
            qReg->IndexedSBC(2 * valueLength, indexLength, valueLength, valueLength - 1, carryIndex, toLoad);
            qReg->X(2 * valueLength, partLength);
        }
        qReg->IndexedSBC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);
        qReg->X(carryIndex);
        qReg->H(partStart, 2);

        // Flip the phase of the input state at the beginning of the iteration. Only in a quaternary Grover's search,
        // we have an exact result at the end of each Grover's iteration, so we consider this an exact input for the
        // next iteration. (See the beginning of the loop, for what happens if we have more than one matching quadrant.
        qReg->ZeroPhaseFlip(partStart, 2);
        qReg->H(partStart, 2);
        // qReg->PhaseFlip();

        std::cout << "Partial search result key: ";
        for (j = indexLength - 1; j >= 0; j--) {
            if (key & (1U << j)) {
                std::cout << "1";
            } else {
                std::cout << "0";
            }
        }
        std::cout << std::endl;
    }

    if (!foundPerm && (i == (indexLength / 2))) {
        // Here, we hit the maximum iterations, but there might be no match in the array, or there might be more than
        // one match.
        bitCapInt key = qReg->MReg(2 * valueLength, indexLength);
        if (toLoad[key] == TARGET_VALUE) {
            foundPerm = true;
        }
    }
    if (!foundPerm && (i > 0)) {
        // If we measured an invalid value in fewer than the full iterations, or if we returned an invalid value on the
        // last iteration, we back the index up one iteration, 2 index qubits. We check the 8 boundary values. If we
        // have more than one match in the ordered list, one of our 8 boundary values is necessarily a match, since the
        // match repetitions must cross the boundary between two quadrants. If none of the 8 match, a match necessarily
        // does not exist in the ordered list.
        // This can only happen on the first iteration if the single highest and lowest values in the list cannot bound
        // the match, in which case we know a match does not exist in the list.
        bitLenInt fixedLength = i * 2;
        bitLenInt unfixedLength = indexLength - fixedLength;
        bitCapInt fixedLengthMask = ((1 << fixedLength) - 1) << unfixedLength;
        bitCapInt checkIncrement = 1 << (unfixedLength - 2);
        bitCapInt key = (qReg->MReg(2 * valueLength, indexLength)) & (fixedLengthMask);
        for (i = 0; i < 4; i++) {
            // (We could either manipulate the quantum bits directly to check this, or rely on auxiliary classical
            // computing components, as need and efficiency dictate).
            if (toLoad[key | (i * checkIncrement)] == TARGET_VALUE) {
                foundPerm = true;
                qReg->SetReg(2 * valueLength, indexLength, key | (i * checkIncrement));
                break;
            }
        }
    }

    if (!foundPerm) {
        std::cout << "Value is not in array.";
    } else {
        qReg->IndexedADC(2 * valueLength, indexLength, 0, valueLength - 1, carryIndex, toLoad);
        // (If we have more than one match, this REQUIRE_THAT needs to instead check that any of the matches are
        // returned. This could be done by only requiring a match to the value register, but we want to show here that
        // the index is correct.)
    }
    free(toLoad);

    std::cout << "Full index/value pair:";
    bitCapInt endState = qReg->MReg(0, 20);
    for (j = 19; j >= 0; j--) {
        if (endState & (1U << j)) {
            std::cout << "1";
        } else {
            std::cout << "0";
        }
    }
    std::cout << std::endl;
}
