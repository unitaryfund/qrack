//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qinterface.hpp"

#if !ENABLE_ALU
#error ALU has not been enabled
#endif

namespace Qrack {

// Arithmetic:

/** Add integer (without sign) */
void QInterface::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        if (bi_and_1(toAdd) != 0) {
            X(start);
        }
        return;
    }

    std::vector<bitLenInt> bits(length);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            MACInvert(std::vector<bitLenInt>(bits.begin() + i, bits.begin() + i + j + 1U), ONE_CMPLX, ONE_CMPLX,
                start + ((i + j + 1U) % length));
        }
    }
}

/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt start, bitLenInt length)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INC(invToSub, start, length);
}

void QInterface::INCDECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (!length) {
        return;
    }

    std::vector<bitLenInt> bits(length + 1U);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }
    bits[length] = carryIndex;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        X(start + i);
        for (bitLenInt j = 0U; j < (length - i); ++j) {
            const bitLenInt target = start + (((i + j + 1U) == length) ? carryIndex : ((i + j + 1U) % length));
            MACInvert(
                std::vector<bitLenInt>(bits.begin() + i, bits.begin() + i + j + 1U), ONE_CMPLX, ONE_CMPLX, target);
        }
    }
}

void QInterface::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        bi_increment(&toAdd, 1U);
    }

    INCDECC(toAdd, start, length, carryIndex);
}

/// Subtract integer (without sign, with carry)
void QInterface::DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    const bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        bi_increment(&toSub, 1U);
    }

    const bitCapInt invToSub = pow2(length) - toSub;
    INCDECC(invToSub, start, length, carryIndex);
}

/** Add integer (without sign, with controls) */
void QInterface::CINC(bitCapInt toAdd, bitLenInt start, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    if (!controls.size()) {
        INC(toAdd, start, length);
        return;
    }

    if (!length) {
        return;
    }

    if (length == 1U) {
        if (bi_and_1(toAdd) != 0) {
            MCInvert(controls, ONE_CMPLX, ONE_CMPLX, start);
        }
        return;
    }

    for (const bitLenInt& control : controls) {
        X(control);
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (bi_and_1(toAdd >> i) == 0) {
            continue;
        }
        MACInvert(controls, ONE_CMPLX, ONE_CMPLX, start + i);
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {
            std::vector<bitLenInt> bits(controls.size() + length);
            std::copy(controls.begin(), controls.end(), bits.begin());
            for (bitLenInt k = 0U; k < (j + 1U); ++k) {
                bits[controls.size() + k] = start + i + k;
            }
            MACInvert(std::vector<bitLenInt>(bits.begin(), bits.begin() + controls.size() + j + 1U), ONE_CMPLX,
                ONE_CMPLX, start + ((i + j + 1U) % length));
        }
    }

    for (const bitLenInt& control : controls) {
        X(control);
    }
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, const std::vector<bitLenInt>& controls)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    CINC(invToSub, inOutStart, length, controls);
}

/** Add a classical integer to the register, with sign and without carry. */
void QInterface::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt signMask = pow2(length - 1U);
    INC(signMask, start, length);
    INCDECC(toAdd & ~signMask, start, length, overflowIndex);
    if (bi_compare_0(toAdd & signMask) == 0) {
        DEC(signMask, start, length);
    }
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    const bitCapInt invToSub = pow2(length) - toSub;
    INCS(invToSub, start, length, overflowIndex);
}

/**
 * Multiplication modulo N by integer, (out of place)
 */
void QInterface::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    std::vector<bitLenInt> controls(1);
    for (bitLenInt i = 0U; i < length; ++i) {
        controls[0] = inStart + i;
        bitCapInt partMul;
        bi_div_mod(toMul * pow2(i), modN, NULL, &partMul);
        if (bi_compare_0(partMul) == 0) {
            continue;
        }
        CINC(partMul, outStart, oLength, controls);
    }

    if (isPow2) {
        return;
    }

    bitCapInt diffPow;
    bi_div_mod(pow2(length), modN, &diffPow, NULL);
    const bitLenInt lDiff = log2(diffPow);
    controls[0] = inStart + length - (lDiff + 1U);
    for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
        DEC(modN, inStart, length);
        X(controls[0]);
        CDEC(modN, outStart, oLength, controls);
        X(controls[0]);
    }
    for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
        INC(modN, inStart, length);
    }
}

/**
 * Inverse of multiplication modulo N by integer, (out of place)
 */
void QInterface::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    bitCapInt diffPow;
    bi_div_mod(pow2(length), modN, &diffPow, NULL);
    const bitLenInt lDiff = log2(diffPow);
    std::vector<bitLenInt> controls{ (bitLenInt)(inStart + length - (lDiff + 1U)) };

    if (!isPow2) {
        for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
            DEC(modN, inStart, length);
        }
        for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
            X(controls[0]);
            CINC(modN, outStart, oLength, controls);
            X(controls[0]);
            INC(modN, inStart, length);
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        controls[0] = inStart + i;
        bitCapInt partMul;
        bi_div_mod(toMul * pow2(i), modN, NULL, &partMul);
        if (bi_compare_0(partMul) == 0) {
            continue;
        }
        CDEC(partMul, outStart, oLength, controls);
    }
}

/**
 * Controlled multiplication modulo N by integer, (out of place)
 */
void QInterface::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    for (bitLenInt i = 0U; i < length; ++i) {
        lControls[controls.size()] = inStart + i;
        bitCapInt partMul;
        bi_div_mod(toMul * pow2(i), modN, NULL, &partMul);
        if (bi_compare_0(partMul) == 0) {
            continue;
        }
        CINC(partMul, outStart, oLength, lControls);
    }

    if (isPow2) {
        return;
    }

    bitCapInt diffPow;
    bi_div_mod(pow2(length), modN, &diffPow, NULL);
    const bitLenInt lDiff = log2(diffPow);
    lControls[controls.size()] = inStart + length - (lDiff + 1U);
    for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
        CDEC(modN, inStart, length, controls);
        X(lControls[controls.size()]);
        CDEC(modN, outStart, oLength, lControls);
        X(lControls[controls.size()]);
    }
    for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
        CINC(modN, inStart, length, controls);
    }
}

/**
 * Inverse of controlled multiplication modulo N by integer, (out of place)
 */
void QInterface::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const std::vector<bitLenInt>& controls)
{
    const bool isPow2 = isPowerOfTwo(modN);
    const bitLenInt oLength = isPow2 ? log2(modN) : (log2(modN) + 1U);
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    bitCapInt diffPow;
    bi_div_mod(pow2(length), modN, &diffPow, NULL);
    const bitLenInt lDiff = log2(diffPow);
    lControls[controls.size()] = inStart + length - (lDiff + 1U);

    if (!isPow2) {
        for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
            CDEC(modN, inStart, length, controls);
        }
        for (bitCapInt i = ZERO_BCI; bi_compare(i, diffPow) < 0; bi_increment(&i, 1U)) {
            X(lControls[controls.size()]);
            CINC(modN, outStart, oLength, lControls);
            X(lControls[controls.size()]);
            CINC(modN, inStart, length, controls);
        }
    }

    for (bitLenInt i = 0U; i < length; ++i) {
        lControls[controls.size()] = inStart + i;
        bitCapInt partMul;
        bi_div_mod(toMul * pow2(i), modN, NULL, &partMul);
        if (bi_compare_0(partMul) == 0) {
            continue;
        }
        CDEC(partMul, outStart, oLength, lControls);
    }
}

/// Quantum analog of classical "Full Adder" gate
void QInterface::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer

    // Assume outputBit is in 0 state.
    CCNOT(inputBit1, inputBit2, carryOut);
    CNOT(inputBit1, inputBit2);
    CCNOT(inputBit2, carryInSumOut, carryOut);
    CNOT(inputBit2, carryInSumOut);
    CNOT(inputBit1, inputBit2);
}

/// Inverse of FullAdd
void QInterface::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
    // (CNOT and CCNOT are self-inverse.)

    // Assume outputBit is in 0 state.
    CNOT(inputBit1, inputBit2);
    CNOT(inputBit2, carryInSumOut);
    CCNOT(inputBit2, carryInSumOut, carryOut);
    CNOT(inputBit1, inputBit2);
    CCNOT(inputBit1, inputBit2, carryOut);
}

/// Quantum analog of classical "Full Adder" gate
void QInterface::CFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    std::vector<bitLenInt> cBits(controls.size() + 2U);
    std::copy(controls.begin(), controls.end(), cBits.begin());

    // Assume outputBit is in 0 state.
    cBits[controls.size()] = inputBit1;
    cBits[controls.size() + 1U] = inputBit2;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);

    cBits[controls.size()] = inputBit2;
    cBits[controls.size() + 1U] = carryInSumOut;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
    MCInvert(std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX,
        carryInSumOut);

    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
}

/// Inverse of FullAdd
void QInterface::CIFullAdd(const std::vector<bitLenInt>& controls, bitLenInt inputBit1, bitLenInt inputBit2,
    bitLenInt carryInSumOut, bitLenInt carryOut)
{
    // See https://quantumcomputing.stackexchange.com/questions/1654/how-do-i-add-11-using-a-quantum-computer
    // Quantum computing is reversible! Simply perform the inverse operations in reverse order!
    // (CNOT and CCNOT are self-inverse.)

    std::vector<bitLenInt> cBits(controls.size() + 2U);
    std::copy(controls.begin(), controls.end(), cBits.begin());

    // Assume outputBit is in 0 state.
    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controls.size()] = inputBit2;
    MCInvert(std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX,
        carryInSumOut);

    cBits[controls.size() + 1U] = carryInSumOut;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);

    cBits[controls.size()] = inputBit1;
    MCInvert(
        std::vector<bitLenInt>(cBits.begin(), cBits.begin() + controls.size() + 1U), ONE_CMPLX, ONE_CMPLX, inputBit2);
    cBits[controls.size() + 1U] = inputBit2;
    MCInvert(cBits, ONE_CMPLX, ONE_CMPLX, carryOut);
}

void QInterface::ADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    FullAdd(input1, input2, carry, output);

    if (length == 1U) {
        Swap(carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    for (bitLenInt i = 1U; i < end; ++i) {
        FullAdd(input1 + i, input2 + i, output + i, output + i + 1U);
    }
    FullAdd(input1 + end, input2 + end, output + end, carry);
}

void QInterface::IADC(bitLenInt input1, bitLenInt input2, bitLenInt output, bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        Swap(carry, output);
        IFullAdd(input1, input2, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    IFullAdd(input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        IFullAdd(input1 + i, input2 + i, output + i, output + i + 1U);
    }
    IFullAdd(input1, input2, carry, output);
}

void QInterface::CADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
    bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    CFullAdd(controls, input1, input2, carry, output);

    if (length == 1) {
        CSwap(controls, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    for (bitLenInt i = 1; i < end; ++i) {
        CFullAdd(controls, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CFullAdd(controls, input1 + end, input2 + end, output + end, carry);
}

void QInterface::CIADC(const std::vector<bitLenInt>& controls, bitLenInt input1, bitLenInt input2, bitLenInt output,
    bitLenInt length, bitLenInt carry)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        CSwap(controls, carry, output);
        CIFullAdd(controls, input1, input2, carry, output);
        return;
    }

    // Otherwise, length > 1.
    const bitLenInt end = length - 1U;
    CIFullAdd(controls, input1 + end, input2 + end, output + end, carry);
    for (bitLenInt i = (end - 1); i > 0U; i--) {
        CIFullAdd(controls, input1 + i, input2 + i, output + i, output + i + 1U);
    }
    CIFullAdd(controls, input1, input2, carry, output);
}

} // namespace Qrack
