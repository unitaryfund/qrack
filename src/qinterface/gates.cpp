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

#include "qinterface.hpp"

#define C_SQRT1_2 complex(SQRT1_2_R1, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, SQRT1_2_R1)
#define C_SQRT_I complex(SQRT1_2_R1, SQRT1_2_R1)
#define C_SQRT_N_I complex(SQRT1_2_R1, -SQRT1_2_R1)
#define ONE_PLUS_I_DIV_2 complex((real1)(ONE_R1 / 2), (real1)(ONE_R1 / 2))
#define ONE_MINUS_I_DIV_2 complex((real1)(ONE_R1 / 2), (real1)(-ONE_R1 / 2))

#define GATE_1_BIT(gate, mtrx00, mtrx01, mtrx10, mtrx11)                                                               \
    void QInterface::gate(bitLenInt qubit)                                                                             \
    {                                                                                                                  \
        const complex mtrx[4] = { mtrx00, mtrx01, mtrx10, mtrx11 };                                                    \
        Mtrx(mtrx, qubit);                                                                                             \
    }

#define GATE_1_PHASE(gate, topLeft, bottomRight)                                                                       \
    void QInterface::gate(bitLenInt qubit) { Phase(topLeft, bottomRight, qubit); }

#define GATE_1_INVERT(gate, topRight, bottomLeft)                                                                      \
    void QInterface::gate(bitLenInt qubit) { Invert(topRight, bottomLeft, qubit); }

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QInterface::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Apply a single bit transformation that only effects phase.
void QInterface::Phase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex)
{
    if ((randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft)) && IS_NORM_0(topLeft - bottomRight)) {
        return;
    }

    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    Mtrx(mtrx, qubitIndex);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase.
void QInterface::Invert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    Mtrx(mtrx, qubitIndex);
}

/// Apply a single bit transformation that only effects phase, with arbitrary control bits.
void QInterface::MCPhase(
    const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    MCMtrx(controls, controlLen, mtrx, target);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control bits.
void QInterface::MCInvert(
    const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    MCMtrx(controls, controlLen, mtrx, target);
}

/// Apply a single bit transformation that only effects phase, with arbitrary (anti-)control bits.
void QInterface::MACPhase(
    const bitLenInt* controls, bitLenInt controlLen, complex topLeft, complex bottomRight, bitLenInt target)
{
    if (IS_NORM_0(ONE_CMPLX - topLeft) && IS_NORM_0(ONE_CMPLX - bottomRight)) {
        return;
    }

    MACWrapper(controls, controlLen, [this, topLeft, bottomRight, target](const bitLenInt* lc, bitLenInt lcLen) {
        MCPhase(lc, lcLen, topLeft, bottomRight, target);
    });
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary
/// (anti-)control bits.
void QInterface::MACInvert(
    const bitLenInt* controls, bitLenInt controlLen, complex topRight, complex bottomLeft, bitLenInt target)
{
    MACWrapper(controls, controlLen, [this, topRight, bottomLeft, target](const bitLenInt* lc, bitLenInt lcLen) {
        MCInvert(lc, lcLen, topRight, bottomLeft, target);
    });
}

void QInterface::MACMtrx(const bitLenInt* controls, bitLenInt controlLen, const complex* mtrx, bitLenInt target)
{
    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        MACPhase(controls, controlLen, mtrx[0], mtrx[3], target);
    } else if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        MACInvert(controls, controlLen, mtrx[1], mtrx[2], target);
    } else {
        MACWrapper(controls, controlLen,
            [this, mtrx, target](const bitLenInt* lc, bitLenInt lcLen) { MCMtrx(lc, lcLen, mtrx, target); });
    }
}

/// Apply 1/(2^N) phase rotation
void QInterface::PhaseRootN(bitLenInt n, bitLenInt qubit)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        Z(qubit);
        return;
    }
    if (n == 2) {
        Phase(ONE_CMPLX, I_CMPLX, qubit);
        return;
    }
    if (n == 3) {
        Phase(ONE_CMPLX, C_SQRT_I, qubit);
        return;
    }

    Phase(ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), qubit);
}

/// Apply inverse 1/(2^N) phase rotation
void QInterface::IPhaseRootN(bitLenInt n, bitLenInt qubit)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        Z(qubit);
        return;
    }
    if (n == 2) {
        Phase(ONE_CMPLX, -I_CMPLX, qubit);
        return;
    }
    if (n == 3) {
        Phase(ONE_CMPLX, C_SQRT_N_I, qubit);
        return;
    }

    Phase(ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(-ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), qubit);
}

/// NOT gate, which is also Pauli x matrix
GATE_1_INVERT(X, ONE_CMPLX, ONE_CMPLX);

/// Apply Pauli Y matrix to bit
GATE_1_INVERT(Y, -I_CMPLX, I_CMPLX);

/// Apply Pauli Z matrix to bit
GATE_1_PHASE(Z, ONE_CMPLX, -ONE_CMPLX);

/// Hadamard gate
GATE_1_BIT(H, C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, -C_SQRT1_2);

/// Y-basis transformation gate
GATE_1_BIT(SH, C_SQRT1_2, C_SQRT1_2, C_I_SQRT1_2, -C_I_SQRT1_2);

/// Inverse Y-basis transformation gate
GATE_1_BIT(HIS, C_SQRT1_2, -C_I_SQRT1_2, C_SQRT1_2, C_I_SQRT1_2);

/// Square root of Hadamard gate
GATE_1_BIT(SqrtH,
    complex((real1)((ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)), (real1)((-ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1))),
    complex((real1)(SQRT1_2_R1 / 2), (real1)(-SQRT1_2_R1 / 2)),
    complex((real1)(SQRT1_2_R1 / 2), (real1)(-SQRT1_2_R1 / 2)),
    complex((real1)((-ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1)), (real1)((ONE_R1 + SQRT2_R1) / (2 * SQRT2_R1))));

/// Square root of NOT gate
GATE_1_BIT(SqrtX, ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2);

/// Inverse square root of NOT gate
GATE_1_BIT(ISqrtX, ONE_MINUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_MINUS_I_DIV_2);

/// Phased square root of NOT gate
GATE_1_BIT(SqrtXConjT, ONE_PLUS_I_DIV_2, -C_I_SQRT1_2, C_SQRT1_2, ONE_PLUS_I_DIV_2);

/// Inverse phased square root of NOT gate
GATE_1_BIT(ISqrtXConjT, ONE_MINUS_I_DIV_2, C_SQRT1_2, C_I_SQRT1_2, ONE_MINUS_I_DIV_2);

/// Apply Pauli Y matrix to bit
GATE_1_BIT(SqrtY, ONE_PLUS_I_DIV_2, -ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2, ONE_PLUS_I_DIV_2);

/// Apply Pauli Y matrix to bit
GATE_1_BIT(ISqrtY, ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2, -ONE_MINUS_I_DIV_2, ONE_MINUS_I_DIV_2);

/// Apply 1/4 phase rotation
void QInterface::S(bitLenInt qubit) { PhaseRootN(2U, qubit); }

/// Apply inverse 1/4 phase rotation
void QInterface::IS(bitLenInt qubit) { IPhaseRootN(2U, qubit); }

/// Apply 1/8 phase rotation
void QInterface::T(bitLenInt qubit) { PhaseRootN(3U, qubit); }

/// Apply inverse 1/8 phase rotation
void QInterface::IT(bitLenInt qubit) { IPhaseRootN(3U, qubit); }

/// Apply controlled S gate to bit
void QInterface::CS(bitLenInt control, bitLenInt target) { CPhaseRootN(2U, control, target); }

/// Apply (anti-)controlled S gate to bit
void QInterface::AntiCS(bitLenInt control, bitLenInt target) { AntiCPhaseRootN(2U, control, target); }

/// Apply controlled IS gate to bit
void QInterface::CIS(bitLenInt control, bitLenInt target) { CIPhaseRootN(2U, control, target); }

/// Apply (anti-)controlled IS gate to bit
void QInterface::AntiCIS(bitLenInt control, bitLenInt target) { AntiCIPhaseRootN(2U, control, target); }

/// Apply controlled T gate to bit
void QInterface::CT(bitLenInt control, bitLenInt target) { CPhaseRootN(3U, control, target); }

/// Apply controlled IT gate to bit
void QInterface::CIT(bitLenInt control, bitLenInt target) { CIPhaseRootN(3U, control, target); }

/// Controlled not
void QInterface::CNOT(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MCInvert(controls, 1, ONE_CMPLX, ONE_CMPLX, target);
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MCInvert(controls, 1, -I_CMPLX, I_CMPLX, target);
}

/// Apply doubly-controlled Pauli Z matrix to bit
void QInterface::CCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MCInvert(controls, 2, -I_CMPLX, I_CMPLX, target);
}

/// "Anti-doubly-controlled Y" - Apply Pauli Y if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MACInvert(controls, 2, -I_CMPLX, I_CMPLX, target);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCY(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MACInvert(controls, 1, -I_CMPLX, I_CMPLX, target);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MCPhase(controls, 1, ONE_CMPLX, -ONE_CMPLX, target);
}

/// Apply doubly-controlled Pauli Z matrix to bit
void QInterface::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MCPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, target);
}

/// "Anti-doubly-controlled Z" - Apply Pauli Z if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MACPhase(controls, 2, ONE_CMPLX, -ONE_CMPLX, target);
}

/// "Anti-controlled Z" - Apply Pauli Z if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCZ(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MACPhase(controls, 1, ONE_CMPLX, -ONE_CMPLX, target);
}

/// Apply controlled Hadamard matrix to bit
void QInterface::CH(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    const complex h[4] = { complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(ONE_R1 / SQRT2_R1, ZERO_R1),
        complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(-ONE_R1 / SQRT2_R1, ZERO_R1) };
    MCMtrx(controls, 1, h, target);
}

/// Apply (anti-)controlled Hadamard matrix to bit
void QInterface::AntiCH(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    const complex h[4] = { complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(ONE_R1 / SQRT2_R1, ZERO_R1),
        complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(-ONE_R1 / SQRT2_R1, ZERO_R1) };
    MACMtrx(controls, 1, h, target);
}

/// Doubly-controlled not
void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MCInvert(controls, 2, ONE_CMPLX, ONE_CMPLX, target);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const bitLenInt controls[2] = { control1, control2 };
    MACInvert(controls, 2, ONE_CMPLX, ONE_CMPLX, target);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MACInvert(controls, 1, ONE_CMPLX, ONE_CMPLX, target);
}

/// Apply controlled "PhaseRootN" gate to bit
void QInterface::CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        CZ(control, target);
        return;
    }

    const bitLenInt controls[1] = { control };

    if (n == 2) {
        MCPhase(controls, 1, ONE_CMPLX, I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MCPhase(controls, 1, ONE_CMPLX, C_SQRT_I, target);
        return;
    }

    MCPhase(controls, 1, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
}

/// Apply controlled "IPhaseRootN" gate to bit
void QInterface::CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        CZ(control, target);
        return;
    }

    const bitLenInt controls[1] = { control };

    if (n == 2) {
        MCPhase(controls, 1, ONE_CMPLX, -I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MCPhase(controls, 1, ONE_CMPLX, C_SQRT_N_I, target);
        return;
    }

    MCPhase(
        controls, 1, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(-ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
}

/// Apply (anti-)controlled "PhaseRootN" gate to bit
void QInterface::AntiCPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        AntiCZ(control, target);
        return;
    }

    const bitLenInt controls[1] = { control };

    if (n == 2) {
        MACPhase(controls, 1, ONE_CMPLX, I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MACPhase(controls, 1, ONE_CMPLX, C_SQRT_I, target);
        return;
    }

    MACPhase(
        controls, 1, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
}

/// Apply (anti-)controlled "IPhaseRootN" gate to bit
void QInterface::AntiCIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        AntiCZ(control, target);
        return;
    }

    const bitLenInt controls[1] = { control };

    if (n == 2) {
        MACPhase(controls, 1, ONE_CMPLX, -I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MACPhase(controls, 1, ONE_CMPLX, C_SQRT_N_I, target);
        return;
    }

    MACPhase(
        controls, 1, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(-ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
}

void QInterface::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    for (bitLenInt bit_pos = 0U; bit_pos < controlLen; bit_pos++) {
        X(controls[bit_pos]);
    }
    const bitCapInt maxI = pow2(controlLen);
    for (bitCapInt lcv = 0U; lcv < maxI; lcv++) {
        const bitCapInt index = pushApartBits(lcv, mtrxSkipPowers, mtrxSkipLen) | mtrxSkipValueMask;
        MCMtrx(controls, controlLen, mtrxs + (bitCapIntOcl)(index * 4U), qubitIndex);

        if ((lcv + 1U) == maxI) {
            continue;
        }

        const bitCapInt lcvDiff = lcv ^ (lcv + ONE_BCI);
        for (bitLenInt bit_pos = 0U; bit_pos < controlLen; bit_pos++) {
            if ((lcvDiff >> bit_pos) & ONE_BCI) {
                X(controls[bit_pos]);
            }
        }
    }
}

void QInterface::PhaseFlip() { Phase(-ONE_CMPLX, -ONE_CMPLX, 0); }

void QInterface::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    const bitLenInt min1 = length - 1U;
    std::unique_ptr<bitLenInt[]> controls(new bitLenInt[min1]);
    for (bitLenInt i = 0; i < min1; i++) {
        controls[i] = start + i;
    }
    MACPhase(controls.get(), min1, -ONE_CMPLX, ONE_CMPLX, start + min1);
}

void QInterface::XMask(bitCapInt mask)
{
    bitCapInt v = mask;
    while (mask) {
        v = v & (v - ONE_BCI);
        X(log2(mask ^ v));
        mask = v;
    }
}

void QInterface::YMask(bitCapInt mask)
{
    bitLenInt bit = log2(mask);
    if (pow2(bit) == mask) {
        Y(bit);
        return;
    }

    ZMask(mask);
    XMask(mask);

    if (randGlobalPhase) {
        return;
    }

    int parity = 0;
    bitCapInt v = mask;
    while (v) {
        v = v & (v - ONE_BCI);
        parity = (parity + 1) & 3;
    }

    if (parity == 1) {
        Phase(I_CMPLX, I_CMPLX, 0);
    } else if (parity == 2) {
        PhaseFlip();
    } else if (parity == 3) {
        Phase(-I_CMPLX, -I_CMPLX, 0);
    }
}

void QInterface::ZMask(bitCapInt mask)
{
    bitCapInt v = mask;
    while (mask) {
        v = v & (v - ONE_BCI);
        Z(log2(mask ^ v));
        mask = v;
    }
}

void QInterface::Swap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    CNOT(q1, q2);
    CNOT(q2, q1);
    CNOT(q1, q2);
}

void QInterface::ISwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    S(q1);
    S(q2);
    H(q1);
    CNOT(q1, q2);
    CNOT(q2, q1);
    H(q2);
}

void QInterface::SqrtSwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    CNOT(q1, q2);
    H(q1);
    IT(q2);
    T(q1);
    H(q2);
    H(q1);
    CNOT(q1, q2);
    H(q1);
    H(q2);
    IT(q1);
    H(q1);
    CNOT(q1, q2);
    IS(q1);
    S(q2);
}

void QInterface::ISqrtSwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    IS(q2);
    S(q1);
    CNOT(q1, q2);
    H(q1);
    T(q1);
    H(q2);
    H(q1);
    CNOT(q1, q2);
    H(q1);
    H(q2);
    IT(q1);
    T(q2);
    H(q1);
    CNOT(q1, q2);
}

void QInterface::CSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    if (!controlLen) {
        Swap(q1, q2);
        return;
    }

    if (q1 == q2) {
        return;
    }

    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen + 1U]());
    std::copy(controls, controls + controlLen, lControls.get());

    lControls[controlLen] = q1;
    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    lControls[controlLen] = q2;
    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q1);

    lControls[controlLen] = q1;
    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }

    CSwap(controls, controlLen, q1, q2);

    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }
}

void QInterface::CSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    if (!controlLen) {
        SqrtSwap(q1, q2);
        return;
    }

    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen + 1U]);
    std::copy(controls, controls + controlLen, lControls.get());
    lControls[controlLen] = q1;

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    const complex had[4] = { C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, -C_SQRT1_2 };
    MCMtrx(lControls.get(), controlLen, had, q1);

    const complex it[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(lControls.get(), controlLen, it, q2);

    const complex t[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(lControls.get(), controlLen, t, q1);

    MCMtrx(lControls.get(), controlLen, had, q2);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCMtrx(lControls.get(), controlLen, had, q2);

    MCMtrx(lControls.get(), controlLen, it, q1);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    const complex is[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };
    MCMtrx(lControls.get(), controlLen, is, q1);

    const complex s[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(lControls.get(), controlLen, s, q2);
}

void QInterface::CISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::unique_ptr<bitLenInt[]> lControls(new bitLenInt[controlLen + 1U]);
    std::copy(controls, controls + controlLen, lControls.get());
    lControls[controlLen] = q1;

    const complex is[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };
    MCMtrx(lControls.get(), controlLen, is, q2);

    const complex s[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(lControls.get(), controlLen, s, q1);

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    const complex had[4] = { C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, -C_SQRT1_2 };
    MCMtrx(lControls.get(), controlLen, had, q1);

    const complex t[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(lControls.get(), controlLen, t, q1);

    MCMtrx(lControls.get(), controlLen, had, q2);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCMtrx(lControls.get(), controlLen, had, q2);

    const complex it[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(lControls.get(), controlLen, it, q1);

    MCMtrx(lControls.get(), controlLen, t, q2);

    MCMtrx(lControls.get(), controlLen, had, q1);

    MCInvert(lControls.get(), controlLen + 1U, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }

    CSqrtSwap(controls, controlLen, q1, q2);

    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }
}

void QInterface::AntiCISqrtSwap(const bitLenInt* controls, bitLenInt controlLen, bitLenInt q1, bitLenInt q2)
{
    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }

    CISqrtSwap(controls, controlLen, q1, q2);

    for (bitLenInt i = 0U; i < controlLen; i++) {
        X(controls[i]);
    }
}

void QInterface::PhaseParity(real1_f radians, bitCapInt mask)
{
    if (!mask) {
        return;
    }

    std::vector<bitLenInt> qubits;
    bitCapInt v = mask;
    while (mask) {
        v = v & (v - ONE_BCI);
        qubits.push_back(log2(mask ^ v));
        mask = v;
    }

    const int end = (int)(qubits.size() - 1);
    for (int i = 0; i < end; i++) {
        CNOT(qubits[i], qubits[i + 1U]);
    }
    real1 cosine = (real1)cos(radians / 2);
    real1 sine = (real1)sin(radians / 2);
    Phase(cosine - I_CMPLX * sine, cosine + I_CMPLX * sine, qubits[end]);
    for (int i = (end - 1U); i >= 0; i--) {
        CNOT(qubits[i], qubits[i + 1U]);
    }
}

void QInterface::TimeEvolve(Hamiltonian h, real1_f timeDiff_f)
{
    real1 timeDiff = (real1)timeDiff_f;

    if (abs(timeDiff) <= REAL1_EPSILON) {
        return;
    }

    // Exponentiation of an arbitrary serial string of gates, each HamiltonianOp component times timeDiff, e^(-i * H *
    // t) as e^(-i * H_(N - 1) * t) * e^(-i * H_(N - 2) * t) * ... e^(-i * H_0 * t)

    for (size_t i = 0; i < h.size(); i++) {
        HamiltonianOpPtr op = h[i];
        complex* opMtrx = op->matrix.get();

        bitCapIntOcl maxJ = 4;
        if (op->uniform) {
            maxJ *= pow2Ocl(op->controlLen);
        }
        std::unique_ptr<complex[]> mtrx(new complex[maxJ]);

        for (bitCapIntOcl j = 0; j < maxJ; j++) {
            mtrx[j] = opMtrx[j] * (-timeDiff);
        }

        if (op->toggles) {
            for (bitLenInt j = 0; j < op->controlLen; j++) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        if (op->uniform) {
            std::unique_ptr<complex[]> expMtrx(new complex[maxJ]);
            for (bitCapIntOcl j = 0; j < pow2(op->controlLen); j++) {
                exp2x2(mtrx.get() + (j * 4U), expMtrx.get() + (j * 4U));
            }
            UniformlyControlledSingleBit(op->controls, op->controlLen, op->targetBit, expMtrx.get());
        } else {
            complex timesI[4] = { I_CMPLX * mtrx[0], I_CMPLX * mtrx[1], I_CMPLX * mtrx[2], I_CMPLX * mtrx[3] };
            complex toApply[4];
            exp2x2(timesI, toApply);
            if (op->controlLen == 0) {
                Mtrx(toApply, op->targetBit);
            } else if (op->anti) {
                MACMtrx(op->controls, op->controlLen, toApply, op->targetBit);
            } else {
                MCMtrx(op->controls, op->controlLen, toApply, op->targetBit);
            }
        }

        if (op->toggles) {
            for (bitLenInt j = 0; j < op->controlLen; j++) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }
    }
}

} // namespace Qrack
