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
        const complex mtrx[4]{ mtrx00, mtrx01, mtrx10, mtrx11 };                                                       \
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
    const std::vector<bitLenInt> controls{ control };
    MCInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    MCInvert(controls, -I_CMPLX, I_CMPLX, target);
}

/// Apply doubly-controlled Pauli Z matrix to bit
void QInterface::CCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MCInvert(controls, -I_CMPLX, I_CMPLX, target);
}

/// "Anti-doubly-controlled Y" - Apply Pauli Y if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCY(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MACInvert(controls, -I_CMPLX, I_CMPLX, target);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCY(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    MACInvert(controls, -I_CMPLX, I_CMPLX, target);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    MCPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
}

/// Apply doubly-controlled Pauli Z matrix to bit
void QInterface::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MCPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
}

/// "Anti-doubly-controlled Z" - Apply Pauli Z if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MACPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
}

/// "Anti-controlled Z" - Apply Pauli Z if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCZ(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    MACPhase(controls, ONE_CMPLX, -ONE_CMPLX, target);
}

/// Apply controlled Hadamard matrix to bit
void QInterface::CH(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    const complex h[4]{ complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(ONE_R1 / SQRT2_R1, ZERO_R1),
        complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(-ONE_R1 / SQRT2_R1, ZERO_R1) };
    MCMtrx(controls, h, target);
}

/// Apply (anti-)controlled Hadamard matrix to bit
void QInterface::AntiCH(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    const complex h[4]{ complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(ONE_R1 / SQRT2_R1, ZERO_R1),
        complex(ONE_R1 / SQRT2_R1, ZERO_R1), complex(-ONE_R1 / SQRT2_R1, ZERO_R1) };
    MACMtrx(controls, h, target);
}

/// Doubly-controlled not
void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MCInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control1, control2 };
    MACInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    const std::vector<bitLenInt> controls{ control };
    MACInvert(controls, ONE_CMPLX, ONE_CMPLX, target);
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

    const std::vector<bitLenInt> controls{ control };

    if (n == 2) {
        MCPhase(controls, ONE_CMPLX, I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MCPhase(controls, ONE_CMPLX, C_SQRT_I, target);
        return;
    }

    MCPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
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

    const std::vector<bitLenInt> controls{ control };

    if (n == 2) {
        MCPhase(controls, ONE_CMPLX, -I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MCPhase(controls, ONE_CMPLX, C_SQRT_N_I, target);
        return;
    }

    MCPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(-ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
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

    const std::vector<bitLenInt> controls{ control };

    if (n == 2) {
        MACPhase(controls, ONE_CMPLX, I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MACPhase(controls, ONE_CMPLX, C_SQRT_I, target);
        return;
    }

    MACPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
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

    const std::vector<bitLenInt> controls{ control };

    if (n == 2) {
        MACPhase(controls, ONE_CMPLX, -I_CMPLX, target);
        return;
    }
    if (n == 3) {
        MACPhase(controls, ONE_CMPLX, C_SQRT_N_I, target);
        return;
    }

    MACPhase(controls, ONE_CMPLX, pow(-ONE_CMPLX, (complex)((real1)(-ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))))), target);
}

void QInterface::UniformlyControlledSingleBit(const std::vector<bitLenInt>& controls, bitLenInt qubitIndex,
    complex const* mtrxs, const std::vector<bitCapInt>& mtrxSkipPowers, bitCapInt mtrxSkipValueMask)
{
    for (bitLenInt bit_pos = 0U; bit_pos < controls.size(); ++bit_pos) {
        X(controls[bit_pos]);
    }
    const bitCapInt maxI = pow2(controls.size()) - ONE_BCI;
    for (bitCapInt lcv = 0U; lcv < maxI; ++lcv) {
        const bitCapInt index = pushApartBits(lcv, mtrxSkipPowers) | mtrxSkipValueMask;
        MCMtrx(controls, mtrxs + (bitCapIntOcl)(index * 4U), qubitIndex);

        const bitCapInt lcvDiff = lcv ^ (lcv + ONE_BCI);
        for (bitLenInt bit_pos = 0U; bit_pos < controls.size(); ++bit_pos) {
            if ((lcvDiff >> bit_pos) & ONE_BCI) {
                X(controls[bit_pos]);
            }
        }
    }
    const bitCapInt index = pushApartBits(maxI, mtrxSkipPowers) | mtrxSkipValueMask;
    MCMtrx(controls, mtrxs + (bitCapIntOcl)(index * 4U), qubitIndex);
}

void QInterface::PhaseFlip() { Phase(-ONE_CMPLX, -ONE_CMPLX, 0); }

void QInterface::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    if (length == 1U) {
        Phase(-ONE_CMPLX, ONE_CMPLX, start);
        return;
    }

    std::vector<bitLenInt> controls(length - 1U);
    for (bitLenInt i = 0U; i < controls.size(); ++i) {
        controls[i] = start + i;
    }
    MACPhase(controls, -ONE_CMPLX, ONE_CMPLX, start + controls.size());
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
        Phase(I_CMPLX, I_CMPLX, 0U);
    } else if (parity == 2) {
        PhaseFlip();
    } else if (parity == 3) {
        Phase(-I_CMPLX, -I_CMPLX, 0U);
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

    Swap(q1, q2);
    CZ(q1, q2);
    S(q1);
    S(q2);
}

void QInterface::IISwap(bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    IS(q2);
    IS(q1);
    CZ(q1, q2);
    Swap(q1, q2);
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

void QInterface::CSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (!controls.size()) {
        Swap(q1, q2);
        return;
    }

    if (q1 == q2) {
        return;
    }

    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());

    lControls[controls.size()] = q1;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    lControls[controls.size()] = q2;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q1);

    lControls[controls.size()] = q1;
    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = 0U;
    for (bitLenInt i = 0U; i < controls.size(); ++i) {
        m |= pow2(controls[i]);
    }

    XMask(m);
    CSwap(controls, q1, q2);
    XMask(m);
}

void QInterface::CSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (!controls.size()) {
        SqrtSwap(q1, q2);
        return;
    }

    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    lControls[controls.size()] = q1;

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    const complex had[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, -C_SQRT1_2 };
    MCMtrx(controls, had, q1);

    const complex it[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(controls, it, q2);

    const complex t[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(controls, t, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(controls, had, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, it, q1);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };
    MCMtrx(controls, is, q1);

    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(controls, s, q2);
}

void QInterface::CISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    if (q1 == q2) {
        return;
    }

    // https://quantumcomputing.stackexchange.com/questions/2228/how-to-implement-the-square-root-of-swap-gate-on-the-ibm-q-composer
    std::vector<bitLenInt> lControls(controls.size() + 1U);
    std::copy(controls.begin(), controls.end(), lControls.begin());
    lControls[controls.size()] = q1;

    const complex is[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -I_CMPLX };
    MCMtrx(controls, is, q2);

    const complex s[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, I_CMPLX };
    MCMtrx(controls, s, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    const complex had[4]{ C_SQRT1_2, C_SQRT1_2, C_SQRT1_2, -C_SQRT1_2 };
    MCMtrx(controls, had, q1);

    const complex t[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_I };
    MCMtrx(controls, t, q1);

    MCMtrx(controls, had, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);

    MCMtrx(controls, had, q1);

    MCMtrx(controls, had, q2);

    const complex it[4]{ ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, C_SQRT_N_I };
    MCMtrx(controls, it, q1);

    MCMtrx(controls, t, q2);

    MCMtrx(controls, had, q1);

    MCInvert(lControls, ONE_CMPLX, ONE_CMPLX, q2);
}

void QInterface::AntiCSqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = 0U;
    for (bitLenInt i = 0U; i < controls.size(); ++i) {
        m |= pow2(controls[i]);
    }

    XMask(m);
    CSqrtSwap(controls, q1, q2);
    XMask(m);
}

void QInterface::AntiCISqrtSwap(const std::vector<bitLenInt>& controls, bitLenInt q1, bitLenInt q2)
{
    bitCapInt m = 0U;
    for (bitLenInt i = 0U; i < controls.size(); ++i) {
        m |= pow2(controls[i]);
    }

    XMask(m);
    CISqrtSwap(controls, q1, q2);
    XMask(m);
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

    const bitLenInt end = qubits.size() - 1U;
    for (bitLenInt i = 0; i < end; ++i) {
        CNOT(qubits[i], qubits[i + 1U]);
    }
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    Phase(cosine - I_CMPLX * sine, cosine + I_CMPLX * sine, qubits[end]);
    for (bitLenInt i = 0; i < end; ++i) {
        CNOT(qubits[end - (i + 1U)], qubits[end - i]);
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

    for (size_t i = 0U; i < h.size(); ++i) {
        HamiltonianOpPtr op = h[i];
        complex* opMtrx = op->matrix.get();

        bitCapIntOcl maxJ = 4U;
        if (op->uniform) {
            maxJ *= pow2Ocl(op->controls.size());
        }
        std::unique_ptr<complex[]> mtrx(new complex[maxJ]);

        for (bitCapIntOcl j = 0U; j < maxJ; ++j) {
            mtrx[j] = opMtrx[j] * (-timeDiff);
        }

        if (op->toggles.size()) {
            for (bitLenInt j = 0U; j < op->controls.size(); ++j) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        if (op->uniform) {
            std::unique_ptr<complex[]> expMtrx(new complex[maxJ]);
            for (bitCapIntOcl j = 0U; j < pow2(op->controls.size()); ++j) {
                exp2x2(mtrx.get() + (j * 4U), expMtrx.get() + (j * 4U));
            }
            UniformlyControlledSingleBit(op->controls, op->targetBit, expMtrx.get());
        } else {
            complex timesI[4U]{ I_CMPLX * mtrx[0U], I_CMPLX * mtrx[1U], I_CMPLX * mtrx[2U], I_CMPLX * mtrx[3U] };
            complex toApply[4U];
            exp2x2(timesI, toApply);
            if (op->controls.size() == 0U) {
                Mtrx(toApply, op->targetBit);
            } else if (op->anti) {
                MACMtrx(op->controls, toApply, op->targetBit);
            } else {
                MCMtrx(op->controls, toApply, op->targetBit);
            }
        }

        if (op->toggles.size()) {
            for (bitLenInt j = 0U; j < op->controls.size(); ++j) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }
    }
}

void QInterface::DepolarizingChannelWeak1Qb(bitLenInt qubit, real1_f lambda)
{
    if (lambda <= ZERO_R1) {
        return;
    }

    // Original qubit, Z->X basis
    H(qubit);

    // Allocate an ancilla
    const bitLenInt ancilla = Allocate(1U);
    // Partially entangle with the ancilla
    CRY(2 * asin(std::pow(lambda, ONE_R1 / 4)), qubit, ancilla);
    // Partially collapse the original state
    M(ancilla);
    // The ancilla is fully separable, after measurement.
    Dispose(ancilla, 1U);

    // Uncompute
    H(qubit);

    // Original qubit might be below separability threshold
    TrySeparate(qubit);
}

bitLenInt QInterface::DepolarizingChannelStrong1Qb(bitLenInt qubit, real1_f lambda)
{
    // Original qubit, Z->X basis
    H(qubit);

    // Allocate an ancilla
    const bitLenInt ancilla = Allocate(1U);
    // Partially entangle with the ancilla
    CRY(2 * asin(std::pow(lambda, ONE_R1 / 4)), qubit, ancilla);

    // Uncompute
    H(qubit);

    return ancilla;
}

} // namespace Qrack
