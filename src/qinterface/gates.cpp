//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qinterface.hpp"

#define SINGLE_PHASE(gate, topLeft, bottomRight, doNorm)                                                               \
    void QInterface::gate(bitLenInt qubit) { ApplySinglePhase(topLeft, bottomRight, doNorm, qubit); }

#define SINGLE_INVERT(gate, topRight, bottomLeft, doNorm)                                                              \
    void QInterface::gate(bitLenInt qubit) { ApplySingleInvert(topRight, bottomLeft, doNorm, qubit); }

#define SINGLE_BIT(gate, mtrx00, mtrx01, mtrx10, mtrx11, doNorm)                                                       \
    void QInterface::gate(bitLenInt qubit)                                                                             \
    {                                                                                                                  \
        const complex mtrx[4] = { mtrx00, mtrx01, mtrx10, mtrx11 };                                                    \
        ApplySingleBit(mtrx, doNorm, qubit);                                                                           \
    }

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QInterface::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Apply a single bit transformation that only effects phase.
void QInterface::ApplySinglePhase(
    const complex topLeft, const complex bottomRight, bool doCalcNorm, bitLenInt qubitIndex)
{
    const complex mtrx[4] = { topLeft, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), bottomRight };
    ApplySingleBit(mtrx, doCalcNorm, qubitIndex);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase.
void QInterface::ApplySingleInvert(
    const complex topRight, const complex bottomLeft, bool doCalcNorm, bitLenInt qubitIndex)
{
    const complex mtrx[4] = { complex(ZERO_R1, ZERO_R1), topRight, bottomLeft, complex(ZERO_R1, ZERO_R1) };
    ApplySingleBit(mtrx, doCalcNorm, qubitIndex);
}

/// Apply a single bit transformation that only effects phase, with arbitrary control bits.
void QInterface::ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    const complex mtrx[4] = { topLeft, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), bottomRight };
    ApplyControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control bits.
void QInterface::ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    const complex mtrx[4] = { complex(ZERO_R1, ZERO_R1), topRight, bottomLeft, complex(ZERO_R1, ZERO_R1) };
    ApplyControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that only effects phase, with arbitrary (anti-)control bits.
void QInterface::ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    const complex mtrx[4] = { topLeft, complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), bottomRight };
    ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary
/// (anti-)control bits.
void QInterface::ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    const complex mtrx[4] = { complex(ZERO_R1, ZERO_R1), topRight, bottomLeft, complex(ZERO_R1, ZERO_R1) };
    ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
}

/// General unitary gate
void QInterface::U(bitLenInt target, real1 theta, real1 phi, real1 lambda)
{
    real1 cos0 = cos(theta / 2);
    real1 sin0 = sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex(-cos(lambda), -sin(lambda)),
        sin0 * complex(cos(phi), sin(phi)), cos0 * complex(cos(phi + lambda), sin(phi + lambda)) };
    ApplySingleBit(uGate, true, target);
}

/// Hadamard gate
SINGLE_BIT(H, complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
    complex(-M_SQRT1_2, ZERO_R1), true);

/// Square root of Hadamard gate
SINGLE_BIT(SqrtH, complex((ONE_R1 + M_SQRT2) / (2 * M_SQRT2), (-ONE_R1 + M_SQRT2) / (2 * M_SQRT2)),
    complex(M_SQRT1_2 / 2, -M_SQRT1_2 / 2), complex(M_SQRT1_2 / 2, -M_SQRT1_2 / 2),
    complex((-ONE_R1 + M_SQRT2) / (2 * M_SQRT2), (ONE_R1 + M_SQRT2) / (2 * M_SQRT2)), true);

/// Apply 1/4 phase rotation
void QInterface::S(bitLenInt qubit) { PhaseRootN(2U, qubit); }

/// Apply inverse 1/4 phase rotation
void QInterface::IS(bitLenInt qubit) { IPhaseRootN(2U, qubit); }

/// Apply 1/8 phase rotation
void QInterface::T(bitLenInt qubit) { PhaseRootN(3U, qubit); }

/// Apply inverse 1/8 phase rotation
void QInterface::IT(bitLenInt qubit) { IPhaseRootN(3U, qubit); }

/// Apply 1/(2^N) phase rotation
void QInterface::PhaseRootN(bitLenInt n, bitLenInt qubit)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        Z(qubit);
    }

    ApplySinglePhase(
        complex(ONE_R1, ZERO_R1), pow(complex(-ONE_R1, ZERO_R1), ONE_R1 / (ONE_BCI << (n - 1U))), true, qubit);
}

/// Apply inverse 1/(2^N) phase rotation
void QInterface::IPhaseRootN(bitLenInt n, bitLenInt qubit)
{
    if (n == 0) {
        return;
    }
    if (n == 1) {
        Z(qubit);
    }

    ApplySinglePhase(
        complex(ONE_R1, ZERO_R1), pow(complex(-ONE_R1, ZERO_R1), -ONE_R1 / (ONE_BCI << (n - 1U))), true, qubit);
}

/// NOT gate, which is also Pauli x matrix
SINGLE_INVERT(X, complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), false);

/// Apply Pauli Y matrix to bit
SINGLE_INVERT(Y, complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1), false);

/// Square root of NOT gate
SINGLE_BIT(SqrtX, complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
    complex(ONE_R1 / 2, ONE_R1 / 2), true);

/// Inverse square root of NOT gate
SINGLE_BIT(ISqrtX, complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
    complex(ONE_R1 / 2, -ONE_R1 / 2), true);

/// Apply Pauli Y matrix to bit
SINGLE_BIT(SqrtY, complex(ONE_R1 / 2, ONE_R1 / 2), complex(-ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
    complex(ONE_R1 / 2, ONE_R1 / 2), true);

/// Apply Pauli Y matrix to bit
SINGLE_BIT(ISqrtY, complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2), complex(-ONE_R1 / 2, ONE_R1 / 2),
    complex(ONE_R1 / 2, -ONE_R1 / 2), true);

/// Apply Pauli Z matrix to bit
SINGLE_PHASE(Z, complex(ONE_R1, ZERO_R1), complex(-ONE_R1, ZERO_R1), false);

/// Doubly-controlled not
void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    ApplyControlledSingleInvert(controls, 2, target, complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1));
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    ApplyAntiControlledSingleInvert(controls, 2, target, complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1));
}

/// Controlled not
void QInterface::CNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSingleInvert(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1));
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyAntiControlledSingleInvert(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1));
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSingleInvert(controls, 1, target, complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1));
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(-ONE_R1, ZERO_R1));
}

/// Apply controlled S gate to bit
void QInterface::CS(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ONE_R1));
}

/// Apply controlled IS gate to bit
void QInterface::CIS(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(ZERO_R1, -ONE_R1));
}

/// Apply controlled T gate to bit
void QInterface::CT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(M_SQRT1_2, M_SQRT1_2));
}

/// Apply controlled IT gate to bit
void QInterface::CIT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, complex(ONE_R1, ZERO_R1), complex(M_SQRT1_2, -M_SQRT1_2));
}

/// Apply controlled "PhaseRootN" gate to bit
void QInterface::CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(
        controls, 1, target, complex(ONE_R1, ZERO_R1), pow(complex(-ONE_R1, ZERO_R1), ONE_R1 / (ONE_BCI << (n - 1U))));
}

/// Apply controlled "IPhaseRootN" gate to bit
void QInterface::CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(
        controls, 1, target, complex(ONE_R1, ZERO_R1), pow(complex(-ONE_R1, ZERO_R1), -ONE_R1 / (ONE_BCI << (n - 1U))));
}

void QInterface::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
    bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    bitCapInt index;
    for (bitCapInt lcv = 0; lcv < pow2(controlLen); lcv++) {
        index = pushApartBits(lcv, mtrxSkipPowers, mtrxSkipLen) | mtrxSkipValueMask;
        for (bitLenInt bit_pos = 0; bit_pos < controlLen; bit_pos++) {
            if (!((lcv >> bit_pos) & 1)) {
                X(controls[bit_pos]);
            }
        }

        ApplyControlledSingleBit(controls, controlLen, qubitIndex, &(mtrxs[index * 4U]));

        for (bitLenInt bit_pos = 0; bit_pos < controlLen; bit_pos++) {
            if (!((lcv >> bit_pos) & 1)) {
                X(controls[bit_pos]);
            }
        }
    }
}

void QInterface::TimeEvolve(Hamiltonian h, real1 timeDiff)
{
    // Exponentiation of an arbitrary serial string of gates, each HamiltonianOp component times timeDiff, e^(-i * H *
    // t) as e^(-i * H_(N - 1) * t) * e^(-i * H_(N - 2) * t) * ... e^(-i * H_0 * t)

    for (bitLenInt i = 0; i < h.size(); i++) {
        HamiltonianOpPtr op = h[i];
        complex* opMtrx = op->matrix.get();
        complex* mtrx;

        bitCapInt maxJ = 4;
        if (op->uniform) {
            maxJ *= pow2(op->controlLen);
        }
        mtrx = new complex[maxJ];

        for (bitCapInt j = 0; j < maxJ; j++) {
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
            complex* expMtrx = new complex[maxJ];
            for (bitCapInt j = 0; j < pow2(op->controlLen); j++) {
                exp2x2(mtrx + (j * 4U), expMtrx + (j * 4U));
            }
            UniformlyControlledSingleBit(op->controls, op->controlLen, op->targetBit, expMtrx);
            delete[] expMtrx;
        } else {
            Exp(op->controls, op->controlLen, op->targetBit, mtrx, op->anti);
        }

        if (op->toggles) {
            for (bitLenInt j = 0; j < op->controlLen; j++) {
                if (op->toggles[j]) {
                    X(op->controls[j]);
                }
            }
        }

        delete[] mtrx;
    }
}

} // namespace Qrack
