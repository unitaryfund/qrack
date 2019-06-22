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
void QInterface::H(bitLenInt qubit)
{
    const complex had[4] = { complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
        complex(-M_SQRT1_2, ZERO_R1) };
    ApplySingleBit(had, true, qubit);
}

/// Apply 1/4 phase rotation
void QInterface::S(bitLenInt qubit)
{
    ApplySinglePhase(complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ONE_R1), false, qubit);
}

/// Apply inverse 1/4 phase rotation
void QInterface::IS(bitLenInt qubit)
{
    ApplySinglePhase(complex(ONE_R1, ZERO_R1), complex(ZERO_R1, -ONE_R1), false, qubit);
}

/// Apply 1/8 phase rotation
void QInterface::T(bitLenInt qubit)
{
    ApplySinglePhase(complex(ONE_R1, ZERO_R1), complex(M_SQRT1_2, M_SQRT1_2), true, qubit);
}

/// Apply inverse 1/8 phase rotation
void QInterface::IT(bitLenInt qubit)
{
    ApplySinglePhase(complex(ONE_R1, ZERO_R1), complex(M_SQRT1_2, -M_SQRT1_2), true, qubit);
}

/// NOT gate, which is also Pauli x matrix
void QInterface::X(bitLenInt qubit)
{
    ApplySingleInvert(complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), false, qubit);
}

/// Apply Pauli Y matrix to bit
void QInterface::Y(bitLenInt qubit)
{
    ApplySingleInvert(complex(ZERO_R1, -ONE_R1), complex(ZERO_R1, ONE_R1), false, qubit);
}

/// Apply Pauli Z matrix to bit
void QInterface::Z(bitLenInt qubit)
{
    ApplySinglePhase(complex(ONE_R1, ZERO_R1), complex(-ONE_R1, ZERO_R1), false, qubit);
}

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

void QInterface::UniformlyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
{
    for (bitCapInt index = 0; index < (1U << controlLen); index++) {
        for (bitLenInt bit_pos = 0; bit_pos < controlLen; bit_pos++) {
            if (!((index >> bit_pos) & 1)) {
                X(controls[bit_pos]);
            }
        }

        ApplyControlledSingleBit(controls, controlLen, qubitIndex, &(mtrxs[index * 4U]));

        for (bitLenInt bit_pos = 0; bit_pos < controlLen; bit_pos++) {
            if (!((index >> bit_pos) & 1)) {
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
            maxJ *= 1U << op->controlLen;
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
            for (bitCapInt j = 0; j < (1U << op->controlLen); j++) {
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
