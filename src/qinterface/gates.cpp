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

#define C_SQRT1_2 complex(M_SQRT1_2, ZERO_R1)
#define C_I_SQRT1_2 complex(ZERO_R1, M_SQRT1_2)
#define ONE_PLUS_I_DIV_2 complex(ONE_R1 / 2, ONE_R1 / 2)
#define ONE_MINUS_I_DIV_2 complex(ONE_R1 / 2, -ONE_R1 / 2)

#define GATE_1_BIT(gate, mtrx00, mtrx01, mtrx10, mtrx11)                                                               \
    void QInterface::gate(bitLenInt qubit)                                                                             \
    {                                                                                                                  \
        const complex mtrx[4] = { mtrx00, mtrx01, mtrx10, mtrx11 };                                                    \
        ApplySingleBit(mtrx, qubit);                                                                                   \
    }

#define GATE_1_PHASE(gate, topLeft, bottomRight)                                                                       \
    void QInterface::gate(bitLenInt qubit) { ApplySinglePhase(topLeft, bottomRight, qubit); }

#define GATE_1_INVERT(gate, topRight, bottomLeft)                                                                      \
    void QInterface::gate(bitLenInt qubit) { ApplySingleInvert(topRight, bottomLeft, qubit); }

namespace Qrack {

/// Set individual bit to pure |0> (false) or |1> (true) state
void QInterface::SetBit(bitLenInt qubit1, bool value)
{
    if (value != M(qubit1)) {
        X(qubit1);
    }
}

/// Apply a single bit transformation that only effects phase.
void QInterface::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex)
{
    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    ApplySingleBit(mtrx, qubitIndex);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase.
void QInterface::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    ApplySingleBit(mtrx, qubitIndex);
}

/// Apply a single bit transformation that only effects phase, with arbitrary control bits.
void QInterface::ApplyControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    ApplyControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary control bits.
void QInterface::ApplyControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    ApplyControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that only effects phase, with arbitrary (anti-)control bits.
void QInterface::ApplyAntiControlledSinglePhase(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topLeft, const complex bottomRight)
{
    const complex mtrx[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
}

/// Apply a single bit transformation that reverses bit probability and might effect phase, with arbitrary
/// (anti-)control bits.
void QInterface::ApplyAntiControlledSingleInvert(const bitLenInt* controls, const bitLenInt& controlLen,
    const bitLenInt& target, const complex topRight, const complex bottomLeft)
{
    const complex mtrx[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    ApplyAntiControlledSingleBit(controls, controlLen, target, mtrx);
}

/// General unitary gate
void QInterface::U(bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    real1 cos0 = cos(theta / 2);
    real1 sin0 = sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex(-cos(lambda), -sin(lambda)),
        sin0 * complex(cos(phi), sin(phi)), cos0 * complex(cos(phi + lambda), sin(phi + lambda)) };
    ApplySingleBit(uGate, target);
}

/// Controlled general unitary gate
void QInterface::CU(bitLenInt* controls, bitLenInt controlLen, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    real1 cos0 = cos(theta / 2);
    real1 sin0 = sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex(-cos(lambda), -sin(lambda)),
        sin0 * complex(cos(phi), sin(phi)), cos0 * complex(cos(phi + lambda), sin(phi + lambda)) };
    ApplyControlledSingleBit(controls, controlLen, target, uGate);
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
        ApplySinglePhase(ONE_CMPLX, I_CMPLX, qubit);
        return;
    }

    ApplySinglePhase(ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))), qubit);
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
        ApplySinglePhase(ONE_CMPLX, -I_CMPLX, qubit);
        return;
    }

    ApplySinglePhase(ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))), qubit);
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
GATE_1_BIT(SqrtH, complex((ONE_R1 + M_SQRT2) / (2 * M_SQRT2), (-ONE_R1 + M_SQRT2) / (2 * M_SQRT2)),
    complex(M_SQRT1_2 / 2, -M_SQRT1_2 / 2), complex(M_SQRT1_2 / 2, -M_SQRT1_2 / 2),
    complex((-ONE_R1 + M_SQRT2) / (2 * M_SQRT2), (ONE_R1 + M_SQRT2) / (2 * M_SQRT2)));

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

/// Apply controlled IS gate to bit
void QInterface::CIS(bitLenInt control, bitLenInt target) { CIPhaseRootN(2U, control, target); }

/// Apply controlled T gate to bit
void QInterface::CT(bitLenInt control, bitLenInt target) { CPhaseRootN(3U, control, target); }

/// Apply controlled IT gate to bit
void QInterface::CIT(bitLenInt control, bitLenInt target) { CIPhaseRootN(3U, control, target); }

/// Controlled not
void QInterface::CNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSingleInvert(controls, 1, target, ONE_CMPLX, ONE_CMPLX);
}

/// Apply controlled Pauli Y matrix to bit
void QInterface::CY(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSingleInvert(controls, 1, target, -I_CMPLX, I_CMPLX);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CZ(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, ONE_CMPLX, -ONE_CMPLX);
}

/// Apply doubly-controlled Pauli Z matrix to bit
void QInterface::CCZ(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    ApplyControlledSinglePhase(controls, 2, target, ONE_CMPLX, -ONE_CMPLX);
}

/// Apply controlled Pauli Z matrix to bit
void QInterface::CH(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    const complex h[4] = { complex(ONE_R1 / sqrt((real1)2), ZERO_R1), complex(ONE_R1 / sqrt((real1)2), ZERO_R1),
        complex(ONE_R1 / sqrt((real1)2), ZERO_R1), complex(-ONE_R1 / sqrt((real1)2), ZERO_R1) };
    ApplyControlledSingleBit(controls, 1, target, h);
}

/// Doubly-controlled not
void QInterface::CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    ApplyControlledSingleInvert(controls, 2, target, ONE_CMPLX, ONE_CMPLX);
}

/// "Anti-doubly-controlled not" - Apply "not" if control bits are both zero, do not apply if either control bit is one.
void QInterface::AntiCCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target)
{
    bitLenInt controls[2] = { control1, control2 };
    ApplyAntiControlledSingleInvert(controls, 2, target, ONE_CMPLX, ONE_CMPLX);
}

/// "Anti-controlled not" - Apply "not" if control bit is zero, do not apply if control bit is one.
void QInterface::AntiCNOT(bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyAntiControlledSingleInvert(controls, 1, target, ONE_CMPLX, ONE_CMPLX);
}

/// Apply controlled "PhaseRootN" gate to bit
void QInterface::CPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))));
}

/// Apply controlled "IPhaseRootN" gate to bit
void QInterface::CIPhaseRootN(bitLenInt n, bitLenInt control, bitLenInt target)
{
    bitLenInt controls[1] = { control };
    ApplyControlledSinglePhase(controls, 1, target, ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / (bitCapIntOcl)(pow2(n - 1U))));
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

        ApplyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs + (bitCapIntOcl)(index * 4U));

        for (bitLenInt bit_pos = 0; bit_pos < controlLen; bit_pos++) {
            if (!((lcv >> bit_pos) & 1)) {
                X(controls[bit_pos]);
            }
        }
    }
}

void QInterface::TimeEvolve(Hamiltonian h, real1_f timeDiff_f)
{
    real1 timeDiff = (real1)timeDiff_f;
    
    // Exponentiation of an arbitrary serial string of gates, each HamiltonianOp component times timeDiff, e^(-i * H *
    // t) as e^(-i * H_(N - 1) * t) * e^(-i * H_(N - 2) * t) * ... e^(-i * H_0 * t)

    for (bitLenInt i = 0; i < h.size(); i++) {
        HamiltonianOpPtr op = h[i];
        complex* opMtrx = op->matrix.get();
        complex* mtrx;

        bitCapIntOcl maxJ = 4;
        if (op->uniform) {
            maxJ *= pow2Ocl(op->controlLen);
        }
        mtrx = new complex[maxJ];

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
            complex* expMtrx = new complex[maxJ];
            for (bitCapIntOcl j = 0; j < pow2(op->controlLen); j++) {
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
