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

namespace Qrack {

/// General unitary gate
void QInterface::U(bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    Mtrx(uGate, target);
}

/// Controlled general unitary gate
void QInterface::CU(
    const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    MCMtrx(controls, controlLen, uGate, target);
}

/// (Anti-)Controlled general unitary gate
void QInterface::AntiCU(
    const bitLenInt* controls, bitLenInt controlLen, bitLenInt target, real1_f theta, real1_f phi, real1_f lambda)
{
    const real1 cos0 = (real1)cos(theta / 2);
    const real1 sin0 = (real1)sin(theta / 2);
    const complex uGate[4] = { complex(cos0, ZERO_R1), sin0 * complex((real1)(-cos(lambda)), (real1)(-sin(lambda))),
        sin0 * complex((real1)cos(phi), (real1)sin(phi)),
        cos0 * complex((real1)cos(phi + lambda), (real1)sin(phi + lambda)) };
    MACMtrx(controls, controlLen, uGate, target);
}

/// "Azimuth, Inclination"
void QInterface::AI(bitLenInt target, real1_f azimuth, real1_f inclination)
{
    real1 cosineA = (real1)cos(azimuth);
    real1 sineA = (real1)sin(azimuth);
    real1 cosineI = (real1)cos(inclination / 2);
    real1 sineI = (real1)sin(inclination / 2);
    complex expA = complex(cosineA, sineA);
    complex expNegA = complex(cosineA, -sineA);
    complex mtrx[4] = { cosineI, -expNegA * sineI, expA * sineI, cosineI };
    Mtrx(mtrx, target);
}

/// Inverse "Azimuth, Inclination"
void QInterface::IAI(bitLenInt target, real1_f azimuth, real1_f inclination)
{
    real1 cosineA = (real1)cos(azimuth);
    real1 sineA = (real1)sin(azimuth);
    real1 cosineI = (real1)cos(inclination / 2);
    real1 sineI = (real1)sin(inclination / 2);
    complex expA = complex(cosineA, sineA);
    complex expNegA = complex(cosineA, -sineA);
    complex mtrx[4] = { cosineI, -expNegA * sineI, expA * sineI, cosineI };
    complex invMtrx[4];
    inv2x2(mtrx, invMtrx);
    Mtrx(invMtrx, target);
}

/// Uniformly controlled y axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli y axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRY(
    const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex, const real1* angles)
{
    bitCapIntOcl permCount = pow2Ocl(controlLen);
    std::unique_ptr<complex[]> pauliRYs(new complex[4U * permCount]);

    for (bitCapIntOcl i = 0; i < permCount; i++) {
        real1 cosine = (real1)cos(angles[i] / 2);
        real1 sine = (real1)sin(angles[i] / 2);

        pauliRYs[0U + 4U * i] = complex(cosine, ZERO_R1);
        pauliRYs[1U + 4U * i] = complex(-sine, ZERO_R1);
        pauliRYs[2U + 4U * i] = complex(sine, ZERO_R1);
        pauliRYs[3U + 4U * i] = complex(cosine, ZERO_R1);
    }

    UniformlyControlledSingleBit(controls, controlLen, qubitIndex, pauliRYs.get());
}

/// Uniformly controlled z axis rotation gate - Rotates as e^(-i*\theta_k/2) around Pauli z axis for each permutation
/// "k" of the control bits.
void QInterface::UniformlyControlledRZ(
    const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex, const real1* angles)
{
    bitCapIntOcl permCount = pow2Ocl(controlLen);
    std::unique_ptr<complex[]> pauliRZs(new complex[4U * permCount]);

    for (bitCapIntOcl i = 0; i < permCount; i++) {
        real1 cosine = (real1)cos(angles[i] / 2);
        real1 sine = (real1)sin(angles[i] / 2);

        pauliRZs[0U + 4U * i] = complex(cosine, -sine);
        pauliRZs[1U + 4U * i] = ZERO_CMPLX;
        pauliRZs[2U + 4U * i] = ZERO_CMPLX;
        pauliRZs[3U + 4U * i] = complex(cosine, sine);
    }

    UniformlyControlledSingleBit(controls, controlLen, qubitIndex, pauliRZs.get());
}

/// "Phase shift gate" - Rotates as e^(-i*\theta/2) around |1> state
void QInterface::RT(real1_f radians, bitLenInt qubit)
{
    Phase(ONE_CMPLX, complex((real1)cos(radians / 2), (real1)sin(radians / 2)), qubit);
}

/// x axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::RX(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRX[4] = { complex(cosine, ZERO_R1), complex(ZERO_R1, -sine), complex(ZERO_R1, -sine),
        complex(cosine, ZERO_R1) };
    Mtrx(pauliRX, qubit);
}

/// y axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli y axis
void QInterface::RY(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRY[4] = { cosine, -sine, sine, cosine };
    Mtrx(pauliRY, qubit);
}

/// z axis rotation gate - Rotates as e^(-i*\theta/2) around Pauli z axis
void QInterface::RZ(real1_f radians, bitLenInt qubit)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    Phase(complex(cosine, -sine), complex(cosine, sine), qubit);
}

/// Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis
void QInterface::CRZ(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const bitLenInt controls[1] = { control };
    MCPhase(controls, 1, complex(cosine, -sine), complex(cosine, sine), target);
}

#if ENABLE_ROT_API
/// Exponentiate identity operator
void QInterface::Exp(real1_f radians, bitLenInt qubit)
{
    complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Phase(phaseFac, phaseFac, qubit);
}

/// Imaginary exponentiate of arbitrary single bit gate
void QInterface::Exp(
    const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubit, const complex* matrix2x2, bool antiCtrled)
{
    complex timesI[4];
    for (bitLenInt i = 0; i < 4; i++) {
        timesI[i] = I_CMPLX * matrix2x2[i];
    }
    complex toApply[4];
    exp2x2(timesI, toApply);
    if (controlLen == 0) {
        Mtrx(toApply, qubit);
    } else if (antiCtrled) {
        MACMtrx(controls, controlLen, toApply, qubit);
    } else {
        MCMtrx(controls, controlLen, toApply, qubit);
    }
}

/// Exponentiate Pauli X operator
void QInterface::ExpX(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Invert(phaseFac, phaseFac, qubit);
}

/// Exponentiate Pauli Y operator
void QInterface::ExpY(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Invert(phaseFac * -I_CMPLX, phaseFac * I_CMPLX, qubit);
}

/// Exponentiate Pauli Z operator
void QInterface::ExpZ(real1_f radians, bitLenInt qubit)
{
    const complex phaseFac = complex((real1)cos(radians), (real1)sin(radians));
    Phase(phaseFac, -phaseFac, qubit);
}

/// Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta/2) around |1> state
void QInterface::CRT(real1_f radians, bitLenInt control, bitLenInt target)
{
    const bitLenInt controls[1] = { control };
    MCPhase(controls, 1, ONE_CMPLX, complex((real1)cos(radians / 2), (real1)sin(radians / 2)), target);
}

/// Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta/2) around Pauli x axis
void QInterface::CRX(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRX[4] = { complex(cosine, ZERO_R1), complex(ZERO_R1, sine), complex(ZERO_R1, sine),
        complex(cosine, ZERO_R1) };
    const bitLenInt controls[1] = { control };
    MCMtrx(controls, 1, pauliRX, target);
}

/// Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis
void QInterface::CRY(real1_f radians, bitLenInt control, bitLenInt target)
{
    const real1 cosine = (real1)cos(radians / 2);
    const real1 sine = (real1)sin(radians / 2);
    const complex pauliRY[4] = { complex(cosine, ZERO_R1), complex(-sine, ZERO_R1), complex(sine, ZERO_R1),
        complex(cosine, ZERO_R1) };
    const bitLenInt controls[1] = { control };
    MCMtrx(controls, 1, pauliRY, target);
}
#endif
} // namespace Qrack
