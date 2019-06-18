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

// Arithmetic:
/// Subtract integer (without sign)
void QInterface::DEC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt invToSub = (1U << length) - toSub;
    INC(invToSub, inOutStart, length);
}

/**
 * Subtract an integer from the register, with sign and without carry. Because the register length is an arbitrary
 * number of bits, the sign bit position on the integer to add is variable. Hence, the integer to add is specified as
 * cast to an unsigned format, with the sign bit assumed to be set at the appropriate position before the cast.
 */
void QInterface::DECS(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
{
    bitCapInt invToSub = (1U << length) - toSub;
    INCS(invToSub, inOutStart, length, overflowIndex);
}

/// Subtract integer (without sign, with controls)
void QInterface::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    bitCapInt invToSub = (1U << length) - toSub;
    CINC(invToSub, inOutStart, length, controls, controlLen);
}

/// Subtract BCD integer (without sign)
void QInterface::DECBCD(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length)
{
    bitCapInt invToSub = intPow(10U, length / 4U) - toSub;
    INCBCD(invToSub, inOutStart, length);
}

} // namespace Qrack
