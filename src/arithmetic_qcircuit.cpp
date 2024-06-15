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

#include "qcircuit.hpp"

#if !ENABLE_ALU
#error ALU has not been enabled
#endif

namespace Qrack {

// Arithmetic:
/** Add integer (without sign) */
void QCircuit::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    if (!length) {
        return;
    }

    const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };

    if (length == 1U) {
        if (bi_and_1(toAdd)) {
            AppendGate(std::make_shared<QCircuitGate>(start, x));
        }
        return;
    }

    std::vector<bitLenInt> bits(length);
    for (bitLenInt i = 0U; i < length; ++i) {
        bits[i] = start + i;
    }

    const bitLenInt lengthMin1 = length - 1U;

    for (bitLenInt i = 0U; i < length; ++i) {
        if (!bi_and_1(toAdd >> i)) {
            continue;
        }
        const complex x[4]{ ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        AppendGate(std::make_shared<QCircuitGate>(start + i, x));
        for (bitLenInt j = 0U; j < (lengthMin1 - i); ++j) {

            // gather up arguments for QCircuitGate creation
            bitCapInt permutationOfControlsToActivateGate = ZERO_BCI;
            bitLenInt targetQubitIndex = start + ((i + j + 1U) % length);
            const std::set<bitLenInt> controlQubitIndices{ bits.begin() + i, bits.begin() + i + j + 1U };

            AppendGate(std::make_shared<QCircuitGate>(
                targetQubitIndex, x, controlQubitIndices, permutationOfControlsToActivateGate));
        }
    }
}
} // namespace Qrack
