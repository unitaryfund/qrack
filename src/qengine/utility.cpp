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

#include "qfactory.hpp"

namespace Qrack {

QInterfacePtr QEngineCPU::Clone()
{
    Finish();

    QInterfacePtr clone = CreateQuantumInterface(QINTERFACE_CPU, qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, false, 0, (hardware_rand_generator == NULL) ? false : true, isSparse);
    if (stateVec) {
        std::dynamic_pointer_cast<QEngineCPU>(clone)->stateVec->copy(stateVec);
    }
    return clone;
}

} // namespace Qrack
