//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qinterface.hpp"
#include "qunit.hpp"

#if ENABLE_OPENCL
#include "qunit_opencl.hpp"
#endif

namespace Qrack {

template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, Ts ... args)
{
    switch (engine) {
    case QUNIT_ENGINE_SOFTWARE:
        return std::make_shared<QUnit>(args...);
#if ENABLE_OPENCL
    case QUNIT_ENGINE_OPENCL:
        return std::make_shared<QUnitOCL>(args...);
#endif
    default:
        return NULL;
    }
}

} // namespace Qrack
