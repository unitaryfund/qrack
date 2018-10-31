//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include "qengine_cpu.hpp"

#if ENABLE_OPENCL
#include "qengine_opencl_multi.hpp"
#include "qunitmulti.hpp"
#endif

namespace Qrack {

/** Factory method to create specific engine implementations. */
template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, QInterfaceEngine subengine, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
    case QINTERFACE_OPENCL_MULTI:
        return std::make_shared<QEngineOCLMulti>(args...);
    case QINTERFACE_QUNITMULTI:
        return std::make_shared<QUnitMulti>(args...);
#endif
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(subengine, args...);
    default:
        return NULL;
    }
}

template <typename... Ts>
QEnginePtr CreateQuantumEngine(QInterfaceEngine engine, QInterfaceEngine subengine, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
    default:
        return NULL;
    }
}

} // namespace Qrack
