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

#pragma once

#include "qengine_cpu.hpp"
#include "qpager.hpp"

#if ENABLE_OPENCL
#include "qengine_opencl.hpp"
#include "qhybrid.hpp"
#include "qunitmulti.hpp"
#else
#include "qunit.hpp"
#endif

namespace Qrack {

/** Factory method to create specific engine implementations. */
template <typename... Ts>
QInterfacePtr CreateQuantumInterface(
    QInterfaceEngine engine, QInterfaceEngine subengine1, QInterfaceEngine subengine2, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(subengine1, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(subengine1, subengine2, args...);
#if ENABLE_OPENCL
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, QInterfaceEngine subengine, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
#endif
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(subengine, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(subengine, args...);
#if ENABLE_OPENCL
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts> QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine, Ts... args)
{
    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

} // namespace Qrack
