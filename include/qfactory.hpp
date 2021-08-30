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

#pragma once

#include "qengine_cpu.hpp"
#include "qpager.hpp"
#include "qstabilizerhybrid.hpp"

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
    QInterfaceEngine engine1, QInterfaceEngine engine2, QInterfaceEngine engine3, Ts... args)
{
    QInterfaceEngine engine = engine1;
    std::vector<QInterfaceEngine> engines = { engine2, engine3 };

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(engines, args...);
#endif
    default:
        return NULL;
    }
}

template <typename... Ts>
QInterfacePtr CreateQuantumInterface(QInterfaceEngine engine1, QInterfaceEngine engine2, Ts... args)
{
    QInterfaceEngine engine = engine1;
    std::vector<QInterfaceEngine> engines = { engine2 };

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(engines, args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(engines, args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(engines, args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        return std::make_shared<QUnitMulti>(engines, args...);
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
    case QINTERFACE_QPAGER:
        return std::make_shared<QPager>(args...);
    case QINTERFACE_STABILIZER_HYBRID:
        return std::make_shared<QStabilizerHybrid>(args...);
    case QINTERFACE_QUNIT:
        return std::make_shared<QUnit>(args...);
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

template <typename... Ts> QInterfacePtr CreateQuantumInterface(std::vector<QInterfaceEngine> engines, Ts... args)
{
    QInterfaceEngine engine = engines[0];
    engines.erase(engines.begin());

    switch (engine) {
    case QINTERFACE_CPU:
        return std::make_shared<QEngineCPU>(args...);
    case QINTERFACE_QPAGER:
        if (engines.size()) {
            return std::make_shared<QPager>(engines, args...);
        }
        return std::make_shared<QPager>(args...);
    case QINTERFACE_STABILIZER_HYBRID:
        if (engines.size()) {
            return std::make_shared<QStabilizerHybrid>(engines, args...);
        }
        return std::make_shared<QStabilizerHybrid>(args...);
    case QINTERFACE_QUNIT:
        if (engines.size()) {
            return std::make_shared<QUnit>(engines, args...);
        }
        return std::make_shared<QUnit>(args...);
#if ENABLE_OPENCL
    case QINTERFACE_OPENCL:
        return std::make_shared<QEngineOCL>(args...);
    case QINTERFACE_HYBRID:
        return std::make_shared<QHybrid>(args...);
    case QINTERFACE_QUNIT_MULTI:
        if (engines.size()) {
            return std::make_shared<QUnitMulti>(engines, args...);
        }
        return std::make_shared<QUnitMulti>(args...);
#endif
    default:
        return NULL;
    }
}

} // namespace Qrack
