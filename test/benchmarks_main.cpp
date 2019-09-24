//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "qfactory.hpp"

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

using namespace Qrack;

enum QInterfaceEngine testEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubEngineType = QINTERFACE_CPU;
enum QInterfaceEngine testSubSubEngineType = QINTERFACE_CPU;
qrack_rand_gen_ptr rng;
bool enable_normalization = false;
bool disable_hardware_rng = false;
bool async_time = false;
int device_id = -1;
bitLenInt max_qubits = 24;
bool single_qubit_run = false;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool qengine = false;
    bool qfusion = false;
    bool qunit = false;
    bool qunit_qfusion = false;
    bool cpu = false;
    bool opencl_single = false;

    int mxQbts = 24;

    using namespace Catch::clara;

    /*
     * Allow specific layers and processor types to be enabled.
     */
    auto cli = session.cli() | Opt(qengine)["--layer-qengine"]("Enable Basic QEngine tests") |
        Opt(qfusion)["--layer-qfusion"]("Enable gate fusion (without QUnit) tests") |
        Opt(qunit)["--layer-qunit"]("Enable QUnit (without gate fusion) implementation tests") |
        Opt(qunit_qfusion)["--layer-qunit-qfusion"]("Enable gate fusion tests under the QUnit layer") |
        Opt(cpu)["--proc-cpu"]("Enable the CPU-based implementation tests") |
        Opt(opencl_single)["--proc-opencl-single"]("Single (parallel) processor OpenCL tests") |
        Opt(async_time)["--async-time"]("Time based on asynchronous return") |
        Opt(enable_normalization)["--enable-normalization"](
            "Enable state vector normalization. (Usually not "
            "necessary, though might benefit accuracy at very high circuit depth.)") |
        Opt(disable_hardware_rng)["--disable-hardware-rng"]("Modern Intel chips provide an instruction for hardware "
                                                            "random number generation, which this option turns off. "
                                                            "(Hardware generation is on by default, if available.)") |
        Opt(device_id, "device-id")["-d"]["--device-id"]("Opencl device ID (\"-1\" for default device)") |
        Opt(mxQbts, "max-qubits")["-m"]["--max-qubits"](
            "Maximum qubits for test (default value 24, enter \"-1\" for automatic selection)") |
        Opt(single_qubit_run)["--single"]("Only run single (maximum) qubit count for tests");

    session.cli(cli);

    /* Set some defaults for convenience. */
    session.configData().useColour = Catch::UseColour::No;
    session.configData().reporterNames = { "compact" };
    session.configData().rngSeed = std::time(0);

    // session.configData().abortAfter = 1;

    /* Parse the command line. */
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    session.config().stream() << "Random Seed: " << session.configData().rngSeed;

    if (disable_hardware_rng) {
        session.config().stream() << std::endl;
    } else {
        session.config().stream() << " (Overridden by hardware generation!)" << std::endl;
    }

    if (!qengine && !qfusion && !qunit && !qunit_qfusion) {
        qfusion = true;
        qunit = true;
        qunit_qfusion = true;
        qengine = true;
    }

    if (!cpu && !opencl_single) {
        cpu = true;
        opencl_single = true;
    }

    if (mxQbts == -1) {
        // If we're talking about a particular OpenCL device,
        // we have an API designed to tell us device capabilities and limitations,
        // like maximum RAM allocation.
        if (opencl_single) {
#if ENABLE_OPENCL
            // Make sure the context singleton is initialized.
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset();

            DeviceContextPtr device_context = OCLEngine::Instance()->GetDeviceContextPtr(device_id);
            size_t maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / sizeof(complex);
            size_t maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / sizeof(complex);

            // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
            max_qubits = log2(maxAlloc);
            if ((3 * (1U << max_qubits)) > maxMem ) {
                max_qubits = log2(maxMem / 3);
            }
#else
        // With OpenCL tests disabled, it's ambiguous what device we want to set the limit by.
        // If we're not talking about the OpenCL resources of a single device,
        // maximum allocation becomes a notoriously thorny matter.
        // For any case besides the above, we just use the default.
#endif
        }
    } else {
        max_qubits = mxQbts;
    }

    int num_failed = 0;

    if (num_failed == 0 && qengine) {
        /* Perform the run against the default (software) variant. */
        if (num_failed == 0 && cpu) {
            testEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            testSubSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_OPENCL;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qfusion) {
        testEngineType = QINTERFACE_QFUSION;
        testSubEngineType = QINTERFACE_CPU;
        testSubSubEngineType = QINTERFACE_CPU;
        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QFusion -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QFusion -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_QFUSION;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qunit) {
        testEngineType = QINTERFACE_QUNIT;
        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QUnit -> QEngine -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QUnit -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qunit_qfusion) {
        testEngineType = QINTERFACE_QUNIT;
        testSubEngineType = QINTERFACE_QFUSION;
        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QUnit -> QFusion -> CPU ############" << std::endl;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QUnit -> QFusion -> OpenCL ############" << std::endl;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    return num_failed;
}

QInterfaceTestFixture::QInterfaceTestFixture()
{
    uint32_t rngSeed = Catch::getCurrentContext().getConfig()->rngSeed();

    if (rngSeed == 0) {
        rngSeed = std::time(0);
    }

    qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>();
    rng->seed(rngSeed);

    qftReg = NULL;
}
