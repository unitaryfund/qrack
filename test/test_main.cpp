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
bool sparse = false;
int device_id = -1;
bitLenInt max_qubits = 24;
std::string mOutputFileName;
std::ofstream mOutputFile;
bool isBinaryOutput;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool qengine = false;
    bool qpager = false;
    bool qunit = false;
    bool qunit_qpager = false;
    bool cpu = false;
    bool opencl_single = false;
    bool opencl_multi = false;
    bool hybrid = false;
    bool hybrid_multi = false;
    bool stabilizer = false;

    using namespace Catch::clara;

    /*
     * Allow specific layers and processor types to be enabled.
     */
    auto cli = session.cli() | Opt(qengine)["--layer-qengine"]("Enable Basic QEngine tests") |
        Opt(qpager)["--layer-qpager"]("Enable QPager implementation tests") |
        Opt(qunit)["--layer-qunit"]("Enable QUnit implementation tests") |
        Opt(qunit_qpager)["--layer-qunit-qpager"]("Enable QUnit with QPager implementation tests") |
        Opt(cpu)["--proc-cpu"]("Enable the CPU-based implementation tests") |
        Opt(opencl_single)["--proc-opencl-single"]("Single (parallel) processor OpenCL tests") |
        Opt(opencl_multi)["--proc-opencl-multi"]("Multiple processor OpenCL tests") |
        Opt(hybrid_multi)["--proc-hybrid-multi"]("Multiple processor hybrid CPU/OpenCL tests") |
        Opt(hybrid)["--proc-hybrid"]("Enable CPU/OpenCL hybrid implementation tests") |
        Opt(stabilizer)["--proc-stabilizer"]("Enable (hybrid) stabilizer implementation tests") |
        Opt(disable_hardware_rng)["--disable-hardware-rng"]("Modern Intel chips provide an instruction for hardware "
                                                            "random number generation, which this option turns off. "
                                                            "(Hardware generation is on by default, if available.)") |
        Opt(device_id, "device-id")["-d"]["--device-id"]("Opencl device ID (\"-1\" for default device)");

    session.cli(cli);

    /* Set some defaults for convenience. */
    session.configData().useColour = Catch::UseColour::No;
    session.configData().rngSeed = std::time(0);

    // session.configData().abortAfter = 1;

    /* Parse the command line. */
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    // If we're talking about a particular OpenCL device,
    // we have an API designed to tell us device capabilities and limitations,
    // like maximum RAM allocation.
#if ENABLE_OPENCL
    if (opencl_single || hybrid) {
        // Make sure the context singleton is initialized.
        CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset();

        DeviceContextPtr device_context = OCLEngine::Instance()->GetDeviceContextPtr(device_id);
        size_t maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / sizeof(complex);
        size_t maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() / sizeof(complex);

        // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
        max_qubits = log2(maxAlloc);
        if ((QEngineOCL::OclMemDenom * pow2(max_qubits)) > maxMem) {
            max_qubits = log2(maxMem / QEngineOCL::OclMemDenom);
        }
    }
#endif

    session.config().stream() << "Random Seed: " << session.configData().rngSeed;

#if ENABLE_RDRAND
    if (!disable_hardware_rng) {
        session.config().stream() << " (Overridden by hardware generation!)";
    }
#endif
    session.config().stream() << std::endl;

    if (!qengine && !qpager && !qunit && !qunit_qpager) {
        qunit = true;
        qengine = true;
        // Unstable:
        // qpager = true;
        // qunit_qpager = true;
    }

    if (!cpu && !opencl_single && !opencl_multi && !hybrid && !hybrid_multi && !stabilizer) {
        cpu = true;
        opencl_single = true;
        opencl_multi = true;
        hybrid = true;
        hybrid_multi = true;
        stabilizer = true;
    }

    int num_failed = 0;

    if (num_failed == 0 && qengine) {
        /* Perform the run against the default (software) variant. */
        if (num_failed == 0 && cpu) {
            testEngineType = QINTERFACE_CPU;
            testSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_OPENCL;
            testSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QHybrid ############" << std::endl;
            testEngineType = QINTERFACE_HYBRID;
            testSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QStabilizerHybrid -> QHybrid ############" << std::endl;
            testEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#else
        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QStabilizerHybrid -> QEngineCPU ############" << std::endl;
            testEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qpager) {
        testEngineType = QINTERFACE_QPAGER;
        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QPager -> QEngine -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QPager -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
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
            num_failed = session.run();
        }

        if (num_failed == 0 && cpu) {
            session.config().stream() << "############ QUnit -> QEngine -> CPU (Sparse) ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            sparse = true;
            num_failed = session.run();
            sparse = false;
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            session.config().stream() << "############ QUnit -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid) {
            session.config().stream() << "############ QUnit -> QHybrid ############" << std::endl;
            testSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_HYBRID, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QHybrid ############" << std::endl;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && opencl_multi) {
            session.config().stream() << "############ QUnitMulti -> QEngineOCL ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid_multi) {
            session.config().stream() << "############ QUnitMulti -> QHybrid ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && hybrid_multi && stabilizer) {
            session.config().stream() << "############ QUnitMulti -> QStabilizerHybrid -> QHybrid ############"
                                      << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_HYBRID;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#else
        if (num_failed == 0 && stabilizer) {
            session.config().stream() << "############ QUnit -> QStabilizerHybrid -> QEngineCPU ############"
                                      << std::endl;
            testSubEngineType = QINTERFACE_STABILIZER_HYBRID;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && qunit_qpager) {
        testEngineType = QINTERFACE_QUNIT;
        testSubEngineType = QINTERFACE_QPAGER;
        if (num_failed == 0 && cpu) {
            testSubSubEngineType = QINTERFACE_CPU;
            session.config().stream() << "############ QUnit -> QPager -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && opencl_single) {
            testSubSubEngineType = QINTERFACE_OPENCL;
            session.config().stream() << "############ QUnit -> QPager -> OpenCL ############" << std::endl;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && opencl_multi) {
            session.config().stream() << "############ QUnitMulti -> QPager (OpenCL) ############" << std::endl;
            testEngineType = QINTERFACE_QUNIT_MULTI;
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

    std::cout << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;

    if (rngSeed == 0) {
        rngSeed = std::time(0);
    }

    qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>();
    rng->seed(rngSeed);

    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 20, 0, rng, ONE_CMPLX,
        enable_normalization, true, false, device_id, !disable_hardware_rng, sparse);
}
