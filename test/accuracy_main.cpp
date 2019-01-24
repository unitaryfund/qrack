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
bool disable_normalization = false;
bool async_time = false;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool qengine = false;
    bool qfusion = false;
    bool qunit = false;
    bool qunit_qfusion = false;
    bool cpu = false;
    bool opencl_single = false;
    bool opencl_multi = false;

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
        Opt(opencl_multi)["--proc-opencl-multi"]("Multiprocessor OpenCL tests") |
        Opt(disable_normalization)["--disable-normalization"]("Disable state vector normalization. (Usually less "
                                                              "accurate computation. Usually makes QEngine types "
                                                              "faster and QUnit types slower.)");

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

    session.config().stream() << "Random Seed: " << session.configData().rngSeed << std::endl;

    if (!qengine && !qfusion && !qunit && !qunit_qfusion) {
        qfusion = true;
        qunit = true;
        qunit_qfusion = true;
        qengine = true;
    }

    if (!cpu && !opencl_single && !opencl_multi) {
        cpu = true;
        opencl_single = true;
        opencl_multi = true;
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

#if 0
        if (num_failed == 0 && opencl_multi) {
            session.config().stream() << "############ QUnitMulti ############" << std::endl;
            testEngineType = QINTERFACE_QUNITMULTI;
            testSubEngineType = QINTERFACE_OPENCL;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(QINTERFACE_OPENCL, 1, 0).reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif

#endif
    }

    if (num_failed == 0 && qunit_qfusion) {
        testEngineType = QINTERFACE_QUNIT;
        if (num_failed == 0 && qunit_qfusion) {
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

    if (disable_normalization) {
        qftReg = CreateQuantumInterface(
            testEngineType, testSubEngineType, testSubSubEngineType, 20, 0, rng, complex(ONE_R1, ZERO_R1), false);
    } else {
        qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 20, 0, rng);
    }
}
