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
std::shared_ptr<std::default_random_engine> rng;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool disable_qengine = false;
    bool disable_opencl = false;
    bool disable_cpu = false;
    bool disable_qunit = false;
    bool disable_opencl_multi = false;
    bool disable_single = false;
    bool disable_qfusion = false;

    using namespace Catch::clara;

    /*
     * Allow disabling running OpenCL tests on the command line, even if
     * supported.
     */
    auto cli = session.cli() | Opt(disable_qengine)["--disable-qengine"]("Disable basic QEngine tests") |
        Opt(disable_opencl)["--disable-opencl"]("Disable OpenCL even if supported") |
        Opt(disable_qunit)["--disable-qunit"]("Disable QUnit implementation tests") |
        Opt(disable_cpu)["--disable-cpu"]("Disable the CPU-based implementation tests") |
        Opt(disable_opencl_multi)["--disable-opencl-multi"]("Disable multiple device OpenCL tests") |
        Opt(disable_single)["--disable-single"]("Disable single processor engine tests") |
        Opt(disable_qfusion)["--disable-qfusion"]("Disable gate fusion tests");

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

    int num_failed = 0;

    if (num_failed == 0 && !disable_qengine) {
        /* Perform the run against the default (software) variant. */
        if (num_failed == 0 && !disable_cpu && !disable_single) {
            session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && !disable_opencl && !disable_single) {
            session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_OPENCL;
            testSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(testEngineType, testSubEngineType, 1, 0)
                .reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && !disable_qfusion) {
        testEngineType = QINTERFACE_QFUSION;
        testSubEngineType = QINTERFACE_CPU;
        if (num_failed == 0 && !disable_cpu && !disable_single) {
            session.config().stream() << "############ QFusion -> CPU ############" << std::endl;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && !disable_opencl && !disable_single) {
            session.config().stream() << "############ QFusion -> OpenCL ############" << std::endl;
            testEngineType = QINTERFACE_QFUSION;
            testSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }
#endif
    }

    if (num_failed == 0 && !disable_qunit) {
        testEngineType = QINTERFACE_QUNIT;
        if (!disable_cpu && !disable_single) {
            session.config().stream() << "############ QUnit -> QEngine -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && !disable_opencl && !disable_single) {
            session.config().stream() << "############ QUnit -> QEngine -> OpenCL ############" << std::endl;
            testSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(testEngineType, testSubEngineType, 1, 0)
                .reset(); /* Get the OpenCL banner out of the way. */
            num_failed = session.run();
        }

        if (num_failed == 0 && !disable_opencl && !disable_opencl_multi) {
            session.config().stream() << "############ QUnitMulti ############" << std::endl;
            testEngineType = QINTERFACE_QUNITMULTI;
            testSubEngineType = QINTERFACE_OPENCL;
            num_failed = session.run();
        }
#endif

        if (num_failed == 0 && !disable_qfusion) {
        testEngineType = QINTERFACE_QUNIT;
        if (!disable_cpu && !disable_single) {
            session.config().stream() << "############ QUnit -> QFusion -> CPU ############" << std::endl;
            testSubEngineType = QINTERFACE_QFUSION;
            testSubSubEngineType = QINTERFACE_CPU;
            num_failed = session.run();
        }

#if ENABLE_OPENCL
        if (num_failed == 0 && !disable_opencl && !disable_single) {
            session.config().stream() << "############ QUnit -> QFusion -> OpenCL ############" << std::endl; 
            testSubEngineType = QINTERFACE_QFUSION;
            testSubSubEngineType = QINTERFACE_OPENCL;
            CreateQuantumInterface(testEngineType, testSubEngineType, 1, 0)
                .reset(); /* Get the OpenCL banner out of the way. */
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

    std::shared_ptr<std::default_random_engine> rng = std::make_shared<std::default_random_engine>();
    rng->seed(rngSeed);

    qftReg = CreateQuantumInterface(testEngineType, testSubEngineType, testSubSubEngineType, 20, 0, rng);
}
