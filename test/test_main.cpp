#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "qregister.hpp"

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

using namespace Qrack;

#if ENABLE_OPENCL
#include "qregister_opencl.hpp"
#endif

/*
 * Default engine type to run the tests with. Global because catch doesn't
 * support parameterization.
 */
enum CoherentUnitEngine testEngineType = COHERENT_UNIT_ENGINE_SOFTWARE;

int main(int argc, char* argv[])
{
    Catch::Session session;

    bool disable_opencl = false;
    bool disable_software = false;
    bool disable_opencl_optimized = false;
    bool disable_software_optimized = false;

    using namespace Catch::clara;

    /*
     * Allow disabling running OpenCL tests on the command line, even if
     * supported.
     */
    auto cli = session.cli() | Opt(disable_opencl)["--disable-opencl"]("Disable OpenCL even if supported") |
        Opt(disable_software)["--disable-software"]("Disable the software implementation tests") |
        Opt(disable_opencl_optimized)["--disable-opencl-optimized"](
            "Disable the optimized OpenCL implementation tests") |
        Opt(disable_software_optimized)["--disable-software-optimized"](
            "Disable the optimized software implementation tests");
    ;

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

    /* Perform the run against the default (software) variant. */
    if (!disable_software) {
        session.config().stream() << "Executing test suite using the Software Implementation" << std::endl;
        num_failed = session.run();
    }

    if (num_failed == 0 && !disable_software_optimized) {
        session.config().stream() << "Executing test suite using the Optimized Software Implementation" << std::endl;
        testEngineType = COHERENT_UNIT_ENGINE_SOFTWARE_SEPARATED;
        num_failed = session.run();
    }

#if ENABLE_OPENCL
    if (num_failed == 0 && !disable_opencl) {
        session.config().stream() << "Executing test suite using the OpenCL Implementation" << std::endl;
        testEngineType = COHERENT_UNIT_ENGINE_OPENCL;
        delete CreateCoherentUnit(testEngineType, 1, 0); /* Get the OpenCL banner out of the way. */
        num_failed = session.run();
    }

    if (num_failed == 0 && !disable_opencl_optimized) {
        session.config().stream() << "Executing test suite using the Optimized OpenCL Implementation" << std::endl;
        testEngineType = COHERENT_UNIT_ENGINE_OPENCL_SEPARATED;
        num_failed = session.run();
    }
#endif

    return num_failed;
}

CoherentUnitTestFixture::CoherentUnitTestFixture()
{
    uint32_t rngSeed = Catch::getCurrentContext().getConfig()->rngSeed();

    if (rngSeed == 0) {
        rngSeed = std::time(0);
    }

    qftReg.reset(CreateCoherentUnit(testEngineType, 20, 0));
    qftReg->SetRandomSeed(rngSeed);
}
