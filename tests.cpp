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

    using namespace Catch::clara;

    /*
     * Allow disabling running OpenCL tests on the command line, even if
     * supported.
     */
    auto cli = session.cli() | Opt(disable_opencl)["--disable-opencl"]("Disable OpenCL even if supported");
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

    /* Perform the run against the default (software) variant. */
    int num_failed = session.run();

#if ENABLE_OPENCL
    if (num_failed == 0 && !disable_opencl) {
        session.config().stream() << "Executing test suite using OpenCL" << std::endl;
        testEngineType = COHERENT_UNIT_ENGINE_OPENCL;
        delete CreateCoherentUnit(testEngineType, 1, 0); /* Get the OpenCL banner out of the way. */
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
