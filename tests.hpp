#pragma once

#include "catch.hpp"

/*
 * Enumerated list of supported engines.
 *
 * Not currently published since selection isn't supported by the API.
 */
enum CoherentUnitEngine {
    COHERENT_UNIT_ENGINE_SOFTWARE = 0,
    COHERENT_UNIT_ENGINE_OPENCL,

    COHERENT_UNIT_ENGINE_MAX
};

/*
 * A fixture to create a unique CoherentUnit test, of the appropriate type, for
 * each executing test case.
 */
class CoherentUnitTestFixture {
protected:
    std::unique_ptr<Qrack::CoherentUnit> qftReg;

public:
    CoherentUnitTestFixture();
};


