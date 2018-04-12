#pragma once

/** Enumerated list of supported engines. */
enum CoherentUnitEngine {
    COHERENT_UNIT_ENGINE_SOFTWARE_SERIAL = 0,
    COHERENT_UNIT_ENGINE_SOFTWARE_PARALLEL,
    COHERENT_UNIT_ENGINE_SOFTWARE = COHERENT_UNIT_ENGINE_SOFTWARE_PARALLEL,

    COHERENT_UNIT_ENGINE_SOFTWARE_SEPARATED,

    COHERENT_UNIT_ENGINE_OPENCL,

    COHERENT_UNIT_ENGINE_OPENCL_SEPARATED,

    COHERENT_UNIT_ENGINE_MAX
};

/** Create a CoherentUnit leveraging the specified engine. */
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState);


