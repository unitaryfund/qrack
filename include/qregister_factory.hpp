#pragma once

#include "qregister.hpp"
#include "separatedunit.hpp"

#if ENABLE_OPENCL
#include "qregister_opencl.hpp"
#endif

namespace Qrack {
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState);
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState,
    std::shared_ptr<std::default_random_engine> rgp);
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState,
    Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, CoherentUnit& pqs);
} // namespace Qrack
