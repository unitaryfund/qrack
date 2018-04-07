#include "qregister.hpp"
#include "separatedunit.hpp"

#if ENABLE_OPENCL
#include "qregister_opencl.hpp"
#endif

namespace Qrack {
CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState)
{
    switch (engine) {
    case COHERENT_UNIT_ENGINE_OPTIMIZED:
        return new SeparatedUnit(qBitCount, initState);
    case COHERENT_UNIT_ENGINE_SOFTWARE:
        return new CoherentUnit(qBitCount, initState);
#if ENABLE_OPENCL
    case COHERENT_UNIT_ENGINE_OPENCL:
        return new CoherentUnitOCL(qBitCount, initState);
#endif
    default:
        return NULL;
    }
}
} // namespace Qrack
