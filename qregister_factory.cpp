#include "qregister_factory.hpp"

namespace Qrack {

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState)
{
    return CreateCoherentUnit(engine, qBitCount, initState, NULL);
}

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState,
    std::shared_ptr<std::default_random_engine> rgp)
{
    return CreateCoherentUnit(engine, qBitCount, initState, Complex16(-999.0, -999.0), rgp);
}

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, bitLenInt qBitCount, bitCapInt initState,
    Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp)
{
    switch (engine) {
    case COHERENT_UNIT_ENGINE_SOFTWARE:
        return new CoherentUnit(qBitCount, initState, phaseFac, rgp);
    case COHERENT_UNIT_ENGINE_SOFTWARE_SEPARATED:
        return new SeparatedUnit(qBitCount, initState, phaseFac, COHERENT_UNIT_ENGINE_SOFTWARE, rgp);
#if ENABLE_OPENCL
    case COHERENT_UNIT_ENGINE_OPENCL:
        return new CoherentUnitOCL(qBitCount, initState, phaseFac, rgp);
    case COHERENT_UNIT_ENGINE_OPENCL_SEPARATED:
        return new SeparatedUnit(qBitCount, initState, phaseFac, COHERENT_UNIT_ENGINE_OPENCL, rgp);
#endif
    default:
        return NULL;
    }
}

CoherentUnit* CreateCoherentUnit(CoherentUnitEngine engine, CoherentUnit& pqs)
{
    switch (engine) {
    case COHERENT_UNIT_ENGINE_SOFTWARE:
        return new CoherentUnit(pqs);
    case COHERENT_UNIT_ENGINE_SOFTWARE_SEPARATED:
        return new SeparatedUnit(pqs);
#if ENABLE_OPENCL
    case COHERENT_UNIT_ENGINE_OPENCL:
        return new CoherentUnitOCL(pqs);
    case COHERENT_UNIT_ENGINE_OPENCL_SEPARATED:
        return new SeparatedUnit(pqs);
#endif
    default:
        return NULL;
    }
}
} // namespace Qrack
