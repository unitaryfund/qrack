//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "qregister.hpp"

namespace Qrack {

class OCLEngine;

class CoherentUnitOCL : public CoherentUnit {
public:
    CoherentUnitOCL(bitLenInt qBitCount);
    CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState);
    CoherentUnitOCL(const CoherentUnitOCL& pqs);

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    //virtual void ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);
    //virtual void SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length);

protected:
    OCLEngine* clObj;
    cl::CommandQueue queue;
    cl::Buffer stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;

    virtual void InitOCL();
    virtual void ReInitOCL();
    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);
};
} // namespace Qrack
