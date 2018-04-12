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
    CoherentUnitOCL(bitLenInt qBitCount, Complex16 phaseFac);
    CoherentUnitOCL(bitLenInt qBitCount, std::shared_ptr<std::default_random_engine> rgp);
    CoherentUnitOCL(bitLenInt qBitCount, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);
    CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState);
    CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac);
    CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp);
    CoherentUnitOCL(
        bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp);
    CoherentUnitOCL(const CoherentUnitOCL& pqs);
    CoherentUnitOCL(const CoherentUnit& pqs);

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    virtual unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);
    virtual unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);
    virtual unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

protected:
    CoherentUnitOCL();

    OCLEngine* clObj;
    cl::CommandQueue queue;
    cl::Buffer stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;
    cl::Buffer loadBuffer;

    virtual void InitOCL();
    virtual void ReInitOCL();
    virtual void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);
};
} // namespace Qrack
