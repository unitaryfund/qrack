//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

namespace Qrack {

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "qunit.hpp"

class OCLEngine;

/** OpenCL enhanced QUnit implementation. */
class QUnitOCL : public QUnit
{
protected:
    OCLEngine* clObj;
    cl::CommandQueue queue;
    cl::Buffer stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;
    cl::Buffer loadBuffer;

public:

    QUnitOCL(
        bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp)
        : QUnit(qBitCount, initState, phaseFac, rgp)
    {
        InitOCL();
    }

    /* Operations that have an improved implementation. */
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual unsigned char SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values);
    virtual unsigned char AdcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);
    virtual unsigned char SbcSuperposeReg8(
        bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);

protected:
    void InitOCL();
    void ReInitOCL();
    void ResetStateVec(std::unique_ptr<Complex16[]> nStateVec);

    void DispatchCall(cl::Kernel *call, bitCapInt (&bciArgs)[BCI_ARG_LEN], Complex16 *nVec = NULL, size_t nVecLen = 0, unsigned char* values = NULL);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount, const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);

    /* A couple utility functions used by the operations above. */
    void ROx(cl::Kernel *call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INTC(cl::Kernel* call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex);
}
};
