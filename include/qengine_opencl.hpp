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

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "qengine_cpu.hpp"

namespace Qrack {

class OCLEngine;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCL : public QEngineCPU
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

    QEngineOCL(
        bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr)
        : QEngineCPU(qBitCount, initState, rgp)
    {
        InitOCL();
    }

    /* Operations that have an improved implementation. */
    virtual void X(bitLenInt qubit); //Inherited overload
    virtual void X(bitLenInt start, bitLenInt length);
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);

protected:
    static const int BCI_ARG_LEN = 10;

    void InitOCL();
    void ReInitOCL();
    void ResetStateVec(Complex16 *nStateVec);

    void DispatchCall(cl::Kernel *call, bitCapInt (&bciArgs)[BCI_ARG_LEN], Complex16 *nVec = NULL, unsigned char* values = NULL);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount, const bitCapInt* qPowersSorted, bool doCalcNorm);

    /* Utility functions used by the operations above. */
    void ROx(cl::Kernel *call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INT(cl::Kernel* call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length);
    void INTC(cl::Kernel* call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex);

    unsigned char OpSuperposeReg8(cl::Kernel *call, bitCapInt carryIn, bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values);
};

}
