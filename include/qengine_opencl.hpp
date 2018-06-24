//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
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
#include <CL/cl2.hpp>
#endif

#include "qengine_cpu.hpp"

namespace Qrack {

typedef std::shared_ptr<cl::CommandQueue> CommandQueuePtr;
typedef std::shared_ptr<cl::Buffer> BufferPtr;

class OCLEngine;

class QEngineOCL;

typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCL : public QEngineCPU {
protected:
    int deviceID;
    OCLEngine* clObj;
    CommandQueuePtr queue;
    BufferPtr stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;

public:
    QEngineOCL(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr,
        int devID = -1, bool partialInit = false)
        : QEngineCPU(qBitCount, initState, rgp, complex(-999.0, -999.0), partialInit)
    {
        InitOCL(devID);
    }

    QEngineOCL(QEngineOCLPtr toCopy)
        : QEngineCPU(toCopy->GetQubitCount(), 0, toCopy->rand_generator, complex(-999.0, -999.0), false)
    {
        doNormalize = toCopy->doNormalize;
        InitOCL(toCopy->deviceID);
        CopyState(toCopy);
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
    }

    virtual BufferPtr GetStateBufferPtr() { return stateBuffer; }
    virtual CommandQueuePtr GetQueuePtr() { return queue; }

    /* Operations that have an improved implementation. */
    virtual void Swap(bitLenInt qubit1, bitLenInt qubit2); // Inherited overload
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);
    using QEngineCPU::Cohere;
    virtual bitLenInt Cohere(QEngineOCLPtr toCopy);
    virtual bitLenInt Cohere(QInterfacePtr toCopy) { return Cohere(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    using QEngineCPU::Decohere;
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    using QEngineCPU::X;
    virtual void X(bitLenInt start, bitLenInt length);
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    virtual real1 Prob(bitLenInt qubit);

    virtual int GetDeviceID() { return deviceID; }
    virtual void SetDevice(const int& dID);

protected:
    static const int BCI_ARG_LEN = 10;

    void InitOCL(int devID);
    void ReInitOCL();
    void ResetStateVec(complex* nStateVec);

    void DecohereDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);
    void DispatchCall(cl::Kernel* call, bitCapInt (&bciArgs)[BCI_ARG_LEN], complex* nVec = NULL,
        unsigned char* values = NULL, bitCapInt valuesLength = 0);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm);

    /* Utility functions used by the operations above. */
    void ROx(cl::Kernel* call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INT(cl::Kernel* call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length);
    void INTC(cl::Kernel* call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);

    bitCapInt OpIndexed(cl::Kernel* call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
};

} // namespace Qrack
