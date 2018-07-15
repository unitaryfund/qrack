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

#include "qengine_cpu.hpp"

namespace Qrack {

typedef std::shared_ptr<cl::Buffer> BufferPtr;

class OCLEngine;

class QEngineOCL;

typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCL : public QEngineCPU {
protected:
    //bool doNormalize;
    //complex* stateVec;
    int deviceID;
    DeviceContextPtr device_context;
    cl::CommandQueue queue;
    cl::Context context;
    // stateBuffer is allocated as a shared_ptr, because it's the only buffer that will be acted on outside of
    // QEngineOCL itself, specifically by QEngineOCLMulti.
    BufferPtr stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;

public:
    /**
     * Initialize a Qrack::QEngineOCL object. Specify the number of qubits and an initial permutation state.
     * Additionally, optionally specify a pointer to a random generator engine object, a device ID from the list of
     * devices in the OCLEngine singleton, and a boolean that is set to "true" to initialize the state vector of the
     * object to zero norm.
     *
     * "devID" is the index of an OpenCL device in the OCLEngine singleton, to select the device to run this engine on.
     * "partialInit" is usually only set to true when this object is one of several collected in a
     * Qrack::QEngineOCLMulti object, in which case this Qrack::QEngineOCL object might not contain the amplitude of the
     * overall permutation state of the combined object.
     */

    QEngineOCL(bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp = nullptr,
        int devID = -1, bool partialInit = false);
    QEngineOCL(QEngineOCLPtr toCopy);

    virtual void SetQubitCount(bitLenInt qb)
    {
        qubitCount = qb;
        maxQPower = 1 << qubitCount;
    }

    virtual complex* GetStateVector() { return stateVec; }
    virtual void SetPermutation(bitCapInt perm);

    /* Operations that have an improved implementation. */
    using QInterface::Swap;
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);
    using QInterface::Cohere;
    virtual bitLenInt Cohere(QEngineOCLPtr toCopy);
    virtual bitLenInt Cohere(QInterfacePtr toCopy) { return Cohere(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    using QInterface::Decohere;
    virtual void Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    using QInterface::X;
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

    virtual void SetQuantumState(complex* inputState);

    virtual void NormalizeState(real1 nrm = -999.0);
    virtual void UpdateRunningNorm();

protected:
    static const int BCI_ARG_LEN = 10;

    void InitOCL(int devID);
    void ReInitOCL();
    void ResetStateVec(complex* nStateVec);
    void ResetStateVec(complex* nStateVec, BufferPtr nStateBuffer);

    void DecohereDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);
    void DispatchCall(OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], complex* nVec = NULL,
        unsigned char* values = NULL, bitCapInt valuesLength = 0);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm);

    /* Utility functions used by the operations above. */
    void ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INT(OCLAPI api_call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length);
    void INTC(OCLAPI api_call, bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);

    bitCapInt OpIndexed(OCLAPI api_call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
};

} // namespace Qrack
