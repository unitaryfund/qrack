//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#include "common/oclengine.hpp"
#include "qengine.hpp"

namespace Qrack {

typedef std::shared_ptr<cl::Buffer> BufferPtr;

class OCLEngine;

class QEngineOCL;

typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

/**
 * OpenCL enhanced QEngineCPU implementation.
 *
 * QEngineOCL exposes asynchronous void-return public methods, wherever possible. While QEngine public methods run on a
 * secondary accelerator, such as a GPU, other code can be executed on the CPU at the same time. If only one (CPU)
 * OpenCL device is available, this engine type is still compatible with most CPUs, and this implementation will still
 * usually give a very significant performance boost over the non-OpenCL QEngineCPU implementation.
 *
 * Each QEngineOCL queues an independent event list of chained asynchronous methods. Multiple QEngineOCL instances may
 * share a single device. Any one QEngineOCL instance (or QEngineCPU instance) is NOT safe to access from multiple
 * threads, but different QEngineOCL instances may be accessed in respective threads. When a public method with a
 * non-void return type is called, (such as Prob() or M() variants,) the engine wait list of OpenCL events will first be
 * finished, then the return value will be calculated based on all public method calls dispatched up to that point.
 * Asynchronous method dispatch is "transparent," in the sense that no explicit consideration for synchronization should
 * be necessary. The programmer benefits from knowing that void-return methods attempt asynchronous execution, but
 * asynchronous methods are always joined, in order of dispatch, before any and all non-void-return methods give their
 * results.
 */
class QEngineOCL : public QEngine {
protected:
    complex* stateVec;
    int deviceID;
    DeviceContextPtr device_context;
    std::vector<EventVecPtr> wait_refs;
    cl::CommandQueue queue;
    cl::Context context;
    // stateBuffer is allocated as a shared_ptr, because it's the only buffer that will be acted on outside of
    // QEngineOCL itself, specifically by QEngineOCLMulti.
    BufferPtr stateBuffer;
    BufferPtr cmplxBuffer;
    BufferPtr realBuffer;
    BufferPtr ulongBuffer;
    BufferPtr nrmBuffer;
    BufferPtr powersBuffer;
    real1* nrmArray;
    size_t nrmGroupCount;
    size_t nrmGroupSize;
    size_t maxWorkItems;
    size_t maxMem;
    size_t maxAlloc;
    unsigned int procElemCount;
    bool unlockHostMem;

public:
    /**
     * Initialize a Qrack::QEngineOCL object. Specify the number of qubits and an initial permutation state.
     * Additionally, optionally specify a pointer to a random generator engine object, a device ID from the list of
     * devices in the OCLEngine singleton, and a boolean that is set to "true" to initialize the state vector of the
     * object to zero norm.
     *
     * "devID" is the index of an OpenCL device in the OCLEngine singleton, to select the device to run this engine on.
     * If "useHostMem" is set false, as by default, the QEngineOCL will attempt to allocate the state vector object
     * only on device memory. If "useHostMem" is set true, general host RAM will be used for the state vector buffers.
     * If the state vector is too large to allocate only on device memory, the QEngineOCL will attempt to fall back to
     * allocating it in general host RAM.
     *
     * \warning "useHostMem" is not conscious of allocation by other QEngineOCL instances on the same device. Attempting
     * to allocate too much device memory across too many QEngineOCL instances, for which each instance would have
     * sufficient device resources on its own, will probably cause the program to crash (and may lead to general system
     * instability). For safety, "useHostMem" can be turned on.
     */

    QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = complex(-999.0, -999.0), bool doNorm = true, bool randomGlobalPhase = true,
        bool useHostMem = false, int devID = -1);
    QEngineOCL(QEngineOCLPtr toCopy);
    ~QEngineOCL()
    {
        clFinish();

        FreeStateVec();

        if (nrmArray) {
            free(nrmArray);
        }
    }

    virtual void SetQubitCount(bitLenInt qb);

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = complex(-999.0, -999.0));
    virtual void CopyState(QInterfacePtr orig);
    virtual real1 ProbAll(bitCapInt fullRegister);

    virtual void UniformlyControlledSingleBit(
        const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs);

    /* Operations that have an improved implementation. */
    using QEngine::X;
    virtual void X(bitLenInt start, bitLenInt length);
    using QEngine::Swap;
    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length);

    using QEngine::Compose;
    virtual bitLenInt Compose(QEngineOCLPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    virtual bitLenInt Compose(QEngineOCLPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy), start);
    }
    virtual void Compose(OCLAPI apiCall, bitCapInt* bciArgs, QEngineOCLPtr toCopy);
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void DEC(bitCapInt toSub, bitLenInt start, bitLenInt length);
    virtual void CDEC(
        bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECS(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void DECSC(
        bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
    virtual void INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void DECBCDC(bitCapInt toSub, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    virtual real1 Prob(bitLenInt qubit);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual void ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    virtual void ProbMaskAll(const bitCapInt& mask, real1* probsArray);

    virtual void PhaseFlip();
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);

    virtual void SetDevice(const int& dID, const bool& forceReInit = false);
    virtual int GetDeviceID() { return deviceID; }

    virtual void SetQuantumState(complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    complex GetAmplitude(bitCapInt perm);

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QEngineOCL>(toCompare));
    }
    virtual bool ApproxCompare(QEngineOCLPtr toCompare);

    virtual void NormalizeState(real1 nrm = -999.0);
    virtual void UpdateRunningNorm();
    virtual void Finish() { clFinish(); };

    virtual QInterfacePtr Clone();

protected:
    static const int BCI_ARG_LEN = 10;

    void InitOCL(int devID);
    void ResetStateVec(complex* nStateVec, BufferPtr nStateBuffer);
    virtual complex* AllocStateVec(bitCapInt elemCount, bool doForceAlloc = false);
    virtual void FreeStateVec() {
        if (stateVec) {
            free(stateVec);
        }
    }
    virtual BufferPtr MakeStateVecBuffer(complex* nStateVec);

    real1 ParSum(real1* toSum, bitCapInt maxI);

    /**
     * Finishes the asynchronous wait event list or queue of OpenCL events.
     *
     * By default (doHard = false) only the wait event list of this engine is finished. If doHard = true, the entire
     * device queue is finished, (which might be shared by other QEngineOCL instances).
     */
    virtual void clFinish(bool doHard = false);

    size_t FixWorkItemCount(size_t maxI, size_t wic);
    size_t FixGroupSize(size_t wic, size_t gs);

    // CL_MAP_READ = (1 << 0); CL_MAP_WRITE = (1 << 1);
    /**
     * Locks synchronization between the state vector buffer and general RAM, so the state vector can be directly read
     * and/or written to.
     *
     * OpenCL buffers, even when allocated on "host" general RAM, are not safe to read from or write to unless "mapped."
     * When mapped, a buffer cannot be used by OpenCL kernels. If the state vector needs to be directly manipulated, it
     * needs to be temporarily mapped, and this can be accomplished with LockSync(). When direct reading from or writing
     * to the state vector is done, before performing other OpenCL operations on it, it must be unmapped with
     * UnlockSync().
     */
    void LockSync(cl_int flags = (CL_MAP_READ | CL_MAP_WRITE));
    /**
     * Unlocks synchronization between the state vector buffer and general RAM, so the state vector can be operated on
     * with OpenCL kernels and operations.
     *
     * OpenCL buffers, even when allocated on "host" general RAM, are not safe to read from or write to unless "mapped."
     * When mapped, a buffer cannot be used by OpenCL kernels. If the state vector needs to be directly manipulated, it
     * needs to be temporarily mapped, and this can be accomplished with LockSync(). When direct reading from or writing
     * to the state vector is done, before performing other OpenCL operations on it, it must be unmapped with
     * UnlockSync().
     */
    void UnlockSync();
    void Sync();

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);
    void ArithmeticCall(
        OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], unsigned char* values = NULL, bitCapInt valuesLength = 0);
    void CArithmeticCall(OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], bitCapInt* controlPowers,
        const bitLenInt controlLen, unsigned char* values = NULL, bitCapInt valuesLength = 0);

    void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm);

    void ApplyM(bitCapInt mask, bool result, complex nrm);
    void ApplyM(bitCapInt mask, bitCapInt result, complex nrm);

    /* Utility functions used by the operations above. */
    cl::Event QueueCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0);
    void ApplyMx(OCLAPI api_call, bitCapInt* bciArgs, complex nrm);
    real1 Probx(OCLAPI api_call, bitCapInt* bciArgs);
    void ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INT(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length);
    void CINT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
        const bitLenInt* controls, const bitLenInt controlLen);
    void INTC(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void INTS(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt overflowIndex);
    void INTSC(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void INTSC(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt overflowIndex, const bitLenInt carryIndex);
    void INTBCD(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length);
    void INTBCDC(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void xMULx(OCLAPI api_call, bitCapInt* bciArgs, BufferPtr controlBuffer);
    void MULx(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
        const bitLenInt length);
    void CMULx(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
        const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen);
    void PhaseFlipX(OCLAPI api_call, bitCapInt* bciArgs);

    bitCapInt OpIndexed(OCLAPI api_call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
};

} // namespace Qrack
