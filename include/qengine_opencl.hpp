//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
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

#include <list>
#include <mutex>

#include "common/oclengine.hpp"
#include "qengine.hpp"

#define BCI_ARG_LEN 10
#define CMPLX_NORM_LEN 6
#define REAL_ARG_LEN 2

namespace Qrack {

enum SPECIAL_2X2 { NONE = 0, PAULIX, PAULIZ, INVERT, PHASE };

typedef std::shared_ptr<cl::Buffer> BufferPtr;

class QEngineOCL;

typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

struct QueueItem {
    OCLAPI api_call;
    size_t workItemCount;
    size_t localGroupSize;
    std::vector<BufferPtr> buffers;
    size_t localBuffSize;

    QueueItem(OCLAPI ac, size_t wic, size_t lgs, std::vector<BufferPtr> b, size_t lbs)
        : api_call(ac)
        , workItemCount(wic)
        , localGroupSize(lgs)
        , buffers(b)
        , localBuffSize(lbs)
    {
    }
};

struct PoolItem {
    BufferPtr cmplxBuffer;
    BufferPtr realBuffer;
    BufferPtr ulongBuffer;

    std::shared_ptr<real1> probArray;
    std::shared_ptr<real1> angleArray;
    complex* otherStateVec;

    PoolItem(cl::Context& context)
        : probArray(NULL)
        , angleArray(NULL)
        , otherStateVec(NULL)
    {
        cmplxBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(complex) * CMPLX_NORM_LEN);
        realBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(real1) * REAL_ARG_LEN);
        ulongBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * BCI_ARG_LEN);
    }
};

typedef std::shared_ptr<PoolItem> PoolItemPtr;

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
class QEngineOCL : virtual public QEngine {
protected:
    bitCapIntOcl maxQPowerOcl;
    complex* stateVec;
    int deviceID;
    DeviceContextPtr device_context;
    std::vector<EventVecPtr> wait_refs;
    std::list<QueueItem> wait_queue_items;
    std::mutex queue_mutex;
    cl::CommandQueue queue;
    cl::Context context;
    // stateBuffer is allocated as a shared_ptr, because it's the only buffer that will be acted on outside of
    // QEngineOCL itself, specifically by QEngineOCLMulti.
    BufferPtr stateBuffer;
    BufferPtr nrmBuffer;
    BufferPtr powersBuffer;
    std::vector<PoolItemPtr> poolItems;
    real1* nrmArray;
    size_t nrmGroupCount;
    size_t nrmGroupSize;
    size_t maxWorkItems;
    size_t maxMem;
    size_t maxAlloc;
    unsigned int procElemCount;
    bool unlockHostMem;
    cl_int lockSyncFlags;
    bool usingHostRam;
    complex permutationAmp;

public:
    /// 1 / OclMemDenom is the maximum fraction of total OCL device RAM that a single state vector should occupy, by
    /// design of the QEngine.
    static const bitCapIntOcl OclMemDenom = 3U;

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
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int devID = -1, bool useHardwareRNG = true, bool ignored = false,
        real1 norm_thresh = REAL1_EPSILON, std::vector<int> ignored2 = {}, bitLenInt ignored3 = 0);

    virtual ~QEngineOCL() { ZeroAmplitudes(); }

    virtual void ZeroAmplitudes()
    {
        clDump();
        runningNorm = ZERO_R1;
        ResetStateBuffer(NULL);
        FreeStateVec();
    }

    virtual void SetQubitCount(bitLenInt qb)
    {
        QEngine::SetQubitCount(qb);
        maxQPowerOcl = (bitCapIntOcl)maxQPower;
    }

    virtual void FreeStateVec(complex* sv = NULL)
    {
        bool doReset = false;
        if (sv == NULL) {
            sv = stateVec;
            doReset = true;
        }

        if (sv) {
#if defined(_WIN32)
            _aligned_free(sv);
#else
            free(sv);
#endif
        }

        if (doReset) {
            stateVec = NULL;
        }
    }

    virtual void CopyStateVec(QInterfacePtr src)
    {
        Finish();
        src->Finish();

        LockSync(CL_MAP_WRITE);
        src->GetQuantumState(stateVec);
        UnlockSync();
    }

    virtual void GetAmplitudePage(complex* pagePtr, const bitCapInt offset, const bitCapInt length);
    virtual void SetAmplitudePage(const complex* pagePtr, const bitCapInt offset, const bitCapInt length);
    virtual void SetAmplitudePage(
        QEnginePtr pageEnginePtr, const bitCapInt srcOffset, const bitCapInt dstOffset, const bitCapInt length);
    virtual void ShuffleBuffers(QEnginePtr engine);

    bitCapIntOcl GetMaxSize() { return maxAlloc / sizeof(complex); };

    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);
    virtual real1 ProbAll(bitCapInt fullRegister);

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask);
    virtual void UniformParityRZ(const bitCapInt& mask, const real1& angle);

    /* Operations that have an improved implementation. */
    using QEngine::X;
    virtual void X(bitLenInt target);
    using QEngine::Z;
    virtual void Z(bitLenInt target);
    using QEngine::ApplySingleInvert;
    virtual void ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex);
    using QEngine::ApplySinglePhase;
    virtual void ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex);

    using QEngine::Compose;
    virtual bitLenInt Compose(QEngineOCLPtr toCopy);
    virtual bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    virtual bitLenInt Compose(QEngineOCLPtr toCopy, bitLenInt start);
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy), start);
    }
    virtual void Decompose(bitLenInt start, QInterfacePtr dest);
    virtual void Dispose(bitLenInt start, bitLenInt length);
    virtual void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen);
    virtual void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
    virtual void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen);
    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    virtual bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, unsigned char* values, bool resetValue = true);
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);
    virtual void Hash(bitLenInt start, bitLenInt length, unsigned char* values);

    virtual real1 Prob(bitLenInt qubit);
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation);
    virtual void ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray);
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation);
    virtual void ProbMaskAll(const bitCapInt& mask, real1* probsArray);
    virtual real1 ProbParity(const bitCapInt& mask);
    virtual bool ForceMParity(const bitCapInt& mask, bool result, bool doForce = true);

    virtual void PhaseFlip();
    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length);
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);

    virtual void SetDevice(const int& dID, const bool& forceReInit = false);
    virtual int GetDeviceID() { return deviceID; }

    virtual void SetQuantumState(const complex* inputState);
    virtual void GetQuantumState(complex* outputState);
    virtual void GetProbs(real1* outputProbs);
    virtual complex GetAmplitude(bitCapInt perm);
    virtual void SetAmplitude(bitCapInt perm, complex amp);

    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        return ApproxCompare(std::dynamic_pointer_cast<QEngineOCL>(toCompare));
    }
    virtual bool ApproxCompare(QEngineOCLPtr toCompare);

    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG);
    virtual void Finish() { clFinish(); };
    virtual bool isFinished() { return (wait_queue_items.size() == 0); };

    virtual QInterfacePtr Clone();

    void PopQueue(cl_event event, cl_int type);
    void DispatchQueue(cl_event event, cl_int type);

protected:
    virtual complex* AllocStateVec(bitCapInt elemCount, bool doForceAlloc = false);
    virtual void ResetStateVec(complex* sv);
    virtual void ResetStateBuffer(BufferPtr nStateBuffer);
    virtual BufferPtr MakeStateVecBuffer(complex* nStateVec);
    virtual void ReinitBuffer();

    virtual void Compose(OCLAPI apiCall, bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy);

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex);
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex);

    void InitOCL(int devID);
    PoolItemPtr GetFreePoolItem();

    real1 ParSum(real1* toSum, bitCapIntOcl maxI);

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

    /**
     * Finishes the asynchronous wait event list or queue of OpenCL events.
     *
     * By default (doHard = false) only the wait event list of this engine is finished. If doHard = true, the entire
     * device queue is finished, (which might be shared by other QEngineOCL instances).
     */
    virtual void clFinish(bool doHard = false);

    /**
     * Dumps the remaining asynchronous wait event list or queue of OpenCL events, for the current queue.
     */
    virtual void clDump();

    size_t FixWorkItemCount(size_t maxI, size_t wic);
    size_t FixGroupSize(size_t wic, size_t gs);

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);
    void ArithmeticCall(OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], unsigned char* values = NULL,
        bitCapIntOcl valuesLength = 0);
    void CArithmeticCall(OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], bitCapIntOcl* controlPowers,
        const bitLenInt controlLen, unsigned char* values = NULL, bitCapIntOcl valuesLength = 0);

    using QEngine::Apply2x2;
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, SPECIAL_2X2::NONE, norm_thresh);
    }
    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special, real1 norm_thresh = REAL1_DEFAULT_ARG);

    virtual void ApplyM(bitCapInt mask, bool result, complex nrm);
    virtual void ApplyM(bitCapInt mask, bitCapInt result, complex nrm);

    /* Utility functions used by the operations above. */
    void QueueCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0);
    void WaitCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0);
    EventVecPtr ResetWaitEvents(bool waitQueue = true);
    void ApplyMx(OCLAPI api_call, bitCapIntOcl* bciArgs, complex nrm);
    real1 Probx(OCLAPI api_call, bitCapIntOcl* bciArgs);
    void ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length);
    void INT(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length);
    void CINT(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length,
        const bitLenInt* controls, const bitLenInt controlLen);
    void INTC(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void INTS(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt overflowIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt overflowIndex, const bitLenInt carryIndex);
    void INTBCD(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length);
    void INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt length,
        const bitLenInt carryIndex);
    void xMULx(OCLAPI api_call, bitCapIntOcl* bciArgs, BufferPtr controlBuffer);
    void MULx(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
        const bitLenInt length);
    void MULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, const bitLenInt inOutStart,
        const bitLenInt carryStart, const bitLenInt length);
    void CMULx(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
        const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen);
    void CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, const bitLenInt inOutStart,
        const bitLenInt carryStart, const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen);
    void FullAdx(
        bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call);
    void PhaseFlipX(OCLAPI api_call, bitCapIntOcl* bciArgs);

    bitCapIntOcl OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values);

    void ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size, EventVecPtr waitVec);
};

} // namespace Qrack
