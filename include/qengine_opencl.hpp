//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qrack_types.hpp"

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#include "common/oclengine.hpp"
#include "qengine.hpp"

#include <list>
#include <mutex>

#define BCI_ARG_LEN 10
#define CMPLX_NORM_LEN 6
#define REAL_ARG_LEN 2

namespace Qrack {

enum SPECIAL_2X2 { NONE = 0, PAULIX, PAULIZ, INVERT, PHASE };

class bad_alloc : public std::bad_alloc {
private:
    std::string m;

public:
    bad_alloc(std::string message)
        : m(message)
    {
        // Intentionally left blank.
    }

    const char* what() const noexcept { return m.c_str(); }
};

typedef std::shared_ptr<cl::Buffer> BufferPtr;

class QEngineOCL;

typedef std::shared_ptr<QEngineOCL> QEngineOCLPtr;

struct QueueItem {
    OCLAPI api_call;
    size_t workItemCount;
    size_t localGroupSize;
    size_t deallocSize;
    std::vector<BufferPtr> buffers;
    size_t localBuffSize;
    bool isSetDoNorm;
    bool isSetRunningNorm;
    bool doNorm;
    real1 runningNorm;

    QueueItem()
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(false)
        , isSetRunningNorm(true)
        , doNorm(false)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(OCLAPI ac, size_t wic, size_t lgs, size_t ds, std::vector<BufferPtr> b, size_t lbs)
        : api_call(ac)
        , workItemCount(wic)
        , localGroupSize(lgs)
        , deallocSize(ds)
        , buffers(b)
        , localBuffSize(lbs)
        , isSetDoNorm(false)
        , isSetRunningNorm(false)
        , doNorm(false)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(bool doNrm)
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(true)
        , isSetRunningNorm(false)
        , doNorm(doNrm)
        , runningNorm(ONE_R1)
    {
    }

    QueueItem(real1_f runningNrm)
        : api_call()
        , workItemCount(0U)
        , localGroupSize(0U)
        , deallocSize(0U)
        , buffers()
        , localBuffSize(0U)
        , isSetDoNorm(false)
        , isSetRunningNorm(true)
        , doNorm(false)
        , runningNorm(runningNrm)
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

    BufferPtr MakeBuffer(const cl::Context& context, cl_mem_flags flags, size_t size, void* host_ptr = NULL)
    {
        cl_int error;
        BufferPtr toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error != CL_SUCCESS) {
            if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                throw bad_alloc("CL_MEM_OBJECT_ALLOCATION_FAILURE in PoolItem::MakeBuffer()");
            }
            if (error == CL_OUT_OF_HOST_MEMORY) {
                throw bad_alloc("CL_OUT_OF_HOST_MEMORY in PoolItem::MakeBuffer()");
            }
            if (error == CL_INVALID_BUFFER_SIZE) {
                throw bad_alloc("CL_INVALID_BUFFER_SIZE in PoolItem::MakeBuffer()");
            }
            throw std::runtime_error("OpenCL error code on buffer allocation attempt: " + std::to_string(error));
        }

        return toRet;
    }

    PoolItem(cl::Context& context)
        : probArray(NULL)
        , angleArray(NULL)
        , otherStateVec(NULL)
    {
        cmplxBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(complex) * CMPLX_NORM_LEN);
        realBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(real1) * REAL_ARG_LEN);
        ulongBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * BCI_ARG_LEN);
    }

    ~PoolItem() {}
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
class QEngineOCL : public QEngine {
protected:
    bool usingHostRam;
    bool unlockHostMem;
    size_t nrmGroupCount;
    size_t nrmGroupSize;
    size_t totalOclAllocSize;
    int64_t deviceID;
    cl_map_flags lockSyncFlags;
    complex permutationAmp;
    complex* stateVec;
    std::mutex queue_mutex;
    cl::CommandQueue queue;
    cl::Context context;
    // stateBuffer is allocated as a shared_ptr, because it's the only buffer that will be acted on outside of
    // QEngineOCL itself, specifically by QEngineOCLMulti.
    BufferPtr stateBuffer;
    BufferPtr nrmBuffer;
    BufferPtr powersBuffer;
    DeviceContextPtr device_context;
    std::vector<EventVecPtr> wait_refs;
    std::list<QueueItem> wait_queue_items;
    std::vector<PoolItemPtr> poolItems;
    std::unique_ptr<real1, void (*)(real1*)> nrmArray;

#if defined(__APPLE__)
    real1* _aligned_nrm_array_alloc(bitCapIntOcl allocSize)
    {
        void* toRet;
        posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
        return (real1*)toRet;
    }
#endif

    // For std::function, cl_int use might discard int qualifiers.
    void tryOcl(std::string message, std::function<int()> oclCall)
    {
        if (oclCall() == CL_SUCCESS) {
            // Success
            return;
        }

        // Soft finish (just for this QEngineOCL)
        clFinish();

        if (oclCall() == CL_SUCCESS) {
            // Success after clearing QEngineOCL queue
            return;
        }

        // Hard finish (for the unique OpenCL device)
        clFinish(true);

        cl_int error = oclCall();
        if (error == CL_SUCCESS) {
            // Success after clearing all queues for the OpenCL device
            return;
        }

        // We're fatally blocked. Clean up and throw to exit.
        FreeAll();
        throw std::runtime_error(message + ", error code: " + std::to_string(error));
    }

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
        bool useHostMem = false, int64_t devID = -1, bool useHardwareRNG = true, bool ignored = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> ignored2 = {}, bitLenInt ignored4 = 0U,
        real1_f ignored3 = FP_NORM_EPSILON_F);

    ~QEngineOCL()
    {
        // clDump() clears the callback queue.
        clDump();
        // clDump() doesn't wait on any kernel being run, though.
        clFinish();
        // Make sure we track device allocation.
        FreeAll();
    }

    bool IsZeroAmplitude() { return !stateBuffer; }
    real1_f FirstNonzeroPhase()
    {
        if (!stateBuffer) {
            return ZERO_R1_F;
        }

        return QInterface::FirstNonzeroPhase();
    }

    void FreeAll();
    void ZeroAmplitudes();
    void FreeStateVec(complex* sv = NULL);
    void CopyStateVec(QEnginePtr src);

    void GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length);
    void SetAmplitudePage(
        QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length);
    void ShuffleBuffers(QEnginePtr engine);
    QEnginePtr CloneEmpty();

    void QueueSetDoNormalize(bool doNorm) { AddQueueItem(QueueItem(doNorm)); }
    void QueueSetRunningNorm(real1_f runningNrm) { AddQueueItem(QueueItem(runningNrm)); }
    void AddQueueItem(const QueueItem& item)
    {
        bool isBase;
        // For lock_guard:
        if (true) {
            std::lock_guard<std::mutex> lock(queue_mutex);
            isBase = !wait_queue_items.size();
            wait_queue_items.push_back(item);
        }

        if (isBase) {
            DispatchQueue();
        }
    }
    void QueueCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0U, size_t deallocSize = 0U)
    {
        AddQueueItem(QueueItem(api_call, workItemCount, localGroupSize, deallocSize, args, localBuffSize));
    }

    bitCapIntOcl GetMaxSize() { return device_context->GetMaxAlloc() / sizeof(complex); };

    void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG);

    void UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
        const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask);
    void UniformParityRZ(bitCapInt mask, real1_f angle);
    void CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle);

    /* Operations that have an improved implementation. */
    using QEngine::X;
    void X(bitLenInt target);
    using QEngine::Z;
    void Z(bitLenInt target);
    using QEngine::Invert;
    void Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex);
    using QEngine::Phase;
    void Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex);

    void XMask(bitCapInt mask)
    {
        if (!mask) {
            return;
        }

        if (!(mask & (mask - ONE_BCI))) {
            X(log2(mask));
            return;
        }

        BitMask((bitCapIntOcl)mask, OCL_API_X_MASK);
    }
    void PhaseParity(real1_f radians, bitCapInt mask)
    {
        if (!mask) {
            return;
        }

        if (!(mask & (mask - ONE_BCI))) {
            complex phaseFac = std::polar(ONE_R1, (real1)(radians / 2));
            Phase(ONE_CMPLX / phaseFac, phaseFac, log2(mask));
            return;
        }

        BitMask((bitCapIntOcl)mask, OCL_API_PHASE_PARITY, radians);
    }

    using QEngine::Compose;
    bitLenInt Compose(QEngineOCLPtr toCopy);
    bitLenInt Compose(QInterfacePtr toCopy) { return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy)); }
    bitLenInt Compose(QEngineOCLPtr toCopy, bitLenInt start);
    bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start)
    {
        return Compose(std::dynamic_pointer_cast<QEngineOCL>(toCopy), start);
    }
    using QEngine::Decompose;
    void Decompose(bitLenInt start, QInterfacePtr dest);
    void Dispose(bitLenInt start, bitLenInt length);
    void Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm);

    void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);

#if ENABLE_ALU
    void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length);
    void CINC(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen);
    void INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length);
#endif
    void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length);
    void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen);
    void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen);
    void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);
    void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut);

    bitCapInt IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        const unsigned char* values, bool resetValue = true);
    bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength,
        bitLenInt carryIndex, const unsigned char* values);
    void Hash(bitLenInt start, bitLenInt length, const unsigned char* values);

    void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex);
    void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length);
#endif

    real1_f Prob(bitLenInt qubit);
    real1_f ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation);
    void ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray);
    real1_f ProbMask(bitCapInt mask, bitCapInt permutation);
    void ProbMaskAll(bitCapInt mask, real1* probsArray);
    real1_f ProbParity(bitCapInt mask);
    bool ForceMParity(bitCapInt mask, bool result, bool doForce = true);
    real1_f ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset = 0);

    void SetDevice(int64_t dID);
    int64_t GetDevice() { return deviceID; }

    void SetQuantumState(const complex* inputState);
    void GetQuantumState(complex* outputState);
    void GetProbs(real1* outputProbs);
    complex GetAmplitude(bitCapInt perm);
    void SetAmplitude(bitCapInt perm, complex amp);

    real1_f SumSqrDiff(QInterfacePtr toCompare) { return SumSqrDiff(std::dynamic_pointer_cast<QEngineOCL>(toCompare)); }
    real1_f SumSqrDiff(QEngineOCLPtr toCompare);

    void NormalizeState(
        real1_f nrm = REAL1_DEFAULT_ARG, real1_f norm_thresh = REAL1_DEFAULT_ARG, real1_f phaseArg = ZERO_R1_F);
    ;
    void UpdateRunningNorm(real1_f norm_thresh = REAL1_DEFAULT_ARG);
    void Finish() { clFinish(); };
    bool isFinished() { return !wait_queue_items.size(); };

    QInterfacePtr Clone();

    void PopQueue();
    void DispatchQueue();

protected:
    void AddAlloc(size_t size)
    {
        size_t currentAlloc = OCLEngine::Instance().AddToActiveAllocSize(deviceID, size);
        if (currentAlloc > OCLEngine::Instance().GetMaxActiveAllocSize()) {
            OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
            FreeAll();
            throw bad_alloc("VRAM limits exceeded in QEngineOCL::AddAlloc()");
        }
        totalOclAllocSize += size;
    }
    void SubtractAlloc(size_t size)
    {
        OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
        totalOclAllocSize -= size;
    }

    BufferPtr MakeBuffer(const cl::Context& context, cl_mem_flags flags, size_t size, void* host_ptr = NULL)
    {
        cl_int error;
        BufferPtr toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error == CL_SUCCESS) {
            // Success
            return toRet;
        }

        // Soft finish (just for this QEngineOCL)
        clFinish();

        toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error == CL_SUCCESS) {
            // Success after clearing QEngineOCL queue
            return toRet;
        }

        // Hard finish (for the unique OpenCL device)
        clFinish(true);

        toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
        if (error != CL_SUCCESS) {
            FreeAll();
            if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                throw bad_alloc("CL_MEM_OBJECT_ALLOCATION_FAILURE in QEngineOCL::MakeBuffer()");
            }
            if (error == CL_OUT_OF_HOST_MEMORY) {
                throw bad_alloc("CL_OUT_OF_HOST_MEMORY in QEngineOCL::MakeBuffer()");
            }
            if (error == CL_INVALID_BUFFER_SIZE) {
                throw bad_alloc("CL_INVALID_BUFFER_SIZE in QEngineOCL::MakeBuffer()");
            }
            throw std::runtime_error("OpenCL error code on buffer allocation attempt: " + std::to_string(error));
        }

        return toRet;
    }

    real1_f GetExpectation(bitLenInt valueStart, bitLenInt valueLength);

    complex* AllocStateVec(bitCapInt elemCount, bool doForceAlloc = false);
    void ResetStateVec(complex* sv);
    void ResetStateBuffer(BufferPtr nStateBuffer);
    BufferPtr MakeStateVecBuffer(complex* nStateVec);
    void ReinitBuffer();

    void Compose(OCLAPI apiCall, const bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy);

    void InitOCL(int64_t devID);
    PoolItemPtr GetFreePoolItem();

    real1_f ParSum(real1* toSum, bitCapIntOcl maxI);

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
    void LockSync(cl_map_flags flags = (CL_MAP_READ | CL_MAP_WRITE));
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
    void clFinish(bool doHard = false);

    /**
     * Flushes the OpenCL event queue, and checks for errors.
     */
    void clFlush()
    {
        tryOcl("Failed to flush queue", [&] { return queue.flush(); });
    }

    /**
     * Dumps the remaining asynchronous wait event list or queue of OpenCL events, for the current queue.
     */
    void clDump();

    size_t FixWorkItemCount(size_t maxI, size_t wic);
    size_t FixGroupSize(size_t wic, size_t gs);

    void DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr dest);

    using QEngine::Apply2x2;
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, real1_f norm_thresh = REAL1_DEFAULT_ARG)
    {
        Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, SPECIAL_2X2::NONE, norm_thresh);
    }
    void Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
        const bitCapIntOcl* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special,
        real1_f norm_thresh = REAL1_DEFAULT_ARG);

    void BitMask(bitCapIntOcl mask, OCLAPI api_call, real1_f phase = (real1_f)PI_R1);

    void ApplyM(bitCapInt mask, bool result, complex nrm);
    void ApplyM(bitCapInt mask, bitCapInt result, complex nrm);

    /* Utility functions used by the operations above. */
    void WaitCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args,
        size_t localBuffSize = 0U);
    EventVecPtr ResetWaitEvents(bool waitQueue = true);
    void ApplyMx(OCLAPI api_call, const bitCapIntOcl* bciArgs, complex nrm);
    real1_f Probx(OCLAPI api_call, const bitCapIntOcl* bciArgs);

    void ArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], const unsigned char* values = NULL,
        bitCapIntOcl valuesLength = 0U);
    void CArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], bitCapIntOcl* controlPowers,
        bitLenInt controlLen, const unsigned char* values = NULL, bitCapIntOcl valuesLength = 0U);
    void ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length);

#if ENABLE_ALU
    void INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INCDECSC(
        bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex);
#if ENABLE_BCD
    void INCDECBCDC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif

    void INT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length);
    void CINT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, const bitLenInt* controls,
        bitLenInt controlLen);
    void INTC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INTS(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
    void INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex,
        bitLenInt carryIndex);
#if ENABLE_BCD
    void INTBCD(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length);
    void INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex);
#endif
    void xMULx(OCLAPI api_call, const bitCapIntOcl* bciArgs, BufferPtr controlBuffer);
    void MULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length);
    void MULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart, bitLenInt carryStart,
        bitLenInt length);
    void CMULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        const bitLenInt* controls, bitLenInt controlLen);
    void CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart, bitLenInt carryStart,
        bitLenInt length, const bitLenInt* controls, bitLenInt controlLen);
    void FullAdx(
        bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call);
    void PhaseFlipX(OCLAPI api_call, const bitCapIntOcl* bciArgs);

    bitCapIntOcl OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
        bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values);
#endif

    void ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size);
};

} // namespace Qrack
