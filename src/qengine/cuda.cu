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

#include "qengine_cuda.hpp"

#include <algorithm>
#include <thread>

namespace Qrack {

// Mask definition for Apply2x2()
#define APPLY2X2_DEFAULT 0x00
#define APPLY2X2_NORM 0x01
#define APPLY2X2_SINGLE 0x02
#define APPLY2X2_DOUBLE 0x04
#define APPLY2X2_WIDE 0x08
#define APPLY2X2_X 0x10
#define APPLY2X2_Z 0x20
#define APPLY2X2_PHASE 0x40
#define APPLY2X2_INVERT 0x80

// These are commonly used emplace patterns, for OpenCL buffer I/O.
#define DISPATCH_BLOCK_WRITE(buff, offset, length, array)                                                              \
    tryCuda("Failed to write buffer", [&] {                                                                            \
        cudaError_t err = cudaStreamSynchronize(queue);                                                                \
        if (err != cudaSuccess) {                                                                                      \
            return err;                                                                                                \
        }                                                                                                              \
        return cudaMemcpy((void*)((complex*)(buff) + offset), (void*)(array), length, cudaMemcpyHostToDevice);         \
    });                                                                                                                \
    wait_refs.clear();

#define DISPATCH_TEMP_WRITE(buff, size, array, event)                                                                  \
    tryCuda("Failed to write buffer", [&] {                                                                            \
        cudaError_t error = cudaMemcpyAsync(buff, array, size, cudaMemcpyHostToDevice, queue);                         \
        if (error != cudaSuccess) {                                                                                    \
            return error;                                                                                              \
        }                                                                                                              \
        return cudaEventRecord(event);                                                                                 \
    });

#define DISPATCH_WRITE(buff, size, array)                                                                              \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->push_back(createCudaEvent());                                                         \
    tryCuda(                                                                                                           \
        "Failed to enqueue buffer write",                                                                              \
        [&] {                                                                                                          \
            cudaError_t error = cudaMemcpyAsync(buff, (void*)(array), size, cudaMemcpyHostToDevice, queue);            \
            if (error != cudaSuccess) {                                                                                \
                return error;                                                                                          \
            }                                                                                                          \
            return cudaEventRecord(device_context->wait_events->back());                                               \
        },                                                                                                             \
        true);                                                                                                         \
    device_context->UnlockWaitEvents();

#define DISPATCH_BLOCK_READ(buff, offset, length, array)                                                               \
    tryCuda("Failed to read buffer", [&] {                                                                             \
        cudaError_t err = cudaStreamSynchronize(queue);                                                                \
        if (err != cudaSuccess) {                                                                                      \
            return err;                                                                                                \
        }                                                                                                              \
        return cudaMemcpy((void*)(array), (void*)((complex*)(buff) + offset), length, cudaMemcpyDeviceToHost);         \
    });                                                                                                                \
    wait_refs.clear();

#define WAIT_REAL1_SUM(buff, size, array, sumPtr)                                                                      \
    clFinish();                                                                                                        \
    tryCuda("Failed to enqueue buffer read", [&] {                                                                     \
        cudaError_t err = cudaStreamSynchronize(queue);                                                                \
        if (err != cudaSuccess) {                                                                                      \
            return err;                                                                                                \
        }                                                                                                              \
        return cudaMemcpy((void*)((array).get()), buff, sizeof(real1) * size, cudaMemcpyDeviceToHost);                 \
    });                                                                                                                \
    *(sumPtr) = ParSum(array.get(), size);

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateBuffer) {                                                                                                \
        return;                                                                                                        \
    }

QEngineCUDA::QEngineCUDA(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool doNorm, bool randomGlobalPhase, bool useHostMem, int64_t devID, bool useHardwareRNG, bool ignored,
    real1_f norm_thresh, std::vector<int64_t> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , unlockHostMem(false)
    , callbackError(cudaSuccess)
    , nrmGroupSize(0U)
    , totalOclAllocSize(0U)
    , deviceID(devID)
    , wait_refs()
    , nrmArray(NULL, [](real1* r) {})
{
    InitOCL(devID);
    clFinish();
    if (qubitCount) {
        SetPermutation(initState, phaseFac);
    } else {
        ZeroAmplitudes();
    }
}

void QEngineCUDA::FreeAll()
{
    ZeroAmplitudes();

    powersBuffer = NULL;
    nrmBuffer = NULL;
    nrmArray = NULL;

    SubtractAlloc(totalOclAllocSize);
}

void QEngineCUDA::ZeroAmplitudes()
{
    clDump();
    runningNorm = ZERO_R1;

    if (!stateBuffer) {
        return;
    }

    ResetStateBuffer(NULL);
    FreeStateVec();

    SubtractAlloc(sizeof(complex) * maxQPowerOcl);
}

void QEngineCUDA::CopyStateVec(QEnginePtr src)
{
    if (qubitCount != src->GetQubitCount()) {
        throw std::invalid_argument("QEngineCUDA::CopyStateVec argument size differs from this!");
    }

    if (src->IsZeroAmplitude()) {
        ZeroAmplitudes();
        return;
    }

    if (stateBuffer) {
        clDump();
    } else {
        ReinitBuffer();
    }

    LockSync(CL_MAP_WRITE);
    src->GetQuantumState(stateVec.get());
    UnlockSync();

    runningNorm = src->GetRunningNorm();
}

void QEngineCUDA::GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCUDA::GetAmplitudePage range is out-of-bounds!");
    }

    if (!stateBuffer) {
        std::fill(pagePtr, pagePtr + length, ZERO_CMPLX);
        return;
    }

    ResetWaitEvents();
    DISPATCH_BLOCK_READ(*stateBuffer, offset, sizeof(complex) * length, pagePtr);
}

void QEngineCUDA::SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (isBadPermRange(offset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCUDA::SetAmplitudePage range is out-of-bounds!");
    }

    if (!stateBuffer) {
        ReinitBuffer();
        if (length != maxQPowerOcl) {
            ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
        }
    }

    ResetWaitEvents();
    DISPATCH_BLOCK_WRITE(*stateBuffer, offset, sizeof(complex) * length, pagePtr);

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineCUDA::SetAmplitudePage(
    QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
{
    if (isBadPermRange(dstOffset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCUDA::SetAmplitudePage source range is out-of-bounds!");
    }

    QEngineCUDAPtr pageEngineOclPtr = std::dynamic_pointer_cast<QEngineCUDA>(pageEnginePtr);

    if (isBadPermRange(srcOffset, length, maxQPowerOcl)) {
        throw std::invalid_argument("QEngineCUDA::SetAmplitudePage source range is out-of-bounds!");
    }

    BufferPtr oStateBuffer = pageEngineOclPtr->stateBuffer;

    if (!stateBuffer && !oStateBuffer) {
        return;
    }

    if (!oStateBuffer) {
        if (length == maxQPower) {
            ZeroAmplitudes();
        } else {
            ClearBuffer(stateBuffer, dstOffset, length);
            runningNorm = REAL1_DEFAULT_ARG;
        }

        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    pageEngineOclPtr->clFinish();

    EventVecPtr waitVec = ResetWaitEvents();

    cudaEvent_t copyEvent = createCudaEvent();
    tryCuda("Failed to enqueue buffer copy", [&] {
        cudaError_t error =
            cudaMemcpyAsync(*oStateBuffer, *stateBuffer, sizeof(complex) * srcOffset, cudaMemcpyDeviceToDevice, queue);
        if (error != cudaSuccess) {
            return error;
        }
        return cudaEventRecord(copyEvent, queue);
    });
    cudaEventSynchronize(copyEvent);

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineCUDA::ShuffleBuffers(QEnginePtr engine)
{
    if (qubitCount != engine->GetQubitCount()) {
        throw std::invalid_argument("QEngineCUDA::ShuffleBuffers argument size differs from this!");
    }

    QEngineCUDAPtr engineOcl = std::dynamic_pointer_cast<QEngineCUDA>(engine);

    if (!stateBuffer && !(engineOcl->stateBuffer)) {
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    if (!(engineOcl->stateBuffer)) {
        engineOcl->ReinitBuffer();
        engineOcl->ClearBuffer(engineOcl->stateBuffer, 0U, engineOcl->maxQPowerOcl);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
        0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeArgsEvent);
    cudaEventSynchronize(writeArgsEvent);

    engineOcl->clFinish();
    QueueCall(OCL_API_SHUFFLEBUFFERS, nrmGroupCount, nrmGroupSize,
        { stateBuffer, engineOcl->stateBuffer, poolItem->ulongBuffer });
    engineOcl->wait_refs.emplace_back(device_context->wait_events);

    runningNorm = REAL1_DEFAULT_ARG;
    engineOcl->runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineCUDA::LockSync(cl_map_flags flags)
{
    lockSyncFlags = flags;
    ResetWaitEvents();

    if (stateVec) {
        unlockHostMem = true;
        tryCuda("Failed to map buffer", [&] {
            cudaError_t error = cudaStreamSynchronize(queue);
            if (error != cudaSuccess) {
                return error;
            }
            return cudaMemcpy(
                (void*)(stateVec.get()), *stateBuffer, sizeof(complex) * maxQPowerOcl, cudaMemcpyDeviceToHost);
        });
        wait_refs.clear();
    } else {
        unlockHostMem = false;
        stateVec = AllocStateVec(maxQPowerOcl);
        if (lockSyncFlags & CL_MAP_READ) {
            DISPATCH_BLOCK_READ(*stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, stateVec.get());
        }
    }
}

void QEngineCUDA::UnlockSync()
{
    EventVecPtr waitVec = ResetWaitEvents();

    if (unlockHostMem) {
        cudaEvent_t unmapEvent = createCudaEvent();
        tryCuda("Failed to unmap buffer", [&] {
            cudaError_t error = cudaMemcpyAsync(
                *stateBuffer, (void*)(stateVec.get()), sizeof(complex) * maxQPowerOcl, cudaMemcpyHostToDevice, queue);
            if (error != cudaSuccess) {
                return error;
            }
            return cudaEventRecord(unmapEvent, queue);
        });
        cudaEventSynchronize(unmapEvent);
        wait_refs.clear();
    } else {
        if (lockSyncFlags & CL_MAP_WRITE) {
            DISPATCH_BLOCK_WRITE(*stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, stateVec.get())
        }
        FreeStateVec();
    }

    lockSyncFlags = 0;
}

void QEngineCUDA::clFinish(bool doHard)
{
    if (!device_context) {
        return;
    }

    checkCallbackError();

    while (wait_queue_items.size() > 1) {
        device_context->WaitOnAllEvents();
        PopQueue();
        checkCallbackError();
    }

    if (doHard) {
        tryCuda("Failed to finish queue", [&] { return cudaStreamSynchronize(queue); });
    } else {
        device_context->WaitOnAllEvents();
        checkCallbackError();
    }

    wait_refs.clear();
}

void QEngineCUDA::clDump()
{
    if (!device_context) {
        return;
    }

    wait_queue_items.clear();
    wait_refs.clear();
    device_context->WaitOnAllEvents();
    checkCallbackError();
}

PoolItemPtr QEngineCUDA::GetFreePoolItem()
{
    std::lock_guard<std::mutex> lock(queue_mutex);

    checkCallbackError();

    while (wait_queue_items.size() >= poolItems.size()) {
        poolItems.push_back(std::make_shared<PoolItem>());
    }

    return poolItems[wait_queue_items.size()];
}

EventVecPtr QEngineCUDA::ResetWaitEvents(bool waitQueue)
{
    if (waitQueue) {
        while (wait_queue_items.size() > 1) {
            device_context->WaitOnAllEvents();
            PopQueue();
            checkCallbackError();
        }
    }

    EventVecPtr waitVec = device_context->ResetWaitEvents();
    if (waitVec->size()) {
        wait_refs.emplace_back(waitVec);
    }
    return wait_refs.size() ? wait_refs.back() : std::make_shared<EventVec>();
}

void QEngineCUDA::WaitCall(
    OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args, size_t localBuffSize)
{
    QueueCall(api_call, workItemCount, localGroupSize, args, localBuffSize);
    clFinish();
}

void CUDART_CB _PopQueue(cudaStream_t stream, cudaError_t status, void* user_data)
{
    ((QEngineCUDA*)user_data)->PopQueue();
}

void QEngineCUDA::PopQueue()
{
    // For lock_guard scope
    if (true) {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (poolItems.size()) {
            poolItems.front()->probArray = NULL;
            poolItems.front()->angleArray = NULL;

            SubtractAlloc(wait_queue_items.front().deallocSize);

            if (poolItems.size() > 1) {
                rotate(poolItems.begin(), poolItems.begin() + 1, poolItems.end());
            }
        }

        if (!wait_queue_items.size()) {
            return;
        }
        wait_queue_items.pop_front();
    }

    if (callbackError != cudaSuccess) {
        wait_queue_items.clear();
        wait_refs.clear();
        return;
    }

    DispatchQueue();
}

void QEngineCUDA::DispatchQueue()
{
    QueueItem item;

    if (true) {
        std::lock_guard<std::mutex> lock(queue_mutex);

        if (!wait_queue_items.size()) {
            return;
        }

        item = wait_queue_items.front();

        while (item.isSetDoNorm || item.isSetRunningNorm) {
            if (item.isSetDoNorm) {
                doNormalize = item.doNorm;
            }
            if (item.isSetRunningNorm) {
                runningNorm = item.runningNorm;
            }

            wait_queue_items.pop_front();
            if (!wait_queue_items.size()) {
                return;
            }
            item = wait_queue_items.front();
        }
    }

    std::vector<BufferPtr> args = item.buffers;

    // We have to reserve the kernel, because its argument hooks are unique. The same kernel therefore can't be used by
    // other QEngineCUDA instances, until we're done queueing it.
    OCLDeviceCall ocl = device_context->Reserve(item.api_call);

    // Load the arguments.
    for (unsigned int i = 0U; i < args.size(); ++i) {
        ocl.call.setArg(i, *args[i]);
    }

    // For all of our kernels, if a local memory buffer is used, there is always only one, as the last argument.
    if (item.localBuffSize) {
#if ENABLE_SNUCL
        ocl.call.setArg(args.size(), cl::__local(item.localBuffSize));
#else
        ocl.call.setArg(args.size(), cl::Local(item.localBuffSize));
#endif
    }

    // Dispatch the primary kernel, to apply the gate.
    EventVecPtr kernelWaitVec = ResetWaitEvents(false);
    device_context->LockWaitEvents();
    cudaError_t error = queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(item.workItemCount), // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(), // vector of events to wait for
        &(device_context->wait_events->back())); // handle to wait for the kernel
    if (error == cudaSuccess) {
        cudaStreamAddCallback(queue, _PopQueue, (void*)this, 0);
        device_context->wait_events->push_back(createCudaEvent());
        cudaEventRecord(device_context->wait_events->back(), queue);
    }
    device_context->UnlockWaitEvents();
    if (error != cudaSuccess) {
        // We're fatally blocked, since we can't make any blocking calls like clFinish() in a callback.
        callbackError = error;
        wait_queue_items.clear();
        wait_refs.clear();
    }
}

void QEngineCUDA::SetDevice(int64_t dID)
{
    if (!(CUDAEngine::Instance().GetDeviceCount())) {
        throw std::runtime_error("Tried to initialize QEngineCUDA, but no available OpenCL devices.");
    }

    const bool didInit = (nrmArray != NULL);

    clFinish();

    const DeviceContextPtr nDeviceContext = CUDAEngine::Instance().GetDeviceContextPtr(dID);
    const int64_t defDevId = (int)CUDAEngine::Instance().GetDefaultDeviceID();

    std::unique_ptr<complex[]> copyVec = NULL;

    if (!didInit) {
        AddAlloc(sizeof(complex) * maxQPowerOcl);
    } else if ((dID == deviceID) || ((dID == -1) && (deviceID == defDevId)) ||
        ((deviceID == -1) && (dID == defDevId))) {
        // If we're "switching" to the device we already have, don't reinitialize.
        return;
    } else if (stateBuffer && !stateVec) {
        // This copies the contents of stateBuffer to host memory, to load into a buffer in the new context.
        copyVec = std::unique_ptr<complex[]>(new complex[maxQPowerOcl]);
        GetQuantumState(copyVec.get());
    }

    device_context = nDeviceContext;
    deviceID = dID;

    // If the user wants not to use host RAM, but we can't allocate enough on the device, fall back to host RAM anyway.
#if ENABLE_OCL_MEM_GUARDS
    const size_t stateVecSize = maxQPowerOcl * sizeof(complex);
    // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
    if (stateVecSize > device_context->GetMaxAlloc()) {
        throw bad_alloc("VRAM limits exceeded in QEngineCUDA::SetDevice()");
    }
    usingHostRam = (useHostRam || ((OclMemDenom * stateVecSize) > device_context->GetGlobalSize()));
#endif

    const bitCapIntOcl oldNrmVecAlignSize = nrmGroupSize ? (nrmGroupCount / nrmGroupSize) : 0U;
    nrmGroupCount = device_context->GetPreferredConcurrency();
    nrmGroupSize = device_context->GetPreferredSizeMultiple();
    if (nrmGroupSize > device_context->GetMaxWorkGroupSize()) {
        nrmGroupSize = device_context->GetMaxWorkGroupSize();
    }
    // constrain to a power of two
    nrmGroupSize = (size_t)pow2(log2(nrmGroupSize));

    const size_t nrmArrayAllocSize =
        (!nrmGroupSize || ((sizeof(real1) * nrmGroupCount / nrmGroupSize) < QRACK_ALIGN_SIZE))
        ? QRACK_ALIGN_SIZE
        : (sizeof(real1) * nrmGroupCount / nrmGroupSize);

    const bool doResize = (nrmGroupCount / nrmGroupSize) != oldNrmVecAlignSize;

    nrmBuffer = NULL;
    if (didInit && doResize) {
        nrmArray = NULL;
        SubtractAlloc(oldNrmVecAlignSize);
    }

    if (!didInit || doResize) {
        AddAlloc(nrmArrayAllocSize);
#if defined(__ANDROID__)
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            new real1[nrmArrayAllocSize / sizeof(real1)], [](real1* r) { delete r; });
#elif defined(_WIN32) && !defined(__CYGWIN__)
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            (real1*)_aligned_malloc(nrmArrayAllocSize, QRACK_ALIGN_SIZE), [](real1* c) { _aligned_free(c); });
#else
        nrmArray = std::unique_ptr<real1, void (*)(real1*)>(
            (real1*)aligned_alloc(QRACK_ALIGN_SIZE, nrmArrayAllocSize), [](real1* c) { free(c); });
#endif
    }
    nrmBuffer = MakeBuffer(CL_MEM_READ_WRITE, nrmArrayAllocSize);

    poolItems.clear();
    poolItems.push_back(std::make_shared<PoolItem>());

    if (!didInit) {
        AddAlloc(sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));
    }
    powersBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));
}

real1_f QEngineCUDA::ParSum(real1* toSum, bitCapIntOcl maxI)
{
    // This interface is potentially parallelizable, but, for now, better performance is probably given by implementing
    // it as a serial loop.
    real1 totSum = ZERO_R1;
    for (bitCapIntOcl i = 0U; i < maxI; ++i) {
        totSum += toSum[i];
    }

    return (real1_f)totSum;
}

void QEngineCUDA::InitOCL(int64_t devID)
{
    cudaError_t error;
    error = cudaStreamCreate(&queue);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream!");
    }
    SetDevice(devID);
}

void QEngineCUDA::ResetStateBuffer(BufferPtr nStateBuffer) { stateBuffer = nStateBuffer; }

void QEngineCUDA::SetPermutation(bitCapInt perm, complex phaseFac)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    ClearBuffer(stateBuffer, 0U, maxQPowerOcl);

    // If "permutationAmp" amp is in (read-only) use, this method completely supersedes that application anyway.

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        permutationAmp = GetNonunitaryPhase();
    } else {
        permutationAmp = phaseFac;
    }

    ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->push_back(createCudaEvent());
    tryCuda(
        "Failed to enqueue buffer write",
        [&] {
            cudaError_t error = cudaMemcpyAsync((void*)((complex*)(*stateBuffer) + (size_t)perm),
                (void*)&permutationAmp, sizeof(complex), cudaMemcpyDeviceToDevice, queue);
            if (error != cudaSuccess) {
                return error;
            }
            return cudaEventRecord(device_context->wait_events->back(), queue);
        },
        true);
    device_context->UnlockWaitEvents();

    QueueSetRunningNorm(ONE_R1_F);
}

/// NOT gate, which is also Pauli x matrix
void QEngineCUDA::X(bitLenInt qubit)
{
    const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    const bitCapIntOcl qPowers[1] = { pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::PAULIX);
}

/// Apply Pauli Z matrix to bit
void QEngineCUDA::Z(bitLenInt qubit)
{
    const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    const bitCapIntOcl qPowers[1] = { pow2Ocl(qubit) };
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PAULIZ);
}

void QEngineCUDA::Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
{
    if ((randGlobalPhase || IS_NORM_0(ONE_CMPLX - topRight)) && IS_NORM_0(topRight - bottomLeft)) {
        X(qubitIndex);
        return;
    }

    const complex pauliX[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    const bitCapIntOcl qPowers[1] = { pow2Ocl(qubitIndex) };
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::INVERT);
}

void QEngineCUDA::Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
{
    if (randGlobalPhase || IS_NORM_0(ONE_CMPLX - topLeft)) {
        if (IS_NORM_0(topLeft - bottomRight)) {
            return;
        }

        if (IS_NORM_0(topLeft + bottomRight)) {
            Z(qubitIndex);
            return;
        }
    }

    const complex pauliZ[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    const bitCapIntOcl qPowers[1] = { pow2Ocl(qubitIndex) };
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PHASE);
}

void QEngineCUDA::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
    const bitCapIntOcl* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special, real1_f norm_thresh)
{
    CHECK_ZERO_SKIP();

    if ((offset1 >= maxQPowerOcl) || (offset2 >= maxQPowerOcl)) {
        throw std::invalid_argument(
            "QEngineCUDA::Apply2x2 offset1 and offset2 parameters must be within allocated qubit bounds!");
    }

    for (bitLenInt i = 0U; i < bitCount; ++i) {
        if (qPowersSorted[i] >= maxQPowerOcl) {
            throw std::invalid_argument(
                "QEngineCUDA::Apply2x2 parameter qPowersSorted array values must be within allocated qubit bounds!");
        }
    }

    const bool skipNorm = !doNormalize || (abs(ONE_R1 - runningNorm) <= FP_NORM_EPSILON);
    const bool isXGate = skipNorm && (special == SPECIAL_2X2::PAULIX);
    const bool isZGate = skipNorm && (special == SPECIAL_2X2::PAULIZ);
    const bool isInvertGate = skipNorm && (special == SPECIAL_2X2::INVERT);
    const bool isPhaseGate = skipNorm && (special == SPECIAL_2X2::PHASE);

    // Are we going to calculate the normalization factor, on the fly? We can't, if this call doesn't iterate through
    // every single permutation amplitude.
    bool doApplyNorm = doNormalize && (bitCount == 1) && (runningNorm > ZERO_R1) && !isXGate && !isZGate &&
        !isInvertGate && !isPhaseGate;
    doCalcNorm = doCalcNorm && (doApplyNorm || (runningNorm <= ZERO_R1));
    doApplyNorm &= (runningNorm != ONE_R1);

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    ResetWaitEvents();

    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    const bitCapIntOcl maxI = maxQPowerOcl >> bitCount;
    bitCapIntOcl bciArgs[5] = { offset2, offset1, maxI, bitCount, 0U };

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // In an efficient OpenCL kernel, every single byte loaded comes at a significant execution time premium.
    // We handle single and double bit gates as special cases, for many reasons. Given that we have already separated
    // these out as special cases, since we know the bit count, we can eliminate the qPowersSorted buffer, by loading
    // its one or two values into the bciArgs buffer, of the same type. This gives us a significant execution time
    // savings.
    size_t bciArgsSize = 4;
    if (bitCount == 1) {
        // Single bit gates offsets are always 0 and target bit power. Hence, we overwrite one of the bit offset
        // arguments.
        if (ngc == maxI) {
            bciArgsSize = 3;
            bciArgs[2] = qPowersSorted[0] - ONE_BCI;
        } else {
            bciArgsSize = 4;
            bciArgs[3] = qPowersSorted[0] - ONE_BCI;
        }
    } else if (bitCount == 2) {
        // Double bit gates include both controlled and swap gates. To reuse the code for both cases, we need two offset
        // arguments. Hence, we cannot easily overwrite either of the bit offset arguments.
        bciArgsSize = 5;
        bciArgs[3] = qPowersSorted[0] - ONE_BCI;
        bciArgs[4] = qPowersSorted[1] - ONE_BCI;
    }
    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * bciArgsSize, bciArgs, writeArgsEvent);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    cmplx[4] = complex(doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1, ZERO_R1);
    cmplx[5] = (real1)norm_thresh;

    BufferPtr locCmplxBuffer;
    cudaEvent_t writeGateEvent = createCudaEvent();
    if (!isXGate && !isZGate) {
        DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), sizeof(complex) * CMPLX_NORM_LEN, cmplx, writeGateEvent);
    }

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    BufferPtr locPowersBuffer;
    cudaEvent_t writeControlsEvent = createCudaEvent();
    if (bitCount > 2) {
        if (doCalcNorm) {
            locPowersBuffer = powersBuffer;
        } else {
            locPowersBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * bitCount);
        }
        if (sizeof(bitCapInt) == sizeof(bitCapIntOcl)) {
            DISPATCH_TEMP_WRITE(*locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent);
        } else {
            DISPATCH_TEMP_WRITE(*locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent);
        }
    }

    // We load the appropriate kernel, that does/doesn't CALCULATE the norm, and does/doesn't APPLY the norm.
    unsigned char kernelMask = APPLY2X2_DEFAULT;
    if (bitCount == 1) {
        kernelMask |= APPLY2X2_SINGLE;
        if (isXGate) {
            kernelMask |= APPLY2X2_X;
        } else if (isZGate) {
            kernelMask |= APPLY2X2_Z;
        } else if (isInvertGate) {
            kernelMask |= APPLY2X2_INVERT;
        } else if (isPhaseGate) {
            kernelMask |= APPLY2X2_PHASE;
        } else if (doCalcNorm) {
            kernelMask |= APPLY2X2_NORM;
        }
    } else if (bitCount == 2) {
        kernelMask |= APPLY2X2_DOUBLE;
    }
    if (ngc == maxI) {
        kernelMask |= APPLY2X2_WIDE;
    }

    OCLAPI api_call;
    switch (kernelMask) {
    case APPLY2X2_DEFAULT:
        api_call = OCL_API_APPLY2X2;
        break;
    case APPLY2X2_SINGLE:
        api_call = OCL_API_APPLY2X2_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_X:
        api_call = OCL_API_X_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_Z:
        api_call = OCL_API_Z_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_INVERT:
        api_call = OCL_API_INVERT_SINGLE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_PHASE:
        api_call = OCL_API_PHASE_SINGLE;
        break;
    case APPLY2X2_NORM | APPLY2X2_SINGLE:
        api_call = OCL_API_APPLY2X2_NORM_SINGLE;
        break;
    case APPLY2X2_DOUBLE:
        api_call = OCL_API_APPLY2X2_DOUBLE;
        break;
    case APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_X:
        api_call = OCL_API_X_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_Z:
        api_call = OCL_API_Z_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_INVERT:
        api_call = OCL_API_INVERT_SINGLE_WIDE;
        break;
    case APPLY2X2_SINGLE | APPLY2X2_WIDE | APPLY2X2_PHASE:
        api_call = OCL_API_PHASE_SINGLE_WIDE;
        break;
    case APPLY2X2_NORM | APPLY2X2_SINGLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_NORM_SINGLE_WIDE;
        break;
    case APPLY2X2_DOUBLE | APPLY2X2_WIDE:
        api_call = OCL_API_APPLY2X2_DOUBLE_WIDE;
        break;
    default:
        throw std::runtime_error("Invalid APPLY2X2 kernel selected!");
    }

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    if (!isXGate && !isZGate) {
        cudaEventSynchronize(writeGateEvent);
    }
    if (bitCount > 2) {
        cudaEventSynchronize(writeControlsEvent);
    }
    wait_refs.clear();

    if (isXGate || isZGate) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
    } else if (doCalcNorm) {
        if (bitCount > 2) {
            QueueCall(api_call, ngc, ngs,
                { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, locPowersBuffer, nrmBuffer },
                sizeof(real1) * ngs);
        } else {
            QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, nrmBuffer },
                sizeof(real1) * ngs);
        }
    } else {
        if (bitCount > 2) {
            QueueCall(
                api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer, locPowersBuffer });
        } else {
            QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->cmplxBuffer, poolItem->ulongBuffer });
        }
    }

    if (doApplyNorm) {
        QueueSetRunningNorm(ONE_R1_F);
    }

    if (!doCalcNorm) {
        return;
    }

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);
    if (runningNorm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
    }
}

void QEngineCUDA::BitMask(bitCapIntOcl mask, OCLAPI api_call, real1_f phase)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::BitMask mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ mask;

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, mask, otherMask, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    const bool isPhaseParity = (api_call == OCL_API_PHASE_PARITY);
    if (isPhaseParity) {
        const complex phaseFac = std::polar(ONE_R1, (real1)(phase / 2));
        const complex cmplxArray[2] = { phaseFac, ONE_CMPLX / phaseFac };
        cudaEvent_t writePhaseEvent = createCudaEvent();
        DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), 2U * sizeof(complex), cmplxArray, writePhaseEvent);
        cudaEventSynchronize(writePhaseEvent);
    }

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    wait_refs.clear();

    if (isPhaseParity) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
    }
}

void QEngineCUDA::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (!controlLen) {
        Mtrx(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    if (qubitIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::UniformlyControlledSingleBit qubitIndex is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(
        controls, controlLen, qubitCount, "QEngineCUDA::UniformlyControlledSingleBit control is out-of-bounds!");

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    const bitCapIntOcl maxI = maxQPowerOcl >> ONE_BCI;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxI, pow2Ocl(qubitIndex), controlLen, mtrxSkipLen,
        (bitCapIntOcl)mtrxSkipValueMask, 0U, 0U, 0U, 0U, 0U };
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs);

    BufferPtr nrmInBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(real1));
    const real1 nrm = (runningNorm > ZERO_R1) ? ONE_R1 / (real1)sqrt(runningNorm) : ONE_R1;
    DISPATCH_WRITE(*nrmInBuffer, sizeof(real1), &nrm);

    const size_t sizeDiff = sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen);
    AddAlloc(sizeDiff);
    BufferPtr uniformBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeDiff);

    DISPATCH_WRITE(*uniformBuffer, sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen), mtrxs);

    std::unique_ptr<bitCapIntOcl[]> qPowers(new bitCapIntOcl[controlLen + mtrxSkipLen]);
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        qPowers[i] = pow2Ocl(controls[i]);
    }
    for (bitLenInt i = 0U; i < mtrxSkipLen; ++i) {
        qPowers[controlLen + i] = (bitCapIntOcl)mtrxSkipPowers[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    DISPATCH_WRITE(*powersBuffer, sizeof(bitCapIntOcl) * (controlLen + mtrxSkipLen), qPowers.get());

    // We call the kernel, with global buffers and one local buffer.
    WaitCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    uniformBuffer.reset();
    qPowers.reset();

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    SubtractAlloc(sizeDiff);
}

void QEngineCUDA::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::UniformParityRZ mask out-of-bounds!");
    }

    CHECK_ZERO_SKIP();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };
    const real1 cosine = (real1)cos(angle);
    const real1 sine = (real1)sin(angle);
    const complex phaseFacs[3] = { complex(cosine, sine), complex(cosine, -sine),
        (runningNorm > ZERO_R1) ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1 };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    cudaEvent_t writeNormEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent);
    DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), sizeof(complex) * 3, &phaseFacs, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    cudaEventSynchronize(writeNormEvent);
    wait_refs.clear();

    QueueCall((abs(ONE_R1 - runningNorm) <= FP_NORM_EPSILON) ? OCL_API_UNIFORMPARITYRZ : OCL_API_UNIFORMPARITYRZ_NORM,
        ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineCUDA::CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
{
    if (!controlLen) {
        UniformParityRZ(mask, angle);
        return;
    }

    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::CUniformParityRZ mask out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount, "QEngineCUDA::CUniformParityRZ control is out-of-bounds!");

    CHECK_ZERO_SKIP();

    bitCapIntOcl controlMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controlLen);
    BufferPtr controlBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers.get());
    controlPowers.reset();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> controlLen), (bitCapIntOcl)mask,
        controlMask, controlLen, 0U, 0U, 0U, 0U, 0U, 0U };
    const real1 cosine = (real1)cos(angle);
    const real1 sine = (real1)sin(angle);
    const complex phaseFacs[2] = { complex(cosine, sine), complex(cosine, -sine) };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    cudaEvent_t writeNormEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, writeArgsEvent);
    DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), sizeof(complex) * 2, &phaseFacs, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    cudaEventSynchronize(writeNormEvent);
    wait_refs.clear();

    QueueCall(OCL_API_CUNIFORMPARITYRZ, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer, controlBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineCUDA::ApplyMx(OCLAPI api_call, const bitCapIntOcl* bciArgs, complex nrm)
{
    CHECK_ZERO_SKIP();

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    cudaEvent_t writeNormEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent);
    BufferPtr locCmplxBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(complex));
    DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), sizeof(complex), &nrm, writeNormEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    cudaEventSynchronize(writeNormEvent);
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineCUDA::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapIntOcl powerTest = result ? (bitCapIntOcl)qPower : 0U;

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), (bitCapIntOcl)qPower,
        powerTest, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    ApplyMx(OCL_API_APPLYM, bciArgs, nrm);
}

void QEngineCUDA::ApplyM(bitCapInt mask, bitCapInt result, complex nrm)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::ApplyM mask out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)result, 0U, 0U, 0U, 0U,
        0U, 0U, 0U };

    ApplyMx(OCL_API_APPLYMREG, bciArgs, nrm);
}

void QEngineCUDA::Compose(OCLAPI apiCall, const bitCapIntOcl* bciArgs, QEngineCUDAPtr toCopy)
{
    if (!toCopy->qubitCount) {
        return;
    }

    if (!stateBuffer || !toCopy->stateBuffer) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return;
    }

    if (!qubitCount) {
        clFinish();
        SetQubitCount(toCopy->qubitCount);
        toCopy->clFinish();
        runningNorm = toCopy->runningNorm;
        stateVec = AllocStateVec(toCopy->maxQPowerOcl);
        stateBuffer = MakeStateVecBuffer(stateVec);

        cudaEvent_t copyEvent = createCudaEvent();
        tryCuda("Failed to enqueue buffer copy", [&] {
            cudaError_t error = cudaMemcpyAsync(
                *stateBuffer, *(toCopy->stateBuffer), sizeof(complex) * maxQPowerOcl, cudaMemcpyDeviceToDevice, queue);
            if (error != cudaSuccess) {
                return error;
            }
            return cudaEventRecord(copyEvent, queue);
        });
        cudaEventSynchronize(copyEvent);

        return;
    }

    const bitCapIntOcl oMaxQPower = maxQPowerOcl;
    const bitCapIntOcl nMaxQPower = bciArgs[0];
    const bitCapIntOcl nQubitCount = bciArgs[1] + toCopy->qubitCount;
    const size_t nStateVecSize = nMaxQPower * sizeof(complex);
#if ENABLE_OCL_MEM_GUARDS
    if (nStateVecSize > device_context->GetMaxAlloc()) {
        throw bad_alloc("VRAM limits exceeded in QEngineCUDA::Compose()");
    }
#endif

    if (doNormalize) {
        NormalizeState();
    }
    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    ResetWaitEvents();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 7, bciArgs, writeArgsEvent);

    AddAlloc(sizeof(complex) * nMaxQPower);

    SetQubitCount(nQubitCount);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    cudaEventSynchronize(writeArgsEvent);
    wait_refs.clear();

    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    toCopy->clFinish();

    QueueCall(apiCall, ngc, ngs, { stateBuffer, toCopy->stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    toCopy->wait_refs.emplace_back(device_context->wait_events);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeof(complex) * oMaxQPower);
}

bitLenInt QEngineCUDA::Compose(QEngineCUDAPtr toCopy)
{
    const bitLenInt result = qubitCount;

    const bitCapIntOcl oQubitCount = toCopy->qubitCount;
    const bitCapIntOcl nQubitCount = qubitCount + oQubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = maxQPowerOcl - ONE_BCI;
    const bitCapIntOcl endMask = (toCopy->maxQPowerOcl - ONE_BCI) << (bitCapIntOcl)qubitCount;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, startMask, endMask, 0U, 0U, 0U, 0U, 0U, 0U };

    OCLAPI api_call;
    if (nMaxQPower <= nrmGroupCount) {
        api_call = OCL_API_COMPOSE_WIDE;
    } else {
        api_call = OCL_API_COMPOSE;
    }

    Compose(api_call, bciArgs, toCopy);

    return result;
}

bitLenInt QEngineCUDA::Compose(QEngineCUDAPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QEngineCUDA::Compose start index is out-of-bounds!");
    }

    const bitLenInt result = start;

    const bitLenInt oQubitCount = toCopy->qubitCount;
    const bitLenInt nQubitCount = qubitCount + oQubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = pow2Ocl(start) - ONE_BCI;
    const bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    const bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, oQubitCount, startMask, midMask, endMask, start,
        0U, 0U, 0U };

    Compose(OCL_API_COMPOSE_MID, bciArgs, toCopy);

    return result;
}

void QEngineCUDA::DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCUDAPtr destination)
{
    // "Dispose" is basically the same as decompose, except "Dispose" throws the removed bits away.

    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    if (!stateBuffer) {
        SetQubitCount(qubitCount - length);
        if (destination) {
            destination->ZeroAmplitudes();
        }
        return;
    }

    if (destination && !destination->stateBuffer) {
        // Reinitialize stateVec RAM
        destination->SetPermutation(0U);
    }

    if (doNormalize) {
        NormalizeState();
    }
    if (destination && destination->doNormalize) {
        destination->NormalizeState();
    }

    const bitLenInt nLength = qubitCount - length;

    if (!nLength) {
        if (destination != NULL) {
            destination->stateVec = stateVec;
            destination->stateBuffer = stateBuffer;
            stateBuffer = NULL;
            stateVec = NULL;
        }
        SetQubitCount(0U);
        // This will be cleared by the destructor:
        SubtractAlloc(sizeof(complex) * pow2Ocl(qubitCount));
        stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);

        return;
    }

    const bitCapIntOcl partPower = pow2Ocl(length);
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);
    const bitCapIntOcl oMaxQPower = maxQPowerOcl;
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0U, 0U, 0U, 0U, 0U, 0U };

    const size_t remainderDiff = 2 * sizeof(real1) * remainderPower;
    AddAlloc(remainderDiff);

    // The "remainder" bits will always be maintained.
    BufferPtr probBuffer1 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(probBuffer1, 0U, remainderPower >> ONE_BCI);
    BufferPtr angleBuffer1 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(angleBuffer1, 0U, remainderPower >> ONE_BCI);

    // The removed "part" is only necessary for Decompose.
    BufferPtr probBuffer2, angleBuffer2;
    const size_t partDiff = 2 * sizeof(real1) * partPower;
    if (destination) {
        AddAlloc(2 * sizeof(real1) * partPower);
        probBuffer2 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(probBuffer2, 0U, partPower >> ONE_BCI);
        angleBuffer2 = MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(angleBuffer2, 0U, partPower >> ONE_BCI);
    }

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    const bitCapIntOcl largerPower = partPower > remainderPower ? partPower : remainderPower;

    const size_t ngc = FixWorkItemCount(largerPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Call the kernel that calculates bit probability and angle, retaining both parts.
    if (destination) {
        QueueCall(OCL_API_DECOMPOSEPROB, ngc, ngs,
            { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1, probBuffer2, angleBuffer2 });
    } else {
        QueueCall(OCL_API_DISPOSEPROB, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1 });
    }

    SetQubitCount(nLength);

    // If we Decompose, calculate the state of the bit system removed.
    if (destination) {
        bciArgs[0] = partPower;

        destination->clFinish();

        poolItem = GetFreePoolItem();
        ResetWaitEvents();
        DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

        const size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        const size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        const size_t oNStateVecSize = maxQPowerOcl * sizeof(complex);

        WaitCall(OCL_API_DECOMPOSEAMP, ngc2, ngs2,
            { probBuffer2, angleBuffer2, poolItem->ulongBuffer, destination->stateBuffer });

        probBuffer2.reset();
        angleBuffer2.reset();

        SubtractAlloc(partDiff);

        if (!(destination->useHostRam) && destination->stateVec &&
            oNStateVecSize <= destination->device_context->GetMaxAlloc() &&
            (2 * oNStateVecSize) <= destination->device_context->GetGlobalSize()) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cudaEvent_t copyEvent = createCudaEvent();
            tryCuda("Failed to enqueue buffer copy", [&] {
                cudaError_t error = cudaMemcpyAsync(
                    *nSB, *(destination->stateBuffer), sizeof(complex) * maxQPowerOcl, cudaMemcpyDeviceToDevice, queue);
                if (error != cudaSuccess) {
                    return error;
                }
                return cudaEventRecord(copyEvent, queue);
            });
            cudaEventSynchronize(copyEvent);

            destination->stateBuffer = nSB;
            destination->stateVec = NULL;
        }
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPowerOcl;
    poolItem = GetFreePoolItem();
    ResetWaitEvents();
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    const size_t ngc3 = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs3 = FixGroupSize(ngc, nrmGroupSize);

    const size_t nStateVecSize = maxQPowerOcl * sizeof(complex);

    if (!useHostRam && stateVec && ((OclMemDenom * nStateVecSize) <= device_context->GetGlobalSize())) {
        FreeStateVec();
    }
    // Drop references to state vector buffer, which we're done with.
    ResetStateBuffer(NULL);
    SubtractAlloc(sizeof(complex) * oMaxQPower);

    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    // Tell QueueCall to track deallocation:
    QueueCall(OCL_API_DECOMPOSEAMP, ngc3, ngs3, { probBuffer1, angleBuffer1, poolItem->ulongBuffer, stateBuffer }, 0U,
        remainderDiff);
}

void QEngineCUDA::Decompose(bitLenInt start, QInterfacePtr destination)
{
    DecomposeDispose(start, destination->GetQubitCount(), std::dynamic_pointer_cast<QEngineCUDA>(destination));
}

void QEngineCUDA::Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QEngineCUDAPtr)NULL); }

void QEngineCUDA::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (!length) {
        return;
    }

    if (!stateBuffer) {
        SetQubitCount(qubitCount - length);
        return;
    }

    if (length == qubitCount) {
        // This will be cleared by the destructor:
        stateVec = AllocStateVec(2);
        stateBuffer = MakeStateVecBuffer(stateVec);
        SubtractAlloc(sizeof(complex) * (pow2Ocl(qubitCount) - 2U));
        SetQubitCount(1);
        return;
    }

    if (doNormalize) {
        NormalizeState();
    }

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    const bitLenInt nLength = qubitCount - length;
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);
    const size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    const bitCapIntOcl skipMask = pow2Ocl(start) - ONE_BCI;
    const bitCapIntOcl disposedRes = (bitCapIntOcl)disposedPerm << (bitCapIntOcl)start;

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { remainderPower, length, skipMask, disposedRes, 0U, 0U, 0U, 0U, 0U, 0U };

    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    SetQubitCount(nLength);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    AddAlloc(sizeof(complex) * maxQPowerOcl);
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    QueueCall(OCL_API_DISPOSE, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeDiff);
}

bitLenInt QEngineCUDA::Allocate(bitLenInt start, bitLenInt length)
{
    if (!length) {
        return start;
    }

    QEngineCUDAPtr nQubits = std::make_shared<QEngineCUDA>(length, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);
    return Compose(nQubits, start);
}

real1_f QEngineCUDA::Probx(OCLAPI api_call, const bitCapIntOcl* bciArgs)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    return clampProb((real1_f)oneChance);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1_f QEngineCUDA::Prob(bitLenInt qubit)
{
    if (qubit >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::Prob qubit index parameter must be within allocated qubit bounds!");
    }

    if (qubitCount == 1) {
        return ProbAll(1);
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    const bitCapIntOcl qPower = pow2Ocl(qubit);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), qPower, 0U, 0U, 0U, 0U, 0U, 0U,
        0U, 0U };

    return Probx(OCL_API_PROB, bciArgs);
}

real1_f QEngineCUDA::CtrlOrAntiProb(bool controlState, bitLenInt control, bitLenInt target)
{
    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    real1_f controlProb = Prob(control);
    if (!controlState) {
        controlProb = ONE_R1 - controlProb;
    }
    if (controlProb <= FP_NORM_EPSILON) {
        return ZERO_R1;
    }
    if ((ONE_R1 - controlProb) <= FP_NORM_EPSILON) {
        return Prob(target);
    }

    if (target >= qubitCount) {
        throw std::invalid_argument(
            "QEngineCUDA::CtrlOrAntiProb target index parameter must be within allocated qubit bounds!");
    }

    const bitCapIntOcl qPower = pow2Ocl(target);
    const bitCapIntOcl qControlPower = pow2Ocl(control);
    const bitCapIntOcl qControlMask = controlState ? qControlPower : 0U;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> 2U), qPower, qControlPower, qControlMask,
        0U, 0U, 0U, 0U, 0U, 0U };

    real1_f oneChance = Probx(OCL_API_CPROB, bciArgs);
    oneChance /= controlProb;

    return clampProb((real1_f)oneChance);
}

// Returns probability of permutation of the register
real1_f QEngineCUDA::ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
{
    if (!start && qubitCount == length) {
        return ProbAll(permutation);
    }

    const bitCapIntOcl perm = (bitCapIntOcl)permutation << (bitCapIntOcl)start;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), perm, start, length, 0U, 0U, 0U,
        0U, 0U, 0U };

    return Probx(OCL_API_PROBREG, bciArgs);
}

void QEngineCUDA::ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray)
{
    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl maxJ = maxQPowerOcl >> length;

    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(probsArray, probsArray + lengthPower, ZERO_R1);
        return;
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, start, length, 0U, 0U, 0U, 0U, 0U, 0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    AddAlloc(sizeof(real1) * lengthPower);
    BufferPtr probsBuffer = MakeBuffer(CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBREGALL, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probsBuffer });

    ResetWaitEvents();
    DISPATCH_BLOCK_READ(*probsBuffer, 0U, sizeof(real1) * lengthPower, probsArray);

    probsBuffer.reset();

    SubtractAlloc(sizeof(real1) * lengthPower);
}

// Returns probability of permutation of the register
real1_f QEngineCUDA::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::ProbMask mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1_F;
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (length = 0U; v; ++length) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), (bitCapIntOcl)mask,
        (bitCapIntOcl)permutation, length, 0U, 0U, 0U, 0U, 0U, 0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[length]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, skipPowers.get());
    skipPowers.reset();

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASK, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer, qPowersBuffer },
        sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    return clampProb((real1_f)oneChance);
}

void QEngineCUDA::ProbMaskAll(bitCapInt mask, real1* probsArray)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::ProbMaskAll mask out-of-bounds!");
    }

    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length;
    std::vector<bitCapIntOcl> powersVec;
    for (length = 0U; v; ++length) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        powersVec.push_back((v ^ oldV) & oldV);
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl maxJ = maxQPowerOcl >> length;

    if (!stateBuffer) {
        std::fill(probsArray, probsArray + lengthPower, ZERO_R1);
        return;
    }

    if ((lengthPower * lengthPower) < nrmGroupCount) {
        // With "lengthPower" count of threads, compared to a redundancy of "lengthPower" with full utilization, this is
        // close to the point where it becomes more efficient to rely on iterating through ProbReg calls.
        QEngine::ProbMaskAll(mask, probsArray);
        return;
    }

    v = (~(bitCapIntOcl)mask) & (maxQPowerOcl - ONE_BCI); // count the number of bits set in v
    bitCapIntOcl skipPower;
    bitLenInt skipLength = 0U; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (skipLength = 0U; v; ++skipLength) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        skipPowersVec.push_back(skipPower);
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, length, skipLength, 0U, 0U, 0U, 0U, 0U, 0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    size_t sizeDiff = sizeof(real1) * lengthPower + sizeof(bitCapIntOcl) * length + sizeof(bitCapIntOcl) * skipLength;
    AddAlloc(sizeDiff);

    BufferPtr probsBuffer = MakeBuffer(CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    std::unique_ptr<bitCapIntOcl[]> powers(new bitCapIntOcl[length]);
    std::copy(powersVec.begin(), powersVec.end(), powers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, powers.get());
    powers.reset();

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[skipLength]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qSkipPowersBuffer =
        MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * skipLength, skipPowers.get());
    skipPowers.reset();

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASKALL, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probsBuffer, qPowersBuffer, qSkipPowersBuffer });

    ResetWaitEvents();
    DISPATCH_BLOCK_READ(*probsBuffer, 0U, sizeof(real1) * lengthPower, probsArray);

    probsBuffer.reset();
    qPowersBuffer.reset();
    qSkipPowersBuffer.reset();

    SubtractAlloc(sizeDiff);
}

real1_f QEngineCUDA::ProbParity(bitCapInt mask)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::ProbParity mask out-of-bounds!");
    }

    // If no bits in mask:
    if (!mask) {
        return ZERO_R1_F;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    return Probx(OCL_API_PROBPARITY, bciArgs);
}

bool QEngineCUDA::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
    if (mask >= maxQPowerOcl) {
        throw std::invalid_argument("QEngineCUDA::ForceMParity mask out-of-bounds!");
    }

    if (!stateBuffer || !mask) {
        return false;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return ForceM(log2(mask), result, doForce);
    }

    if (!doForce) {
        result = (Rand() <= ProbParity(mask));
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)(result ? ONE_BCI : 0U),
        0U, 0U, 0U, 0U, 0U, 0U, 0U };

    runningNorm = Probx(OCL_API_FORCEMPARITY, bciArgs);

    if (!doNormalize) {
        NormalizeState();
    }

    return result;
}

real1_f QEngineCUDA::ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset)
{
    if (length == 1U) {
        return Prob(bits[0]);
    }

    if (!stateBuffer || !length) {
        return ZERO_R1_F;
    }

    if (doNormalize) {
        NormalizeState();
    }

    std::unique_ptr<bitCapIntOcl[]> bitPowers(new bitCapIntOcl[length]);
    for (bitLenInt p = 0U; p < length; ++p) {
        bitPowers[p] = pow2Ocl(bits[p]);
    }

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    BufferPtr bitMapBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length);
    DISPATCH_WRITE(*bitMapBuffer, sizeof(bitCapIntOcl) * length, bitPowers.get());
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, length, (bitCapIntOcl)offset, 0U, 0U, 0U, 0U, 0U, 0U,
        0U };
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_EXPPERM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, bitMapBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    real1_f expectation;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &expectation);

    return expectation;
}

real1_f QEngineCUDA::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    LockSync(CL_MAP_READ);
    for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
        const bitCapIntOcl outputInt = (i & outputMask) >> valueStart;
        const real1 prob = norm(stateVec.get()[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return (real1_f)average;
}

void QEngineCUDA::ArithmeticCall(
    OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], const unsigned char* values, bitCapIntOcl valuesPower)
{
    CArithmeticCall(api_call, bciArgs, NULL, 0U, values, valuesPower);
}
void QEngineCUDA::CArithmeticCall(OCLAPI api_call, const bitCapIntOcl (&bciArgs)[BCI_ARG_LEN],
    bitCapIntOcl* controlPowers, bitLenInt controlLen, const unsigned char* values, bitCapIntOcl valuesPower)
{
    CHECK_ZERO_SKIP();

    size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    if (controlLen) {
        sizeDiff += sizeof(bitCapIntOcl) * controlLen;
    }
    if (values) {
        sizeDiff += sizeof(unsigned char) * valuesPower;
    }
    AddAlloc(sizeDiff);

    ResetWaitEvents();

    // Allocate a temporary nStateVec, or use the one supplied.
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen) {
        controlBuffer =
            MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers);
    }

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen) {
        device_context->LockWaitEvents();
        device_context->wait_events->push_back(createCudaEvent());
        tryCuda(
            "Failed to enqueue buffer copy",
            [&] {
                return cudaMemcpyAsync(
                    *nStateBuffer, *stateBuffer, sizeof(complex) * maxQPowerOcl, cudaMemcpyDeviceToDevice, queue);
            },
            true);
        device_context->UnlockWaitEvents();
    } else {
        ClearBuffer(nStateBuffer, 0U, maxQPowerOcl);
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * BCI_ARG_LEN, bciArgs);

    const bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    std::vector<BufferPtr> oclArgs = { stateBuffer, poolItem->ulongBuffer, nStateBuffer };

    BufferPtr loadBuffer;
    if (values) {
        loadBuffer =
            MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, (void*)values);
        oclArgs.push_back(loadBuffer);
    }
    if (controlLen) {
        oclArgs.push_back(controlBuffer);
    }

    QueueCall(api_call, ngc, ngs, oclArgs);

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeDiff);
}

void QEngineCUDA::ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::ROx range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    shift %= length;
    if (!shift) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~regMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, start, shift, length, 0U,
        0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineCUDA::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

#if ENABLE_ALU
/// Add or Subtract integer (without sign or carry)
void QEngineCUDA::INT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INT range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & ~(regMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, start, toMod, 0U, 0U, 0U,
        0U };

    ArithmeticCall(api_call, bciArgs);
}

/// Add or Subtract integer (without sign or carry, with controls)
void QEngineCUDA::CINT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CINT range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount, "QEngineCUDA::CINT control is out-of-bounds!");

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl regMask = lengthMask << start;

    bitCapIntOcl controlMask = 0U;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controlLen);

    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (regMask | controlMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> controlLen), regMask, otherMask,
        lengthPower, start, toMod, controlLen, controlMask, 0U, 0U };

    CArithmeticCall(api_call, bciArgs, controlPowers.get(), controlLen);
}

/** Increment integer (without sign, with carry) */
void QEngineCUDA::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INT(OCL_API_INC, (bitCapIntOcl)toAdd, start, length);
}

void QEngineCUDA::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        INC(toAdd, inOutStart, length);
        return;
    }

    CINT(OCL_API_CINC, (bitCapIntOcl)toAdd, inOutStart, length, controls, controlLen);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineCUDA::INTC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(regMask | carryMask));
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), regMask, otherMask,
        lengthPower, carryMask, start, toMod, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/// Common driver method behing INCC and DECC
void QEngineCUDA::INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    INTC(OCL_API_INCDECC, (bitCapIntOcl)toMod, inOutStart, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineCUDA::INTS(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTS range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTS overflowIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl regMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ regMask;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, overflowMask, start,
        toMod, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineCUDA::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, (bitCapIntOcl)toAdd, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineCUDA::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex,
    bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTSC range is out-of-bounds!");
    }

    if (overflowIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTSC overflowIndex is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTSC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inOutMask = lengthMask << start;
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask,
        lengthPower, overflowMask, carryMask, start, toMod, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineCUDA::INCDECSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_1, (bitCapIntOcl)toAdd, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineCUDA::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTSC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTSC carryIndex is out-of-bounds!");
    }

    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl lengthPower = pow2Ocl(length);
    const bitCapIntOcl inOutMask = (lengthPower - ONE_BCI) << start;
    const bitCapIntOcl otherMask = pow2MaskOcl(qubitCount) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask,
        lengthPower, carryMask, start, toMod, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineCUDA::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_2, (bitCapIntOcl)toAdd, start, length, carryIndex);
}

#if ENABLE_BCD
/// Add or Subtract integer (BCD)
void QEngineCUDA::INTBCD(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTBCD range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, inOutMask, otherMask, start, toMod, nibbleCount, 0U, 0U,
        0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineCUDA::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, (bitCapIntOcl)toAdd, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineCUDA::INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::INTBCDC range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::INTBCDC carryIndex is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    const bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (!toMod) {
        return;
    }

    const bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask,
        carryMask, start, toMod, nibbleCount, 0U, 0U, 0U };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineCUDA::INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTBCDC(OCL_API_INCDECBCDC, (bitCapIntOcl)toAdd, start, length, carryIndex);
}
#endif

/** Multiply by integer */
void QEngineCUDA::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(carryStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        SetReg(inOutStart, length, 0U);
        return;
    }

    MULx(OCL_API_MUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length);
}

/** Divide by integer */
void QEngineCUDA::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (!toDiv) {
        throw std::runtime_error("DIV by zero");
    }

    MULx(OCL_API_DIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length);
}

/** Multiplication modulo N by integer, (out of place) */
void QEngineCUDA::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(outStart, length, 0U);

    MULModx(OCL_API_MULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

void QEngineCUDA::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    MULModx(OCL_API_IMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Raise a classical base to a quantum power, modulo N, (out of place) */
void QEngineCUDA::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (base == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    MULModx(OCL_API_POWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Quantum analog of classical "Full Adder" gate */
void QEngineCUDA::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_FULLADD);
}

/** Inverse of FullAdd */
void QEngineCUDA::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_IFULLADD);
}

void QEngineCUDA::FullAdx(
    bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call)
{
    CHECK_ZERO_SKIP();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)2U), pow2Ocl(inputBit1),
        pow2Ocl(inputBit2), pow2Ocl(carryInSumOut), pow2Ocl(carryOut), 0U, 0U, 0U, 0U, 0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    wait_refs.clear();

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

/** Controlled multiplication by integer */
void QEngineCUDA::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (!controlLen) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 1) {
        return;
    }

    CMULx(OCL_API_CMUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled division by integer */
void QEngineCUDA::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (!toDiv) {
        throw std::runtime_error("DIV by zero");
    }

    if (toDiv == 1) {
        return;
    }

    CMULx(OCL_API_CDIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineCUDA::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (!controlLen) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        return;
    }

    CMULModx(
        OCL_API_CMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls, controlLen);
}

void QEngineCUDA::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (!controlLen) {
        IMULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    const bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (!toMul) {
        return;
    }

    CMULModx(OCL_API_CIMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls,
        controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineCUDA::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (!controlLen) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0U);

    CMULModx(
        OCL_API_CPOWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length, controls, controlLen);
}

void QEngineCUDA::xMULx(OCLAPI api_call, const bitCapIntOcl* bciArgs, BufferPtr controlBuffer)
{
    CHECK_ZERO_SKIP();

    ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    std::shared_ptr<complex> nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ClearBuffer(nStateBuffer, 0U, maxQPowerOcl);

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 10U, bciArgs);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    if (controlBuffer) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer, controlBuffer });
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    }

    stateVec = nStateVec;
    ResetStateBuffer(nStateBuffer);
}

void QEngineCUDA::MULx(
    OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::MULx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::MULx range is out-of-bounds!");
    }

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << (bitCapIntOcl)inOutStart;
    const bitCapIntOcl carryMask = lowMask << (bitCapIntOcl)carryStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(carryStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inOutMask, carryMask,
        otherMask, length, inOutStart, carryStart, skipMask, 0U };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineCUDA::MULModx(
    OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (isBadBitRange(inStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::MULModx range is out-of-bounds!");
    }

    if (isBadBitRange(outStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::MULModx range is out-of-bounds!");
    }

    if (!toMod) {
        return;
    }

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inMask = lowMask << (bitCapIntOcl)inStart;
    const bitCapIntOcl modMask = (isPowerOfTwo(modN) ? modN : pow2Ocl(log2(modN) + 1U)) - ONE_BCI;
    const bitCapIntOcl outMask = modMask << (bitCapIntOcl)outStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(outStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inMask, outMask,
        otherMask, length, inStart, outStart, skipMask, modN };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineCUDA::CMULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart,
    bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CMULx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CMULx range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount, "QEngineCUDA::CMULx control is out-of-bounds!");

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    bitCapIntOcl controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask | controlMask);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)(controlLen + length)),
        toMod, controlLen, controlMask, inOutMask, carryMask, otherMask, length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controlLen * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

void QEngineCUDA::CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart,
    bitLenInt carryStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (isBadBitRange(inOutStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CMULModx range is out-of-bounds!");
    }

    if (isBadBitRange(carryStart, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CMULModx range is out-of-bounds!");
    }

    ThrowIfQbIdArrayIsBad(controls, controlLen, qubitCount, "QEngineCUDA::CMULModx control is out-of-bounds!");

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    bitCapIntOcl controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; ++i) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0U; i < length; ++i) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, toMod, controlLen, controlMask, inOutMask, carryMask,
        modN, length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controlLen * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineCUDA::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values, bool resetValue)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::IndexedLDA range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::IndexedLDA range is out-of-bounds!");
    }

    if (!stateBuffer) {
        return 0U;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, 0U);
    }

    const bitLenInt valueBytes = (valueLength + 7) / 8;
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> valueLength), indexStart, inputMask,
        valueStart, valueBytes, valueLength, 0U, 0U, 0U, 0U };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return 0U;
#endif
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapIntOcl QEngineCUDA::OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    if (isBadBitRange(indexStart, indexLength, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::OpIndexed range is out-of-bounds!");
    }

    if (isBadBitRange(valueStart, valueLength, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::OpIndexed range is out-of-bounds!");
    }

    if (carryIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::OpIndexed carryIndex is out-of-bounds!");
    }

    if (!stateBuffer) {
        return 0U;
    }

    bool carryRes = M(carryIndex);
    // The carry has to first to be measured for its input value.
    if (carryRes) {
        /*
         * If the carry is set, we flip the carry bit. We always initially
         * clear the carry after testing for carry in.
         */
        carryIn ^= ONE_BCI;
        X(carryIndex);
    }

    const bitLenInt valueBytes = (valueLength + 7) / 8;
    const bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    const bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    const bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    const bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), indexStart, inputMask,
        valueStart, outputMask, otherMask, carryIn, carryMask, lengthPower, valueBytes };

    ArithmeticCall(api_call, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

#if ENABLE_VM6502Q_DEBUG
    return (bitCapInt)(GetExpectation(valueStart, valueLength) + (real1_f)0.5f);
#else
    return 0U;
#endif
}

/** Add based on an indexed load from classical memory */
bitCapInt QEngineCUDA::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDADC, 0U, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineCUDA::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDSBC, 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Set 8 bit register bits based on read from classical memory */
void QEngineCUDA::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    const bitLenInt bytes = (length + 7) / 8;
    const bitCapIntOcl inputMask = bitRegMaskOcl(start, length);
    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, start, inputMask, bytes, 0U, 0U, 0U, 0U, 0U, 0U };

    ArithmeticCall(OCL_API_HASH, bciArgs, values, pow2Ocl(length) * bytes);
}

void QEngineCUDA::PhaseFlipX(OCLAPI api_call, const bitCapIntOcl* bciArgs)
{
    CHECK_ZERO_SKIP();

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cudaEvent_t writeArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeArgsEvent);
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

void QEngineCUDA::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::CPhaseFlipIfLess range is out-of-bounds!");
    }

    if (flagIndex >= qubitCount) {
        throw std::invalid_argument("QEngineCUDA::CPhaseFlipIfLess flagIndex is out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        pow2Ocl(flagIndex), (bitCapIntOcl)greaterPerm, start, 0U, 0U, 0U, 0U, 0U };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineCUDA::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCUDA::PhaseFlipIfLess range is out-of-bounds!");
    }

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        (bitCapIntOcl)greaterPerm, start, 0U, 0U, 0U, 0U, 0U, 0U };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}
#endif

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineCUDA::SetQuantumState(const complex* inputState)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    ResetWaitEvents();
    DISPATCH_BLOCK_WRITE(*stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, inputState);

    UpdateRunningNorm();
}

complex QEngineCUDA::GetAmplitude(bitCapInt perm)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QEngineCUDA::GetAmplitude argument out-of-bounds!");
    }

    // WARNING: Does not normalize!
    if (!stateBuffer) {
        return ZERO_CMPLX;
    }

    complex amp;
    DISPATCH_BLOCK_READ(*stateBuffer, (bitCapIntOcl)perm, sizeof(complex), &amp);

    return amp;
}

void QEngineCUDA::SetAmplitude(bitCapInt perm, complex amp)
{
    if (perm >= maxQPower) {
        throw std::invalid_argument("QEngineCUDA::SetAmplitude argument out-of-bounds!");
    }

    if (!stateBuffer && !norm(amp)) {
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0U, maxQPowerOcl);
    }

    permutationAmp = amp;

    if (runningNorm != REAL1_DEFAULT_ARG) {
        runningNorm += norm(amp) - norm(permutationAmp);
    }

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->push_back(createCudaEvent());
    tryCuda(
        "Failed to enqueue buffer write",
        [&] {
            cudaError_t error = cudaMemcpyAsync((void*)((complex*)(*stateBuffer) + maxQPowerOcl),
                (void*)&permutationAmp, sizeof(complex), cudaMemcpyHostToDevice, queue);
            if (error != cudaSuccess) {
                return error;
            }
            return cudaEventRecord(device_context->wait_events->back(), queue);
        },
        true);
    device_context->UnlockWaitEvents();
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineCUDA::GetQuantumState(complex* outputState)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(outputState, outputState + maxQPowerOcl, ZERO_CMPLX);
        return;
    }

    ResetWaitEvents();
    DISPATCH_BLOCK_READ(*stateBuffer, 0U, sizeof(complex) * maxQPowerOcl, outputState);
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineCUDA::GetProbs(real1* outputProbs) { ProbRegAll(0U, qubitCount, outputProbs); }

real1_f QEngineCUDA::SumSqrDiff(QEngineCUDAPtr toCompare)
{
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }

    if (!stateBuffer && !toCompare->stateBuffer) {
        return ZERO_R1_F;
    }

    if (!stateBuffer) {
        toCompare->UpdateRunningNorm();
        return (real1_f)(toCompare->runningNorm);
    }

    if (!toCompare->stateBuffer) {
        UpdateRunningNorm();
        return (real1_f)runningNorm;
    }

    if (randGlobalPhase) {
        real1_f lPhaseArg = FirstNonzeroPhase();
        real1_f rPhaseArg = toCompare->FirstNonzeroPhase();
        NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, rPhaseArg - lPhaseArg);
    }

    toCompare->clFinish();

    const bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

    ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    const size_t partInnerSize = ngc / ngs;

    AddAlloc(sizeof(complex) * partInnerSize);
    BufferPtr locCmplxBuffer = MakeBuffer(CL_MEM_READ_ONLY, sizeof(complex) * partInnerSize);

    QueueCall(OCL_API_APPROXCOMPARE, ngc, ngs,
        { stateBuffer, toCompare->stateBuffer, poolItem->ulongBuffer, locCmplxBuffer }, sizeof(complex) * ngs);

    std::unique_ptr<complex[]> partInner(new complex[partInnerSize]);

    clFinish();
    tryCuda("Failed to read buffer", [&] {
        cudaError_t error = cudaStreamSynchronize(queue);
        if (error != cudaSuccess) {
            return error;
        }
        return cudaMemcpy(
            (void*)(partInner.get()), *locCmplxBuffer, sizeof(complex) * partInnerSize, cudaMemcpyDeviceToHost);
    });
    locCmplxBuffer.reset();
    SubtractAlloc(sizeof(complex) * partInnerSize);

    complex totInner = ZERO_CMPLX;
    for (size_t i = 0; i < partInnerSize; ++i) {
        totInner += partInner[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(totInner));
}

QInterfacePtr QEngineCUDA::Clone()
{
    if (!stateBuffer) {
        return CloneEmpty();
    }

    QEngineCUDAPtr copyPtr = std::make_shared<QEngineCUDA>(qubitCount, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);

    cudaEvent_t copyEvent = createCudaEvent();

    copyPtr->clFinish();
    clFinish();

    tryCuda("Failed to enqueue buffer copy", [&] {
        cudaError_t error = cudaMemcpyAsync(
            *(copyPtr->stateBuffer), *stateBuffer, sizeof(complex) * maxQPowerOcl, cudaMemcpyDeviceToDevice, queue);
        if (error != cudaSuccess) {
            return error;
        }
        return cudaEventRecord(copyEvent, queue);
    });
    cudaEventSynchronize(copyEvent);

    copyPtr->runningNorm = runningNorm;

    return copyPtr;
}

QEnginePtr QEngineCUDA::CloneEmpty()
{
    QEngineCUDAPtr copyPtr = std::make_shared<QEngineCUDA>(0U, 0U, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, (real1_f)amplitudeFloor);

    copyPtr->SetQubitCount(qubitCount);

    return copyPtr;
}

void QEngineCUDA::NormalizeState(real1_f nrm, real1_f norm_thresh, real1_f phaseArg)
{
    CHECK_ZERO_SKIP();

    if ((runningNorm == REAL1_DEFAULT_ARG) && (nrm == REAL1_DEFAULT_ARG)) {
        UpdateRunningNorm();
    }

    if (nrm < ZERO_R1) {
        // runningNorm can be set by OpenCL queue pop, so finish first.
        clFinish();
        nrm = (real1_f)runningNorm;
    }
    // We might avoid the clFinish().
    if (nrm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
        return;
    }
    if ((abs(ONE_R1 - nrm) <= FP_NORM_EPSILON) && ((phaseArg * phaseArg) <= FP_NORM_EPSILON)) {
        return;
    }
    // We might have async execution of gates still happening.
    clFinish();

    if (norm_thresh < ZERO_R1) {
        norm_thresh = (real1_f)amplitudeFloor;
    }
    nrm = ONE_R1_F / std::sqrt((real1_s)nrm);

    PoolItemPtr poolItem = GetFreePoolItem();

    complex c_args[2] = { complex((real1)norm_thresh, ZERO_R1), std::polar((real1)nrm, (real1)phaseArg) };
    cudaEvent_t writeRealArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->cmplxBuffer), sizeof(complex) * 2, c_args, writeRealArgsEvent);

    bitCapIntOcl bciArgs[1] = { maxQPowerOcl };
    cudaEvent_t writeBCIArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeBCIArgsEvent);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeRealArgsEvent);
    cudaEventSynchronize(writeBCIArgsEvent);
    wait_refs.clear();

    OCLAPI api_call;
    if (maxQPowerOcl == ngc) {
        api_call = OCL_API_NORMALIZE_WIDE;
    } else {
        api_call = OCL_API_NORMALIZE;
    }

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1_F);
}

void QEngineCUDA::UpdateRunningNorm(real1_f norm_thresh)
{
    if (!stateBuffer) {
        runningNorm = ZERO_R1_F;
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = (real1_f)amplitudeFloor;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    const real1 r1_args[1] = { (real1)norm_thresh };
    cudaEvent_t writeRealArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->realBuffer), sizeof(real1), r1_args, writeRealArgsEvent);

    cudaEvent_t writeBCIArgsEvent = createCudaEvent();
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), &maxQPowerOcl, writeBCIArgsEvent);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    cudaEventSynchronize(writeRealArgsEvent);
    cudaEventSynchronize(writeBCIArgsEvent);
    wait_refs.clear();

    QueueCall(OCL_API_UPDATENORM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    if (runningNorm <= FP_NORM_EPSILON) {
        ZeroAmplitudes();
    }
}

#if defined(__APPLE__)
complex* _aligned_state_vec_alloc(bitCapIntOcl allocSize)
{
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
    return (complex*)toRet;
}
#endif

std::shared_ptr<complex> QEngineOCL::AllocStateVec(bitCapInt elemCount, bool doForceAlloc)
{
    // If we're not using host ram, there's no reason to allocate.
    if (!elemCount || (!doForceAlloc && !stateVec)) {
        return NULL;
    }

#if defined(__ANDROID__)
    return std::shared_ptr<complex>(elemCount);
#else
    // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
    size_t allocSize = sizeof(complex) * (size_t)elemCount;
    if (allocSize < QRACK_ALIGN_SIZE) {
        allocSize = QRACK_ALIGN_SIZE;
    }
#if defined(__APPLE__)
    return std::shared_ptr<complex>(_aligned_state_vec_alloc(allocSize), [](complex* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return std::shared_ptr<complex>(
        (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE), [](complex* c) { _aligned_free(c); });
#else
    return std::shared_ptr<complex>((complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize), [](complex* c) { free(c); });
#endif
#endif
}

BufferPtr QEngineCUDA::MakeStateVecBuffer(std::shared_ptr<complex> nStateVec)
{
    if (!maxQPowerOcl) {
        return NULL;
    }

    if (nStateVec) {
        return MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl, nStateVec.get());
    } else {
        return MakeBuffer(CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl);
    }
}

void QEngineCUDA::ReinitBuffer()
{
    AddAlloc(sizeof(complex) * maxQPowerOcl);
    stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
    ResetStateBuffer(MakeStateVecBuffer(stateVec));
}

void QEngineCUDA::ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size)
{
    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapIntOcl bciArgs[2] = { size, offset };
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent);

    const size_t ngc = FixWorkItemCount(size, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();

    QueueCall(OCL_API_CLEARBUFFER, ngc, ngs, { buff, poolItem->ulongBuffer });
}

} // namespace Qrack
