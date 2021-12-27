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

#include "qengine_opencl.hpp"

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
#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent, error)                                                \
    error = queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &clEvent);                         \
    if (error != CL_SUCCESS) {                                                                                         \
        FreeAll();                                                                                                     \
        throw std::runtime_error("Failed to enqueue buffer write, error code: " + std::to_string(error));              \
    }

#define DISPATCH_LOC_WRITE(buff, size, array, clEvent, error)                                                          \
    error = queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, NULL, &clEvent);                                  \
    if (error != CL_SUCCESS) {                                                                                         \
        FreeAll();                                                                                                     \
        throw std::runtime_error("Failed to enqueue buffer write, error code: " + std::to_string(error));              \
    }

#define DISPATCH_WRITE(waitVec, buff, size, array, error)                                                              \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    error = queue.enqueueWriteBuffer(                                                                                  \
        buff, CL_FALSE, 0, size, array, waitVec.get(), &(device_context->wait_events->back()));                        \
    device_context->UnlockWaitEvents();                                                                                \
    if (error != CL_SUCCESS) {                                                                                         \
        FreeAll();                                                                                                     \
        throw std::runtime_error("Failed to enqueue buffer write, error code: " + std::to_string(error));              \
    }

#define DISPATCH_COPY(waitVec, buff1, buff2, size, error)                                                              \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    error = queue.enqueueCopyBuffer(buff1, buff2, 0, 0, size, waitVec.get(), &(device_context->wait_events->back()));  \
    device_context->UnlockWaitEvents();                                                                                \
    if (error != CL_SUCCESS) {                                                                                         \
        FreeAll();                                                                                                     \
        throw std::runtime_error("Failed to enqueue buffer read, error code: " + std::to_string(error));               \
    }

#define WAIT_REAL1_SUM(buff, size, array, sumPtr, error)                                                               \
    clFinish();                                                                                                        \
    error = queue.enqueueReadBuffer(buff, CL_TRUE, 0, sizeof(real1) * size, array, NULL, NULL);                        \
    if (error != CL_SUCCESS) {                                                                                         \
        FreeAll();                                                                                                     \
        throw std::runtime_error("Failed to enqueue buffer read, error code: " + std::to_string(error));               \
    }                                                                                                                  \
    *(sumPtr) = ParSum(array, size);

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateBuffer) {                                                                                                \
        return;                                                                                                        \
    }

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int devID, bool useHardwareRNG, bool ignored, real1_f norm_thresh,
    std::vector<int> devList, bitLenInt qubitThreshold, real1_f sep_thresh)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , stateVec(NULL)
    , deviceID(devID)
    , wait_refs()
    , nrmArray(NULL)
    , nrmGroupSize(0)
    , totalOclAllocSize(0)
    , unlockHostMem(false)
{
    InitOCL(devID);
    clFinish();
    SetPermutation(initState, phaseFac);
}

void QEngineOCL::GetAmplitudePage(complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (!stateBuffer) {
        std::fill(pagePtr, pagePtr + (bitCapIntOcl)length, ZERO_CMPLX);
        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * offset, sizeof(complex) * length, pagePtr, waitVec.get());
    wait_refs.clear();
}

void QEngineOCL::SetAmplitudePage(const complex* pagePtr, bitCapIntOcl offset, bitCapIntOcl length)
{
    if (!stateBuffer) {
        ReinitBuffer();
        if (length != maxQPowerOcl) {
            ClearBuffer(stateBuffer, 0, maxQPowerOcl);
        }
    }

    EventVecPtr waitVec = ResetWaitEvents();
    cl_int error = queue.enqueueWriteBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * offset, sizeof(complex) * length, pagePtr, waitVec.get());
    wait_refs.clear();
    if (error != CL_SUCCESS) {
        FreeAll();
        throw std::runtime_error("Failed to write buffer, error code: " + std::to_string(error));
    }

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::SetAmplitudePage(
    QEnginePtr pageEnginePtr, bitCapIntOcl srcOffset, bitCapIntOcl dstOffset, bitCapIntOcl length)
{
    QEngineOCLPtr pageEngineOclPtr = std::dynamic_pointer_cast<QEngineOCL>(pageEnginePtr);
    BufferPtr oStateBuffer = pageEngineOclPtr->stateBuffer;

    if (!stateBuffer && !oStateBuffer) {
        return;
    }

    if (!oStateBuffer) {
        if (length == maxQPower) {
            ZeroAmplitudes();
        } else {
            ClearBuffer(stateBuffer, dstOffset, length);
        }

        runningNorm = ZERO_R1;

        return;
    }

    cl_int error;

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl);
    }

    pageEngineOclPtr->clFinish();

    EventVecPtr waitVec = ResetWaitEvents();

    cl::Event copyEvent;
    error = queue.enqueueCopyBuffer(*oStateBuffer, *stateBuffer, sizeof(complex) * srcOffset,
        sizeof(complex) * dstOffset, sizeof(complex) * length, waitVec.get(), &copyEvent);
    if (error != CL_SUCCESS) {
        FreeAll();
        throw std::runtime_error("Failed to enqueue buffer copy, error code: " + std::to_string(error));
    }
    copyEvent.wait();

    runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::ShuffleBuffers(QEnginePtr engine)
{
    QEngineOCLPtr engineOcl = std::dynamic_pointer_cast<QEngineOCL>(engine);

    if (!stateBuffer && !(engineOcl->stateBuffer)) {
        return;
    }

    cl_int error;

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl);
    }

    if (!(engineOcl->stateBuffer)) {
        engineOcl->ReinitBuffer();
        engineOcl->ClearBuffer(engineOcl->stateBuffer, 0, engineOcl->maxQPowerOcl);
    }

    engineOcl->clFinish();

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, error);

    WaitCall(OCL_API_SHUFFLEBUFFERS, nrmGroupCount, nrmGroupSize,
        { stateBuffer, engineOcl->stateBuffer, poolItem->ulongBuffer });

    runningNorm = REAL1_DEFAULT_ARG;
    engineOcl->runningNorm = REAL1_DEFAULT_ARG;
}

void QEngineOCL::LockSync(cl_map_flags flags)
{
    lockSyncFlags = flags;
    clFinish();

    if (stateVec) {
        unlockHostMem = true;
        queue.enqueueMapBuffer(*stateBuffer, CL_TRUE, flags, 0, sizeof(complex) * maxQPowerOcl, NULL);
    } else {
        unlockHostMem = false;
        stateVec = AllocStateVec(maxQPowerOcl, true);
        if (lockSyncFlags & CL_MAP_READ) {
            queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, stateVec, NULL);
        }
    }
}

void QEngineOCL::UnlockSync()
{
    clFinish();

    if (unlockHostMem) {
        cl::Event unmapEvent;
        queue.enqueueUnmapMemObject(*stateBuffer, stateVec, NULL, &unmapEvent);
        unmapEvent.wait();
    } else {
        if (lockSyncFlags & CL_MAP_WRITE) {
            queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, stateVec, NULL);
        }
        FreeStateVec();
        stateVec = NULL;
    }

    lockSyncFlags = 0;
}

void QEngineOCL::clFinish(bool doHard)
{
    if (!device_context) {
        return;
    }

    while (wait_queue_items.size() > 1) {
        device_context->WaitOnAllEvents();
        PopQueue(NULL, CL_COMPLETE);
    }

    if (doHard) {
        queue.finish();
    } else {
        device_context->WaitOnAllEvents();
    }
    wait_refs.clear();
}

void QEngineOCL::clDump()
{
    if (!device_context) {
        return;
    }

    if (wait_queue_items.size()) {
        device_context->WaitOnAllEvents();
    }

    wait_queue_items.clear();
    wait_refs.clear();
}

size_t QEngineOCL::FixWorkItemCount(size_t maxI, size_t wic)
{
    if (wic > maxI) {
        // Guaranteed to be a power of two
        wic = maxI;
    } else {
        // Otherwise, clamp to a power of two
        size_t power = 2;
        while (power < wic) {
            power <<= ONE_BCI;
        }
        if (power > wic) {
            power >>= ONE_BCI;
        }
        wic = power;
    }
    return wic;
}

size_t QEngineOCL::FixGroupSize(size_t wic, size_t gs)
{
    if (gs > wic) {
        gs = wic;
    }
    size_t frac = wic / gs;
    while ((frac * gs) != wic) {
        gs++;
        frac = wic / gs;
    }
    return gs;
}

PoolItemPtr QEngineOCL::GetFreePoolItem()
{
    std::lock_guard<std::mutex> lock(queue_mutex);

    while (wait_queue_items.size() >= poolItems.size()) {
        poolItems.push_back(std::make_shared<PoolItem>(context));
    }

    return poolItems[wait_queue_items.size()];
}

EventVecPtr QEngineOCL::ResetWaitEvents(bool waitQueue)
{
    if (waitQueue) {
        while (wait_queue_items.size() > 1) {
            device_context->WaitOnAllEvents();
            PopQueue(NULL, CL_COMPLETE);
        }
    }

    wait_refs.emplace_back(device_context->ResetWaitEvents());
    return wait_refs.back();
}

void QEngineOCL::WaitCall(
    OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args, size_t localBuffSize)
{
    QueueCall(api_call, workItemCount, localGroupSize, args, localBuffSize);
    clFinish();
}

void CL_CALLBACK _PopQueue(cl_event event, cl_int type, void* user_data)
{
    ((QEngineOCL*)user_data)->PopQueue(event, type);
}

void QEngineOCL::PopQueue(cl_event event, cl_int type)
{
    queue_mutex.lock();

    poolItems.front()->probArray = NULL;
    poolItems.front()->angleArray = NULL;
    if (poolItems.front()->otherStateVec) {
        FreeStateVec(poolItems.front()->otherStateVec);
        poolItems.front()->otherStateVec = NULL;
    }

    SubtractAlloc(wait_queue_items.front().deallocSize);

    wait_queue_items.pop_front();

    if (poolItems.size() > 1) {
        rotate(poolItems.begin(), poolItems.begin() + 1, poolItems.end());
    }

    queue_mutex.unlock();

    DispatchQueue(event, type);
}

void QEngineOCL::DispatchQueue(cl_event event, cl_int type)
{
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (wait_queue_items.size() == 0) {
        return;
    }

    QueueItem item = wait_queue_items.front();

    while (item.isSetDoNorm || item.isSetRunningNorm) {
        if (item.isSetDoNorm) {
            doNormalize = item.doNorm;
        }
        if (item.isSetRunningNorm) {
            runningNorm = item.runningNorm;
        }

        wait_queue_items.pop_front();
        if (wait_queue_items.size() == 0) {
            return;
        }
        item = wait_queue_items.front();
    }

    std::vector<BufferPtr> args = item.buffers;

    // We have to reserve the kernel, because its argument hooks are unique. The same kernel therefore can't be used by
    // other QEngineOCL instances, until we're done queueing it.
    OCLDeviceCall ocl = device_context->Reserve(item.api_call);

    // Load the arguments.
    for (unsigned int i = 0; i < args.size(); i++) {
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
    device_context->wait_events->emplace_back();
    device_context->wait_events->back().setCallback(CL_COMPLETE, _PopQueue, this);
    cl_int error = queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(item.workItemCount), // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(), // vector of events to wait for
        &(device_context->wait_events->back())); // handle to wait for the kernel

    device_context->UnlockWaitEvents();

    if (error != CL_SUCCESS) {
        FreeAll();
        throw std::runtime_error("Failed to enqueue kernel, error code: " + std::to_string(error));
    }
}

real1_f QEngineOCL::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    complex amp;
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * (bitCapIntOcl)fullRegister, sizeof(complex), &amp, waitVec.get());
    wait_refs.clear();
    return clampProb(norm(amp));
}

void QEngineOCL::SetDevice(int dID, bool forceReInit)
{
    if (!(OCLEngine::Instance()->GetDeviceCount())) {
        FreeAll();
        throw std::runtime_error("Tried to initialize QEngineOCL, but no available OpenCL devices.");
    }

    bool didInit = (nrmArray != NULL);

    clFinish();

    int oldContextId = device_context ? device_context->context_id : 0;
    device_context = OCLEngine::Instance()->GetDeviceContextPtr(dID);

    if (didInit) {
        // If we're "switching" to the device we already have, don't reinitialize.
        if ((!forceReInit) && (oldContextId == device_context->context_id)) {
            deviceID = dID;
            context = device_context->context;
            queue = device_context->queue;

            return;
        }

        if (stateBuffer) {
            // This copies the contents of stateBuffer to host memory, to load into a buffer in the new context.
            LockSync();
        }
    } else {
        AddAlloc(sizeof(complex) * maxQPowerOcl);
    }

    deviceID = dID;
    context = device_context->context;
    queue = device_context->queue;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPLY2X2_NORM_SINGLE);

    bitCapIntOcl oldNrmVecAlignSize = nrmGroupSize ? (nrmGroupCount / nrmGroupSize) : 0;
    nrmGroupSize = ocl.call.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_context->device);
    procElemCount = device_context->GetProcElementCount();
    maxWorkItems = device_context->GetMaxWorkItems();

    // constrain to a power of two
    size_t groupSizePow = ONE_BCI;
    while (groupSizePow <= nrmGroupSize) {
        groupSizePow <<= ONE_BCI;
    }
    groupSizePow >>= ONE_BCI;
    nrmGroupSize = groupSizePow;
    size_t procElemPow = ONE_BCI;
    while (procElemPow <= procElemCount) {
        procElemPow <<= ONE_BCI;
    }
    procElemPow >>= ONE_BCI;
    nrmGroupCount = procElemPow * nrmGroupSize * 4U;
    while (nrmGroupCount > maxWorkItems) {
        nrmGroupCount >>= ONE_BCI;
    }

    // If the user wants to not use general host RAM, but we can't allocate enough on the device, fall back to host RAM
    // anyway.
    maxMem = device_context->GetGlobalSize();
    maxAlloc = device_context->GetMaxAlloc();
#if ENABLE_OCL_MEM_GUARDS
    size_t stateVecSize = maxQPowerOcl * sizeof(complex);
    // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
    if (stateVecSize > maxAlloc) {
        FreeAll();
        throw std::bad_alloc();
    } else if (useHostRam || ((OclMemDenom * stateVecSize) > maxMem)) {
        usingHostRam = true;
    } else {
        usingHostRam = false;
    }
#endif

    size_t nrmArrayAllocSize = (!nrmGroupSize || ((sizeof(real1) * nrmGroupCount / nrmGroupSize) < QRACK_ALIGN_SIZE))
        ? QRACK_ALIGN_SIZE
        : (sizeof(real1) * nrmGroupCount / nrmGroupSize);

    bool doResize = (nrmGroupCount / nrmGroupSize) != oldNrmVecAlignSize;

    if (didInit && doResize) {
        nrmBuffer = NULL;
        FreeAligned(nrmArray);
        nrmArray = NULL;
        SubtractAlloc(oldNrmVecAlignSize);
    }

    if (!didInit || doResize) {
        AddAlloc(nrmArrayAllocSize);
#if defined(__APPLE__)
        posix_memalign((void**)&nrmArray, QRACK_ALIGN_SIZE, nrmArrayAllocSize);
#elif defined(_WIN32) && !defined(__CYGWIN__)
        nrmArray = (real1*)_aligned_malloc(nrmArrayAllocSize, QRACK_ALIGN_SIZE);
#else
        nrmArray = (real1*)aligned_alloc(QRACK_ALIGN_SIZE, nrmArrayAllocSize);
#endif
        nrmBuffer = MakeBuffer(context, CL_MEM_READ_WRITE, nrmArrayAllocSize);
    }

    // create buffers on device (allocate space on GPU)
    if (didInit) {
        if (stateBuffer) {
            if (usingHostRam) {
                ResetStateBuffer(MakeStateVecBuffer(stateVec));
            } else {
                ResetStateBuffer(MakeStateVecBuffer(NULL));
                // In this branch, the QEngineOCL was previously allocated, and now we need to copy its memory to a
                // buffer.
                clFinish();
                queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, stateVec, NULL);

                ResetStateVec(NULL);
            }

            lockSyncFlags = 0;
        }
    } else {
        // In this branch, the QEngineOCL is first being initialized, and no data needs to be copied between device
        // contexts.
        stateVec = AllocStateVec(maxQPowerOcl, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);
    }

    poolItems.clear();
    poolItems.push_back(std::make_shared<PoolItem>(context));

    AddAlloc(sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));
    powersBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * pow2Ocl(QBCAPPOW));
}

real1_f QEngineOCL::ParSum(real1* toSum, bitCapIntOcl maxI)
{
    // This interface is potentially parallelizable, but, for now, better performance is probably given by implementing
    // it as a serial loop.
    real1_f totSum = ZERO_R1;
    for (bitCapIntOcl i = 0; i < maxI; i++) {
        totSum += toSum[i];
    }

    return totSum;
}

void QEngineOCL::InitOCL(int devID) { SetDevice(devID, true); }

void QEngineOCL::ResetStateVec(complex* nStateVec)
{
    if (stateVec) {
        FreeStateVec();
        stateVec = nStateVec;
    }
}

void QEngineOCL::ResetStateBuffer(BufferPtr nStateBuffer) { stateBuffer = nStateBuffer; }

void QEngineOCL::SetPermutation(bitCapInt perm, complex phaseFac)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    ClearBuffer(stateBuffer, 0, maxQPowerOcl);

    // If "permutationAmp" amp is in (read-only) use, this method complicates supersedes that application anyway.

    if (phaseFac == CMPLX_DEFAULT_ARG) {
        permutationAmp = GetNonunitaryPhase();
    } else {
        permutationAmp = phaseFac;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    queue.enqueueWriteBuffer(*stateBuffer, CL_FALSE, sizeof(complex) * (bitCapIntOcl)perm, sizeof(complex),
        &permutationAmp, waitVec.get(), &(device_context->wait_events->back()));
    device_context->UnlockWaitEvents();

    QueueSetRunningNorm(ONE_R1);
}

/// NOT gate, which is also Pauli x matrix
void QEngineOCL::X(bitLenInt qubit)
{
    const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    bitCapIntOcl qPowers[1];
    qPowers[0] = pow2Ocl(qubit);
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::PAULIX);
}

/// Apply Pauli Z matrix to bit
void QEngineOCL::Z(bitLenInt qubit)
{
    const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    bitCapIntOcl qPowers[1];
    qPowers[0] = pow2Ocl(qubit);
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PAULIZ);
}

void QEngineOCL::Invert(complex topRight, complex bottomLeft, bitLenInt qubitIndex)
{
    if ((topRight == bottomLeft) && (randGlobalPhase || (topRight == ONE_CMPLX))) {
        X(qubitIndex);
        return;
    }

    const complex pauliX[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    bitCapIntOcl qPowers[1];
    qPowers[0] = pow2Ocl(qubitIndex);
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::INVERT);
}

void QEngineOCL::Phase(complex topLeft, complex bottomRight, bitLenInt qubitIndex)
{
    if ((topLeft == bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        return;
    }

    if ((topLeft == -bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        Z(qubitIndex);
        return;
    }

    const complex pauliZ[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    bitCapIntOcl qPowers[1];
    qPowers[0] = pow2Ocl(qubitIndex);
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PHASE);
}

void QEngineOCL::Apply2x2(bitCapIntOcl offset1, bitCapIntOcl offset2, const complex* mtrx, bitLenInt bitCount,
    const bitCapIntOcl* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special, real1_f norm_thresh)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    const bool skipNorm = !doNormalize || (runningNorm == ONE_R1);
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
    EventVecPtr waitVec = ResetWaitEvents();

    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    const bitCapIntOcl maxI = maxQPowerOcl >> bitCount;
    bitCapIntOcl bciArgs[5] = { offset2, offset1, maxI, bitCount, 0 };

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
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(
        waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * bciArgsSize, bciArgs, writeArgsEvent, error);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    cmplx[4] = complex(doApplyNorm ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1, ZERO_R1);
    cmplx[5] = (real1)norm_thresh;

    BufferPtr locCmplxBuffer;
    cl::Event writeGateEvent;
    if (!isXGate && !isZGate) {
        DISPATCH_TEMP_WRITE(
            waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * CMPLX_NORM_LEN, cmplx, writeGateEvent, error);
    }

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    BufferPtr locPowersBuffer;
    cl::Event writeControlsEvent;
    if (bitCount > 2) {
        if (doCalcNorm) {
            locPowersBuffer = powersBuffer;
        } else {
            locPowersBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * bitCount);
        }
        if (sizeof(bitCapInt) == sizeof(bitCapIntOcl)) {
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent, error);
        } else {
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent, error);
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
        FreeAll();
        throw("Invalid APPLY2X2 kernel selected!");
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    if (!isXGate && !isZGate) {
        writeGateEvent.wait();
    }
    if (bitCount > 2) {
        writeControlsEvent.wait();
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
        QueueSetRunningNorm(ONE_R1);
    }

    if (!doCalcNorm) {
        return;
    }

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm, error);
    if (runningNorm == ZERO_R1) {
        ZeroAmplitudes();
    }
}

void QEngineOCL::BitMask(bitCapIntOcl mask, OCLAPI api_call, real1_f phase)
{
    CHECK_ZERO_SKIP();

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ mask;

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, mask, otherMask, 0, 0, 0, 0, 0, 0, 0 };

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    bool isPhaseParity = (api_call == OCL_API_PHASE_PARITY);
    if (isPhaseParity) {
        complex phaseFac = std::polar(ONE_R1, (real1)(phase / 2));
        ;
        complex cmplxArray[2] = { phaseFac, ONE_CMPLX / phaseFac };
        cl::Event writePhaseEvent;
        DISPATCH_TEMP_WRITE(
            waitVec, *(poolItem->cmplxBuffer), 2U * sizeof(complex), cmplxArray, writePhaseEvent, error);
        writePhaseEvent.wait();
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    if (isPhaseParity) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
    }
}

void QEngineOCL::UniformlyControlledSingleBit(const bitLenInt* controls, bitLenInt controlLen, bitLenInt qubitIndex,
    const complex* mtrxs, const bitCapInt* mtrxSkipPowers, bitLenInt mtrxSkipLen, bitCapInt mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        Mtrx(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    cl_int error;

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapIntOcl maxI = maxQPowerOcl >> ONE_BCI;
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxI, pow2Ocl(qubitIndex), controlLen, mtrxSkipLen,
        (bitCapIntOcl)mtrxSkipValueMask, 0, 0, 0, 0, 0 };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, error);

    BufferPtr nrmInBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(real1));
    real1 nrm = (runningNorm > ZERO_R1) ? ONE_R1 / (real1)sqrt(runningNorm) : ONE_R1;
    DISPATCH_WRITE(waitVec, *nrmInBuffer, sizeof(real1), &nrm, error);

    size_t sizeDiff = sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen);
    AddAlloc(sizeDiff);
    BufferPtr uniformBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeDiff);

    DISPATCH_WRITE(waitVec, *uniformBuffer, sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen), mtrxs, error);

    std::unique_ptr<bitCapIntOcl[]> qPowers(new bitCapIntOcl[controlLen + mtrxSkipLen]);
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = pow2Ocl(controls[i]);
    }
    for (bitLenInt i = 0; i < mtrxSkipLen; i++) {
        qPowers[controlLen + i] = (bitCapIntOcl)mtrxSkipPowers[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    DISPATCH_WRITE(waitVec, *powersBuffer, sizeof(bitCapIntOcl) * (controlLen + mtrxSkipLen), qPowers.get(), error);

    // We call the kernel, with global buffers and one local buffer.
    WaitCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    uniformBuffer.reset();
    qPowers.reset();

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm, error);

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::UniformParityRZ(bitCapInt mask, real1_f angle)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, 0, 0, 0, 0, 0, 0, 0, 0 };
    real1 cosine = (real1)cos(angle);
    real1 sine = (real1)sin(angle);
    complex phaseFacs[3] = { complex(cosine, sine), complex(cosine, -sine),
        (runningNorm > ZERO_R1) ? (ONE_R1 / (real1)sqrt(runningNorm)) : ONE_R1 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent, error);
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * 3, &phaseFacs, writeNormEvent, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall((runningNorm == ONE_R1) ? OCL_API_UNIFORMPARITYRZ : OCL_API_UNIFORMPARITYRZ_NORM, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1);
}

void QEngineOCL::CUniformParityRZ(const bitLenInt* controls, bitLenInt controlLen, bitCapInt mask, real1_f angle)
{
    if (!controlLen) {
        return UniformParityRZ(mask, angle);
    }

    CHECK_ZERO_SKIP();

    cl_int error;

    bitCapIntOcl controlMask = 0;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controlLen);
    BufferPtr controlBuffer = MakeBuffer(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers.get());
    controlPowers.reset();

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> controlLen), (bitCapIntOcl)mask, controlMask,
        controlLen, 0, 0, 0, 0, 0, 0 };
    real1 cosine = (real1)cos(angle);
    real1 sine = (real1)sin(angle);
    complex phaseFacs[2] = { complex(cosine, sine), complex(cosine, -sine) };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, writeArgsEvent, error);
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * 2, &phaseFacs, writeNormEvent, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall(OCL_API_CUNIFORMPARITYRZ, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer, controlBuffer });
    QueueSetRunningNorm(ONE_R1);
}

void QEngineOCL::ApplyMx(OCLAPI api_call, bitCapIntOcl* bciArgs, complex nrm)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent, error);
    BufferPtr locCmplxBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(complex));
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex), &nrm, writeNormEvent, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
    QueueSetRunningNorm(ONE_R1);
}

void QEngineOCL::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapIntOcl powerTest = result ? (bitCapIntOcl)qPower : 0;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), (bitCapIntOcl)qPower, powerTest, 0,
        0, 0, 0, 0, 0, 0 };

    ApplyMx(OCL_API_APPLYM, bciArgs, nrm);
}

void QEngineOCL::ApplyM(bitCapInt mask, bitCapInt result, complex nrm)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)result, 0, 0, 0, 0, 0, 0, 0 };

    ApplyMx(OCL_API_APPLYMREG, bciArgs, nrm);
}

void QEngineOCL::Compose(OCLAPI apiCall, bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy)
{
    if (!stateBuffer || !toCopy->stateBuffer) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(qubitCount + toCopy->qubitCount);
        return;
    }

    cl_int error;

    if (doNormalize) {
        NormalizeState();
    }
    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }

    // int toCopyDevID = toCopy->GetDeviceID();
    toCopy->SetDevice(deviceID);

    PoolItemPtr poolItem = GetFreePoolItem();
    EventVecPtr waitVec = ResetWaitEvents();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 7, bciArgs, writeArgsEvent, error);

    bitCapIntOcl oMaxQPower = maxQPowerOcl;
    bitCapIntOcl nMaxQPower = bciArgs[0];
    bitCapIntOcl nQubitCount = bciArgs[1] + toCopy->qubitCount;
    size_t nStateVecSize = nMaxQPower * sizeof(complex);
    maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (nStateVecSize > maxAlloc) {
        FreeAll();
        throw std::bad_alloc();
    }

    AddAlloc(sizeof(complex) * nMaxQPower);

    SetQubitCount(nQubitCount);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);
    const bool forceAlloc = !stateVec && ((OclMemDenom * nStateVecSize) > maxMem);

    writeArgsEvent.wait();
    wait_refs.clear();

    complex* nStateVec = AllocStateVec(maxQPowerOcl, forceAlloc);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    toCopy->clFinish();

    WaitCall(apiCall, ngc, ngs, { stateBuffer, toCopy->stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    // toCopy->SetDevice(toCopyDevID);

    SubtractAlloc(sizeof(complex) * oMaxQPower);
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy)
{
    bitLenInt result = qubitCount;

    bitCapIntOcl oQubitCount = toCopy->qubitCount;
    bitCapIntOcl nQubitCount = qubitCount + oQubitCount;
    bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    bitCapIntOcl startMask = maxQPowerOcl - ONE_BCI;
    bitCapIntOcl endMask = (toCopy->maxQPowerOcl - ONE_BCI) << (bitCapIntOcl)qubitCount;
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, startMask, endMask, 0, 0, 0, 0, 0, 0 };

    OCLAPI api_call;
    if (nMaxQPower <= nrmGroupCount) {
        api_call = OCL_API_COMPOSE_WIDE;
    } else {
        api_call = OCL_API_COMPOSE;
    }

    Compose(api_call, bciArgs, toCopy);

    return result;
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy, bitLenInt start)
{
    bitLenInt result = start;

    bitLenInt oQubitCount = toCopy->qubitCount;
    bitLenInt nQubitCount = qubitCount + oQubitCount;
    bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    bitCapIntOcl startMask = pow2Ocl(start) - ONE_BCI;
    bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, oQubitCount, startMask, midMask, endMask, start, 0, 0,
        0 };

    Compose(OCL_API_COMPOSE_MID, bciArgs, toCopy);

    return result;
}

void QEngineOCL::DecomposeDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr destination)
{
    // "Dispose" is basically the same as decompose, except "Dispose" throws the removed bits away.

    if (length == 0) {
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
        // Reinitialize stateVec RAM, on this device.
        destination->SetDevice(deviceID);
        destination->SetPermutation(0);
    }

    if (doNormalize) {
        NormalizeState();
    }
    if (destination && destination->doNormalize) {
        destination->NormalizeState();
    }

    if (destination) {
        destination->SetDevice(deviceID);
    }

    if (length == qubitCount) {
        if (destination != NULL) {
            destination->ResetStateVec(stateVec);
            destination->stateBuffer = stateBuffer;
            stateVec = NULL;
        }
        // This will be cleared by the destructor:
        SubtractAlloc(sizeof(complex) * (pow2Ocl(qubitCount) - 2U));
        ResetStateVec(AllocStateVec(2));
        stateBuffer = MakeStateVecBuffer(stateVec);
        SetQubitCount(1);
        return;
    }

    cl_int error;

    bitLenInt nLength = qubitCount - length;

    bitCapIntOcl partPower = pow2Ocl(length);
    bitCapIntOcl remainderPower = pow2Ocl(nLength);
    bitCapIntOcl oMaxQPower = maxQPowerOcl;
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    size_t remainderDiff = 2 * sizeof(real1) * remainderPower;
    AddAlloc(remainderDiff);

    // The "remainder" bits will always be maintained.
    BufferPtr probBuffer1 = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(probBuffer1, 0, remainderPower >> ONE_BCI);
    BufferPtr angleBuffer1 = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    ClearBuffer(angleBuffer1, 0, remainderPower >> ONE_BCI);

    // The removed "part" is only necessary for Decompose.
    BufferPtr probBuffer2, angleBuffer2;
    size_t partDiff = 2 * sizeof(real1) * partPower;
    if (destination) {
        AddAlloc(2 * sizeof(real1) * partPower);
        probBuffer2 = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(probBuffer2, 0, partPower >> ONE_BCI);
        angleBuffer2 = MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(real1) * partPower);
        ClearBuffer(angleBuffer2, 0, partPower >> ONE_BCI);
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    bitCapIntOcl largerPower = partPower > remainderPower ? partPower : remainderPower;

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
    if (!destination) {
        clFinish();
    } else {
        bciArgs[0] = partPower;

        destination->clFinish();

        poolItem = GetFreePoolItem();
        EventVecPtr waitVec2 = ResetWaitEvents();
        DISPATCH_WRITE(waitVec2, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, error);

        const size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        const size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        const size_t oNStateVecSize = maxQPowerOcl * sizeof(complex);

        WaitCall(OCL_API_DECOMPOSEAMP, ngc2, ngs2,
            { probBuffer2, angleBuffer2, poolItem->ulongBuffer, destination->stateBuffer });

        probBuffer2.reset();
        angleBuffer2.reset();

        SubtractAlloc(partDiff);

        if (!(destination->useHostRam) && destination->stateVec && oNStateVecSize <= destination->maxAlloc &&
            (2 * oNStateVecSize) <= destination->maxMem) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cl::Event copyEvent;
            error = destination->queue.enqueueCopyBuffer(
                *(destination->stateBuffer), *nSB, 0, 0, sizeof(complex) * destination->maxQPowerOcl, NULL, &copyEvent);
            if (error != CL_SUCCESS) {
                FreeAll();
                throw std::runtime_error("Failed to enqueue buffer copy, error code: " + std::to_string(error));
            }
            copyEvent.wait();

            destination->stateBuffer = nSB;
            FreeAligned(destination->stateVec);
            destination->stateVec = NULL;
        }
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPowerOcl;
    poolItem = GetFreePoolItem();
    EventVecPtr waitVec3 = ResetWaitEvents();
    DISPATCH_WRITE(waitVec3, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, error);

    const size_t ngc3 = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs3 = FixGroupSize(ngc, nrmGroupSize);

    const size_t nStateVecSize = maxQPowerOcl * sizeof(complex);

    clFinish();

    if (!useHostRam && stateVec && ((OclMemDenom * nStateVecSize) <= maxMem)) {
        FreeStateVec();
    }

    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeof(complex) * oMaxQPower);

    // Tell QueueCall to track deallocation:
    QueueCall(OCL_API_DECOMPOSEAMP, ngc3, ngs3, { probBuffer1, angleBuffer1, poolItem->ulongBuffer, stateBuffer }, 0,
        remainderDiff);
}

void QEngineOCL::Decompose(bitLenInt start, QInterfacePtr destination)
{
    DecomposeDispose(start, destination->GetQubitCount(), std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QEngineOCLPtr)NULL); }

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (length == 0) {
        return;
    }

    if (!stateBuffer) {
        SetQubitCount(qubitCount - length);
        return;
    }

    if (length == qubitCount) {
        // This will be cleared by the destructor:
        ResetStateVec(AllocStateVec(2));
        stateBuffer = MakeStateVecBuffer(stateVec);
        SubtractAlloc(sizeof(complex) * (pow2Ocl(qubitCount) - 2U));
        SetQubitCount(1);
        return;
    }

    cl_int error;

    if (doNormalize) {
        NormalizeState();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    bitLenInt nLength = qubitCount - length;
    bitCapIntOcl remainderPower = pow2Ocl(nLength);
    size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    bitCapIntOcl skipMask = pow2Ocl(start) - ONE_BCI;
    bitCapIntOcl disposedRes = (bitCapIntOcl)disposedPerm << (bitCapIntOcl)start;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { remainderPower, length, skipMask, disposedRes, 0, 0, 0, 0, 0, 0 };

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    SetQubitCount(nLength);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    AddAlloc(sizeof(complex) * maxQPowerOcl);
    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    WaitCall(OCL_API_DISPOSE, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    SubtractAlloc(sizeDiff);
}

real1_f QEngineOCL::Probx(OCLAPI api_call, bitCapIntOcl* bciArgs)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance, error);

    return clampProb(oneChance);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1_f QEngineOCL::Prob(bitLenInt qubit)
{
    if (qubitCount == 1) {
        return ProbAll(1);
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    bitCapIntOcl qPower = pow2Ocl(qubit);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROB, bciArgs);
}

// Returns probability of permutation of the register
real1_f QEngineOCL::ProbReg(bitLenInt start, bitLenInt length, bitCapInt permutation)
{
    if (start == 0 && qubitCount == length) {
        return ProbAll(permutation);
    }

    bitCapIntOcl perm = (bitCapIntOcl)permutation << (bitCapIntOcl)start;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), perm, start, length, 0, 0, 0, 0, 0,
        0 };

    return Probx(OCL_API_PROBREG, bciArgs);
}

void QEngineOCL::ProbRegAll(bitLenInt start, bitLenInt length, real1* probsArray)
{
    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl maxJ = maxQPowerOcl >> length;

    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(probsArray, probsArray + lengthPower, ZERO_R1);
        return;
    }

    cl_int error;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, start, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    AddAlloc(sizeof(real1) * lengthPower);
    BufferPtr probsBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBREGALL, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probsBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();
    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, waitVec2.get());
    wait_refs.clear();

    probsBuffer.reset();

    SubtractAlloc(sizeof(real1) * lengthPower);
}

// Returns probability of permutation of the register
real1_f QEngineOCL::ProbMask(bitCapInt mask, bitCapInt permutation)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    cl_int error;

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (length = 0; v; length++) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), (bitCapIntOcl)mask,
        (bitCapIntOcl)permutation, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[length]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, skipPowers.get());
    skipPowers.reset();

    bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASK, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer, qPowersBuffer },
        sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance, error);

    return clampProb(oneChance);
}

void QEngineOCL::ProbMaskAll(bitCapInt mask, real1* probsArray)
{
    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitLenInt length;
    std::vector<bitCapIntOcl> powersVec;
    for (length = 0; v; length++) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        powersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl maxJ = maxQPowerOcl >> length;

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

    cl_int error;

    v = (~(bitCapIntOcl)mask) & (maxQPowerOcl - ONE_BCI); // count the number of bits set in v
    bitCapIntOcl skipPower;
    bitLenInt skipLength = 0; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (skipLength = 0; v; skipLength++) {
        bitCapIntOcl oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        skipPowersVec.push_back(skipPower);
    }

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, length, skipLength, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs, error);

    size_t sizeDiff = sizeof(real1) * lengthPower + sizeof(bitCapIntOcl) * length + sizeof(bitCapIntOcl) * skipLength;
    AddAlloc(sizeDiff);

    BufferPtr probsBuffer = MakeBuffer(context, CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    std::unique_ptr<bitCapIntOcl[]> powers(new bitCapIntOcl[length]);
    std::copy(powersVec.begin(), powersVec.end(), powers.get());
    BufferPtr qPowersBuffer =
        MakeBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, powers.get());
    powers.reset();

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[skipLength]);
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers.get());
    BufferPtr qSkipPowersBuffer = MakeBuffer(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * skipLength, skipPowers.get());
    skipPowers.reset();

    const size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASKALL, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probsBuffer, qPowersBuffer, qSkipPowersBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();
    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, waitVec2.get());
    wait_refs.clear();

    probsBuffer.reset();
    qPowersBuffer.reset();
    qSkipPowersBuffer.reset();

    SubtractAlloc(sizeDiff);
}

real1_f QEngineOCL::ProbParity(bitCapInt mask)
{
    // If no bits in mask:
    if (!mask) {
        return ZERO_R1;
    }

    // If only one bit in mask:
    if (!(mask & (mask - ONE_BCI))) {
        return Prob(log2(mask));
    }

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, 0, 0, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROBPARITY, bciArgs);
}

bool QEngineOCL::ForceMParity(bitCapInt mask, bool result, bool doForce)
{
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)(result ? ONE_BCI : 0), 0, 0,
        0, 0, 0, 0, 0 };

    runningNorm = Probx(OCL_API_FORCEMPARITY, bciArgs);

    if (!doNormalize) {
        NormalizeState();
    }

    return result;
}

real1_f QEngineOCL::ExpectationBitsAll(const bitLenInt* bits, bitLenInt length, bitCapInt offset)
{
    if (length == 1U) {
        return Prob(bits[0]);
    }

    if (!stateBuffer || length == 0) {
        return ZERO_R1;
    }

    if (doNormalize) {
        NormalizeState();
    }

    std::unique_ptr<bitCapIntOcl[]> bitPowers(new bitCapIntOcl[length]);
    for (bitLenInt p = 0; p < length; p++) {
        bitPowers[p] = pow2Ocl(bits[p]);
    }

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    BufferPtr bitMapBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length);
    DISPATCH_WRITE(waitVec, *bitMapBuffer, sizeof(bitCapIntOcl) * length, bitPowers.get(), error);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, length, (bitCapIntOcl)offset, 0, 0, 0, 0, 0, 0, 0 };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, error);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_EXPPERM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, bitMapBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    real1_f expectation;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &expectation, error);

    return expectation;
}

real1_f QEngineOCL::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    LockSync(CL_MAP_READ);
    for (bitCapIntOcl i = 0; i < maxQPower; i++) {
        bitCapIntOcl outputInt = (i & outputMask) >> valueStart;
        real1 prob = norm(stateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return average;
}

void QEngineOCL::ArithmeticCall(
    OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], const unsigned char* values, bitCapIntOcl valuesPower)
{
    CArithmeticCall(api_call, bciArgs, NULL, 0, values, valuesPower);
}
void QEngineOCL::CArithmeticCall(OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], bitCapIntOcl* controlPowers,
    bitLenInt controlLen, const unsigned char* values, bitCapIntOcl valuesPower)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    size_t sizeDiff = sizeof(complex) * maxQPowerOcl;
    if (controlLen) {
        sizeDiff += sizeof(bitCapIntOcl) * controlLen;
    }
    if (values) {
        sizeDiff += sizeof(unsigned char) * valuesPower;
    }
    AddAlloc(sizeDiff);

    EventVecPtr waitVec = ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen) {
        controlBuffer = MakeBuffer(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers);
    }

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen) {
        device_context->LockWaitEvents();
        device_context->wait_events->emplace_back();
        error = queue.enqueueCopyBuffer(*stateBuffer, *nStateBuffer, 0, 0, sizeof(complex) * maxQPowerOcl,
            waitVec.get(), &(device_context->wait_events->back()));
        if (error != CL_SUCCESS) {
            FreeAll();
            throw std::runtime_error("Failed to enqueue buffer copy, error code: " + std::to_string(error));
        }
        device_context->UnlockWaitEvents();
    } else {
        ClearBuffer(nStateBuffer, 0, maxQPowerOcl);
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * BCI_ARG_LEN, bciArgs, error);

    bitCapIntOcl maxI = bciArgs[0];
    const size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    std::vector<BufferPtr> oclArgs = { stateBuffer, poolItem->ulongBuffer, nStateBuffer };

    BufferPtr loadBuffer;
    if (values) {
        loadBuffer = MakeBuffer(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, (void*)values);
        oclArgs.push_back(loadBuffer);
    }
    if (controlLen > 0) {
        oclArgs.push_back(controlBuffer);
    }

    WaitCall(api_call, ngc, ngs, oclArgs);

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    controlBuffer.reset();
    loadBuffer.reset();

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    shift %= length;
    if (shift == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~regMask);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, start, shift, length, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

#if ENABLE_ALU
/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl regMask = lengthMask << start;
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & ~(regMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, start, toMod, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Add or Subtract integer (without sign or carry, with controls)
void QEngineOCL::CINT(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, const bitLenInt* controls,
    bitLenInt controlLen)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl regMask = lengthMask << start;

    bitCapIntOcl controlMask = 0;
    std::unique_ptr<bitCapIntOcl[]> controlPowers(new bitCapIntOcl[controlLen]);
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers.get(), controlPowers.get() + controlLen);

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (regMask | controlMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> controlLen), regMask, otherMask, lengthPower,
        start, toMod, controlLen, controlMask, 0, 0 };

    CArithmeticCall(api_call, bciArgs, controlPowers.get(), controlLen);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INT(OCL_API_INC, (bitCapIntOcl)toAdd, start, length);
}

void QEngineOCL::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    CINT(OCL_API_CINC, (bitCapIntOcl)toAdd, inOutStart, length, controls, controlLen);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl regMask = (lengthPower - ONE_BCI) << start;
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(regMask | carryMask));

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), regMask, otherMask, lengthPower,
        carryMask, start, toMod, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Common driver method behing INCC and DECC
void QEngineOCL::INCDECC(bitCapInt toMod, bitLenInt inOutStart, bitLenInt length, bitLenInt carryIndex)
{
    INTC(OCL_API_INCDECC, (bitCapIntOcl)toMod, inOutStart, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    bitCapIntOcl regMask = lengthMask << start;
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ regMask;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, regMask, otherMask, lengthPower, overflowMask, start, toMod, 0,
        0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, (bitCapIntOcl)toAdd, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex,
    bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl lengthMask = lengthPower - ONE_BCI;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl overflowMask = pow2Ocl(overflowIndex);
    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl inOutMask = lengthMask << start;
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, lengthPower,
        overflowMask, carryMask, start, toMod, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_1, (bitCapIntOcl)toAdd, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl inOutMask = (lengthPower - ONE_BCI) << start;
    bitCapIntOcl otherMask = pow2MaskOcl(qubitCount) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, lengthPower,
        carryMask, start, toMod, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTSC(OCL_API_INCDECSC_2, (bitCapIntOcl)toAdd, start, length, carryIndex);
}

#if ENABLE_BCD
/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        FreeAll();
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ inOutMask;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, inOutMask, otherMask, start, toMod, nibbleCount, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineOCL::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, (bitCapIntOcl)toAdd, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        FreeAll();
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapIntOcl maxPow = intPowOcl(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl inOutMask = bitRegMaskOcl(start, length);
    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), inOutMask, otherMask, carryMask,
        start, toMod, nibbleCount, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCDECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INTBCDC(OCL_API_INCDECBCDC, (bitCapIntOcl)toAdd, start, length, carryIndex);
}
#endif

/** Multiply by integer */
void QEngineOCL::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(carryStart, length, 0);

    bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }

    MULx(OCL_API_MUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length);
}

/** Divide by integer */
void QEngineOCL::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
        FreeAll();
        throw std::runtime_error("DIV by zero");
    }

    MULx(OCL_API_DIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length);
}

/** Multiplication modulo N by integer, (out of place) */
void QEngineOCL::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    SetReg(outStart, length, 0);

    MULModx(OCL_API_MULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

void QEngineOCL::IMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    MULModx(OCL_API_IMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Raise a classical base to a quantum power, modulo N, (out of place) */
void QEngineOCL::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    CHECK_ZERO_SKIP();

    if (base == ONE_BCI) {
        SetReg(outStart, length, ONE_BCI);
        return;
    }

    MULModx(OCL_API_POWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length);
}

/** Quantum analog of classical "Full Adder" gate */
void QEngineOCL::FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_FULLADD);
}

/** Inverse of FullAdd */
void QEngineOCL::IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
{
    FullAdx(inputBit1, inputBit2, carryInSumOut, carryOut, OCL_API_IFULLADD);
}

void QEngineOCL::FullAdx(
    bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut, OCLAPI api_call)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)2U), pow2Ocl(inputBit1),
        pow2Ocl(inputBit2), pow2Ocl(carryInSumOut), pow2Ocl(carryOut), 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent, error);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

/** Controlled multiplication by integer */
void QEngineOCL::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (controlLen == 0) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0);

    bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 1) {
        return;
    }

    CMULx(OCL_API_CMUL, (bitCapIntOcl)toMul, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled division by integer */
void QEngineOCL::CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (toDiv == 0) {
        FreeAll();
        throw std::runtime_error("DIV by zero");
    }

    if (toDiv == 1) {
        return;
    }

    CMULx(OCL_API_CDIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (controlLen == 0) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0);

    bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 0) {
        return;
    }

    CMULModx(
        OCL_API_CMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls, controlLen);
}

void QEngineOCL::CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        IMULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    bitCapIntOcl lowPower = pow2Ocl(length);
    toMul &= (lowPower - ONE_BCI);
    if (toMul == 0) {
        return;
    }

    CMULModx(OCL_API_CIMULMODN_OUT, (bitCapIntOcl)toMul, (bitCapIntOcl)modN, inStart, outStart, length, controls,
        controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    const bitLenInt* controls, bitLenInt controlLen)
{
    CHECK_ZERO_SKIP();

    if (controlLen == 0) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0);

    CMULModx(
        OCL_API_CPOWMODN_OUT, (bitCapIntOcl)base, (bitCapIntOcl)modN, inStart, outStart, length, controls, controlLen);
}

void QEngineOCL::xMULx(OCLAPI api_call, bitCapIntOcl* bciArgs, BufferPtr controlBuffer)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ClearBuffer(nStateBuffer, 0, maxQPowerOcl);

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 10, bciArgs, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    if (controlBuffer) {
        WaitCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer, controlBuffer });
    } else {
        WaitCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    }

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);
}

void QEngineOCL::MULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << (bitCapIntOcl)inOutStart;
    const bitCapIntOcl carryMask = lowMask << (bitCapIntOcl)carryStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(carryStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inOutMask, carryMask,
        otherMask, length, inOutStart, carryStart, skipMask, 0 };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::MULModx(
    OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMod == 0) {
        return;
    }

    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inMask = lowMask << (bitCapIntOcl)inStart;
    const bitCapIntOcl outMask = lowMask << (bitCapIntOcl)outStart;
    const bitCapIntOcl skipMask = pow2MaskOcl(outStart);
    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> length), toMod, inMask, outMask, otherMask,
        length, inStart, outStart, skipMask, modN };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::CMULx(OCLAPI api_call, bitCapIntOcl toMod, bitLenInt inOutStart, bitLenInt carryStart,
    bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    const bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask | controlMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> (bitCapIntOcl)(controlLen + length)), toMod,
        controlLen, controlMask, inOutMask, carryMask, otherMask, length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controlLen * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

void QEngineOCL::CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, bitLenInt inOutStart,
    bitLenInt carryStart, bitLenInt length, const bitLenInt* controls, bitLenInt controlLen)
{
    const bitCapIntOcl lowMask = pow2MaskOcl(length);
    const bitCapIntOcl inOutMask = lowMask << inOutStart;
    const bitCapIntOcl carryMask = lowMask << carryStart;

    std::unique_ptr<bitCapIntOcl[]> skipPowers(new bitCapIntOcl[controlLen + length]);
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        bitCapIntOcl controlPower = pow2Ocl(controls[i]);
        skipPowers[i] = controlPower;
        controlMask |= controlPower;
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers.get(), skipPowers.get() + controlLen + length);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, toMod, controlLen, controlMask, inOutMask, carryMask, modN,
        length, inOutStart, carryStart };

    const size_t sizeDiff = sizeof(bitCapIntOcl) * ((controlLen * 2U) + length);
    AddAlloc(sizeDiff);
    BufferPtr controlBuffer = MakeBuffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeDiff, skipPowers.get());
    skipPowers.reset();

    xMULx(api_call, bciArgs, controlBuffer);

    SubtractAlloc(sizeDiff);
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, const unsigned char* values, bool resetValue)
{
    if (!stateBuffer) {
        return 0U;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, 0);
    }

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> valueLength), indexStart, inputMask,
        valueStart, valueBytes, valueLength, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    return (bitCapInt)(average + (ONE_R1 / 2));
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapIntOcl QEngineOCL::OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
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

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapIntOcl lengthPower = pow2Ocl(valueLength);
    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    bitCapIntOcl outputMask = bitRegMaskOcl(valueStart, valueLength);
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) & (~(inputMask | outputMask | carryMask));
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), indexStart, inputMask, valueStart,
        outputMask, otherMask, carryIn, carryMask, lengthPower, valueBytes };

    ArithmeticCall(api_call, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapIntOcl)(average + (ONE_R1 / 2));
}

/** Add based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDADC, 0, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, const unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDSBC, 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Set 8 bit register bits based on read from classical memory */
void QEngineOCL::Hash(bitLenInt start, bitLenInt length, const unsigned char* values)
{
    bitLenInt bytes = (length + 7) / 8;
    bitCapIntOcl inputMask = bitRegMaskOcl(start, length);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, start, inputMask, bytes, 0, 0, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_HASH, bciArgs, values, pow2Ocl(length) * bytes);
}

void QEngineOCL::PhaseFlipX(OCLAPI api_call, bitCapIntOcl* bciArgs)
{
    CHECK_ZERO_SKIP();

    cl_int error;

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent, error);

    const size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        pow2Ocl(flagIndex), (bitCapIntOcl)greaterPerm, start, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineOCL::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { (bitCapIntOcl)(maxQPowerOcl >> ONE_BCI), bitRegMaskOcl(start, length),
        (bitCapIntOcl)greaterPerm, start, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}
#endif

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(const complex* inputState)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, inputState, waitVec.get());
    wait_refs.clear();

    UpdateRunningNorm();
}

complex QEngineOCL::GetAmplitude(bitCapInt fullRegister)
{
    if (!stateBuffer) {
        return ZERO_CMPLX;
    }

    if (doNormalize) {
        NormalizeState();
    }

    complex amp;
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * (bitCapIntOcl)fullRegister, sizeof(complex), &amp, waitVec.get());
    wait_refs.clear();

    return amp;
}

void QEngineOCL::SetAmplitude(bitCapInt perm, complex amp)
{
    if (doNormalize) {
        NormalizeState();
    }
    clFinish();

    if (!stateBuffer && !norm(amp)) {
        return;
    }

    runningNorm = REAL1_DEFAULT_ARG;

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl);
    }

    permutationAmp = amp;

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    queue.enqueueWriteBuffer(*stateBuffer, CL_FALSE, sizeof(complex) * (bitCapIntOcl)perm, sizeof(complex),
        &permutationAmp, waitVec.get(), &(device_context->wait_events->back()));
    device_context->UnlockWaitEvents();
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineOCL::GetQuantumState(complex* outputState)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        std::fill(outputState, outputState + maxQPowerOcl, ZERO_CMPLX);
        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, outputState, waitVec.get());
    wait_refs.clear();

    clFinish();
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineOCL::GetProbs(real1* outputProbs) { ProbRegAll(0, qubitCount, outputProbs); }

real1_f QEngineOCL::SumSqrDiff(QEngineOCLPtr toCompare)
{
    if (this == toCompare.get()) {
        return ZERO_R1;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }

    if (!stateBuffer && !toCompare->stateBuffer) {
        return ZERO_R1;
    }

    if (!stateBuffer) {
        toCompare->UpdateRunningNorm();
        return toCompare->runningNorm;
    }

    if (!toCompare->stateBuffer) {
        UpdateRunningNorm();
        return runningNorm;
    }

    cl_int error;

    toCompare->clFinish();

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, error);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    const int partInnerSize = ngc / ngs;

    AddAlloc(sizeof(complex) * partInnerSize);
    BufferPtr locCmplxBuffer = MakeBuffer(context, CL_MEM_READ_ONLY, sizeof(complex) * partInnerSize);

    QueueCall(OCL_API_APPROXCOMPARE, ngc, ngs,
        { stateBuffer, toCompare->stateBuffer, poolItem->ulongBuffer, locCmplxBuffer }, sizeof(complex) * nrmGroupSize);

    std::unique_ptr<complex[]> partInner(new complex[partInnerSize]);

    clFinish();
    queue.enqueueReadBuffer(*locCmplxBuffer, CL_TRUE, 0, sizeof(complex) * partInnerSize, partInner.get(), NULL, NULL);

    locCmplxBuffer.reset();

    SubtractAlloc(sizeof(complex) * partInnerSize);

    complex totInner = ZERO_CMPLX;
    for (int i = 0; i < partInnerSize; i++) {
        totInner += partInner[i];
    }

    return ONE_R1 - clampProb(norm(totInner));
}

QInterfacePtr QEngineOCL::Clone()
{
    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, amplitudeFloor);

    copyPtr->clFinish();
    clFinish();
    copyPtr->runningNorm = runningNorm;

    EventVecPtr waitVec = ResetWaitEvents();
    if (stateBuffer) {
        cl_int error;
        DISPATCH_COPY(waitVec, *stateBuffer, *(copyPtr->stateBuffer), sizeof(complex) * maxQPowerOcl, error);
    } else {
        copyPtr->ZeroAmplitudes();
    }
    clFinish();

    return copyPtr;
}

void QEngineOCL::NormalizeState(real1_f nrm, real1_f norm_thresh)
{
    // We might have async execution of gates still happening.
    clFinish();

    CHECK_ZERO_SKIP();

    cl_int error;

    if (nrm < ZERO_R1) {
        nrm = runningNorm;
    }
    if ((nrm <= ZERO_R1) || (nrm == ONE_R1)) {
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    real1 r1_args[2] = { (real1)norm_thresh, (real1)(ONE_R1 / sqrt(nrm)) };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1) * 2, r1_args, writeRealArgsEvent, error);

    bitCapIntOcl bciArgs[1] = { maxQPowerOcl };
    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeBCIArgsEvent, error);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    OCLAPI api_call;
    if (maxQPowerOcl == ngc) {
        api_call = OCL_API_NORMALIZE_WIDE;
    } else {
        api_call = OCL_API_NORMALIZE;
    }

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer });
    QueueSetRunningNorm(ONE_R1);
}

void QEngineOCL::UpdateRunningNorm(real1_f norm_thresh)
{
    if (!stateBuffer) {
        runningNorm = ZERO_R1;
        return;
    }

    cl_int error;

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    real1 r1_args[1] = { (real1)norm_thresh };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1), r1_args, writeRealArgsEvent, error);

    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), &maxQPowerOcl, writeBCIArgsEvent, error);

    const size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    QueueCall(OCL_API_UPDATENORM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm, error);

    if (runningNorm == ZERO_R1) {
        ZeroAmplitudes();
    }
}

complex* QEngineOCL::AllocStateVec(bitCapInt elemCount, bool doForceAlloc)
{
    // If we're not using host ram, there's no reason to allocate.
    if (!doForceAlloc && !stateVec) {
        return NULL;
    }

    size_t allocSize = sizeof(complex) * (bitCapIntOcl)elemCount;
    if (allocSize < QRACK_ALIGN_SIZE) {
        allocSize = QRACK_ALIGN_SIZE;
    }

    // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE, allocSize);
    return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (complex*)_aligned_malloc(allocSize, QRACK_ALIGN_SIZE);
#else
    return (complex*)aligned_alloc(QRACK_ALIGN_SIZE, allocSize);
#endif
}

BufferPtr QEngineOCL::MakeStateVecBuffer(complex* nStateVec)
{
    if (nStateVec) {
        return MakeBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl, nStateVec);
    } else {
        return MakeBuffer(context, CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl);
    }
}

void QEngineOCL::ReinitBuffer()
{
    AddAlloc(sizeof(complex) * maxQPowerOcl);
    ResetStateVec(AllocStateVec(maxQPowerOcl, usingHostRam));
    ResetStateBuffer(MakeStateVecBuffer(stateVec));
}

void QEngineOCL::ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size)
{
    cl_int error;

    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapIntOcl bciArgs[2] = { size, offset };
    cl::Event writeArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent, error);

    const size_t ngc = FixWorkItemCount(size, nrmGroupCount);
    const size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();

    QueueCall(OCL_API_CLEARBUFFER, ngc, ngs, { buff, poolItem->ulongBuffer });
}

} // namespace Qrack
