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

#include <memory>

#include "oclengine.hpp"
#include "qengine_opencl.hpp"
#include "qfactory.hpp"

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
#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)                                                       \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &clEvent);                                 \
    queue.flush();

#define DISPATCH_LOC_WRITE(buff, size, array, clEvent)                                                                 \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, NULL, &clEvent);                                          \
    queue.flush();

#define DISPATCH_WRITE(waitVec, buff, size, array)                                                                     \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &(device_context->wait_events->back()));   \
    device_context->UnlockWaitEvents();                                                                                \
    queue.flush()

#define DISPATCH_READ(waitVec, buff, size, array)                                                                      \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueReadBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &(device_context->wait_events->back()));    \
    device_context->UnlockWaitEvents();                                                                                \
    queue.flush()

#define DISPATCH_COPY(waitVec, buff1, buff2, size)                                                                     \
    device_context->LockWaitEvents();                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueCopyBuffer(buff1, buff2, 0, 0, size, waitVec.get(), &(device_context->wait_events->back()));          \
    device_context->UnlockWaitEvents();                                                                                \
    queue.flush();

#define WAIT_REAL1_SUM(buff, size, array, sumPtr)                                                                      \
    clFinish();                                                                                                        \
    queue.enqueueReadBuffer(buff, CL_TRUE, 0, sizeof(real1) * size, array, NULL, NULL);                                \
    *(sumPtr) = ParSum(array, size);

#define CHECK_ZERO_SKIP()                                                                                              \
    if (!stateBuffer) {                                                                                                \
        return;                                                                                                        \
    }

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int devID, bool useHardwareRNG, bool ignored, real1 norm_thresh,
    std::vector<int> devList, bitLenInt qubitThreshold)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG, norm_thresh)
    , stateVec(NULL)
    , deviceID(devID)
    , wait_refs()
    , nrmArray(NULL)
    , nrmGroupSize(0)
    , unlockHostMem(false)
{
    maxQPowerOcl = pow2Ocl(qubitCount);
    InitOCL(devID);
    clFinish();
    SetPermutation(initState, phaseFac);
}

void QEngineOCL::GetAmplitudePage(complex* pagePtr, const bitCapInt offset, const bitCapInt length)
{
    if (!stateBuffer) {
        std::fill(pagePtr, pagePtr + length, ZERO_CMPLX);
        return;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * offset, sizeof(complex) * length, pagePtr, waitVec.get());
}

void QEngineOCL::SetAmplitudePage(const complex* pagePtr, const bitCapInt offset, const bitCapInt length)
{
    if (!stateBuffer) {
        ReinitBuffer();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueWriteBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * offset, sizeof(complex) * length, pagePtr, waitVec.get());

    runningNorm = ONE_R1;
}

void QEngineOCL::SetAmplitudePage(
    QEnginePtr pageEnginePtr, const bitCapInt srcOffset, const bitCapInt dstOffset, const bitCapInt length)
{
    QEngineOCLPtr pageEngineOclPtr = std::dynamic_pointer_cast<QEngineOCL>(pageEnginePtr);
    BufferPtr oStateBuffer = pageEngineOclPtr->stateBuffer;

    if (!stateBuffer && !oStateBuffer) {
        return;
    }

    if (!oStateBuffer) {
        ClearBuffer(stateBuffer, dstOffset, length, ResetWaitEvents());
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl, ResetWaitEvents());
    }

    clFinish();
    pageEngineOclPtr->clFinish();

    queue.enqueueCopyBuffer(*oStateBuffer, *stateBuffer, sizeof(complex) * srcOffset, sizeof(complex) * dstOffset,
        sizeof(complex) * length);

    queue.finish();

    runningNorm = ONE_R1;
}

void QEngineOCL::ShuffleBuffers(QEnginePtr engine)
{
    QEngineOCLPtr engineOcl = std::dynamic_pointer_cast<QEngineOCL>(engine);

    if (!stateBuffer && !(engineOcl->stateBuffer)) {
        return;
    }

    if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl, ResetWaitEvents());
    }

    if (!(engineOcl->stateBuffer)) {
        engineOcl->ReinitBuffer();
        engineOcl->ClearBuffer(engineOcl->stateBuffer, 0, engineOcl->maxQPowerOcl, engineOcl->ResetWaitEvents());
    }

    size_t halfSize = sizeof(complex) * (maxQPowerOcl >> ONE_BCI);
    cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, halfSize);

    engineOcl->clFinish();
    clFinish();

    queue.enqueueCopyBuffer(*stateBuffer, tempBuffer, halfSize, 0, halfSize);
    queue.enqueueCopyBuffer(*(engineOcl->stateBuffer), *stateBuffer, 0, halfSize, halfSize);
    queue.enqueueCopyBuffer(tempBuffer, *(engineOcl->stateBuffer), 0, 0, halfSize);

    queue.finish();

    runningNorm = ONE_R1;
    engineOcl->runningNorm = ONE_R1;
}

void QEngineOCL::LockSync(cl_int flags)
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
        wait_refs.clear();
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

    wait_queue_items.clear();
    device_context->WaitOnAllEvents();
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

void QEngineOCL::QueueCall(
    OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args, size_t localBuffSize)
{
    QueueItem item(api_call, workItemCount, localGroupSize, args, localBuffSize);

    queue_mutex.lock();
    bool isBase = (wait_queue_items.size() == 0);
    wait_queue_items.push_back(item);
    queue_mutex.unlock();

    if (isBase) {
        DispatchQueue(NULL, CL_COMPLETE);
    }
}

void CL_CALLBACK _PopQueue(cl_event event, cl_int type, void* user_data)
{
    ((QEngineOCL*)user_data)->PopQueue(event, type);
}

void QEngineOCL::PopQueue(cl_event event, cl_int type)
{
    queue_mutex.lock();

    wait_queue_items.pop_front();

    poolItems.front()->probArray = NULL;
    poolItems.front()->angleArray = NULL;
    if (poolItems.front()->otherStateVec) {
        FreeStateVec(poolItems.front()->otherStateVec);
        poolItems.front()->otherStateVec = NULL;
    }

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
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(item.workItemCount), // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(), // vector of events to wait for
        &(device_context->wait_events->back())); // handle to wait for the kernel

    device_context->UnlockWaitEvents();
    queue.flush();
}

real1 QEngineOCL::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    complex amp[1];
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * (bitCapIntOcl)fullRegister, sizeof(complex), amp, waitVec.get());
    wait_refs.clear();
    return norm(amp[0]);
}

void QEngineOCL::SetDevice(const int& dID, const bool& forceReInit)
{
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
    }

    deviceID = dID;
    context = device_context->context;
    queue = device_context->queue;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPLY2X2_NORM_SINGLE);

    bitCapIntOcl oldNrmVecAlignSize = nrmGroupSize ? (nrmGroupCount / nrmGroupSize) : 0;
    nrmGroupSize = ocl.call.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_context->device);
    procElemCount = device_context->device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    maxWorkItems = device_context->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];

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
    maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
#if ENABLE_OCL_MEM_GUARDS
    size_t stateVecSize = maxQPowerOcl * sizeof(complex);
    // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
    if (stateVecSize > maxAlloc) {
        throw "Error: State vector exceeds device maximum OpenCL allocation";
    } else if (useHostRam || ((OclMemDenom * stateVecSize) > maxMem)) {
        usingHostRam = true;
    } else {
        usingHostRam = false;
    }
#endif

    size_t nrmVecAlignSize = ((sizeof(real1) * nrmGroupCount / nrmGroupSize) < QRACK_ALIGN_SIZE)
        ? QRACK_ALIGN_SIZE
        : (sizeof(real1) * nrmGroupCount / nrmGroupSize);

    bool doResize = (nrmGroupCount / nrmGroupSize) != oldNrmVecAlignSize;

    if (didInit && doResize) {
        nrmBuffer = NULL;
        FreeAligned(nrmArray);
        nrmArray = NULL;
    }

    if (!didInit || doResize) {
#if defined(__APPLE__)
        posix_memalign((void**)&nrmArray, QRACK_ALIGN_SIZE, nrmVecAlignSize);
#elif defined(_WIN32) && !defined(__CYGWIN__)
        nrmArray = (real1*)_aligned_malloc(nrmVecAlignSize, QRACK_ALIGN_SIZE);
#else
        nrmArray = (real1*)aligned_alloc(QRACK_ALIGN_SIZE, nrmVecAlignSize);
#endif
        nrmBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, nrmVecAlignSize);
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
    powersBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * pow2(QBCAPPOW));
}

real1 QEngineOCL::ParSum(real1* toSum, bitCapIntOcl maxI)
{
    // This interface is potentially parallelizable, but, for now, better performance is probably given by implementing
    // it as a serial loop.
    real1 totNorm = 0;
    for (bitCapIntOcl i = 0; i < maxI; i++) {
        totNorm += toSum[i];
    }
    return totNorm;
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

    ClearBuffer(stateBuffer, 0, maxQPowerOcl, ResetWaitEvents());

    // If "permutationAmp" amp is in (read-only) use, this method complicates supersedes that application anyway.

    if (phaseFac == complex(-999.0, -999.0)) {
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
    queue.flush();

    runningNorm = ONE_R1;
}

void QEngineOCL::ArithmeticCall(
    OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], unsigned char* values, bitCapIntOcl valuesPower)
{
    CArithmeticCall(api_call, bciArgs, NULL, 0, values, valuesPower);
}

void QEngineOCL::CArithmeticCall(OCLAPI api_call, bitCapIntOcl (&bciArgs)[BCI_ARG_LEN], bitCapIntOcl* controlPowers,
    const bitLenInt controlLen, unsigned char* values, bitCapIntOcl valuesPower)
{
    CHECK_ZERO_SKIP();

    EventVecPtr waitVec = ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen > 0) {
        controlBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * controlLen, controlPowers);
    }

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen > 0) {
        device_context->LockWaitEvents();
        device_context->wait_events->emplace_back();
        queue.enqueueCopyBuffer(*stateBuffer, *nStateBuffer, 0, 0, sizeof(complex) * maxQPowerOcl, waitVec.get(),
            &(device_context->wait_events->back()));
        device_context->UnlockWaitEvents();
        queue.flush();
    } else {
        ClearBuffer(nStateBuffer, 0, maxQPowerOcl, waitVec);
    }

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * BCI_ARG_LEN, bciArgs);

    bitCapIntOcl maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    std::vector<BufferPtr> oclArgs = { stateBuffer, poolItem->ulongBuffer, nStateBuffer };

    BufferPtr loadBuffer;
    if (values) {
        loadBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        oclArgs.push_back(loadBuffer);
    }
    if (controlLen > 0) {
        oclArgs.push_back(controlBuffer);
    }

    WaitCall(api_call, ngc, ngs, oclArgs);

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);
}

/// NOT gate, which is also Pauli x matrix
void QEngineOCL::X(bitLenInt qubit)
{
    const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
    bitCapInt qPowers[1];
    qPowers[0] = pow2(qubit);
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::PAULIX);
}

/// Apply Pauli Z matrix to bit
void QEngineOCL::Z(bitLenInt qubit)
{
    const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };
    bitCapInt qPowers[1];
    qPowers[0] = pow2(qubit);
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PAULIZ);
}

void QEngineOCL::ApplySingleInvert(const complex topRight, const complex bottomLeft, bitLenInt qubitIndex)
{
    if ((topRight == bottomLeft) && (randGlobalPhase || (topRight == ONE_CMPLX))) {
        X(qubitIndex);
        return;
    }

    const complex pauliX[4] = { ZERO_CMPLX, topRight, bottomLeft, ZERO_CMPLX };
    bitCapInt qPowers[1];
    qPowers[0] = pow2(qubitIndex);
    Apply2x2(0U, qPowers[0], pauliX, 1U, qPowers, false, SPECIAL_2X2::INVERT);
}

void QEngineOCL::ApplySinglePhase(const complex topLeft, const complex bottomRight, bitLenInt qubitIndex)
{
    if ((topLeft == bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        return;
    }

    if ((topLeft == -bottomRight) && (randGlobalPhase || (topLeft == ONE_CMPLX))) {
        Z(qubitIndex);
        return;
    }

    const complex pauliZ[4] = { topLeft, ZERO_CMPLX, ZERO_CMPLX, bottomRight };
    bitCapInt qPowers[1];
    qPowers[0] = pow2(qubitIndex);
    Apply2x2(0U, qPowers[0], pauliZ, 1U, qPowers, false, SPECIAL_2X2::PHASE);
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special, real1 norm_thresh)
{
    CHECK_ZERO_SKIP();

    bool skipNorm = !doNormalize || (runningNorm == ONE_R1);
    bool isXGate = skipNorm && (special == SPECIAL_2X2::PAULIX);
    bool isZGate = skipNorm && (special == SPECIAL_2X2::PAULIZ);
    bool isInvertGate = skipNorm && (special == SPECIAL_2X2::INVERT);
    bool isPhaseGate = skipNorm && (special == SPECIAL_2X2::PHASE);

    // Are we going to calculate the normalization factor, on the fly? We can't, if this call doesn't iterate through
    // every single permutation amplitude.
    doCalcNorm = (doCalcNorm || (runningNorm != ONE_R1)) && doNormalize && !isXGate && !isZGate && !isInvertGate &&
        !isPhaseGate && (bitCount == 1);

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec;
    if (doCalcNorm) {
        waitVec = ResetWaitEvents();
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapIntOcl maxI = maxQPowerOcl >> bitCount;
    bitCapIntOcl bciArgs[5] = { (bitCapIntOcl)offset2, (bitCapIntOcl)offset1, maxI, bitCount, 0 };

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

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
            bciArgs[2] = (bitCapIntOcl)(qPowersSorted[0] - 1U);
        } else {
            bciArgsSize = 4;
            bciArgs[3] = (bitCapIntOcl)(qPowersSorted[0] - 1U);
        }
    } else if (bitCount == 2) {
        // Double bit gates include both controlled and swap gates. To reuse the code for both cases, we need two offset
        // arguments. Hence, we cannot easily overwrite either of the bit offset arguments.
        bciArgsSize = 5;
        bciArgs[3] = (bitCapIntOcl)(qPowersSorted[0] - 1U);
        bciArgs[4] = (bitCapIntOcl)(qPowersSorted[1] - 1U);
    }
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * bciArgsSize, bciArgs, writeArgsEvent);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    bool isUnitLength = (runningNorm == ONE_R1) || !(doNormalize && (bitCount == 1));
    cmplx[4] = complex(isUnitLength ? ONE_R1 : (ONE_R1 / std::sqrt(runningNorm)), ZERO_R1);
    cmplx[5] = norm_thresh;

    BufferPtr locCmplxBuffer;
    cl::Event writeGateEvent;
    if (!isXGate && !isZGate) {
        DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * CMPLX_NORM_LEN, cmplx, writeGateEvent);
    }

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    BufferPtr locPowersBuffer;
    cl::Event writeControlsEvent;
    bitCapIntOcl* qPowersSortedOcl = NULL;
    if (bitCount > 2) {
        if (doCalcNorm) {
            locPowersBuffer = powersBuffer;
        } else {
            locPowersBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * bitCount);
        }
        if (sizeof(bitCapInt) == sizeof(bitCapIntOcl)) {
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSorted, writeControlsEvent);
        } else {
            qPowersSortedOcl = new bitCapIntOcl[bitCount];
            for (bitLenInt i = 0; i < bitCount; i++) {
                qPowersSortedOcl[i] = (bitCapIntOcl)qPowersSorted[i];
            }
            DISPATCH_TEMP_WRITE(
                waitVec, *locPowersBuffer, sizeof(bitCapIntOcl) * bitCount, qPowersSortedOcl, writeControlsEvent);
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
        throw("Invalid APPLY2X2 kernel selected!");
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    if (!isXGate && !isZGate) {
        writeGateEvent.wait();
    }
    if (bitCount > 2) {
        writeControlsEvent.wait();
        if (sizeof(bitCapInt) != sizeof(bitCapIntOcl)) {
            delete[] qPowersSortedOcl;
        }
    }
    if (doCalcNorm) {
        wait_refs.clear();
    }

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

    if (doCalcNorm) {
        // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
        // values into a single normalization constant.
        WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);
    } else if ((runningNorm == ZERO_R1) || ((bitCount == 1) && !isXGate && !isZGate && !isInvertGate && !isPhaseGate)) {
        runningNorm = ONE_R1;
    }
}

void QEngineOCL::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
    bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    CHECK_ZERO_SKIP();

    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(mtrxs + (bitCapIntOcl)(mtrxSkipValueMask * 4U), qubitIndex);
        return;
    }

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapIntOcl maxI = maxQPowerOcl >> ONE_BCI;
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxI, pow2Ocl(qubitIndex), controlLen, mtrxSkipLen,
        (bitCapIntOcl)mtrxSkipValueMask, 0, 0, 0, 0, 0 };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs);

    BufferPtr nrmInBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(real1));
    real1 nrm = (real1)(ONE_R1 / std::sqrt(runningNorm));
    DISPATCH_WRITE(waitVec, *nrmInBuffer, sizeof(real1), &nrm);

    BufferPtr uniformBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_READ_ONLY, sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen));

    DISPATCH_WRITE(waitVec, *uniformBuffer, sizeof(complex) * 4U * pow2Ocl(controlLen + mtrxSkipLen), mtrxs);

    bitCapIntOcl* qPowers = new bitCapIntOcl[controlLen + mtrxSkipLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = pow2Ocl(controls[i]);
    }
    for (bitLenInt i = 0; i < mtrxSkipLen; i++) {
        qPowers[controlLen + i] = (bitCapIntOcl)mtrxSkipPowers[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    DISPATCH_WRITE(waitVec, *powersBuffer, sizeof(bitCapIntOcl) * (controlLen + mtrxSkipLen), qPowers);

    // We call the kernel, with global buffers and one local buffer.
    WaitCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    delete[] qPowers;
}

void QEngineOCL::ApplyMx(OCLAPI api_call, bitCapIntOcl* bciArgs, complex nrm)
{
    CHECK_ZERO_SKIP();

    // We don't actually have to wait, so this is empty:
    EventVecPtr waitVec;
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 3, bciArgs, writeArgsEvent);
    BufferPtr locCmplxBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(complex));
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex), &nrm, writeNormEvent);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });

    runningNorm = ONE_R1;
}

void QEngineOCL::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapIntOcl powerTest = result ? (bitCapIntOcl)qPower : 0;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, (bitCapIntOcl)qPower, powerTest, 0, 0, 0, 0, 0, 0,
        0 };

    ApplyMx(OCL_API_APPLYM, bciArgs, nrm);
}

void QEngineOCL::ApplyM(bitCapInt mask, bitCapInt result, complex nrm)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, (bitCapIntOcl)mask, (bitCapIntOcl)result, 0, 0, 0, 0, 0, 0, 0 };

    ApplyMx(OCL_API_APPLYMREG, bciArgs, nrm);
}

void QEngineOCL::Compose(OCLAPI apiCall, bitCapIntOcl* bciArgs, QEngineOCLPtr toCopy)
{
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
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 7, bciArgs, writeArgsEvent);

    bitCapIntOcl nMaxQPower = bciArgs[0];
    bitCapIntOcl nQubitCount = bciArgs[1] + toCopy->qubitCount;
    size_t nStateVecSize = nMaxQPower * sizeof(complex);
    maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    if (nStateVecSize > maxAlloc) {
        throw "Error: State vector exceeds device maximum OpenCL allocation";
    }

    SetQubitCount(nQubitCount);

    size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);
    bool forceAlloc = !stateVec && ((OclMemDenom * nStateVecSize) > maxMem);

    writeArgsEvent.wait();

    complex* nStateVec = AllocStateVec(maxQPowerOcl, forceAlloc);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    toCopy->Finish();

    WaitCall(apiCall, ngc, ngs, { stateBuffer, toCopy->stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    // toCopy->SetDevice(toCopyDevID);
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

    if (doNormalize) {
        NormalizeState();
    }
    if (destination && destination->doNormalize) {
        destination->NormalizeState();
    }

    // int destinationDevID = 0;
    if (destination) {
        // destinationDevID = destination->GetDeviceID();
        destination->SetDevice(deviceID);
    }

    if (length == qubitCount) {
        if (destination != NULL) {
            destination->ResetStateVec(stateVec);
            destination->stateBuffer = stateBuffer;
            stateVec = NULL;
            // destination->SetDevice(destinationDevID);
        }
        // This will be cleared by the destructor:
        ResetStateVec(AllocStateVec(2));
        stateBuffer = MakeStateVecBuffer(stateVec);
        SetQubitCount(1);
        return;
    }

    bitLenInt nLength = qubitCount - length;

    OCLAPI api_call = OCL_API_DECOMPOSEPROB;

    bitCapIntOcl partPower = pow2Ocl(length);
    bitCapIntOcl remainderPower = pow2Ocl(nLength);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // The "remainder" bits will always be maintained.
    BufferPtr probBuffer1 = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);
    BufferPtr angleBuffer1 = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(real1) * remainderPower);

    // The removed "part" is only necessary for Decompose.
    BufferPtr probBuffer2 = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(real1) * partPower);
    BufferPtr angleBuffer2 = std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(real1) * partPower);

    // Call the kernel that calculates bit probability and angle, retaining both parts.
    QueueCall(api_call, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1, probBuffer2, angleBuffer2 });

    SetQubitCount(nLength);

    // If we Decompose, calculate the state of the bit system removed.
    if (destination == NULL) {
        clFinish();
    } else {
        destination->Finish();

        bciArgs[0] = partPower;

        poolItem = GetFreePoolItem();
        EventVecPtr waitVec2 = ResetWaitEvents();
        DISPATCH_WRITE(waitVec2, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

        size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        size_t oNStateVecSize = maxQPowerOcl * sizeof(complex);

        WaitCall(OCL_API_DECOMPOSEAMP, ngc2, ngs2,
            { probBuffer2, angleBuffer2, poolItem->ulongBuffer, destination->stateBuffer });

        if (!(destination->useHostRam) && destination->stateVec && oNStateVecSize <= destination->maxAlloc &&
            (2 * oNStateVecSize) <= destination->maxMem) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cl::Event copyEvent;
            destination->queue.enqueueCopyBuffer(
                *(destination->stateBuffer), *nSB, 0, 0, sizeof(complex) * destination->maxQPowerOcl, NULL, &copyEvent);
            copyEvent.wait();
            wait_refs.clear();

            destination->stateBuffer = nSB;
            FreeAligned(destination->stateVec);
            destination->stateVec = NULL;
        }

        // destination->SetDevice(destinationDevID);
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPowerOcl;
    poolItem = GetFreePoolItem();
    EventVecPtr waitVec3 = ResetWaitEvents();
    DISPATCH_WRITE(waitVec3, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    ngs = FixGroupSize(ngc, nrmGroupSize);

    size_t nStateVecSize = maxQPowerOcl * sizeof(complex);

    clFinish();

    if (!useHostRam && stateVec && ((OclMemDenom * nStateVecSize) <= maxMem)) {
        FreeStateVec();
    }

    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    QueueCall(OCL_API_DECOMPOSEAMP, ngc, ngs, { probBuffer1, angleBuffer1, poolItem->ulongBuffer, stateBuffer });
}

void QEngineOCL::Decompose(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    DecomposeDispose(start, length, std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length) { DecomposeDispose(start, length, (QEngineOCLPtr)NULL); }

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length, bitCapInt disposedPerm)
{
    if (length == 0) {
        return;
    }

    if (length == qubitCount) {
        // This will be cleared by the destructor:
        ResetStateVec(AllocStateVec(2));
        stateBuffer = MakeStateVecBuffer(stateVec);
        SetQubitCount(1);
        return;
    }

    if (doNormalize) {
        NormalizeState();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    bitLenInt nLength = qubitCount - length;
    bitCapIntOcl remainderPower = pow2Ocl(nLength);
    bitCapIntOcl skipMask = pow2Ocl(start) - ONE_BCI;
    bitCapIntOcl disposedRes = (bitCapIntOcl)disposedPerm << (bitCapIntOcl)start;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { remainderPower, length, skipMask, disposedRes, 0, 0, 0, 0, 0, 0 };

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    SetQubitCount(nLength);

    size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    WaitCall(OCL_API_DISPOSE, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);
}

real1 QEngineOCL::Probx(OCLAPI api_call, bitCapIntOcl* bciArgs)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    bitCapIntOcl maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    if (oneChance > ONE_R1)
        oneChance = ONE_R1;

    return clampProb(oneChance);
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1 QEngineOCL::Prob(bitLenInt qubit)
{
    if (qubitCount == 1) {
        return ProbAll(1);
    }

    bitCapIntOcl qPower = pow2Ocl(qubit);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROB, bciArgs);
}

// Returns probability of permutation of the register
real1 QEngineOCL::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    if (start == 0 && qubitCount == length) {
        return ProbAll(permutation);
    }

    bitCapIntOcl perm = (bitCapIntOcl)permutation << (bitCapIntOcl)start;

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> length, perm, start, length, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROBREG, bciArgs);
}

void QEngineOCL::ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray)
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, start, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    BufferPtr probsBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBREGALL, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, probsBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();

    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, waitVec2.get());
    wait_refs.clear();
}

// Returns probability of permutation of the register
real1 QEngineOCL::ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (!stateBuffer) {
        return ZERO_R1;
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitCapIntOcl oldV;
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> length, (bitCapIntOcl)mask, (bitCapIntOcl)permutation, length,
        0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    bitCapIntOcl* skipPowers = new bitCapIntOcl[length];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    BufferPtr qPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, skipPowers);

    bitCapIntOcl maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASK, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer, qPowersBuffer },
        sizeof(real1) * ngs);

    real1 oneChance;
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    delete[] skipPowers;

    return clampProb(oneChance);
}

void QEngineOCL::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    if (doNormalize) {
        NormalizeState();
    }

    bitCapIntOcl v = (bitCapIntOcl)mask; // count the number of bits set in v
    bitCapIntOcl oldV;
    bitLenInt length;
    std::vector<bitCapIntOcl> powersVec;
    for (length = 0; v; length++) {
        oldV = v;
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

    v = (~(bitCapIntOcl)mask) & (maxQPowerOcl - ONE_BCI); // count the number of bits set in v
    bitCapIntOcl skipPower;
    bitLenInt skipLength = 0; // c accumulates the total bits set in v
    std::vector<bitCapIntOcl> skipPowersVec;
    for (skipLength = 0; v; skipLength++) {
        oldV = v;
        v &= v - ONE_BCI; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        skipPowersVec.push_back(skipPower);
    }

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, length, skipLength, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 4, bciArgs);

    BufferPtr probsBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    bitCapIntOcl* powers = new bitCapIntOcl[length];
    std::copy(powersVec.begin(), powersVec.end(), powers);

    BufferPtr qPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * length, powers);

    bitCapIntOcl* skipPowers = new bitCapIntOcl[skipLength];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    BufferPtr qSkipPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapIntOcl) * skipLength, skipPowers);

    size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_PROBMASKALL, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probsBuffer, qPowersBuffer, qSkipPowersBuffer });

    EventVecPtr waitVec2 = ResetWaitEvents();

    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, waitVec2.get());
    wait_refs.clear();

    delete[] powers;
    delete[] skipPowers;
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

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length)
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
void QEngineOCL::CINT(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt* controls, const bitLenInt controlLen)
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
    bitCapIntOcl* controlPowers = new bitCapIntOcl[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (regMask | controlMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> controlLen, regMask, otherMask, lengthPower, start, toMod,
        controlLen, controlMask, 0, 0 };

    CArithmeticCall(api_call, bciArgs, controlPowers, controlLen);

    delete[] controlPowers;
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INT(OCL_API_INC, (bitCapIntOcl)toAdd, start, length);
}

void QEngineOCL::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    CINT(OCL_API_CINC, (bitCapIntOcl)toAdd, inOutStart, length, controls, controlLen);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(
    OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, regMask, otherMask, lengthPower, carryMask, start,
        toMod, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Common driver method behing INCC and DECC
void QEngineOCL::INCDECC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTC(OCL_API_INCDECC, (bitCapIntOcl)toMod, inOutStart, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(
    OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
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
void QEngineOCL::INCS(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, (bitCapIntOcl)toAdd, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt overflowIndex, const bitLenInt carryIndex)
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, inOutMask, otherMask, lengthPower, overflowMask,
        carryMask, start, toMod, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length,
    const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
{
    INTSC(OCL_API_INCDECSC_1, (bitCapIntOcl)toAdd, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(
    OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapIntOcl carryMask = pow2Ocl(carryIndex);
    bitCapIntOcl lengthPower = pow2Ocl(length);
    bitCapIntOcl inOutMask = (lengthPower - ONE_BCI) << start;
    bitCapIntOcl otherMask = pow2MaskOcl(qubitCount) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, inOutMask, otherMask, lengthPower, carryMask, start,
        toMod, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTSC(OCL_API_INCDECSC_2, (bitCapIntOcl)toAdd, start, length, carryIndex);
}

/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
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
void QEngineOCL::INCBCD(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, (bitCapIntOcl)toAdd, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(
    OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapIntOcl nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> ONE_BCI, inOutMask, otherMask, carryMask, start, toMod,
        nibbleCount, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCDECBCDC(
    bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTBCDC(OCL_API_INCDECBCDC, (bitCapIntOcl)toAdd, start, length, carryIndex);
}

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
        throw "DIV by zero";
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

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)2U, pow2Ocl(inputBit1), pow2Ocl(inputBit2),
        pow2Ocl(carryInSumOut), pow2Ocl(carryOut), 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

/** Controlled multiplication by integer */
void QEngineOCL::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
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
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DIV(toDiv, inOutStart, carryStart, length);
        return;
    }

    if (toDiv == 0) {
        throw "DIV by zero";
    }

    if (toDiv == 1) {
        return;
    }

    CMULx(OCL_API_CDIV, (bitCapIntOcl)toDiv, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
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
    bitLenInt* controls, bitLenInt controlLen)
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
    bitLenInt* controls, bitLenInt controlLen)
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

    EventVecPtr waitVec = ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPowerOcl);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    ClearBuffer(nStateBuffer, 0, maxQPowerOcl, waitVec);

    PoolItemPtr poolItem = GetFreePoolItem();
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 10, bciArgs);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    if (controlBuffer) {
        WaitCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer, controlBuffer });
    } else {
        WaitCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nStateBuffer });
    }

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);
}

void QEngineOCL::MULx(
    OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt carryStart, const bitLenInt length)
{
    bitCapIntOcl lowMask = pow2MaskOcl(length);
    bitCapIntOcl inOutMask = lowMask << (bitCapIntOcl)inOutStart;
    bitCapIntOcl carryMask = lowMask << (bitCapIntOcl)carryStart;
    bitCapIntOcl skipMask = pow2MaskOcl(carryStart);
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)length, toMod, inOutMask, carryMask, otherMask,
        length, inOutStart, carryStart, skipMask, 0 };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::MULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, const bitLenInt inStart,
    const bitLenInt outStart, const bitLenInt length)
{
    if (toMod == 0) {
        return;
    }

    bitCapIntOcl lowMask = pow2MaskOcl(length);
    bitCapIntOcl inMask = lowMask << (bitCapIntOcl)inStart;
    bitCapIntOcl outMask = lowMask << (bitCapIntOcl)outStart;
    bitCapIntOcl skipMask = pow2MaskOcl(outStart);
    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inMask | outMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)length, toMod, inMask, outMask, otherMask,
        length, inStart, outStart, skipMask, modN };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::CMULx(OCLAPI api_call, bitCapIntOcl toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
    const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapIntOcl lowMask = pow2MaskOcl(length);
    bitCapIntOcl inOutMask = lowMask << inOutStart;
    bitCapIntOcl carryMask = lowMask << carryStart;

    bitCapIntOcl* skipPowers = new bitCapIntOcl[controlLen + length];
    bitCapIntOcl* controlPowers = new bitCapIntOcl[controlLen];
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapIntOcl otherMask = (maxQPowerOcl - ONE_BCI) ^ (inOutMask | carryMask | controlMask);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)(controlLen + length), toMod, controlLen,
        controlMask, inOutMask, carryMask, otherMask, length, inOutStart, carryStart };

    BufferPtr controlBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(bitCapIntOcl) * ((controlLen * 2) + length), skipPowers);

    xMULx(api_call, bciArgs, controlBuffer);

    delete[] skipPowers;
    delete[] controlPowers;
}

void QEngineOCL::CMULModx(OCLAPI api_call, bitCapIntOcl toMod, bitCapIntOcl modN, const bitLenInt inOutStart,
    const bitLenInt carryStart, const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapIntOcl lowMask = pow2MaskOcl(length);
    bitCapIntOcl inOutMask = lowMask << inOutStart;
    bitCapIntOcl carryMask = lowMask << carryStart;

    bitCapIntOcl* skipPowers = new bitCapIntOcl[controlLen + length];
    bitCapIntOcl* controlPowers = new bitCapIntOcl[controlLen];
    bitCapIntOcl controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = pow2Ocl(controls[i]);
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = pow2Ocl(carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, toMod, controlLen, controlMask, inOutMask, carryMask, modN,
        length, inOutStart, carryStart };

    BufferPtr controlBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(bitCapIntOcl) * ((controlLen * 2) + length), skipPowers);

    xMULx(api_call, bciArgs, controlBuffer);

    delete[] skipPowers;
    delete[] controlPowers;
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, unsigned char* values, bool resetValue)
{
    if (!stateBuffer) {
        return 0U;
    }

    if (resetValue) {
        SetReg(valueStart, valueLength, 0);
    }

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapIntOcl inputMask = bitRegMaskOcl(indexStart, indexLength);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)valueLength, indexStart, inputMask, valueStart,
        valueBytes, valueLength, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, pow2Ocl(indexLength) * valueBytes);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    return (bitCapInt)(average + (ONE_R1 / 2));
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapIntOcl QEngineOCL::OpIndexed(OCLAPI api_call, bitCapIntOcl carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
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
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)ONE_BCI, indexStart, inputMask, valueStart,
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
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDADC, 0, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    return OpIndexed(OCL_API_INDEXEDSBC, 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Set 8 bit register bits based on read from classical memory */
void QEngineOCL::Hash(bitLenInt start, bitLenInt length, unsigned char* values)
{
    bitLenInt bytes = (length + 7) / 8;
    bitCapIntOcl inputMask = bitRegMaskOcl(start, length);
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, start, inputMask, bytes, 0, 0, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_HASH, bciArgs, values, pow2Ocl(length) * bytes);
}

void QEngineOCL::PhaseFlipX(OCLAPI api_call, bitCapIntOcl* bciArgs)
{
    CHECK_ZERO_SKIP();

    // We don't actually have to wait, so this is empty:
    EventVecPtr waitVec;
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 5, bciArgs, writeArgsEvent);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer });
}

void QEngineOCL::PhaseFlip()
{
    // This gate has no physical consequence. We only enable it for "book-keeping," if the engine is not using global
    // phase offsets.
    if (!randGlobalPhase) {
        bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        PhaseFlipX(OCL_API_PHASEFLIP, bciArgs);
    }
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineOCL::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)length, pow2Ocl(start), length, 0, 0, 0, 0, 0,
        0, 0 };

    PhaseFlipX(OCL_API_ZEROPHASEFLIP, bciArgs);
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)ONE_BCI, bitRegMaskOcl(start, length),
        pow2Ocl(flagIndex), (bitCapIntOcl)greaterPerm, start, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineOCL::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl >> (bitCapIntOcl)ONE_BCI, bitRegMaskOcl(start, length),
        (bitCapIntOcl)greaterPerm, start, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(const complex* inputState)
{
    clDump();

    if (!stateBuffer) {
        ReinitBuffer();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPowerOcl, inputState, waitVec.get());

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

    complex amp[1];
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(
        *stateBuffer, CL_TRUE, sizeof(complex) * (bitCapIntOcl)fullRegister, sizeof(complex), amp, waitVec.get());
    wait_refs.clear();
    return amp[0];
}

void QEngineOCL::SetAmplitude(bitCapInt perm, complex amp)
{
    if (doNormalize) {
        NormalizeState();
    }

    runningNorm -= norm(GetAmplitude(perm));
    runningNorm += norm(amp);
    if (runningNorm <= min_norm) {
        ZeroAmplitudes();
        return;
    } else if (!stateBuffer) {
        ReinitBuffer();
        ClearBuffer(stateBuffer, 0, maxQPowerOcl, ResetWaitEvents());
    }

    // "permutationAmp" might be in use, so we clFinish(), first, to guarantee it is not.
    clFinish();
    permutationAmp = amp;

    EventVecPtr waitVec = ResetWaitEvents();
    device_context->LockWaitEvents();
    device_context->wait_events->emplace_back();
    queue.enqueueWriteBuffer(*stateBuffer, CL_FALSE, sizeof(complex) * (bitCapIntOcl)perm, sizeof(complex),
        &permutationAmp, waitVec.get(), &(device_context->wait_events->back()));
    device_context->UnlockWaitEvents();
    queue.flush();
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
    queue.flush();
    clFinish();
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineOCL::GetProbs(real1* outputProbs) { ProbRegAll(0, qubitCount, outputProbs); }

bool QEngineOCL::ApproxCompare(QEngineOCLPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }

    toCompare->Finish();

    bitCapIntOcl bciArgs[BCI_ARG_LEN] = { maxQPowerOcl, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs);

    QueueCall(OCL_API_APPROXCOMPARE, nrmGroupCount, nrmGroupSize,
        { stateBuffer, toCompare->stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * nrmGroupSize);

    real1 sumSqrErr = 0;
    WAIT_REAL1_SUM(*nrmBuffer, nrmGroupCount / nrmGroupSize, nrmArray, &sumSqrErr);

    return sumSqrErr < approxcompare_error;
}

QInterfacePtr QEngineOCL::Clone()
{
    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(qubitCount, 0, rand_generator, ONE_CMPLX, doNormalize,
        randGlobalPhase, useHostRam, deviceID, hardware_rand_generator != NULL, false, amplitudeFloor);

    copyPtr->Finish();
    copyPtr->runningNorm = runningNorm;

    EventVecPtr waitVec = ResetWaitEvents();
    DISPATCH_COPY(waitVec, *stateBuffer, *(copyPtr->stateBuffer), sizeof(complex) * maxQPowerOcl);
    Finish();

    return copyPtr;
}

void QEngineOCL::NormalizeState(real1 nrm, real1 norm_thresh)
{
    // We might have async execution of gates still happening.
    clFinish();

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

    real1 r1_args[2] = { norm_thresh, (real1)ONE_R1 / std::sqrt(nrm) };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1) * 2, r1_args, writeRealArgsEvent);

    bitCapIntOcl bciArgs[1] = { maxQPowerOcl };
    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), bciArgs, writeBCIArgsEvent);

    size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

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

    runningNorm = ONE_R1;
}

void QEngineOCL::UpdateRunningNorm(real1 norm_thresh)
{
    if (!stateBuffer) {
        return;
    }

    if (norm_thresh < ZERO_R1) {
        norm_thresh = amplitudeFloor;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    real1 r1_args[1] = { norm_thresh };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1), r1_args, writeRealArgsEvent);

    runningNorm = ONE_R1;

    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl), &maxQPowerOcl, writeBCIArgsEvent);

    size_t ngc = FixWorkItemCount(maxQPowerOcl, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    QueueCall(OCL_API_UPDATENORM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    if (runningNorm <= min_norm) {
        ZeroAmplitudes();
    }
}

complex* QEngineOCL::AllocStateVec(bitCapInt elemCount, bool doForceAlloc)
{
    // If we're not using host ram, there's no reason to allocate.
    if (!doForceAlloc && !stateVec) {
        return NULL;
    }

    // elemCount is always a power of two, but might be smaller than QRACK_ALIGN_SIZE
#if defined(__APPLE__)
    void* toRet;
    posix_memalign(&toRet, QRACK_ALIGN_SIZE,
        ((sizeof(complex) * (bitCapIntOcl)elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE
                                                                         : sizeof(complex) * (bitCapIntOcl)elemCount);
    return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (complex*)_aligned_malloc(((sizeof(complex) * (bitCapIntOcl)elemCount) < QRACK_ALIGN_SIZE)
            ? QRACK_ALIGN_SIZE
            : sizeof(complex) * (bitCapIntOcl)elemCount,
        QRACK_ALIGN_SIZE);
#else
    return (complex*)aligned_alloc(QRACK_ALIGN_SIZE,
        ((sizeof(complex) * (bitCapIntOcl)elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE
                                                                         : sizeof(complex) * (bitCapIntOcl)elemCount);
#endif
}

BufferPtr QEngineOCL::MakeStateVecBuffer(complex* nStateVec)
{
    if (nStateVec) {
        return std::make_shared<cl::Buffer>(
            context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl, nStateVec);
    } else {
        return std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(complex) * maxQPowerOcl);
    }
}

void QEngineOCL::ReinitBuffer()
{
    ResetStateVec(AllocStateVec(maxQPower, usingHostRam));
    ResetStateBuffer(MakeStateVecBuffer(stateVec));
}

void QEngineOCL::ClearBuffer(BufferPtr buff, bitCapIntOcl offset, bitCapIntOcl size, EventVecPtr waitVec)
{
    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapIntOcl bciArgs[2] = { size, offset };
    cl::Event writeArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapIntOcl) * 2, bciArgs, writeArgsEvent);

    size_t ngc = FixWorkItemCount(size, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();

    QueueCall(OCL_API_CLEARBUFFER, ngc, ngs, { buff, poolItem->ulongBuffer });
}

} // namespace Qrack
