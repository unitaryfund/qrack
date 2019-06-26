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

// These are commonly used emplace patterns, for OpenCL buffer I/O.
#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)                                                       \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &clEvent);                                 \
    queue.flush();

#define DISPATCH_LOC_WRITE(buff, size, array, clEvent)                                                                 \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, NULL, &clEvent);                                          \
    queue.flush();

#define DISPATCH_WRITE(waitVec, buff, size, array)                                                                     \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &(device_context->wait_events->back()));   \
    queue.flush()

#define DISPATCH_READ(waitVec, buff, size, array)                                                                      \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueReadBuffer(buff, CL_FALSE, 0, size, array, waitVec.get(), &(device_context->wait_events->back()));    \
    queue.flush()

#define DISPATCH_FILL(waitVec, buff, size, value)                                                                      \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueFillBuffer(buff, value, 0, size, waitVec.get(), &(device_context->wait_events->back()));              \
    queue.flush()

#define WAIT_COPY(buff1, buff2, size)                                                                                  \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueCopyBuffer(buff1, buff2, 0, 0, size, NULL, &(device_context->wait_events->back()));                   \
    device_context->wait_events->back().wait();                                                                        \
    device_context->wait_events->pop_back()

#define WAIT_REAL1_SUM(buff, size, array, sumPtr)                                                                      \
    clFinish();                                                                                                        \
    queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * (size));                                     \
    *(sumPtr) = ParSum(array, size);                                                                                   \
    device_context->wait_events->emplace_back();                                                                       \
    queue.enqueueUnmapMemObject(buff, array, NULL, &(device_context->wait_events->back()));

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int devID, bool useHardwareRNG)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG)
    , deviceID(devID)
    , wait_refs()
    , nrmArray(NULL)
    , unlockHostMem(false)
{
    InitOCL(devID);
    SetPermutation(initState, phaseFac);
}

void QEngineOCL::LockSync(cl_int flags)
{
    clFinish();

    if (stateVec) {
        unlockHostMem = true;
    } else {
        unlockHostMem = false;
        stateVec = AllocStateVec(maxQPower, true);
        BufferPtr nStateBuffer = MakeStateVecBuffer(stateVec);
        WAIT_COPY(*stateBuffer, *nStateBuffer, sizeof(complex) * maxQPower);
        stateBuffer = nStateBuffer;
    }

    queue.enqueueMapBuffer(*stateBuffer, CL_TRUE, flags, 0, sizeof(complex) * maxQPower, NULL);
}

void QEngineOCL::UnlockSync()
{
    EventVecPtr waitVec = ResetWaitEvents();
    cl::Event unmapEvent;
    queue.enqueueUnmapMemObject(*stateBuffer, stateVec, waitVec.get(), &unmapEvent);
    unmapEvent.wait();
    wait_refs.clear();

    if (!unlockHostMem) {
        BufferPtr nStateBuffer = MakeStateVecBuffer(NULL);
        WAIT_COPY(*stateBuffer, *nStateBuffer, sizeof(complex) * maxQPower);

        stateBuffer = nStateBuffer;
        FreeStateVec();
        stateVec = NULL;
    }
}

void QEngineOCL::clFinish(bool doHard)
{
    if (device_context == NULL) {
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

size_t QEngineOCL::FixWorkItemCount(size_t maxI, size_t wic)
{
    if (wic > maxI) {
        // Guaranteed to be a power of two
        wic = maxI;
    } else {
        // Otherwise, clamp to a power of two
        size_t power = 2;
        while (power < wic) {
            power <<= 1U;
        }
        if (power > wic) {
            power >>= 1U;
        }
        wic = power;
    }
    return wic;
}

size_t QEngineOCL::FixGroupSize(size_t wic, size_t gs)
{
    if (gs > (wic / procElemCount)) {
        gs = (wic / procElemCount);
        if (gs == 0) {
            gs = 1;
        }
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

    if (wait_queue_items.size() == poolItems.size()) {
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

CL_CALLBACK void _PopQueue(cl_event event, cl_int type, void* user_data)
{
    ((QEngineOCL*)user_data)->PopQueue(event, type);
}

void QEngineOCL::PopQueue(cl_event event, cl_int type)
{
    queue_mutex.lock();
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
        ocl.call.setArg(args.size(), cl::Local(item.localBuffSize));
    }

    // Dispatch the primary kernel, to apply the gate.
    cl::Event kernelEvent;
    kernelEvent.setCallback(CL_COMPLETE, _PopQueue, this);
    EventVecPtr kernelWaitVec = ResetWaitEvents(false);
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(item.workItemCount), // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(), // vector of events to wait for
        &kernelEvent); // handle to wait for the kernel

    queue.flush();

    device_context->wait_events->push_back(kernelEvent);
}

void QEngineOCL::CopyState(QInterfacePtr orig)
{
    QEngineOCLPtr src = std::dynamic_pointer_cast<QEngineOCL>(orig);

    /* Set the size and reset the stateVec to the correct size. */
    SetQubitCount(orig->GetQubitCount());

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);
    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    src->LockSync(CL_MAP_READ);
    LockSync(CL_MAP_WRITE);
    runningNorm = src->runningNorm;
    std::copy(src->stateVec, src->stateVec + (1 << (src->qubitCount)), stateVec);
    src->UnlockSync();
    UnlockSync();
}

real1 QEngineOCL::ProbAll(bitCapInt fullRegister)
{
    if (doNormalize) {
        NormalizeState();
    }

    complex amp[1];
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, sizeof(complex) * fullRegister, sizeof(complex), amp, waitVec.get());
    wait_refs.clear();
    return norm(amp[0]);
}

void QEngineOCL::SetDevice(const int& dID, const bool& forceReInit)
{
    bool didInit = (nrmArray != NULL);

    complex* nStateVec = NULL;

    if (didInit) {
        // If we're "switching" to the device we already have, don't reinitialize.
        if ((!forceReInit) && (dID == deviceID)) {
            return;
        }

        // In this branch, the QEngineOCL was previously allocated, and now we need to copy its memory to a buffer
        // that's accessible in a new device. (The old buffer is definitely not accessible to the new device.)
        nStateVec = AllocStateVec(maxQPower, true);
        LockSync(CL_MAP_READ);
        std::copy(stateVec, stateVec + maxQPower, nStateVec);
        UnlockSync();

        // We're about to switch to a new device, so finish the queue, first.
        clFinish(true);
    }

    int oldDeviceID = deviceID;
    device_context = OCLEngine::Instance()->GetDeviceContextPtr(dID);
    deviceID = device_context->context_id;
    context = device_context->context;
    cl::CommandQueue oldQueue = queue;
    queue = device_context->queue;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPLY2X2_NORM_SINGLE);
    clFinish(true);

    bitCapInt oldNrmGroupCount = nrmGroupCount;
    nrmGroupSize = ocl.call.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device_context->device);
    procElemCount = device_context->device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    // If the user wants to not use general host RAM, but we can't allocate enough on the device, fall back to host RAM
    // anyway.
    maxMem = device_context->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    maxAlloc = device_context->device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    size_t stateVecSize = maxQPower * sizeof(complex);
    bool usingHostRam;
    // Device RAM should be large enough for 2 times the size of the stateVec, plus some excess.
    if (useHostRam && !(stateVecSize > maxAlloc || (3 * stateVecSize) > maxMem)) {
        usingHostRam = true;
    } else {
        usingHostRam = false;
    }

    // constrain to a power of two
    size_t procElemPow = 1;
    while (procElemPow < procElemCount) {
        procElemPow <<= 1U;
    }
    procElemCount = procElemPow;
    maxWorkItems = device_context->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
    nrmGroupCount = maxWorkItems;
    size_t nrmGroupPow = 2;
    while (nrmGroupPow <= nrmGroupCount) {
        nrmGroupPow <<= 1U;
    }
    nrmGroupCount = nrmGroupPow >> 1U;
    if (nrmGroupSize > (nrmGroupCount / procElemCount)) {
        nrmGroupSize = (nrmGroupCount / procElemCount);
        if (nrmGroupSize == 0) {
            nrmGroupSize = 1;
        }
    }
    size_t frac = nrmGroupCount / nrmGroupSize;
    while ((frac * nrmGroupSize) != nrmGroupCount) {
        nrmGroupSize++;
        frac = nrmGroupCount / nrmGroupSize;
    }

    size_t nrmVecAlignSize =
        ((sizeof(real1) * nrmGroupCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : (sizeof(real1) * nrmGroupCount);

    if (didInit && (nrmGroupCount != oldNrmGroupCount)) {
        nrmBuffer = NULL;
        FreeAligned(nrmArray);
        nrmArray = NULL;
    }

    if (!didInit || (nrmGroupCount != oldNrmGroupCount)) {
#if defined(__APPLE__)
        posix_memalign((void**)&nrmArray, QRACK_ALIGN_SIZE, nrmVecAlignSize);
#elif defined(_WIN32) && !defined(__CYGWIN__)
        nrmArray = (real1*)_aligned_malloc(nrmVecAlignSize, QRACK_ALIGN_SIZE);
#else
        nrmArray = (real1*)aligned_alloc(QRACK_ALIGN_SIZE, nrmVecAlignSize);
#endif
    }

    // create buffers on device (allocate space on GPU)
    if (didInit) {
        // In this branch, the QEngineOCL was previously allocated, and now we need to copy its memory to a buffer
        // that's accessible in a new device. (The old buffer is definitely not accessible to the new device.)
        if (!stateVec) {
            // We did not have host allocation, so we copied from device-local memory to host memory, above.
            // Now, we copy to the new device's memory.
            stateBuffer = MakeStateVecBuffer(NULL);
            queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(complex) * maxQPower, nStateVec);
            FreeAligned(nStateVec);
        } else {
            // We had host allocation; we will continue to have it.
            ResetStateVec(nStateVec);
            ResetStateBuffer(MakeStateVecBuffer(nStateVec));
        }
    } else {
        // In this branch, the QEngineOCL is first being initialized, and no data needs to be copied between device
        // contexts.
        stateVec = AllocStateVec(maxQPower, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);
    }

    poolItems.clear();
    poolItems.push_back(std::make_shared<PoolItem>(context));
    powersBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * sizeof(bitCapInt) * 16);

    if ((!didInit) || (oldDeviceID != deviceID) || (nrmGroupCount != oldNrmGroupCount)) {
        nrmBuffer =
            std::make_shared<cl::Buffer>(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, nrmVecAlignSize, nrmArray);
        EventVecPtr waitVec = ResetWaitEvents();
        // GPUs can't always tolerate uninitialized host memory, even if they're not reading from it
        DISPATCH_FILL(waitVec, *nrmBuffer, sizeof(real1) * nrmGroupCount, ZERO_R1);
    }
}

real1 QEngineOCL::ParSum(real1* toSum, bitCapInt maxI)
{
    // This interface is potentially parallelizable, but, for now, better performance is probably given by implementing
    // it as a serial loop.
    real1 totNorm = 0;
    for (bitCapInt i = 0; i < maxI; i++) {
        totNorm += toSum[i];
    }
    return totNorm;
}

void QEngineOCL::InitOCL(int devID) { SetDevice(devID, true); }

void QEngineOCL::ResetStateBuffer(BufferPtr nStateBuffer) { stateBuffer = nStateBuffer; }

void QEngineOCL::SetPermutation(bitCapInt perm, complex phaseFac)
{
    EventVecPtr waitVec = ResetWaitEvents();

    cl::Event fillEvent1;
    queue.enqueueFillBuffer(
        *stateBuffer, complex(ZERO_R1, ZERO_R1), 0, sizeof(complex) * maxQPower, waitVec.get(), &fillEvent1);
    queue.flush();

    complex amp;
    if (phaseFac == complex(-999.0, -999.0)) {
        amp = GetNonunitaryPhase();
    } else {
        amp = phaseFac;
    }

    fillEvent1.wait();
    wait_refs.clear();

    device_context->wait_events->emplace_back();
    queue.enqueueFillBuffer(
        *stateBuffer, amp, sizeof(complex) * perm, sizeof(complex), NULL, &(device_context->wait_events->back()));
    queue.flush();

    runningNorm = ONE_R1;
}

void QEngineOCL::ArithmeticCall(
    OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], unsigned char* values, bitCapInt valuesPower)
{
    CArithmeticCall(api_call, bciArgs, NULL, 0, values, valuesPower);
}

void QEngineOCL::CArithmeticCall(OCLAPI api_call, bitCapInt (&bciArgs)[BCI_ARG_LEN], bitCapInt* controlPowers,
    const bitLenInt controlLen, unsigned char* values, bitCapInt valuesPower)
{
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen > 0) {
        controlBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * controlLen, controlPowers);
    }

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen > 0) {
        device_context->wait_events->emplace_back();
        queue.enqueueCopyBuffer(*stateBuffer, *nStateBuffer, 0, 0, sizeof(complex) * maxQPower, waitVec.get(),
            &(device_context->wait_events->back()));
        queue.flush();
    } else {
        DISPATCH_FILL(waitVec, *nStateBuffer, sizeof(complex) * maxQPower, complex(ZERO_R1, ZERO_R1));
    }

    bitCapInt maxI = bciArgs[0];
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
    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], pauliX, 1, qPowers, false, SPECIAL_2X2::PAULIX);
}

/// Apply Pauli Z matrix to bit
void QEngineOCL::Z(bitLenInt qubit)
{
    // if (qubit >= qubitCount)
    //     throw std::invalid_argument("operation on bit index greater than total bits.");
    const complex pauliZ[4] = { complex(ONE_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1), complex(ZERO_R1, ZERO_R1),
        complex(-ONE_R1, ZERO_R1) };
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], pauliZ, 1, qPowers, false, SPECIAL_2X2::PAULIZ);
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm, SPECIAL_2X2 special)
{
    bool isXGate = (special == SPECIAL_2X2::PAULIX) && (!doNormalize || (runningNorm == ONE_R1));
    bool isZGate = (special == SPECIAL_2X2::PAULIZ) && (!doNormalize || (runningNorm == ONE_R1));

    // Are we going to calculate the normalization factor, on the fly? We can't, if this call doesn't iterate through
    // every single permutation amplitude.
    doCalcNorm &= doNormalize && (!isXGate) && (!isZGate) && (bitCount == 1);

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec;
    if (doCalcNorm) {
        waitVec = ResetWaitEvents();
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapInt maxI = maxQPower >> bitCount;
    bitCapInt bciArgs[5] = { offset2, offset1, maxI, bitCount, 0 };

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
            bciArgs[2] = qPowersSorted[0] - 1;
        } else {
            bciArgsSize = 4;
            bciArgs[3] = qPowersSorted[0] - 1;
        }
    } else if (bitCount == 2) {
        // Double bit gates include both controlled and swap gates. To reuse the code for both cases, we need two offset
        // arguments. Hence, we cannot easily overwrite either of the bit offset arguments.
        bciArgsSize = 5;
        bciArgs[3] = qPowersSorted[0] - 1;
        bciArgs[4] = qPowersSorted[1] - 1;
    }
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * bciArgsSize, bciArgs, writeArgsEvent);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    bool isUnitLength = (runningNorm == ONE_R1) || !(doNormalize && (bitCount == 1));
    cmplx[4] = complex(isUnitLength ? ONE_R1 : (ONE_R1 / std::sqrt(runningNorm)), ZERO_R1);

    BufferPtr locCmplxBuffer;
    cl::Event writeGateEvent;
    if (!isXGate && !isZGate) {
        DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex) * 5, cmplx, writeGateEvent);
    }

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    BufferPtr locPowersBuffer;
    cl::Event writeControlsEvent;
    if (bitCount > 2) {
        if (doCalcNorm) {
            locPowersBuffer = powersBuffer;
        } else {
            locPowersBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * bitCount);
        }
        DISPATCH_TEMP_WRITE(waitVec, *locPowersBuffer, sizeof(bitCapInt) * bitCount, qPowersSorted, writeControlsEvent);
    }

    // We load the appropriate kernel, that does/doesn't CALCULATE the norm, and does/doesn't APPLY the norm.
    unsigned char kernelMask = APPLY2X2_DEFAULT;
    if (bitCount == 1) {
        kernelMask |= APPLY2X2_SINGLE;
        if (isXGate) {
            kernelMask |= APPLY2X2_X;
        } else if (isZGate) {
            kernelMask |= APPLY2X2_Z;
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
    } else if ((bitCount == 1) && (!isXGate) && (!isZGate)) {
        runningNorm = ONE_R1;
    }
}

void QEngineOCL::UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
    bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
    const bitCapInt& mtrxSkipValueMask)
{
    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(&(mtrxs[mtrxSkipValueMask * 4U]), true, qubitIndex);
        return;
    }

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapInt maxI = maxQPower >> 1;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxI, (bitCapInt)(1 << qubitIndex), controlLen, mtrxSkipLen, mtrxSkipValueMask,
        0, 0, 0, 0, 0 };
    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 5, bciArgs);

    BufferPtr nrmInBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(real1));
    real1 nrm = (real1)(ONE_R1 / std::sqrt(runningNorm));
    DISPATCH_WRITE(waitVec, *nrmInBuffer, sizeof(real1), &nrm);

    BufferPtr uniformBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_READ_ONLY, sizeof(complex) * 4U * (1U << (controlLen + mtrxSkipLen)));

    DISPATCH_WRITE(waitVec, *uniformBuffer, sizeof(complex) * 4 * (1U << (controlLen + mtrxSkipLen)), mtrxs);

    bitCapInt* qPowers = new bitCapInt[controlLen + mtrxSkipLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = 1U << controls[i];
    }
    for (bitLenInt i = 0; i < mtrxSkipLen; i++) {
        qPowers[controlLen + i] = mtrxSkipPowers[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    DISPATCH_WRITE(waitVec, *powersBuffer, sizeof(bitCapInt) * (controlLen + mtrxSkipLen), qPowers);

    // We call the kernel, with global buffers and one local buffer.
    WaitCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer },
        sizeof(real1) * ngs);

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant.
    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);

    delete[] qPowers;
}

void QEngineOCL::ApplyMx(OCLAPI api_call, bitCapInt* bciArgs, complex nrm)
{
    // We don't actually have to wait, so this is empty:
    EventVecPtr waitVec;
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent, writeNormEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 3, bciArgs, writeArgsEvent);
    BufferPtr locCmplxBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(complex));
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->cmplxBuffer), sizeof(complex), &nrm, writeNormEvent);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeNormEvent.wait();

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->cmplxBuffer });
}

void QEngineOCL::ApplyM(bitCapInt qPower, bool result, complex nrm)
{
    bitCapInt powerTest = result ? qPower : 0;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, qPower, powerTest, 0, 0, 0, 0, 0, 0, 0 };

    ApplyMx(OCL_API_APPLYM, bciArgs, nrm);
}

void QEngineOCL::ApplyM(bitCapInt mask, bitCapInt result, complex nrm)
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, mask, result, 0, 0, 0, 0, 0, 0, 0 };

    ApplyMx(OCL_API_APPLYMREG, bciArgs, nrm);
}

void QEngineOCL::Compose(OCLAPI apiCall, bitCapInt* bciArgs, QEngineOCLPtr toCopy)
{
    if (doNormalize) {
        NormalizeState();
    }

    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }

    toCopy->Finish();

    PoolItemPtr poolItem = GetFreePoolItem();

    bitCapInt nMaxQPower = bciArgs[0];
    bitCapInt nQubitCount = bciArgs[1] + toCopy->qubitCount;

    size_t nStateVecSize = nMaxQPower * sizeof(complex);
    if (!stateVec && (nStateVecSize > maxAlloc || (2 * nStateVecSize) > maxMem)) {
        complex* nSV = AllocStateVec(maxQPower, true);
        BufferPtr nSB = MakeStateVecBuffer(nSV);

        WAIT_COPY(*stateBuffer, *nSB, sizeof(complex) * maxQPower);

        stateVec = nSV;
        stateBuffer = nSB;
    }

    EventVecPtr waitVec = ResetWaitEvents();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 7, bciArgs);

    SetQubitCount(nQubitCount);

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    OCLDeviceCall ocl = device_context->Reserve(apiCall);

    BufferPtr otherStateBuffer;
    complex* otherStateVec;
    if (toCopy->deviceID == deviceID) {
        otherStateVec = toCopy->stateVec;
        otherStateBuffer = toCopy->stateBuffer;
    } else {
        toCopy->LockSync(CL_MAP_READ);
        otherStateVec = toCopy->stateVec;
        otherStateBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(complex) * toCopy->maxQPower, otherStateVec);
    }

    runningNorm = ONE_R1;

    WaitCall(apiCall, ngc, ngs, { stateBuffer, otherStateBuffer, poolItem->ulongBuffer, nStateBuffer });

    if (toCopy->deviceID != deviceID) {
        toCopy->UnlockSync();
    }

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy)
{
    bitLenInt result = qubitCount;

    bitCapInt oQubitCount = toCopy->qubitCount;
    bitCapInt nQubitCount = qubitCount + oQubitCount;
    bitCapInt nMaxQPower = 1U << nQubitCount;
    bitCapInt startMask = maxQPower - 1U;
    bitCapInt endMask = (toCopy->maxQPower - 1U) << qubitCount;
    bitCapInt bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, startMask, endMask, 0, 0, 0, 0, 0, 0 };

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
    bitCapInt nMaxQPower = 1U << nQubitCount;
    bitCapInt startMask = (1U << start) - 1U;
    bitCapInt midMask = bitRegMask(start, oQubitCount);
    bitCapInt endMask = ((1U << (qubitCount + oQubitCount)) - 1U) & ~(startMask | midMask);
    bitCapInt bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, oQubitCount, startMask, midMask, endMask, start, 0, 0,
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

    OCLAPI api_call = OCL_API_DECOMPOSEPROB;

    if (doNormalize) {
        NormalizeState();
    }

    bitCapInt partPower = 1U << length;
    bitCapInt remainderPower = 1U << (qubitCount - length);
    bitCapInt bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 4, bciArgs);

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // The "remainder" bits will always be maintained.
    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower]();
    BufferPtr probBuffer1 = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateProb);
    BufferPtr angleBuffer1 = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateAngle);

    // The removed "part" is only necessary for Decompose.
    real1* partStateProb = new real1[partPower]();
    real1* partStateAngle = new real1[partPower]();
    BufferPtr probBuffer2 = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateProb);
    BufferPtr angleBuffer2 = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateAngle);

    // Call the kernel that calculates bit probability and angle, retaining both parts.
    QueueCall(api_call, ngc, ngs,
        { stateBuffer, poolItem->ulongBuffer, probBuffer1, angleBuffer1, probBuffer2, angleBuffer2 });

    EventVecPtr waitVec2 = ResetWaitEvents();

    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    // If we Decompose, calculate the state of the bit system removed.
    if (destination != nullptr) {
        Finish();
        destination->Finish();

        bciArgs[0] = partPower;

        EventVecPtr waitVec2 = ResetWaitEvents();
        DISPATCH_WRITE(waitVec2, *(poolItem->ulongBuffer), sizeof(bitCapInt), bciArgs);

        size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        BufferPtr otherStateBuffer;
        complex* otherStateVec;
        if (destination->deviceID == deviceID) {
            otherStateVec = destination->stateVec;
            otherStateBuffer = destination->stateBuffer;
        } else {
            otherStateVec = AllocStateVec(destination->maxQPower, true);
            otherStateBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                sizeof(complex) * destination->maxQPower, otherStateVec);

            DISPATCH_FILL(
                waitVec2, *otherStateBuffer, sizeof(complex) * destination->maxQPower, complex(ZERO_R1, ZERO_R1));
        }

        WaitCall(
            OCL_API_DECOMPOSEAMP, ngc2, ngs2, { probBuffer2, angleBuffer2, poolItem->ulongBuffer, otherStateBuffer });

        size_t oNStateVecSize = maxQPower * sizeof(complex);

        if (destination->deviceID != deviceID) {
            queue.enqueueMapBuffer(*otherStateBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * destination->maxQPower);
            destination->LockSync(CL_MAP_WRITE);
            std::copy(otherStateVec, otherStateVec + destination->maxQPower, destination->stateVec);
            cl::Event waitUnmap;
            queue.enqueueUnmapMemObject(*otherStateBuffer, otherStateVec, NULL, &waitUnmap);
            waitUnmap.wait();
            destination->UnlockSync();
            FreeAligned(otherStateVec);
        } else if (!(destination->useHostRam) && destination->stateVec && oNStateVecSize <= destination->maxAlloc &&
            (2 * oNStateVecSize) <= destination->maxMem) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cl::Event copyEvent;
            destination->queue.enqueueCopyBuffer(
                *(destination->stateBuffer), *nSB, 0, 0, sizeof(complex) * destination->maxQPower, NULL, &copyEvent);
            copyEvent.wait();
            wait_refs.clear();

            destination->stateBuffer = nSB;
            FreeAligned(destination->stateVec);
            destination->stateVec = NULL;
        }
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPower;
    EventVecPtr waitVec3 = ResetWaitEvents();
    DISPATCH_WRITE(waitVec3, *(poolItem->ulongBuffer), sizeof(bitCapInt), bciArgs);

    ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    ngs = FixGroupSize(ngc, nrmGroupSize);

    size_t nStateVecSize = maxQPower * sizeof(complex);
    if (!useHostRam && stateVec && nStateVecSize <= maxAlloc && (2 * nStateVecSize) <= maxMem) {
        clFinish();
        FreeStateVec();
    }

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    WaitCall(OCL_API_DECOMPOSEAMP, ngc, ngs, { probBuffer1, angleBuffer1, poolItem->ulongBuffer, nStateBuffer });

    ResetStateVec(nStateVec);
    ResetStateBuffer(nStateBuffer);

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
    delete[] partStateProb;
    delete[] partStateAngle;
}

void QEngineOCL::Decompose(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    DecomposeDispose(start, length, std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length)
{
    DecomposeDispose(start, length, (QEngineOCLPtr) nullptr);
}

real1 QEngineOCL::Probx(OCLAPI api_call, bitCapInt* bciArgs)
{
    if (doNormalize) {
        NormalizeState();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 4, bciArgs);

    bitCapInt maxI = bciArgs[0];
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

    bitCapInt qPower = 1U << qubit;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1U, qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROB, bciArgs);
}

// Returns probability of permutation of the register
real1 QEngineOCL::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
    if (start == 0 && qubitCount == length) {
        return ProbAll(permutation);
    }

    bitCapInt perm = permutation << start;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, perm, start, length, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROBREG, bciArgs);
}

void QEngineOCL::ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray)
{
    bitCapInt lengthPower = 1U << length;
    bitCapInt maxJ = maxQPower >> length;

    if (doNormalize) {
        NormalizeState();
    }

    if ((lengthPower * lengthPower) < nrmGroupCount) {
        // With "lengthPower" count of threads, compared to a redundancy of "lengthPower" with full utilization, this is
        // close to the point where it becomes more efficient to rely on iterating through ProbReg calls.
        QEngine::ProbRegAll(start, length, probsArray);
        return;
    }

    bitCapInt bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, start, length, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 4, bciArgs);

    BufferPtr probsBuffer =
        std::make_shared<cl::Buffer>(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

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
    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
        skipPowersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, mask, permutation, length, 0, 0, 0, 0, 0, 0 };

    if (doNormalize) {
        NormalizeState();
    }

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 4, bciArgs);

    bitCapInt* skipPowers = new bitCapInt[length];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    BufferPtr qPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * length, skipPowers);

    bitCapInt maxI = bciArgs[0];
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
    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length;
    std::vector<bitCapInt> powersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1U; // clear the least significant bit set
        powersVec.push_back((v ^ oldV) & oldV);
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt maxJ = maxQPower >> length;

    if (doNormalize) {
        NormalizeState();
    }

    if ((lengthPower * lengthPower) < nrmGroupCount) {
        // With "lengthPower" count of threads, compared to a redundancy of "lengthPower" with full utilization, this is
        // close to the point where it becomes more efficient to rely on iterating through ProbReg calls.
        QEngine::ProbMaskAll(mask, probsArray);
        return;
    }

    v = (~mask) & (maxQPower - 1U); // count the number of bits set in v
    bitCapInt skipPower;
    bitLenInt skipLength = 0; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (skipLength = 0; v; skipLength++) {
        oldV = v;
        v &= v - 1U; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        skipPowersVec.push_back(skipPower);
    }

    bitCapInt bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, length, skipLength, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 4, bciArgs);

    BufferPtr probsBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    bitCapInt* powers = new bitCapInt[length];
    std::copy(powersVec.begin(), powersVec.end(), powers);

    BufferPtr qPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * length, powers);

    bitCapInt* skipPowers = new bitCapInt[skipLength];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    BufferPtr qSkipPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * skipLength, skipPowers);

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

    bitCapInt lengthPower = 1U << length;
    bitCapInt regMask = (lengthPower - 1U) << start;
    bitCapInt otherMask = (maxQPower - 1U) & (~regMask);
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt regMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - 1U) & ~(regMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, toMod, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Add or Subtract integer (without sign or carry, with controls)
void QEngineOCL::CINT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt* controls, const bitLenInt controlLen)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt regMask = lengthMask << start;

    bitCapInt controlMask = 0;
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapInt otherMask = (maxQPower - 1U) ^ (regMask | controlMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> controlLen, regMask, otherMask, lengthPower, start, toMod,
        controlLen, controlMask, 0, 0 };

    CArithmeticCall(api_call, bciArgs, controlPowers, controlLen);

    delete[] controlPowers;
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INT(OCL_API_INC, toAdd, start, length);
}

void QEngineOCL::CINC(
    bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        INC(toAdd, inOutStart, length);
        return;
    }

    CINT(OCL_API_CINC, toAdd, inOutStart, length, controls, controlLen);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt regMask = (lengthPower - 1U) << start;
    bitCapInt otherMask = (maxQPower - 1U) & (~(regMask | carryMask));

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1U, regMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Common driver method behing INCC and DECC
void QEngineOCL::INCDECC(
    bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTC(OCL_API_INCDECC, toMod, inOutStart, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt regMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - 1U) ^ regMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, overflowMask, start, toMod, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCS(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, toAdd, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt overflowIndex, const bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt lengthPower = 1U << length;
    bitCapInt lengthMask = lengthPower - 1U;
    toMod &= lengthMask;
    if (toMod == 0) {
        return;
    }

    bitCapInt overflowMask = 1U << overflowIndex;
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt inOutMask = lengthMask << start;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1U, inOutMask, otherMask, lengthPower, overflowMask, carryMask,
        start, toMod, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length,
    const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
{
    INTSC(OCL_API_INCDECSC_1, toAdd, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt lengthPower = 1U << length;
    bitCapInt inOutMask = (lengthPower - 1U) << start;
    bitCapInt otherMask = ((1U << qubitCount) - 1U) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1U, inOutMask, otherMask, lengthPower, carryMask, start, toMod, 0,
        0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCDECSC(bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTSC(OCL_API_INCDECSC_2, toAdd, start, length, carryIndex);
}

/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    if (length == 0) {
        return;
    }

    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(start, length);
    bitCapInt otherMask = (maxQPower - 1U) ^ inOutMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, inOutMask, otherMask, start, toMod, nibbleCount, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineOCL::INCBCD(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, toAdd, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    if (length == 0) {
        return;
    }

    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }

    bitCapInt maxPow = intPow(10U, nibbleCount);
    toMod %= maxPow;
    if (toMod == 0) {
        return;
    }

    bitCapInt inOutMask = bitRegMask(start, length);
    bitCapInt carryMask = 1U << carryIndex;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1U, inOutMask, otherMask, carryMask, start, toMod, nibbleCount, 0,
        0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCDECBCDC(
    bitCapInt toAdd, const bitLenInt& start, const bitLenInt& length, const bitLenInt& carryIndex)
{
    INTBCDC(OCL_API_INCDECBCDC, toAdd, start, length, carryIndex);
}

/** Multiply by integer */
void QEngineOCL::MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    SetReg(carryStart, length, 0);

    bitCapInt lowPower = 1U << length;
    toMul %= lowPower;
    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }

    MULx(OCL_API_MUL, toMul, inOutStart, carryStart, length);
}

/** Divide by integer */
void QEngineOCL::DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
{
    if (toDiv == 0) {
        throw "DIV by zero";
    }

    MULx(OCL_API_DIV, toDiv, inOutStart, carryStart, length);
}

/** Multiplication modulo N by integer, (out of place) */
void QEngineOCL::MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    if (toMul == 0) {
        SetReg(outStart, length, 0);
        return;
    }

    MULModx(OCL_API_MULMODN_OUT, toMul, modN, inStart, outStart, length);
}

/** Raise a classical base to a quantum power, modulo N, (out of place) */
void QEngineOCL::POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
{
    MULModx(OCL_API_POWMODN_OUT, base, modN, inStart, outStart, length);
}

/** Controlled multiplication by integer */
void QEngineOCL::CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MUL(toMul, inOutStart, carryStart, length);
        return;
    }

    SetReg(carryStart, length, 0);

    bitCapInt lowPower = 1U << length;
    toMul %= lowPower;
    if (toMul == 0) {
        SetReg(inOutStart, length, 0);
        return;
    }

    if (toMul == 1) {
        return;
    }

    CMULx(OCL_API_CMUL, toMul, inOutStart, carryStart, length, controls, controlLen);
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

    CMULx(OCL_API_CDIV, toDiv, inOutStart, carryStart, length, controls, controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        MULModNOut(toMul, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0);

    bitCapInt lowPower = 1U << length;
    toMul %= lowPower;
    if (toMul == 0) {
        return;
    }

    CMULModx(OCL_API_CMULMODN_OUT, toMul, modN, inStart, outStart, length, controls, controlLen);
}

/** Controlled multiplication modulo N by integer, (out of place) */
void QEngineOCL::CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
    bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        POWModNOut(base, modN, inStart, outStart, length);
        return;
    }

    SetReg(outStart, length, 0);

    if (base == 0) {
        return;
    }

    CMULModx(OCL_API_CPOWMODN_OUT, base, modN, inStart, outStart, length, controls, controlLen);
}

void QEngineOCL::xMULx(OCLAPI api_call, bitCapInt* bciArgs, BufferPtr controlBuffer)
{
    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 10, bciArgs);
    DISPATCH_FILL(waitVec, *nStateBuffer, sizeof(complex) * maxQPower, complex(ZERO_R1, ZERO_R1));

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
    OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt carryStart, const bitLenInt length)
{
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;
    bitCapInt skipMask = (1U << carryStart) - 1U;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, toMod, inOutMask, carryMask, otherMask, length, inOutStart,
        carryStart, skipMask, 0 };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::MULModx(OCLAPI api_call, bitCapInt toMod, bitCapInt modN, const bitLenInt inStart,
    const bitLenInt outStart, const bitLenInt length)
{
    SetReg(outStart, length, 0);

    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inMask = lowMask << inStart;
    bitCapInt outMask = lowMask << outStart;
    bitCapInt skipMask = (1U << outStart) - 1U;
    bitCapInt otherMask = (maxQPower - 1U) ^ (inMask | outMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, toMod, inMask, outMask, otherMask, length, inStart,
        outStart, skipMask, modN };

    xMULx(api_call, bciArgs, NULL);
}

void QEngineOCL::CMULx(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
    const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt otherMask = (maxQPower - 1U) ^ (inOutMask | carryMask | controlMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> (controlLen + length), toMod, controlLen, controlMask, inOutMask,
        carryMask, otherMask, length, inOutStart, carryStart };

    BufferPtr controlBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * ((controlLen * 2) + length), skipPowers);

    xMULx(api_call, bciArgs, controlBuffer);

    delete[] skipPowers;
    delete[] controlPowers;
}

void QEngineOCL::CMULModx(OCLAPI api_call, bitCapInt toMod, bitCapInt modN, const bitLenInt inOutStart,
    const bitLenInt carryStart, const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0;
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0; i < length; i++) {
        skipPowers[i + controlLen] = 1U << (carryStart + i);
    }
    std::sort(skipPowers, skipPowers + controlLen + length);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, toMod, controlLen, controlMask, inOutMask, carryMask, modN, length,
        inOutStart, carryStart };

    BufferPtr controlBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * ((controlLen * 2) + length), skipPowers);

    xMULx(api_call, bciArgs, controlBuffer);

    delete[] skipPowers;
    delete[] controlPowers;
}

real1 QEngineOCL::GetExpectation(bitLenInt valueStart, bitLenInt valueLength)
{
    real1 average = ZERO_R1;
    real1 prob;
    real1 totProb = ZERO_R1;
    bitCapInt i, outputInt;
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    LockSync(CL_MAP_READ);
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(stateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    UnlockSync();
    if (totProb > ZERO_R1) {
        average /= totProb;
    }

    return average;
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    SetReg(valueStart, valueLength, 0);
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> valueLength, indexStart, inputMask, valueStart, valueBytes,
        valueLength, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, (1 << indexLength) * valueBytes);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    return (bitCapInt)(average + (ONE_R1 / 2));
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::OpIndexed(OCLAPI api_call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    bool carryRes = M(carryIndex);
    // The carry has to first to be measured for its input value.
    if (carryRes) {
        /*
         * If the carry is set, we flip the carry bit. We always initially
         * clear the carry after testing for carry in.
         */
        carryIn ^= 1U;
        X(carryIndex);
    }

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt lengthPower = 1 << valueLength;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = bitRegMask(indexStart, indexLength);
    bitCapInt outputMask = bitRegMask(valueStart, valueLength);
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, indexStart, inputMask, valueStart, outputMask, otherMask,
        carryIn, carryMask, lengthPower, valueBytes };

    ArithmeticCall(api_call, bciArgs, values, (1 << indexLength) * valueBytes);

    real1 average = ZERO_R1;
#if ENABLE_VM6502Q_DEBUG
    average = GetExpectation(valueStart, valueLength);
#endif

    // Return the expectation value.
    return (bitCapInt)(average + (ONE_R1 / 2));
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

void QEngineOCL::PhaseFlipX(OCLAPI api_call, bitCapInt* bciArgs)
{
    // We don't actually have to wait, so this is empty:
    EventVecPtr waitVec;
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt) * 5, bciArgs, writeArgsEvent);

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
        bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        PhaseFlipX(OCL_API_PHASEFLIP, bciArgs);
    }
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineOCL::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, (1U << start), length, 0, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_ZEROPHASEFLIP, bciArgs);
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, bitRegMask(start, length), 1U << flagIndex, greaterPerm, start,
        0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineOCL::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, bitRegMask(start, length), greaterPerm, start, 0, 0, 0, 0, 0,
        0 };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(const complex* inputState)
{
    LockSync(CL_MAP_WRITE);
    std::copy(inputState, inputState + maxQPower, stateVec);
    runningNorm = ONE_R1;
    UnlockSync();
}

complex QEngineOCL::GetAmplitude(bitCapInt fullRegister)
{
    if (doNormalize) {
        NormalizeState();
    }

    complex amp[1];
    EventVecPtr waitVec = ResetWaitEvents();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, sizeof(complex) * fullRegister, sizeof(complex), amp, waitVec.get());
    wait_refs.clear();
    return amp[0];
}

/// Get pure quantum state, in unsigned int permutation basis
void QEngineOCL::GetQuantumState(complex* outputState)
{
    if (doNormalize) {
        NormalizeState();
    }

    LockSync(CL_MAP_READ);
    std::copy(stateVec, stateVec + maxQPower, outputState);
    UnlockSync();
}

/// Get all probabilities, in unsigned int permutation basis
void QEngineOCL::GetProbs(real1* outputProbs)
{
    if (doNormalize && (runningNorm != ONE_R1)) {
        NormalizeState();
    }

    LockSync(CL_MAP_READ);
    std::transform(stateVec, stateVec + maxQPower, outputProbs, normHelper);
    UnlockSync();
}

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

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPROXCOMPARE);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    DISPATCH_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt), bciArgs);

    BufferPtr otherStateBuffer;
    complex* otherStateVec;
    if (toCompare->deviceID == deviceID) {
        otherStateVec = toCompare->stateVec;
        otherStateBuffer = toCompare->stateBuffer;
    } else {
        otherStateVec = AllocStateVec(toCompare->maxQPower, true);
        toCompare->LockSync(CL_MAP_READ);
        std::copy(toCompare->stateVec, toCompare->stateVec + toCompare->maxQPower, otherStateVec);
        toCompare->UnlockSync();
        otherStateBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * toCompare->maxQPower, otherStateVec);
    }

    QueueCall(OCL_API_APPROXCOMPARE, nrmGroupCount, nrmGroupSize,
        { stateBuffer, otherStateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * nrmGroupSize);

    real1 sumSqrErr;
    WAIT_REAL1_SUM(*nrmBuffer, nrmGroupCount / nrmGroupSize, nrmArray, &sumSqrErr);

    if (toCompare->deviceID != deviceID) {
        FreeAligned(otherStateVec);
    }

    return sumSqrErr < approxcompare_error;
}

QInterfacePtr QEngineOCL::Clone()
{
    clFinish();

    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(
        qubitCount, 0, rand_generator, complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam, deviceID);

    copyPtr->clFinish();

    copyPtr->runningNorm = runningNorm;

    LockSync(CL_MAP_READ);
    copyPtr->LockSync(CL_MAP_WRITE);
    std::copy(stateVec, stateVec + maxQPower, copyPtr->stateVec);
    UnlockSync();
    copyPtr->UnlockSync();

    return copyPtr;
}

void QEngineOCL::NormalizeState(real1 nrm)
{
    // We might have async execution of gates still happening.
    clFinish();

    if (nrm < ZERO_R1) {
        nrm = runningNorm;
    }
    if (nrm == ONE_R1) {
        return;
    }

    PoolItemPtr poolItem = GetFreePoolItem();

    real1 r1_args[2] = { min_norm, (real1)ONE_R1 / std::sqrt(nrm) };
    cl::Event writeRealArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->realBuffer), sizeof(real1) * REAL_ARG_LEN, r1_args, writeRealArgsEvent);

    bitCapInt bciArgs[1] = { maxQPower };
    cl::Event writeBCIArgsEvent;
    DISPATCH_LOC_WRITE(*(poolItem->ulongBuffer), sizeof(bitCapInt), bciArgs, writeBCIArgsEvent);

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Wait for buffer write from limited lifetime objects
    writeRealArgsEvent.wait();
    writeBCIArgsEvent.wait();
    wait_refs.clear();

    OCLAPI api_call;
    if (maxQPower == ngc) {
        api_call = OCL_API_NORMALIZE_WIDE;
    } else {
        api_call = OCL_API_NORMALIZE;
    }

    QueueCall(api_call, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, poolItem->realBuffer });

    runningNorm = ONE_R1;
}

void QEngineOCL::UpdateRunningNorm()
{
    OCLDeviceCall ocl = device_context->Reserve(OCL_API_UPDATENORM);

    runningNorm = ONE_R1;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    EventVecPtr waitVec = ResetWaitEvents();
    PoolItemPtr poolItem = GetFreePoolItem();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->ulongBuffer), sizeof(bitCapInt), bciArgs, writeArgsEvent);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    wait_refs.clear();

    size_t ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    QueueCall(OCL_API_UPDATENORM, ngc, ngs, { stateBuffer, poolItem->ulongBuffer, nrmBuffer }, sizeof(real1) * ngs);

    WAIT_REAL1_SUM(*nrmBuffer, ngc / ngs, nrmArray, &runningNorm);
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
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
    return (complex*)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return (complex*)_aligned_malloc(
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount,
        QRACK_ALIGN_SIZE);
#else
    return (complex*)aligned_alloc(QRACK_ALIGN_SIZE,
        ((sizeof(complex) * elemCount) < QRACK_ALIGN_SIZE) ? QRACK_ALIGN_SIZE : sizeof(complex) * elemCount);
#endif
}

BufferPtr QEngineOCL::MakeStateVecBuffer(complex* nStateVec)
{
    if (nStateVec) {
        return std::make_shared<cl::Buffer>(
            context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, nStateVec);
    } else {
        return std::make_shared<cl::Buffer>(context, CL_MEM_READ_WRITE, sizeof(complex) * maxQPower);
    }
}

} // namespace Qrack
