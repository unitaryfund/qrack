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

#include <memory>

#include "oclengine.hpp"
#include "qengine_opencl.hpp"
#include "qfactory.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

// These are commonly used emplace patterns, for OpenCL buffer I/O.
#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)                                                       \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec, &clEvent);                                       \
    queue.flush();                                                                                                     \
    device_context->wait_events.push_back(clEvent)

#define DISPATCH_WRITE(waitVec, buff, size, array)                                                                     \
    device_context->wait_events.emplace_back();                                                                        \
    queue.enqueueWriteBuffer(buff, CL_FALSE, 0, size, array, waitVec, &(device_context->wait_events.back()));          \
    queue.flush()

#define DISPATCH_READ(waitVec, buff, size, array)                                                                      \
    device_context->wait_events.emplace_back();                                                                        \
    queue.enqueueReadBuffer(buff, CL_FALSE, 0, size, array, waitVec, &(device_context->wait_events.back()));           \
    queue.flush()

#define DISPATCH_FILL(waitVec, buff, size, value)                                                                      \
    device_context->wait_events.emplace_back();                                                                        \
    queue.enqueueFillBuffer(buff, value, 0, size, waitVec, &(device_context->wait_events.back()));                     \
    queue.flush()

#define WAIT_COPY(buff1, buff2, size)                                                                                  \
    device_context->wait_events.emplace_back();                                                                        \
    queue.enqueueCopyBuffer(buff1, buff2, 0, 0, size, NULL, &(device_context->wait_events.back()));                    \
    device_context->wait_events.back().wait();                                                                         \
    device_context->wait_events.pop_back()

#define WAIT_REAL1_SUM(waitVec, buff, size, array, sumPtr)                                                             \
    queue.enqueueMapBuffer(buff, CL_TRUE, CL_MAP_READ, 0, sizeof(real1) * (size), waitVec);                            \
    *(sumPtr) = ParSum(array, size);                                                                                   \
    device_context->wait_events.emplace_back();                                                                        \
    queue.enqueueUnmapMemObject(buff, array, NULL, &(device_context->wait_events.back()))

QEngineOCL::QEngineOCL(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int devID)
    : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem)
    , stateVec(NULL)
    , deviceID(devID)
    , nrmArray(NULL)
    , unlockHostMem(false)
{
    if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");

    runningNorm = ONE_R1;
    SetQubitCount(qBitCount);

    InitOCL(devID);

    SetPermutation(initState, phaseFac);
}

QEngineOCL::QEngineOCL(QEngineOCLPtr toCopy)
    : QEngine(
          toCopy->qubitCount, toCopy->rand_generator, toCopy->doNormalize, toCopy->randGlobalPhase, toCopy->useHostRam)
    , stateVec(NULL)
    , deviceID(-1)
    , nrmArray(NULL)
    , unlockHostMem(false)
{
    CopyState(toCopy);

    InitOCL(toCopy->deviceID);
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
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();
    cl::Event unmapEvent;
    queue.enqueueUnmapMemObject(*stateBuffer, stateVec, &waitVec, &unmapEvent);
    unmapEvent.wait();

    if (!unlockHostMem) {
        BufferPtr nStateBuffer = MakeStateVecBuffer(NULL);

        WAIT_COPY(*stateBuffer, *nStateBuffer, sizeof(complex) * maxQPower);

        stateBuffer = nStateBuffer;
        free(stateVec);
        stateVec = NULL;
    }
}

void QEngineOCL::Sync()
{
    LockSync(CL_MAP_READ);
    UnlockSync();
}

void QEngineOCL::clFinish(bool doHard)
{
    if (device_context == NULL) {
        return;
    }

    if (doHard) {
        queue.finish();
    } else {
        for (unsigned int i = 0; i < (device_context->wait_events.size()); i++) {
            device_context->wait_events[i].wait();
        }
    }
    device_context->wait_events.clear();
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

cl::Event QEngineOCL::QueueCall(
    OCLAPI api_call, size_t workItemCount, size_t localGroupSize, std::vector<BufferPtr> args, size_t localBuffSize)
{
    // We have to reserve the kernel, because its argument hooks are unique. The same kernel therefore can't be used by
    // other QEngineOCL instances, until we're done queueing it.
    OCLDeviceCall ocl = device_context->Reserve(api_call);

    // Load the arguments.
    for (unsigned int i = 0; i < args.size(); i++) {
        ocl.call.setArg(i, *args[i]);
    }

    // For all of our kernels, if a local memory buffer is used, there is always only one, as the last argument.
    if (localBuffSize) {
        ocl.call.setArg(args.size(), cl::Local(localBuffSize));
    }
    
#if ENABLE_RASPBERRYPI
    clFinish();
#endif

    // Dispatch the primary kernel, to apply the gate.
    cl::Event kernelEvent;
    std::vector<cl::Event> kernelWaitVec = device_context->ResetWaitEvents();
    queue.enqueueNDRangeKernel(ocl.call, cl::NullRange, // kernel, offset
        cl::NDRange(workItemCount), // global number of work items
        cl::NDRange(localGroupSize), // local number (per group)
        &kernelWaitVec, // vector of events to wait for
        &kernelEvent); // handle to wait for the kernel

    queue.flush();
    
#if ENABLE_RASPBERRYPI
    clFinish();
#endif

    return kernelEvent;
}

void QEngineOCL::CopyState(QInterfacePtr orig)
{
    QEngineOCLPtr src = std::dynamic_pointer_cast<QEngineOCL>(orig);

    /* Set the size and reset the stateVec to the correct size. */
    SetQubitCount(orig->GetQubitCount());

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);
    ResetStateVec(nStateVec, nStateBuffer);

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
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, sizeof(complex) * fullRegister, sizeof(complex), amp, &waitVec);
    return norm(amp[0]);
}

void QEngineOCL::SetDevice(const int& dID, const bool& forceReInit)
{
    bool didInit = (nrmArray != NULL);

    if (didInit) {
        // If we're "switching" to the device we already have, don't reinitialize.
        if ((!forceReInit) && (dID == deviceID)) {
            return;
        }

        // Otherwise, we're about to switch to a new device, so finish the queue, first.
        clFinish(true);
    }

    int oldDeviceID = deviceID;
    device_context = OCLEngine::Instance()->GetDeviceContextPtr(dID);
    deviceID = device_context->context_id;
    context = device_context->context;
    cl::CommandQueue oldQueue = queue;
    queue = device_context->queue;

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPLY2X2_NORM);
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
    size_t procElemPow = 2;
    while (procElemPow < procElemCount) {
        procElemPow <<= 1U;
    }
    procElemCount = procElemPow;
    nrmGroupCount = procElemCount * 2 * nrmGroupSize;
    maxWorkItems = device_context->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>()[0];
    if (nrmGroupCount > maxWorkItems) {
        nrmGroupCount = maxWorkItems;
    }
    nrmGroupCount = FixWorkItemCount(nrmGroupCount, nrmGroupCount);
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
        ((sizeof(real1) * nrmGroupCount) < ALIGN_SIZE) ? ALIGN_SIZE : (sizeof(real1) * nrmGroupCount);

    if (!didInit) {
#ifdef __APPLE__
        posix_memalign((void**)&nrmArray, ALIGN_SIZE, nrmVecAlignSize);
#else
        nrmArray = (real1*)aligned_alloc(ALIGN_SIZE, nrmVecAlignSize);
#endif
    } else if ((oldDeviceID != deviceID) || (nrmGroupCount != oldNrmGroupCount)) {
        nrmBuffer = NULL;
        free(nrmArray);
        nrmArray = NULL;
#ifdef __APPLE__
        posix_memalign((void**)&nrmArray, ALIGN_SIZE, nrmVecAlignSize);
#else
        nrmArray = (real1*)aligned_alloc(ALIGN_SIZE, nrmVecAlignSize);
#endif
    }

    // create buffers on device (allocate space on GPU)
    if (didInit) {
        // In this branch, the QEngineOCL was previously allocated, and now we need to copy its memory to a buffer
        // that's accessible in a new device. (The old buffer is definitely not accessible to the new device.)

        if (!stateVec) {
            // We did not have host allocation, so we definitely have to copy device-local memory to host memory, then
            // to a new device.
            cl::CommandQueue nQueue = queue;
            queue = oldQueue;

            complex* nStateVec = AllocStateVec(maxQPower, true);
            BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

            WAIT_COPY(*stateBuffer, *nStateBuffer, sizeof(complex) * maxQPower);

            // Host RAM should now by synchronized.
            queue = nQueue;
            if (usingHostRam) {
                // If we're using host RAM from here out, just create the buffer from the array pointer, in the context
                // of the new device/queue.
                stateBuffer = MakeStateVecBuffer(nStateVec);
                stateVec = nStateVec;
            } else {
                // If we're not using host RAM from here, we need to copy into a device memory buffer.
                stateBuffer = MakeStateVecBuffer(NULL);
                queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, nStateVec);
                free(nStateVec);
            }
        } else if (usingHostRam) {
            // We had host allocation; we will continue to have it. Just make the array pointer a buffer in the new
            // context.
            stateBuffer = MakeStateVecBuffer(stateVec);
        } else {
            // We had host allocation; we will no longer have it. Just copy the array pointer into a buffer in the new
            // context.
            stateBuffer = MakeStateVecBuffer(NULL);
            queue.enqueueWriteBuffer(*stateBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, stateVec);
            free(stateVec);
            stateVec = NULL;
        }
    } else {
        // In this branch, the QEngineOCL is first being initialized, and no data needs to be copied between device
        // contexts.
        stateVec = AllocStateVec(maxQPower, usingHostRam);
        stateBuffer = MakeStateVecBuffer(stateVec);
    }

    cmplxBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(complex) * CMPLX_NORM_LEN);
    ulongBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * BCI_ARG_LEN);
    powersBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * sizeof(bitCapInt) * 8);

    if ((!didInit) || (oldDeviceID != deviceID) || (nrmGroupCount != oldNrmGroupCount)) {
        nrmBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * nrmGroupCount, nrmArray);
        // GPUs can't always tolerate uninitialized host memory, even if they're not reading from it
        DISPATCH_FILL(NULL, *nrmBuffer, sizeof(real1) * nrmGroupCount, ZERO_R1);
    }
}

void QEngineOCL::SetQubitCount(bitLenInt qb)
{
    qubitCount = qb;
    maxQPower = 1 << qubitCount;
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

void QEngineOCL::InitOCL(int devID) { SetDevice(devID); }

void QEngineOCL::ResetStateVec(complex* nStateVec, BufferPtr nStateBuffer)
{
    stateBuffer = nStateBuffer;
    if (stateVec) {
        free(stateVec);
        stateVec = nStateVec;
    }
}

void QEngineOCL::SetPermutation(bitCapInt perm, complex phaseFac)
{
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    cl::Event fillEvent1;
    queue.enqueueFillBuffer(
        *stateBuffer, complex(ZERO_R1, ZERO_R1), 0, sizeof(complex) * maxQPower, &waitVec, &fillEvent1);
    queue.flush();

    complex amp;
    if (phaseFac == complex(-999.0, -999.0)) {
        if (randGlobalPhase) {
            real1 angle = Rand() * 2.0 * PI_R1;
            amp = complex(cos(angle), sin(angle));
        } else {
            amp = complex(ONE_R1, ZERO_R1);
        }
    } else {
        amp = phaseFac;
    }

    fillEvent1.wait();

    cl::Event fillEvent2;
    queue.enqueueFillBuffer(*stateBuffer, amp, sizeof(complex) * perm, sizeof(complex), NULL, &fillEvent2);
    queue.flush();
    device_context->wait_events.push_back(fillEvent2);

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
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer;
    BufferPtr controlBuffer;
    if (controlLen > 0) {
        controlBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * controlLen, controlPowers);
    }

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    nStateBuffer = MakeStateVecBuffer(nStateVec);

    if (controlLen > 0) {
        device_context->wait_events.emplace_back();
        queue.enqueueCopyBuffer(*stateBuffer, *nStateBuffer, 0, 0, sizeof(complex) * maxQPower, &waitVec,
            &(device_context->wait_events.back()));
        queue.flush();
    } else {
        DISPATCH_FILL(&waitVec, *nStateBuffer, sizeof(complex) * maxQPower, complex(ZERO_R1, ZERO_R1));
    }

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    std::vector<BufferPtr> oclArgs = { stateBuffer, ulongBuffer, nStateBuffer };

    BufferPtr loadBuffer;
    if (values) {
        loadBuffer = std::make_shared<cl::Buffer>(
            context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        oclArgs.push_back(loadBuffer);
    }
    if (controlLen > 0) {
        oclArgs.push_back(controlBuffer);
    }

    QueueCall(api_call, ngc, ngs, oclArgs).wait();

    ResetStateVec(nStateVec, nStateBuffer);
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapInt maxI = maxQPower >> bitCount;
    bitCapInt bciArgs[BCI_ARG_LEN] = { bitCount, maxI, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs, writeArgsEvent);

    // Load the 2x2 complex matrix and the normalization factor into the complex arguments buffer.
    complex cmplx[CMPLX_NORM_LEN];
    std::copy(mtrx, mtrx + 4, cmplx);

    // Is the vector already normalized, or is this method not appropriate for on-the-fly normalization?
    bool isUnitLength = (runningNorm == ONE_R1) || !(doNormalize && (bitCount == 1));
    cmplx[4] = complex(isUnitLength ? ONE_R1 : (ONE_R1 / std::sqrt(runningNorm)), ZERO_R1);
    size_t cmplxSize = ((isUnitLength && !doCalcNorm) ? 4 : 5);

    cl::Event writeGateEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *cmplxBuffer, sizeof(complex) * cmplxSize, cmplx, writeGateEvent);

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Are we going to calculate the normalization factor, on the fly? We can't, if this call doesn't iterate through
    // every single permutation amplitude.
    doCalcNorm &= doNormalize && (bitCount == 1);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    cl::Event writeControlsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *powersBuffer, sizeof(bitCapInt) * bitCount, qPowersSorted, writeControlsEvent);

    // We load the appropriate kernel, that does/doesn't CALCULATE the norm, and does/doesn't APPLY the norm.
    OCLAPI api_call;
    if (doCalcNorm) {
        api_call = OCL_API_APPLY2X2_NORM;
    } else {
        if (isUnitLength) {
            api_call = OCL_API_APPLY2X2_UNIT;
        } else {
            api_call = OCL_API_APPLY2X2;
        }
    }

    if (doCalcNorm) {
        device_context->wait_events.push_back(QueueCall(api_call, ngc, ngs,
            { stateBuffer, cmplxBuffer, ulongBuffer, powersBuffer, nrmBuffer }, sizeof(real1) * ngs));
    } else {
        device_context->wait_events.push_back(
            QueueCall(api_call, ngc, ngs, { stateBuffer, cmplxBuffer, ulongBuffer, powersBuffer }));
    }

    if (doCalcNorm) {
        // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
        // values into a single normalization constant. We want to do this in a non-blocking, asynchronous way.

        // Until a norm calculation returns, don't change the stateVec length.
        runningNorm = ONE_R1;

        // This kernel is run on a single work group, but it lets us continue asynchronously.
        if (ngc / ngs > 1) {
            device_context->wait_events.push_back(
                QueueCall(OCL_API_NORMSUM, ngc / ngs, ngc / ngs, { nrmBuffer }, sizeof(real1) * ngc / ngs));
        }

        std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

        // Asynchronously, whenever the normalization result is ready, it will be placed in this QEngineOCL's
        // "runningNorm" variable.
        DISPATCH_READ(&waitVec2, *nrmBuffer, sizeof(real1), &runningNorm);
    }

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    writeGateEvent.wait();
    writeControlsEvent.wait();
}

void QEngineOCL::UniformlyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, bitLenInt qubitIndex, const complex* mtrxs)
{
    // If there are no controls, the base case should be the non-controlled single bit gate.
    if (controlLen == 0) {
        ApplySingleBit(mtrxs, true, qubitIndex);
        return;
    }

    // We grab the wait event queue. We will replace it with three new asynchronous events, to wait for.
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    // Arguments are concatenated into buffers by primitive type, such as integer or complex number.

    // Load the integer kernel arguments buffer.
    bitCapInt maxI = maxQPower >> 1;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxI, (bitCapInt)(1 << qubitIndex), controlLen, 0, 0, 0, 0, 0, 0, 0 };
    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 3, bciArgs, writeArgsEvent);

    BufferPtr nrmInBuffer = std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(real1));

    cl::Event writeNormEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *nrmInBuffer, sizeof(real1), &runningNorm, writeNormEvent);

    BufferPtr uniformBuffer =
        std::make_shared<cl::Buffer>(context, CL_MEM_READ_ONLY, sizeof(complex) * 4 * (1U << controlLen));

    cl::Event writeMatricesEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *uniformBuffer, sizeof(complex) * 4 * (1U << controlLen), mtrxs, writeMatricesEvent);

    bitCapInt* qPowers = new bitCapInt[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
    }

    // We have default OpenCL work item counts and group sizes, but we may need to use different values due to the total
    // amount of work in this method call instance.
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    // Load a buffer with the powers of 2 of each bit index involved in the operation.
    cl::Event writeControlsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *powersBuffer, sizeof(bitCapInt) * controlLen, qPowers, writeControlsEvent);

    // We call the kernel, with global buffers and one local buffer.
    device_context->wait_events.push_back(QueueCall(OCL_API_UNIFORMLYCONTROLLED, ngc, ngs,
        { stateBuffer, ulongBuffer, powersBuffer, uniformBuffer, nrmInBuffer, nrmBuffer }, sizeof(real1) * ngs));

    // If we have calculated the norm of the state vector in this call, we need to sum the buffer of partial norm
    // values into a single normalization constant. We want to do this in a non-blocking, asynchronous way.

    // Until a norm calculation returns, don't change the stateVec length.
    runningNorm = ONE_R1;

    // This kernel is run on a single work group, but it lets us continue asynchronously.
    if (ngc / ngs > 1) {
        device_context->wait_events.push_back(
            QueueCall(OCL_API_NORMSUM, ngc / ngs, ngc / ngs, { nrmBuffer }, sizeof(real1) * ngc / ngs));
    }

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    // Asynchronously, whenever the normalization result is ready, it will be placed in this QEngineOCL's
    // "runningNorm" variable.
    DISPATCH_READ(&waitVec2, *nrmBuffer, sizeof(real1), &runningNorm);

    // Wait for buffer write from limited lifetime objects
    writeNormEvent.wait();
    writeMatricesEvent.wait();
    writeArgsEvent.wait();
    writeControlsEvent.wait();

    delete[] qPowers;
}

void QEngineOCL::ApplyMx(OCLAPI api_call, bitCapInt* bciArgs, complex nrm)
{
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 3, bciArgs, writeArgsEvent);
    DISPATCH_WRITE(&waitVec, *cmplxBuffer, sizeof(complex), &nrm);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(QueueCall(api_call, ngc, ngs, { stateBuffer, ulongBuffer, cmplxBuffer }));

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
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

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 7, bciArgs);

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
        otherStateVec = toCopy->AllocStateVec(toCopy->maxQPower, true);
        toCopy->LockSync(CL_MAP_READ);
        std::copy(toCopy->stateVec, toCopy->stateVec + toCopy->maxQPower, otherStateVec);
        toCopy->UnlockSync();
        otherStateBuffer = toCopy->MakeStateVecBuffer(otherStateVec);
    }

    runningNorm = ONE_R1;

    QueueCall(apiCall, ngc, ngs, { stateBuffer, otherStateBuffer, ulongBuffer, nStateBuffer }).wait();

    if (toCopy->deviceID != deviceID) {
        free(otherStateVec);
    }

    ResetStateVec(nStateVec, nStateBuffer);
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy)
{
    bitLenInt result = qubitCount;

    bitCapInt oQubitCount = toCopy->qubitCount;
    bitCapInt nQubitCount = qubitCount + oQubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy->qubitCount)) - 1) << qubitCount;
    bitCapInt bciArgs[BCI_ARG_LEN] = { nMaxQPower, qubitCount, startMask, endMask, 0, 0, 0, 0, 0, 0 };

    Compose(OCL_API_COMPOSE, bciArgs, toCopy);

    return result;
}

bitLenInt QEngineOCL::Compose(QEngineOCLPtr toCopy, bitLenInt start)
{
    bitLenInt result = start;

    bitCapInt oQubitCount = toCopy->qubitCount;
    bitCapInt nQubitCount = qubitCount + oQubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << start) - 1;
    bitCapInt midMask = ((1 << oQubitCount) - 1) << start;
    bitCapInt endMask = ((1 << (qubitCount + oQubitCount)) - 1) & ~(startMask | midMask);
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

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);
    bitCapInt bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs);

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
    device_context->wait_events.push_back(QueueCall(
        api_call, ngc, ngs, { stateBuffer, ulongBuffer, probBuffer1, angleBuffer1, probBuffer2, angleBuffer2 }));

    if ((maxQPower - partPower) <= 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    queue.enqueueMapBuffer(
        *probBuffer1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(real1) * remainderPower, &waitVec2);
    queue.enqueueMapBuffer(*probBuffer2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(real1) * partPower);
    queue.enqueueMapBuffer(*angleBuffer1, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(real1) * remainderPower);
    queue.enqueueMapBuffer(*angleBuffer2, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(real1) * partPower);

    bitCapInt i, j, k;
    i = 0;
    j = 0;
    k = 0;
    while (remainderStateProb[i] < min_norm) {
        i++;
    }
    k = i & ((1U << start) - 1);
    k |= (i ^ k) << (start + length);

    while (partStateProb[j] < min_norm) {
        j++;
    }
    k |= j << start;

    real1 refAngle = arg(GetAmplitude(k));
    real1 angleOffset = refAngle - (remainderStateAngle[i] + partStateAngle[j]);

    for (bitCapInt l = 0; l < partPower; l++) {
        partStateAngle[l] += angleOffset;
    }

    device_context->wait_events.resize(4);
    queue.enqueueUnmapMemObject(*probBuffer1, remainderStateProb, NULL, &(device_context->wait_events[0]));
    queue.enqueueUnmapMemObject(*probBuffer2, partStateProb, NULL, &(device_context->wait_events[1]));
    queue.enqueueUnmapMemObject(*angleBuffer1, remainderStateAngle, NULL, &(device_context->wait_events[2]));
    queue.enqueueUnmapMemObject(*angleBuffer2, partStateAngle, NULL, &(device_context->wait_events[3]));

    // If we Decompose, calculate the state of the bit system removed.
    if (destination != nullptr) {
        bciArgs[0] = partPower;

        std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();
        DISPATCH_WRITE(&waitVec2, *ulongBuffer, sizeof(bitCapInt), bciArgs);

        size_t ngc2 = FixWorkItemCount(partPower, nrmGroupCount);
        size_t ngs2 = FixGroupSize(ngc2, nrmGroupSize);

        BufferPtr otherStateBuffer;
        complex* otherStateVec;
        if (destination->deviceID == deviceID) {
            otherStateVec = destination->stateVec;
            otherStateBuffer = destination->stateBuffer;
        } else {
            otherStateVec = destination->AllocStateVec(destination->maxQPower, true);
            otherStateBuffer = destination->MakeStateVecBuffer(otherStateVec);

            DISPATCH_FILL(
                &waitVec2, *otherStateBuffer, sizeof(complex) * destination->maxQPower, complex(ZERO_R1, ZERO_R1));
        }

        QueueCall(OCL_API_DECOMPOSEAMP, ngc2, ngs2, { probBuffer2, angleBuffer2, ulongBuffer, otherStateBuffer })
            .wait();

        size_t oNStateVecSize = maxQPower * sizeof(complex);

        if (destination->deviceID != deviceID) {
            destination->LockSync(CL_MAP_READ | CL_MAP_WRITE);
            std::copy(otherStateVec, otherStateVec + destination->maxQPower, destination->stateVec);
            destination->UnlockSync();
            free(otherStateVec);
        } else if (!(destination->useHostRam) && destination->stateVec && oNStateVecSize <= destination->maxAlloc &&
            (2 * oNStateVecSize) <= destination->maxMem) {

            BufferPtr nSB = destination->MakeStateVecBuffer(NULL);

            cl::Event copyEvent;
            destination->queue.enqueueCopyBuffer(
                *(destination->stateBuffer), *nSB, 0, 0, sizeof(complex) * destination->maxQPower, NULL, &copyEvent);
            copyEvent.wait();

            destination->stateBuffer = nSB;
            free(destination->stateVec);
            destination->stateVec = NULL;
        }
    }

    // If we either Decompose or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPower;
    std::vector<cl::Event> waitVec3 = device_context->ResetWaitEvents();
    DISPATCH_WRITE(&waitVec3, *ulongBuffer, sizeof(bitCapInt), bciArgs);

    ngc = FixWorkItemCount(maxQPower, nrmGroupCount);
    ngs = FixGroupSize(ngc, nrmGroupSize);

    size_t nStateVecSize = maxQPower * sizeof(complex);
    if (!useHostRam && stateVec && nStateVecSize <= maxAlloc && (2 * nStateVecSize) <= maxMem) {
        clFinish();
        free(stateVec);
        stateVec = NULL;
    }

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    runningNorm = ONE_R1;
    if (destination != nullptr) {
        destination->runningNorm = ONE_R1;
    }

    QueueCall(OCL_API_DECOMPOSEAMP, ngc, ngs, { probBuffer1, angleBuffer1, ulongBuffer, nStateBuffer }).wait();

    ResetStateVec(nStateVec, nStateBuffer);

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

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs);

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(
        QueueCall(api_call, ngc, ngs, { stateBuffer, ulongBuffer, nrmBuffer }, sizeof(real1) * ngs));

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    real1 oneChance;
    WAIT_REAL1_SUM(&waitVec2, *nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    if (oneChance > ONE_R1)
        oneChance = ONE_R1;

    return oneChance;
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1 QEngineOCL::Prob(bitLenInt qubit)
{
    if (qubitCount == 1) {
        return ProbAll(1);
    }

    bitCapInt qPower = 1 << qubit;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    return Probx(OCL_API_PROB, bciArgs);
}

// Returns probability of permutation of the register
real1 QEngineOCL::ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
{
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

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs);

    BufferPtr probsBuffer =
        std::make_shared<cl::Buffer>(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

    size_t ngc = FixWorkItemCount(lengthPower, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(
        QueueCall(OCL_API_PROBREGALL, ngc, ngs, { stateBuffer, ulongBuffer, probsBuffer }));

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, &waitVec2);
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

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs);

    bitCapInt* skipPowers = new bitCapInt[length];
    std::copy(skipPowersVec.begin(), skipPowersVec.end(), skipPowers);

    BufferPtr qPowersBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(bitCapInt) * length, skipPowers);

    bitCapInt maxI = bciArgs[0];
    size_t ngc = FixWorkItemCount(maxI, nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(QueueCall(
        OCL_API_PROBMASK, ngc, ngs, { stateBuffer, ulongBuffer, nrmBuffer, qPowersBuffer }, sizeof(real1) * ngs));

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    real1 oneChance;
    WAIT_REAL1_SUM(&waitVec2, *nrmBuffer, ngc / ngs, nrmArray, &oneChance);

    delete[] skipPowers;

    if (oneChance > ONE_R1)
        oneChance = ONE_R1;

    return oneChance;
}

void QEngineOCL::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length;
    std::vector<bitCapInt> powersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
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

    v = ~mask; // count the number of bits set in v
    bitCapInt skipPower;
    bitCapInt maxPower = powersVec[powersVec.size() - 1];
    bitLenInt skipLength = 0; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (skipLength = 0; v; skipLength++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
        skipPower = (v ^ oldV) & oldV;
        if (skipPower < maxPower) {
            skipPowersVec.push_back(skipPower);
        } else {
            v = 0;
        }
    }

    bitCapInt bciArgs[BCI_ARG_LEN] = { lengthPower, maxJ, length, skipLength, 0, 0, 0, 0, 0, 0 };

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 4, bciArgs);

    BufferPtr probsBuffer =
        std::make_shared<cl::Buffer>(context, CL_MEM_COPY_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(real1) * lengthPower);

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

    device_context->wait_events.push_back(QueueCall(
        OCL_API_PROBMASKALL, ngc, ngs, { stateBuffer, ulongBuffer, probsBuffer, qPowersBuffer, qSkipPowersBuffer }));

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    queue.enqueueReadBuffer(*probsBuffer, CL_TRUE, 0, sizeof(real1) * lengthPower, probsArray, &waitVec2);

    delete[] powers;
    delete[] skipPowers;
}

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QEngineOCL::X(bitLenInt start, bitLenInt length)
{
    if (length == 1) {
        X(start);
        return;
    }

    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ regMask;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, 0, 0, 0, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_X, bciArgs);
}

/// Bitwise swap
void QEngineOCL::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    if (start1 == start2) {
        return;
    }

    bitCapInt reg1Mask = ((1 << length) - 1) << start1;
    bitCapInt reg2Mask = ((1 << length) - 1) << start2;
    bitCapInt otherMask = maxQPower - 1;
    otherMask ^= reg1Mask | reg2Mask;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, reg1Mask, reg2Mask, otherMask, start1, start2, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_SWAP, bciArgs);
}

void QEngineOCL::ROx(OCLAPI api_call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROL, shift, start, length); }

/// "Circular shift right" - shift bits right, and carry first bits.
void QEngineOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length) { ROx(OCL_API_ROR, shift, start, length); }

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & ~(regMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, start, toMod, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/// Add or Subtract integer (without sign or carry, with controls)
void QEngineOCL::CINT(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;

    bitCapInt controlMask = 0U;
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    for (bitLenInt i = 0; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        controlMask |= controlPowers[i];
    }
    std::sort(controlPowers, controlPowers + controlLen);

    bitCapInt otherMask = (maxQPower - 1) ^ (regMask | controlMask);

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

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DEC(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    INT(OCL_API_DEC, toSub, start, length);
}

void QEngineOCL::CDEC(
    bitCapInt toSub, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
{
    if (controlLen == 0) {
        DEC(toSub, inOutStart, length);
        return;
    }

    CINT(OCL_API_CDEC, toSub, inOutStart, length, controls, controlLen);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~(regMask | carryMask));

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, regMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTC(OCL_API_INCC, toAdd, start, length, carryIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTC(OCL_API_DECC, toSub, start, length, carryIndex);
}

/// Add or Subtract integer (with overflow, without carry)
void QEngineOCL::INTS(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ regMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, regMask, otherMask, lengthPower, overflowMask, start, toMod, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCS(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_INCS, toAdd, start, length, overflowIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECS(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex)
{
    INTS(OCL_API_DECS, toSub, start, length, overflowIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length,
    const bitLenInt overflowIndex, const bitLenInt carryIndex)
{
    bitCapInt overflowMask = 1 << overflowIndex;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, overflowMask, carryMask,
        start, toMod, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCSC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex,
    const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTSC(OCL_API_INCSC_1, toAdd, start, length, overflowIndex, carryIndex);
}

/** Subtract integer (with sign, with carry) */
void QEngineOCL::DECSC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt overflowIndex,
    const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTSC(OCL_API_DECSC_1, toSub, start, length, overflowIndex, carryIndex);
}

/// Add or Subtract integer (with sign, with carry)
void QEngineOCL::INTSC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (with sign, with carry) */
void QEngineOCL::INCSC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTSC(OCL_API_INCSC_2, toAdd, start, length, carryIndex);
}

/** Subtract integer (with sign, with carry) */
void QEngineOCL::DECSC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTSC(OCL_API_DECSC_2, toSub, start, length, carryIndex);
}

/// Add or Subtract integer (BCD)
void QEngineOCL::INTBCD(OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ inOutMask;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, inOutMask, otherMask, start, toMod, nibbleCount, 0, 0, 0, 0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD) */
void QEngineOCL::INCBCD(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_INCBCD, toAdd, start, length);
}

/** Subtract integer (BCD) */
void QEngineOCL::DECBCD(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    INTBCD(OCL_API_DECBCD, toSub, start, length);
}

/// Add or Subtract integer (BCD, with carry)
void QEngineOCL::INTBCDC(
    OCLAPI api_call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt nibbleCount = length / 4;
    if (nibbleCount * 4 != length) {
        throw std::invalid_argument("BCD word bit length must be a multiple of 4.");
    }
    bitCapInt inOutMask = ((1 << length) - 1) << start;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ (inOutMask | carryMask);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, inOutMask, otherMask, carryMask, start, toMod, nibbleCount, 0, 0,
        0 };

    ArithmeticCall(api_call, bciArgs);
}

/** Increment integer (BCD, with carry) */
void QEngineOCL::INCBCDC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTBCDC(OCL_API_INCBCDC, toAdd, start, length, carryIndex);
}

/** Subtract integer (BCD, with carry) */
void QEngineOCL::DECBCDC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTBCDC(OCL_API_DECBCDC, toSub, start, length, carryIndex);
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
    bitCapInt lowPower = 1U << length;
    if ((toDiv == 0) || (toDiv >= lowPower)) {
        throw "DIV by zero (or modulo 0 to register size)";
    }

    MULx(OCL_API_DIV, toDiv, inOutStart, carryStart, length);
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

    bitCapInt lowPower = 1U << length;
    if ((toDiv == 0) || (toDiv >= lowPower)) {
        throw "DIV by zero (or modulo 0 to register size)";
    }

    if (toDiv == 1) {
        return;
    }

    CMULx(OCL_API_CDIV, toDiv, inOutStart, carryStart, length, controls, controlLen);
}

void QEngineOCL::xMULx(OCLAPI api_call, bitCapInt* bciArgs, BufferPtr controlBuffer)
{
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = MakeStateVecBuffer(nStateVec);

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 10, bciArgs);
    DISPATCH_FILL(&waitVec, *nStateBuffer, sizeof(complex) * maxQPower, complex(ZERO_R1, ZERO_R1));

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    if (controlBuffer) {
        QueueCall(api_call, ngc, ngs, { stateBuffer, ulongBuffer, nStateBuffer, controlBuffer }).wait();
    } else {
        QueueCall(api_call, ngc, ngs, { stateBuffer, ulongBuffer, nStateBuffer }).wait();
    }

    ResetStateVec(nStateVec, nStateBuffer);
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

void QEngineOCL::CMULx(OCLAPI api_call, bitCapInt toMod, const bitLenInt inOutStart, const bitLenInt carryStart,
    const bitLenInt length, const bitLenInt* controls, const bitLenInt controlLen)
{
    bitCapInt lowMask = (1U << length) - 1U;
    bitCapInt inOutMask = lowMask << inOutStart;
    bitCapInt carryMask = lowMask << carryStart;

    bitCapInt* skipPowers = new bitCapInt[controlLen + length];
    bitCapInt* controlPowers = new bitCapInt[controlLen];
    bitCapInt controlMask = 0U;
    for (bitLenInt i = 0U; i < controlLen; i++) {
        controlPowers[i] = 1U << controls[i];
        skipPowers[i] = controlPowers[i];
        controlMask |= controlPowers[i];
    }
    for (bitLenInt i = 0U; i < length; i++) {
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

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    SetReg(valueStart, valueLength, 0);
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> valueLength, indexStart, inputMask, valueStart, valueBytes,
        valueLength, 0, 0, 0, 0 };

    ArithmeticCall(OCL_API_INDEXEDLDA, bciArgs, values, (1 << indexLength) * valueBytes);

    real1 prob;
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    bitCapInt i, outputInt;
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

    return (bitCapInt)(average + 0.5);
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
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, indexStart, inputMask, valueStart, outputMask, otherMask,
        carryIn, carryMask, lengthPower, valueBytes };

    ArithmeticCall(api_call, bciArgs, values, (1 << indexLength) * valueBytes);

    // At the end, just as a convenience, we return the expectation value for the addition result.
    real1 prob;
    real1 average = ZERO_R1;
    real1 totProb = ZERO_R1;
    bitCapInt i, outputInt;
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

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
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
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt) * 5, bciArgs, writeArgsEvent);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(QueueCall(api_call, ngc, ngs, { stateBuffer, ulongBuffer }));

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
}

void QEngineOCL::PhaseFlip()
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_PHASEFLIP, bciArgs);
}

/// For chips with a zero flag, flip the phase of the state where the register equals zero.
void QEngineOCL::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> length, (1U << start), length, 0, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_ZEROPHASEFLIP, bciArgs);
}

void QEngineOCL::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    bitCapInt regMask = ((1 << length) - 1) << start;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, regMask, 1U << flagIndex, greaterPerm, start, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_CPHASEFLIPIFLESS, bciArgs);
}

void QEngineOCL::PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = ((1 << length) - 1) << start;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower >> 1, regMask, greaterPerm, start, 0, 0, 0, 0, 0, 0 };

    PhaseFlipX(OCL_API_PHASEFLIPIFLESS, bciArgs);
}

/// Set arbitrary pure quantum state, in unsigned int permutation basis
void QEngineOCL::SetQuantumState(complex* inputState)
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
    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();
    queue.enqueueReadBuffer(*stateBuffer, CL_TRUE, sizeof(complex) * fullRegister, sizeof(complex), amp, &waitVec);
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

    OCLDeviceCall ocl = device_context->Reserve(OCL_API_APPROXCOMPARE);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt), bciArgs);

    BufferPtr otherStateBuffer;
    complex* otherStateVec;
    if (toCompare->deviceID == deviceID) {
        otherStateVec = toCompare->stateVec;
        otherStateBuffer = toCompare->stateBuffer;
    } else {
        otherStateVec = toCompare->AllocStateVec(toCompare->maxQPower, true);
        toCompare->LockSync(CL_MAP_READ);
        std::copy(toCompare->stateVec, toCompare->stateVec + toCompare->maxQPower, otherStateVec);
        toCompare->UnlockSync();
        otherStateBuffer = toCompare->MakeStateVecBuffer(otherStateVec);
    }

    device_context->wait_events.push_back(QueueCall(OCL_API_APPROXCOMPARE, nrmGroupCount, nrmGroupSize,
        { stateBuffer, otherStateBuffer, ulongBuffer, nrmBuffer }, sizeof(real1) * nrmGroupSize));

    unsigned int size = (nrmGroupCount / nrmGroupSize);
    if (size == 0) {
        size = 1;
    }

    runningNorm = ZERO_R1;
    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    real1 sumSqrErr;
    WAIT_REAL1_SUM(&waitVec2, *nrmBuffer, size, nrmArray, &sumSqrErr);

    if (toCompare->deviceID != deviceID) {
        free(otherStateVec);
    }

    return sumSqrErr < approxcompare_error;
}

QInterfacePtr QEngineOCL::Clone()
{
    QEngineOCLPtr copyPtr = std::make_shared<QEngineOCL>(
        qubitCount, 0, rand_generator, complex(ONE_R1, ZERO_R1), doNormalize, randGlobalPhase, useHostRam);

    LockSync(CL_MAP_READ);
    copyPtr->LockSync(CL_MAP_WRITE);

    std::copy(stateVec, stateVec + maxQPower, copyPtr->stateVec);

    copyPtr->UnlockSync();
    UnlockSync();

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

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    if (nrm < min_norm) {
        DISPATCH_FILL(&waitVec, *stateBuffer, sizeof(complex) * maxQPower, complex(ZERO_R1, ZERO_R1));
        runningNorm = ZERO_R1;
        return;
    }

    real1 r1_args[2] = { min_norm, (real1)std::sqrt(nrm) };
    BufferPtr argsBuffer =
        std::make_shared<cl::Buffer>(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(real1) * 2, r1_args);

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt), bciArgs, writeArgsEvent);

    size_t ngc = FixWorkItemCount(bciArgs[0], nrmGroupCount);
    size_t ngs = FixGroupSize(ngc, nrmGroupSize);

    device_context->wait_events.push_back(
        QueueCall(OCL_API_NORMALIZE, ngc, ngs, { stateBuffer, ulongBuffer, argsBuffer }));

    runningNorm = ONE_R1;

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    
#if ENABLE_RASPBERRYPI
    clFinish();
#endif
}

void QEngineOCL::UpdateRunningNorm()
{
    OCLDeviceCall ocl = device_context->Reserve(OCL_API_UPDATENORM);

    runningNorm = ONE_R1;

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    std::vector<cl::Event> waitVec = device_context->ResetWaitEvents();

    cl::Event writeArgsEvent;
    DISPATCH_TEMP_WRITE(&waitVec, *ulongBuffer, sizeof(bitCapInt), bciArgs, writeArgsEvent);

    device_context->wait_events.push_back(QueueCall(OCL_API_UPDATENORM, nrmGroupCount, nrmGroupSize,
        { stateBuffer, ulongBuffer, nrmBuffer }, sizeof(real1) * nrmGroupSize));

    unsigned int size = (nrmGroupCount / nrmGroupSize);
    if (size == 0) {
        size = 1;
    }

    // This kernel is run on a single work group, but it lets us continue asynchronously.
    if (size > 1) {
        device_context->wait_events.push_back(
            QueueCall(OCL_API_NORMSUM, size, size, { nrmBuffer }, sizeof(real1) * size));
    }

    std::vector<cl::Event> waitVec2 = device_context->ResetWaitEvents();

    DISPATCH_WRITE(&waitVec2, *nrmBuffer, sizeof(real1), &runningNorm);

    // Wait for buffer write from limited lifetime objects
    writeArgsEvent.wait();
    
#if ENABLE_RASPBERRYPI
    clFinish();
#endif
}

complex* QEngineOCL::AllocStateVec(bitCapInt elemCount, bool doForceAlloc)
{
    // If we're not using host ram, there's no reason to allocate.
    if (!doForceAlloc && !stateVec) {
        return NULL;
    }

        // elemCount is always a power of two, but might be smaller than ALIGN_SIZE
#ifdef __APPLE__
    void* toRet;
    posix_memalign(
        &toRet, ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
    return (complex*)toRet;
#else
    return (complex*)aligned_alloc(
        ALIGN_SIZE, ((sizeof(complex) * elemCount) < ALIGN_SIZE) ? ALIGN_SIZE : sizeof(complex) * elemCount);
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
