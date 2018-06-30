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

#include <memory>

#include "oclengine.hpp"
#include "qengine_opencl.hpp"

namespace Qrack {

typedef std::lock_guard<std::recursive_mutex> LockGuard;

#define CMPLX_NORM_LEN 5

void QEngineOCL::SetDevice(const int& dID)
{
    if (dID >= 0) {
        deviceID = dID % (clObj->GetDeviceCount());
    } else {
        deviceID = -1;
    }
    queue = clObj->GetQueuePtr(deviceID);
    context = *(clObj->GetContextPtr(queue));
    deviceMutexPtr = clObj->GetMutexPtr(queue);
}

void QEngineOCL::InitOCL(int devID)
{
    clObj = OCLEngine::Instance();
    SetDevice(devID);

    // create buffers on device (allocate space on GPU)
    stateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(complex) * maxQPower, stateVec);
    cmplxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(complex) * 5);
    ulongBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * 10);
    nrmBuffer = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE,
        sizeof(real1) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    maxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt));

    LockGuard locked_call(*deviceMutexPtr);
    queue->enqueueMapBuffer(*stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);
}

void QEngineOCL::ReInitOCL()
{
    clObj = OCLEngine::Instance();

    // create buffers on device (allocate space on GPU)
    stateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(complex) * maxQPower, stateVec);

    queue->enqueueMapBuffer(*stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);
}

void QEngineOCL::ResetStateVec(complex* nStateVec)
{
    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    QEngineCPU::ResetStateVec(nStateVec);
    ReInitOCL();
}

void QEngineOCL::DispatchCall(
    cl::Kernel* call, bitCapInt (&bciArgs)[BCI_ARG_LEN], complex* nVec, unsigned char* values, bitCapInt valuesPower)
{
    /* Allocate a temporary nStateVec, or use the one supplied. */
    complex* nStateVec = nVec ? nVec : AllocStateVec(maxQPower);

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(complex) * maxQPower, nStateVec);
    queue->enqueueFillBuffer(*nStateBuffer, complex(0.0, 0.0), 0, sizeof(complex) * maxQPower);
    call->setArg(0, *stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, *nStateBuffer);
    cl::Buffer loadBuffer;
    if (values) {
        loadBuffer =
            cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        call->setArg(3, loadBuffer);
    }
    queue->finish();

    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*nStateBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(complex) * maxQPower);
    if (!nVec) {
        /* a nStateVec wasn't passed in; swap the one allocated here with stateVec */
        ResetStateVec(nStateVec);
    }
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    LockGuard locked_call(*deviceMutexPtr);

    complex cmplx[CMPLX_NORM_LEN];
    real1* nrmParts = nullptr;
    for (int i = 0; i < 4; i++) {
        cmplx[i] = mtrx[i];
    }
    cmplx[4] = complex((doNormalize && (bitCount == 1)) ? (1.0 / sqrt(runningNorm)) : 1.0, 0.0);
    bitCapInt bciArgs[BCI_ARG_LEN] = { bitCount, maxQPower, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < bitCount; i++) {
        bciArgs[4 + i] = qPowersSorted[i];
    }

    /* Slightly different call parameters than the rest of the calls. */
    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue->enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(complex) * CMPLX_NORM_LEN, cmplx);
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    if (doCalcNorm) {
        nrmParts = new real1[CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]();
        queue->enqueueWriteBuffer(
            nrmBuffer, CL_FALSE, 0, sizeof(real1) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
    }
    queue->finish();

    cl::Kernel apply2x2;
    if (doCalcNorm) {
        apply2x2 = *(clObj->GetApply2x2NormPtr(queue));
    } else {
        apply2x2 = *(clObj->GetApply2x2Ptr(queue));
    }
    apply2x2.setArg(0, *stateBuffer);
    apply2x2.setArg(1, cmplxBuffer);
    apply2x2.setArg(2, ulongBuffer);
    if (doCalcNorm) {
        apply2x2.setArg(3, nrmBuffer);
    }
    queue->enqueueNDRangeKernel(apply2x2, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);
    if (doNormalize && doCalcNorm) {
        queue->enqueueReadBuffer(
            nrmBuffer, CL_TRUE, 0, sizeof(real1) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
        runningNorm = 0.0;
        for (unsigned long int i = 0; i < CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE; i++) {
            runningNorm += nrmParts[i];
        }
        delete[] nrmParts;
    }
}

bitLenInt QEngineOCL::Cohere(QEngineOCLPtr toCopy)
{
    LockGuard locked_call(*deviceMutexPtr);

    bitLenInt result = qubitCount;

    if (doNormalize && (runningNorm != 1.0)) {
        NormalizeState();
    }

    if ((toCopy->doNormalize) && (toCopy->runningNorm != 1.0)) {
        toCopy->NormalizeState();
    }

    bitCapInt nQubitCount = qubitCount + toCopy->qubitCount;
    bitCapInt nMaxQPower = 1 << nQubitCount;
    bitCapInt startMask = (1 << qubitCount) - 1;
    bitCapInt endMask = ((1 << (toCopy->qubitCount)) - 1) << qubitCount;
    bitCapInt bciArgs[BCI_ARG_LEN] = { nMaxQPower, startMask, endMask, qubitCount, 0, 0, 0, 0, 0, 0 };

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    BufferPtr stateBuffer2 = std::make_shared<cl::Buffer>(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
        sizeof(complex) * (1 << (toCopy->qubitCount)), toCopy->stateVec);

    complex* nStateVec = AllocStateVec(nMaxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(complex) * nMaxQPower, nStateVec);
    cl::Kernel* call = clObj->GetCoherePtr(queue);
    call->setArg(0, *stateBuffer);
    call->setArg(1, *stateBuffer2);
    call->setArg(2, ulongBuffer);
    call->setArg(3, *nStateBuffer);
    queue->finish();

    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*nStateBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(complex) * nMaxQPower);
    SetQubitCount(nQubitCount);
    ResetStateVec(nStateVec);

    return result;
}

void QEngineOCL::DecohereDispose(bitLenInt start, bitLenInt length, QEngineOCLPtr destination)
{
    // "Dispose" is basically the same as decohere, except "Dispose" throws the removed bits away.

    if (length == 0) {
        return;
    }

    if (doNormalize && (runningNorm != 1.0)) {
        NormalizeState();
    }

    bitCapInt partPower = 1 << length;
    bitCapInt remainderPower = 1 << (qubitCount - length);
    bitCapInt bciArgs[BCI_ARG_LEN] = { partPower, remainderPower, start, length, 0, 0, 0, 0, 0, 0 };

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    // The "remainder" bits will always be maintained.
    real1* remainderStateProb = new real1[remainderPower]();
    real1* remainderStateAngle = new real1[remainderPower];
    cl::Buffer probBuffer1 = cl::Buffer(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateProb);
    cl::Buffer angleBuffer1 = cl::Buffer(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * remainderPower, remainderStateAngle);

    // Depending on whether we Decohere or Dispose, we have optimized kernels.
    cl::Kernel* call;
    if (destination != nullptr) {
        call = clObj->GetDecohereProbPtr(queue);
    } else {
        call = clObj->GetDisposeProbPtr(queue);
    }
    // These arguments are common to both kernels.
    call->setArg(0, *stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, probBuffer1);
    call->setArg(3, angleBuffer1);

    // The removed "part" is only necessary for Decohere.
    real1* partStateProb = nullptr;
    real1* partStateAngle = nullptr;
    cl::Buffer probBuffer2, angleBuffer2;
    if (destination != nullptr) {
        partStateProb = new real1[partPower]();
        partStateAngle = new real1[partPower];
        probBuffer2 =
            cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateProb);
        angleBuffer2 =
            cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(real1) * partPower, partStateAngle);

        call->setArg(4, probBuffer2);
        call->setArg(5, angleBuffer2);
    }

    queue->finish();

    // Call the kernel that calculates bit probability and angle.
    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    if ((maxQPower - partPower) == 0) {
        SetQubitCount(1);
    } else {
        SetQubitCount(qubitCount - length);
    }

    // Wait as long as possible before joining the kernel.
    queue->flush();
    queue->finish();

    call = clObj->GetDecohereAmpPtr(queue);

    // If we Decohere, calculate the state of the bit system removed.
    if (destination != nullptr) {
        bciArgs[0] = partPower;
        queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
        queue->enqueueUnmapMemObject(*(destination->stateBuffer), destination->stateVec);
        queue->finish();

        call->setArg(0, probBuffer2);
        call->setArg(1, angleBuffer2);
        call->setArg(2, ulongBuffer);
        call->setArg(3, *(destination->stateBuffer));
        queue->finish();

        queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
            cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
            cl::NDRange(1)); // local number (per group)

        queue->enqueueMapBuffer(
            *(destination->stateBuffer), CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * partPower);

        delete[] partStateProb;
        delete[] partStateAngle;
    }

    // If we either Decohere or Dispose, calculate the state of the bit system that remains.
    bciArgs[0] = maxQPower;
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    complex* nStateVec = AllocStateVec(maxQPower);
    BufferPtr nStateBuffer = std::make_shared<cl::Buffer>(
        context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(complex) * maxQPower, nStateVec);

    queue->finish();

    call->setArg(0, probBuffer1);
    call->setArg(1, angleBuffer1);
    call->setArg(2, ulongBuffer);
    call->setArg(3, *nStateBuffer);
    queue->finish();

    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);

    ResetStateVec(nStateVec);

    delete[] remainderStateProb;
    delete[] remainderStateAngle;
}

void QEngineOCL::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr destination)
{
    LockGuard locked_call(*deviceMutexPtr);
    DecohereDispose(start, length, std::dynamic_pointer_cast<QEngineOCL>(destination));
}

void QEngineOCL::Dispose(bitLenInt start, bitLenInt length) { DecohereDispose(start, length, (QEngineOCLPtr) nullptr); }

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
real1 QEngineOCL::Prob(bitLenInt qubit)
{
    LockGuard locked_call(*deviceMutexPtr);

    if (doNormalize && (runningNorm != 1.0)) {
        NormalizeState();
    }

    bitCapInt qPower = 1 << qubit;
    real1 oneChance = 0.0;

    int numCores = CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
    real1* oneChanceArray = new real1[numCores]();

    bitCapInt bciArgs[BCI_ARG_LEN] = { maxQPower, qPower, 0, 0, 0, 0, 0, 0, 0, 0 };

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    cl::Buffer oneChanceBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(real1) * numCores, oneChanceArray);

    cl::Kernel* call = clObj->GetProbPtr(queue);
    call->setArg(0, *stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, oneChanceBuffer);
    queue->finish();

    // Note that the global size is 1 (serial). This is because the kernel is not very easily parallelized, but we
    // ultimately want to offload all manipulation of stateVec from host code to OpenCL kernels.
    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*stateBuffer, CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);
    queue->enqueueMapBuffer(oneChanceBuffer, CL_FALSE, CL_MAP_READ, 0, sizeof(real1) * numCores);

    queue->finish();

    for (int i = 0; i < numCores; i++) {
        oneChance += oneChanceArray[i];
    }

    if (oneChance > 1.0)
        oneChance = 1.0;

    return oneChance;
}

// Apply X ("not") gate to each bit in "length," starting from bit index
// "start"
void QEngineOCL::X(bitLenInt start, bitLenInt length)
{
    LockGuard locked_call(*deviceMutexPtr);

    if (length == 1) {
        X(start);
        return;
    }

    bitCapInt regMask = ((1 << length) - 1) << start;
    bitCapInt otherMask = ((1 << qubitCount) - 1) ^ regMask;
    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, 0, 0, 0, 0, 0, 0, 0 };

    DispatchCall(clObj->GetXPtr(queue), bciArgs);
}

void QEngineOCL::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    LockGuard locked_call(*deviceMutexPtr);
    QEngineCPU::Swap(qubit1, qubit2);
}

/// Bitwise swap
void QEngineOCL::Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
{
    if (start1 == start2) {
        return;
    }

    LockGuard locked_call(*deviceMutexPtr);

    bitCapInt reg1Mask = ((1 << length) - 1) << start1;
    bitCapInt reg2Mask = ((1 << length) - 1) << start2;
    bitCapInt otherMask = maxQPower - 1;
    otherMask ^= reg1Mask | reg2Mask;
    bitCapInt bciArgs[10] = { maxQPower, reg1Mask, reg2Mask, otherMask, start1, start2, 0, 0, 0, 0 };

    DispatchCall(clObj->GetSwapPtr(queue), bciArgs);
}

void QEngineOCL::ROx(cl::Kernel* call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);
    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    DispatchCall(call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QEngineOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    LockGuard locked_call(*deviceMutexPtr);
    ROx(clObj->GetROLPtr(queue), shift, start, length);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QEngineOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    LockGuard locked_call(*deviceMutexPtr);
    ROx(clObj->GetRORPtr(queue), shift, start, length);
}

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(cl::Kernel* call, bitCapInt toMod, const bitLenInt start, const bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & ~(regMask);

    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, lengthPower, start, toMod, 0, 0, 0, 0 };

    DispatchCall(call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length)
{
    LockGuard locked_call(*deviceMutexPtr);
    INT(clObj->GetINCPtr(queue), toAdd, start, length);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DEC(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    LockGuard locked_call(*deviceMutexPtr);
    INT(clObj->GetDECPtr(queue), toSub, start, length);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(
    cl::Kernel* call, bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~(regMask | carryMask));

    bitCapInt bciArgs[10] = { maxQPower >> 1, regMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0, 0 };

    DispatchCall(call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCC(bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    LockGuard locked_call(*deviceMutexPtr);

    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTC(clObj->GetINCCPtr(queue), toAdd, start, length, carryIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECC(bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    LockGuard locked_call(*deviceMutexPtr);

    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTC(clObj->GetDECCPtr(queue), toSub, start, length, carryIndex);
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    if (runningNorm <= 0.0) {
        return 0;
    }

    LockGuard locked_call(*deviceMutexPtr);

    SetReg(valueStart, valueLength, 0);
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt bciArgs[10] = { maxQPower >> valueLength, indexStart, inputMask, valueStart, valueBytes, valueLength, 0,
        0, 0 };

    complex* nStateVec = AllocStateVec(maxQPower);
    DispatchCall(clObj->GetLDAPtr(queue), bciArgs, nStateVec, values, (1 << valueLength) * valueBytes);

    real1 prob;
    real1 average = 0.0;
    real1 totProb = 0.0;
    bitCapInt i, outputInt;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > 0.0) {
        average /= totProb;
    }

    ResetStateVec(nStateVec);

    return (unsigned char)(average + 0.5);
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::OpIndexed(cl::Kernel* call, bitCapInt carryIn, bitLenInt indexStart, bitLenInt indexLength,
    bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    if (runningNorm <= 0.0) {
        return 0;
    }

    // The carry has to first to be measured for its input value.
    if (M(carryIndex)) {
        /*
         * If the carry is set, we flip the carry bit. We always initially
         * clear the carry after testing for carry in.
         */
        carryIn = !carryIn;
        X(carryIndex);
    }

    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt lengthPower = 1 << valueLength;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask | carryMask));
    bitCapInt bciArgs[10] = { maxQPower >> 1, indexStart, inputMask, valueStart, outputMask, otherMask, carryIn,
        carryMask, lengthPower, valueBytes };

    complex* nStateVec = AllocStateVec(maxQPower);
    DispatchCall(call, bciArgs, nStateVec, values, (1 << valueLength) * valueBytes);

    // At the end, just as a convenience, we return the expectation value for the addition result.
    real1 prob;
    real1 average = 0.0;
    real1 totProb = 0.0;
    bitCapInt i, outputInt;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    if (totProb > 0.0) {
        average /= totProb;
    }

    // Finally, we dealloc the old state vector and replace it with the one we just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}

/** Add based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    LockGuard locked_call(*deviceMutexPtr);
    return OpIndexed(clObj->GetADCPtr(queue), 0, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
    bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    LockGuard locked_call(*deviceMutexPtr);
    return OpIndexed(clObj->GetSBCPtr(queue), 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

void QEngineOCL::NormalizeState(real1 nrm)
{
    if (nrm < 0.0) {
        nrm = runningNorm;
    }
    if (nrm == 1.0) {
        return;
    }
    LockGuard locked_call(*deviceMutexPtr);
    if (nrm <= 0.0) {
        queue->enqueueFillBuffer(*stateBuffer, complex(0.0, 0.0), 0, sizeof(complex) * maxQPower);
        runningNorm = 0.0;
        queue->finish();
        return;
    }

    real1 r1_args[2] = { min_norm, (real1)sqrt(nrm) };
    cl::Buffer argsBuffer = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(real1) * 2, r1_args);

    bitCapInt bciArgs[10] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queue->enqueueWriteBuffer(ulongBuffer, CL_TRUE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    cl::Kernel* call = clObj->GetNormalizePtr(queue);
    call->setArg(0, *stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, argsBuffer);
    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);

    runningNorm = 1.0;
}

void QEngineOCL::UpdateRunningNorm()
{
    LockGuard locked_call(*deviceMutexPtr);

    real1* nrmParts = new real1[CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]();
    queue->enqueueWriteBuffer(
        nrmBuffer, CL_FALSE, 0, sizeof(real1) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);

    bitCapInt bciArgs[10] = { maxQPower, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    queue->enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    queue->flush();

    queue->enqueueUnmapMemObject(*stateBuffer, stateVec);
    cl::Kernel* call = clObj->GetUpdateNormPtr(queue);

    queue->finish();

    call->setArg(0, *stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, nrmBuffer);
    queue->enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue->enqueueMapBuffer(*stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(complex) * maxQPower);

    queue->enqueueReadBuffer(
        nrmBuffer, CL_TRUE, 0, sizeof(real1) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
    runningNorm = 0.0;
    for (unsigned long int i = 0; i < CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE; i++) {
        runningNorm += nrmParts[i];
    }

    if (runningNorm < min_norm) {
        NormalizeState(0.0);
    }

    delete[] nrmParts;
}

} // namespace Qrack
