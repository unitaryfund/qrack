//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "qengine_opencl.hpp"
#include "oclengine.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

void QEngineOCL::InitOCL()
{
    clObj = OCLEngine::Instance();

    queue = *(clObj->GetQueuePtr());
    cl::Context context = *(clObj->GetContextPtr());

    // create buffers on device (allocate space on GPU)
    stateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, stateVec);
    cmplxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Complex16) * 5);
    ulongBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * 10);
    nrmBuffer = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    maxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt));

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
}

void QEngineOCL::ReInitOCL()
{
    clObj = OCLEngine::Instance();

    queue = *(clObj->GetQueuePtr());
    cl::Context context = *(clObj->GetContextPtr());

    // create buffers on device (allocate space on GPU)
    stateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(Complex16) * maxQPower, stateVec);

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
}

void QEngineOCL::ResetStateVec(Complex16* nStateVec)
{
    queue.enqueueUnmapMemObject(stateBuffer, stateVec);
    QEngineCPU::ResetStateVec(nStateVec);
    ReInitOCL();
}

void QEngineOCL::DispatchCall(cl::Kernel *call, bitCapInt (&bciArgs)[BCI_ARG_LEN], Complex16 *nVec, unsigned char* values, bitCapInt valuesPower)
{
    /* Allocate a temporary nStateVec, or use the one supplied. */
    Complex16* nStateVec = nVec ? nVec : AllocStateVec(maxQPower);
    std::fill(nStateVec, nStateVec + maxQPower, Complex16(0.0, 0.0));

    queue.enqueueUnmapMemObject(stateBuffer, stateVec);
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);

    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(Complex16) * maxQPower, nStateVec);
    call->setArg(0, stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, nStateBuffer);
    cl::Buffer loadBuffer;
    if (values) {
        loadBuffer = cl::Buffer(context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(unsigned char) * valuesPower, values);
        call->setArg(3, loadBuffer);
    }
    queue.finish();

    queue.enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ, 0, sizeof(Complex16) * maxQPower);
    if (!nVec) {
        /* a nStateVec wasn't passed in; swap the one allocated here with stateVec */
        ResetStateVec(nStateVec);
    }
}

void QEngineOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doCalcNorm)
{
    Complex16 cmplx[CMPLX_NORM_LEN];
    double* nrmParts;
    for (int i = 0; i < 4; i++) {
        cmplx[i] = mtrx[i];
    }
    cmplx[4] = Complex16((bitCount == 1) ? (1.0 / runningNorm) : 1.0, 0.0);
    bitCapInt bciArgs[BCI_ARG_LEN] = { bitCount, maxQPower, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < bitCount; i++) {
        bciArgs[4 + i] = qPowersSorted[i];
    }

    /* Slightly different call parameters than the rest of the calls. */
    queue.enqueueUnmapMemObject(stateBuffer, stateVec);
    queue.enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(Complex16) * CMPLX_NORM_LEN, cmplx);
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    if (doCalcNorm) {
        nrmParts = new double[CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE]();
        queue.enqueueWriteBuffer(nrmBuffer, CL_FALSE, 0, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
    }
    queue.finish();

    cl::Kernel apply2x2;
    if (doCalcNorm) {
         apply2x2 = *(clObj->GetApply2x2NormPtr());
    } else {
         apply2x2 = *(clObj->GetApply2x2Ptr());
    }
    apply2x2.setArg(0, stateBuffer);
    apply2x2.setArg(1, cmplxBuffer);
    apply2x2.setArg(2, ulongBuffer);
    if (doCalcNorm) {
        apply2x2.setArg(3, nrmBuffer);
    }
    queue.enqueueNDRangeKernel(apply2x2, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    if (doCalcNorm) {
        queue.enqueueReadBuffer(nrmBuffer, CL_TRUE, 0, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, nrmParts);
        runningNorm = 0.0;
        for (unsigned long int i = 0; i < CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE; i++) {
            runningNorm += nrmParts[i];
        }
        delete[] nrmParts;
        runningNorm = sqrt(runningNorm);
    } else {
        runningNorm = 1.0;
    }

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
    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, 0, 0, 0, 0, 0, 0, 0 };

    DispatchCall(clObj->GetXPtr(), bciArgs);
}

void QEngineOCL::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    QEngineCPU::Swap(qubit1, qubit2);
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
    bitCapInt bciArgs[10] = { maxQPower, reg1Mask, reg2Mask, otherMask, start1, start2, 0, 0, 0, 0 };

    DispatchCall(clObj->GetSwapPtr(), bciArgs);
}

void QEngineOCL::ROx(cl::Kernel *call, bitLenInt shift, bitLenInt start, bitLenInt length)
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
    ROx(clObj->GetROLPtr(), shift, start, length);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QEngineOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    ROx(clObj->GetRORPtr(), shift, start, length);
}

/// Add or Subtract integer (without sign or carry)
void QEngineOCL::INT(cl::Kernel* call,
    bitCapInt toMod, const bitLenInt start, const bitLenInt length)
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
    INT(clObj->GetINCPtr(), toAdd, start, length);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DEC(bitCapInt toSub, const bitLenInt start, const bitLenInt length)
{
    INT(clObj->GetDECPtr(), toSub, start, length);
}

/// Add or Subtract integer (without sign, with carry)
void QEngineOCL::INTC(cl::Kernel* call,
    bitCapInt toMod, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~(regMask | carryMask));

    bitCapInt bciArgs[10] = { maxQPower >> 1, regMask, otherMask, lengthPower, carryMask, start, toMod, 0, 0,
        0 };

    DispatchCall(call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QEngineOCL::INCC(
    bitCapInt toAdd, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    INTC(clObj->GetINCCPtr(), toAdd, start, length, carryIndex);
}

/** Subtract integer (without sign, with carry) */
void QEngineOCL::DECC(
    bitCapInt toSub, const bitLenInt start, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }

    INTC(clObj->GetDECCPtr(), toSub, start, length, carryIndex);
}

/** Set 8 bit register bits based on read from classical memory */
bitCapInt QEngineOCL::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    SetReg(valueStart, valueLength, 0);
    bitLenInt valueBytes = (valueLength + 7) / 8;
    bitCapInt inputMask = ((1 << indexLength) - 1) << indexStart;
    bitCapInt outputMask = ((1 << valueLength) - 1) << valueStart;
    bitCapInt bciArgs[10] = { maxQPower >> valueLength, indexStart, inputMask, valueStart, valueBytes, valueLength, 0, 0, 0 };

    Complex16* nStateVec = AllocStateVec(maxQPower);
    DispatchCall(clObj->GetLDAPtr(), bciArgs, nStateVec, values, (1 << valueLength) * valueBytes);

    double prob, average, totProb;
    totProb = 0.0;
    bitCapInt i, outputInt;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    average /= totProb;

    ResetStateVec(nStateVec);

    return (unsigned char)(average + 0.5);
}

/** Add or Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::OpIndexed(cl::Kernel *call, bitCapInt carryIn,
    bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
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

    Complex16* nStateVec = AllocStateVec(maxQPower);
    DispatchCall(call, bciArgs, nStateVec, values, (1 << valueLength) * valueBytes);

    // At the end, just as a convenience, we return the expectation value for the addition result.
    double prob, average, totProb;
    totProb = 0.0;
    bitCapInt i, outputInt;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> valueStart;
        prob = norm(nStateVec[i]);
        totProb += prob;
        average += prob * outputInt;
    }
    average /= totProb;

    // Finally, we dealloc the old state vector and replace it with the one we just calculated.
    ResetStateVec(nStateVec);

    // Return the expectation value.
    return (bitCapInt)(average + 0.5);
}


/** Add based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    return OpIndexed(clObj->GetADCPtr(), 0, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
bitCapInt QEngineOCL::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    return OpIndexed(clObj->GetSBCPtr(), 1, indexStart, indexLength, valueStart, valueLength, carryIndex, values);
}

} // namespace Qrack
