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

#include "qunit_opencl.hpp"
#include "oclengine.hpp"

namespace Qrack {

#define BCI_ARG_LEN 10
#define CMPLX_NORM_LEN 5

void QUnitOCL::InitOCL()
{
    clObj = OCLEngine::Instance();

    queue = *(clObj->GetQueuePtr());
    cl::Context context = *(clObj->GetContextPtr());

    // create buffers on device (allocate space on GPU)
    stateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(stateVec[0]));
    cmplxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(Complex16) * 5);
    ulongBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt) * 10);
    nrmBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    maxBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(bitCapInt));
    loadBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned char) * 256);

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
}

void QUnitOCL::ReInitOCL()
{
    clObj = OCLEngine::Instance();

    queue = *(clObj->GetQueuePtr());
    cl::Context context = *(clObj->GetContextPtr());

    // create buffers on device (allocate space on GPU)
    stateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(stateVec[0]));

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
}

void QUnitOCL::ResetStateVec(std::unique_ptr<Complex16[]> nStateVec)
{
    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    CoherentUnit::ResetStateVec(std::move(nStateVec));
    ReInitOCL();
}

void QUnitOCL::DispatchCall(cl::Kernel *call, bitCapInt (&bciArgs)[BCI_ARG_LEN], Complex16 *nVec = NULL, size_t nVecLen = 0, unsigned char* values = NULL)
{
    queue.enqueueUnmapMemObject(stateBuffer, stateVec);
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0,
            sizeof(bitCapInt) * BCI_ARG_LEN, bciArgs);
    size_t cmplxSz = nVecLen > 0 ? nVecLen : maxQPower;
    Complex16 *nStateVec = nVec ? nVec : new Complex16[cmplxSz];
    std::fill(nStateVec, nStateVec + cmplxSz, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * cmplxSz, nStateVec);
    call->setArg(0, stateBuffer);
    call->setArg(1, ulongBuffer);
    call->setArg(2, nStateBuffer);
    if (values) {
        queue.enqueueWriteBuffer(loadBuffer, CL_FALSE, 0, sizeof(unsigned char) * 256, values);
        call->setArg(3, loadBuffer);
    }
    queue.finish();

    queue.enqueueNDRangeKernel(*call, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * cmplxSz);

    if (!nVec) {
        ResetStateVec(std::move(nStateVec));
        free(nStateVec);
    }
}

void QUnitOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm)
{
    Complex16 cmplx[CMPLX_NORM_LEN];
    for (int i = 0; i < 4; i++) {
        cmplx[i] = mtrx[i];
    }
    cmplx[4] = Complex16(doApplyNorm ? (1.0 / runningNorm) : 1.0, 0.0);
    bitCapInt bciArgs[BCI_ARG_LEN] = { bitCount, maxQPower, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < bitCount; i++) {
        bciArgs[4 + i] = qPowersSorted[i];
    }

    DispatchCall(clObj->getApply2x2Ptr(), bitCapInt, cmplx, CMPLX_NORM_LEN);

    if (doCalcNorm) {
        UpdateRunningNorm();
    } else {
        runningNorm = 1.0;
    }
}

void QUnitOCL::ROx(cl::Kernel *call, bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);
    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    DispatchCall(call, bciArgs);
}

/// "Circular shift left" - shift bits left, and carry last bits.
void QUnitOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    ROx(clObj->GetROLPtr(), shift, start, length);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QUnitOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    ROx(clObj->GetRORPtr(), shift, start, length);
}

/// Add or Subtract integer (without sign, with carry)
void QUnitOCL::INTC(cl::Kernel* call,
    bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }

    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~(regMask | carryMask));

    bitCapInt bciArgs[10] = { maxQPower >> 1, regMask, otherMask, lengthPower, carryMask, start, toAdd, 0, 0,
        0 };

    DispatchCall(call, bciArgs);
}

/** Increment integer (without sign, with carry) */
void QUnitOCL::INCC(
    bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    INTC(clObj->GetINCCPtr(), toAdd, start, length, carryIndex);
}

/** Subtract integer (without sign, with carry) */
void QUnitOCL::DECC(
    bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    INTC(clObj->GetDECCPtr(), toAdd, start, length, carryIndex);
}

/** Set 8 bit register bits based on read from classical memory */
unsigned char QUnitOCL::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    SetReg(outputStart, 8, 0);
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt bciArgs[10] = { maxQPower >> 8, inputStart, inputMask, outputStart, 0, 0, 0, 0, 0, 0 };

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    DispatchCall(clObj->GetSR8Ptr(), bciArgs, nStateVec.get(), maxQPower, values);

    bitCapInt i, outputInt;
    double prob, average;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> outputStart;
        prob = norm(nStateVec[i]);
        average += prob * outputInt;
    }
    ResetStateVec(std::move(nStateVec));

    return (unsigned char)(average + 0.5);
}

/** Add or Subtract based on an indexed load from classical memory */
unsigned char QUnitOCL::OpSuperposeReg8(cl::Kernel *call, bitCapInt carryIn,
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
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

    bitCapInt lengthPower = 1 << 8;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask));
    bitCapInt bciArgs[10] = { maxQPower >> 1, inputStart, inputMask, outputStart, outputMask, otherMask, carryIn,
        carryMask, lengthPower, 0 };

    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    DispatchCall(call, bciArgs, nStateVec.get(), maxQPower, values);

    // At the end, just as a convenience, we return the expectation value for the addition result.
    double prob, average;
    bitCapInt i, outputInt;
    for (i = 0; i < maxQPower; i++) {
        outputInt = (i & outputMask) >> outputStart;
        prob = norm(nStateVec[i]);
        average += prob * outputInt;
    }

    // Finally, we dealloc the old state vector and replace it with the one we just calculated.
    ResetStateVec(std::move(nStateVec));

    // Return the expectation value.
    return (unsigned char)(average + 0.5);
}

/** Add based on an indexed load from classical memory */
unsigned char QUnitOCL::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    OpSuperposeReg8(clObj->GetADC8Ptr(), 0, inputStart, outputStart, carryIndex, values);
}

/** Subtract based on an indexed load from classical memory */
unsigned char QUnitOCL::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    OpSuperposeReg8(clObj->GetSBC8Ptr(), 1, inputStart, outputStart, carryIndex, values);
}

} // namespace Qrack
