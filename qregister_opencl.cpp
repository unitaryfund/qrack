//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <iostream>

#include "qregister.hpp"
#include "qregister_opencl.hpp"

#include "oclengine.hpp"

namespace Qrack {

/* Modified constructors with the addition of InitOCL(). */
CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount)
    : CoherentUnit(qBitCount)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount, Complex16 phaseFac)
    : CoherentUnit(qBitCount, phaseFac)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, rgp)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState)
    : CoherentUnit(qBitCount, initState)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac)
    : CoherentUnit(qBitCount, initState, phaseFac)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(
    bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, initState, rgp)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(
    bitLenInt qBitCount, bitCapInt initState, Complex16 phaseFac, std::shared_ptr<std::default_random_engine> rgp)
    : CoherentUnit(qBitCount, initState, phaseFac, rgp)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(const CoherentUnitOCL& pqs)
    : CoherentUnit(pqs)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(const CoherentUnit& pqs)
    : CoherentUnit(pqs)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL()
    : CoherentUnit()
{
}

void CoherentUnitOCL::InitOCL()
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

void CoherentUnitOCL::ReInitOCL()
{
    clObj = OCLEngine::Instance();

    queue = *(clObj->GetQueuePtr());
    cl::Context context = *(clObj->GetContextPtr());

    // create buffers on device (allocate space on GPU)
    stateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(stateVec[0]));

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
}

void CoherentUnitOCL::ResetStateVec(std::unique_ptr<Complex16[]> nStateVec)
{
    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    CoherentUnit::ResetStateVec(std::move(nStateVec));
    ReInitOCL();
}

void CoherentUnitOCL::Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
    const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm)
{
    Complex16 cmplx[5];
    for (int i = 0; i < 4; i++) {
        cmplx[i] = mtrx[i];
    }
    cmplx[4] = Complex16(doApplyNorm ? (1.0 / runningNorm) : 1.0, 0.0);
    bitCapInt ulong[10] = { bitCount, maxQPower, offset1, offset2, 0, 0, 0, 0, 0, 0 };
    for (int i = 0; i < bitCount; i++) {
        ulong[4 + i] = qPowersSorted[i];
    }

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(cmplxBuffer, CL_FALSE, 0, sizeof(Complex16) * 5, cmplx);
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, ulong);

    cl::Kernel apply2x2 = *(clObj->GetApply2x2Ptr());
    queue.finish();
    apply2x2.setArg(0, stateBuffer);
    apply2x2.setArg(1, cmplxBuffer);
    apply2x2.setArg(2, ulongBuffer);
    queue.enqueueNDRangeKernel(apply2x2, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(stateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    if (doCalcNorm) {
        UpdateRunningNorm();
    } else {
        runningNorm = 1.0;
    }
}

/// "Circular shift left" - shift bits left, and carry last bits.
void CoherentUnitOCL::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);
    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel rol = *(clObj->GetROLPtr());
    rol.setArg(0, stateBuffer);
    rol.setArg(1, ulongBuffer);
    rol.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(rol, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(std::move(nStateVec));
}

/// "Circular shift right" - shift bits right, and carry first bits.
void CoherentUnitOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    bitCapInt otherMask = (maxQPower - 1) & (~regMask);

    bitCapInt bciArgs[10] = { maxQPower, regMask, otherMask, lengthPower, start, shift, length, 0, 0, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel ror = *(clObj->GetRORPtr());
    ror.setArg(0, stateBuffer);
    ror.setArg(1, ulongBuffer);
    ror.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(ror, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(std::move(nStateVec));
}

/// Add integer (without sign, with carry)
void CoherentUnitOCL::INCC(
    bitCapInt toAdd, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
        toAdd++;
    }
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inOutMask | carryMask));

    bitCapInt bciArgs[10] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, carryMask, inOutStart, toAdd, 0, 0,
        0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel incc = *(clObj->GetINCCPtr());
    incc.setArg(0, stateBuffer);
    incc.setArg(1, ulongBuffer);
    incc.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(incc, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(std::move(nStateVec));
}

/// Subtract integer (without sign, with carry)
void CoherentUnitOCL::DECC(
    bitCapInt toSub, const bitLenInt inOutStart, const bitLenInt length, const bitLenInt carryIndex)
{
    bool hasCarry = M(carryIndex);
    if (hasCarry) {
        X(carryIndex);
    } else {
        toSub++;
    }
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt lengthPower = 1 << length;
    bitCapInt inOutMask = (lengthPower - 1) << inOutStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inOutMask | carryMask));

    bitCapInt bciArgs[10] = { maxQPower >> 1, inOutMask, otherMask, lengthPower, carryMask, inOutStart, toSub, 0, 0,
        0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel decc = *(clObj->GetDECCPtr());
    decc.setArg(0, stateBuffer);
    decc.setArg(1, ulongBuffer);
    decc.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(decc, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(std::move(nStateVec));
}

/// Set 8 bit register bits based on read from classical memory
unsigned char CoherentUnitOCL::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    SetReg(outputStart, 8, 0);
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt bciArgs[10] = { maxQPower >> 8, inputStart, inputMask, outputStart, 0, 0, 0, 0, 0, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    queue.enqueueWriteBuffer(loadBuffer, CL_FALSE, 0, sizeof(unsigned char) * 256, values);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel sr8 = *(clObj->GetSR8Ptr());
    sr8.setArg(0, stateBuffer);
    sr8.setArg(1, ulongBuffer);
    sr8.setArg(2, nStateBuffer);
    sr8.setArg(3, loadBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(sr8, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);

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

/// Add based on an indexed load from classical memory
unsigned char CoherentUnitOCL::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    // The carry has to first to be measured for its input value.
    bitCapInt carryIn = 0;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 1;
        X(carryIndex);
    }

    bitCapInt lengthPower = 1 << 8;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask));
    bitCapInt bciArgs[10] = { maxQPower >> 1, inputStart, inputMask, outputStart, outputMask, otherMask, carryIn,
        carryMask, lengthPower, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    queue.enqueueWriteBuffer(loadBuffer, CL_FALSE, 0, sizeof(unsigned char) * 256, values);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel adc8 = *(clObj->GetADC8Ptr());
    adc8.setArg(0, stateBuffer);
    adc8.setArg(1, ulongBuffer);
    adc8.setArg(2, nStateBuffer);
    adc8.setArg(3, loadBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(adc8, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);

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

/// Subtract based on an indexed load from classical memory
unsigned char CoherentUnitOCL::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    // The carry has to first to be measured for its input value.
    bitCapInt carryIn = 1;
    if (M(carryIndex)) {
        // If the carry is set, we carry 1 in. We always initially clear the carry after testing for carry in.
        carryIn = 0;
        X(carryIndex);
    }

    bitCapInt lengthPower = 1 << 8;
    bitCapInt carryMask = 1 << carryIndex;
    bitCapInt inputMask = 0xff << inputStart;
    bitCapInt outputMask = 0xff << outputStart;
    bitCapInt otherMask = (maxQPower - 1) & (~(inputMask | outputMask));
    bitCapInt bciArgs[10] = { maxQPower >> 1, inputStart, inputMask, outputStart, outputMask, otherMask, carryIn,
        carryMask, lengthPower, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    queue.enqueueWriteBuffer(loadBuffer, CL_FALSE, 0, sizeof(unsigned char) * 256, values);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    std::fill(&(nStateVec[0]), &(nStateVec[0]) + maxQPower, Complex16(0.0, 0.0));
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel sbc8 = *(clObj->GetSBC8Ptr());
    sbc8.setArg(0, stateBuffer);
    sbc8.setArg(1, ulongBuffer);
    sbc8.setArg(2, nStateBuffer);
    sbc8.setArg(3, loadBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(sbc8, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);

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

} // namespace Qrack
