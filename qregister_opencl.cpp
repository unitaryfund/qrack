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

#include "par_for.hpp"

namespace Qrack {

/* Modified constructors with the addition of InitOCL(). */
CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount)
    : CoherentUnit(qBitCount)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(bitLenInt qBitCount, bitCapInt initState)
    : CoherentUnit(qBitCount, initState)
{
    InitOCL();
}

CoherentUnitOCL::CoherentUnitOCL(const CoherentUnitOCL& pqs)
    : CoherentUnit(pqs)
{
    InitOCL();
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

void CoherentUnitOCL::ResetStateVec(std::unique_ptr<Complex16[]>& nStateVec)
{
    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    CoherentUnit::ResetStateVec(nStateVec);
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
    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;
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
    ResetStateVec(nStateVec);
}

/// "Circular shift right" - shift bits right, and carry first bits.
void CoherentUnitOCL::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    bitCapInt regMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitCapInt i;
    for (i = 0; i < length; i++) {
        regMask += 1 << (start + i);
    }
    otherMask -= regMask;
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
    ResetStateVec(nStateVec);
}

/// Add two quantum integers
/** Add integer of "length" bits in "inStart" to integer of "length" bits in "inOutStart," and store result in
 * "inOutStart." */
void CoherentUnitOCL::ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length)
{
    bitCapInt inOutMask = 0;
    bitCapInt inMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitLenInt i;
    for (i = 0; i < length; i++) {
        inOutMask += 1 << (inOutStart + i);
        inMask += 1 << (inStart + i);
    }
    otherMask -= inOutMask + inMask;
    bitCapInt bciArgs[10] = { maxQPower, inOutMask, inMask, otherMask, lengthPower, inOutStart, inStart, 0, 0, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel add = *(clObj->GetADDPtr());
    add.setArg(0, stateBuffer);
    add.setArg(1, ulongBuffer);
    add.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(add, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(nStateVec);
}

/// Subtract two quantum integers
/** Subtract integer of "length" bits in "toSub" from integer of "length" bits in "inOutStart," and store result in
 * "inOutStart." */
void CoherentUnitOCL::SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length)
{
    bitCapInt inOutMask = 0;
    bitCapInt inMask = 0;
    bitCapInt otherMask = (1 << qubitCount) - 1;
    bitCapInt lengthPower = 1 << length;
    bitLenInt i;
    for (i = 0; i < length; i++) {
        inOutMask += 1 << (inOutStart + i);
        inMask += 1 << (toSub + i);
    }
    otherMask -= inOutMask + inMask;
    bitCapInt bciArgs[10] = { maxQPower, inOutMask, inMask, otherMask, lengthPower, inOutStart, toSub, 0, 0, 0 };

    queue.enqueueUnmapMemObject(stateBuffer, &(stateVec[0]));
    queue.enqueueWriteBuffer(ulongBuffer, CL_FALSE, 0, sizeof(bitCapInt) * 10, bciArgs);
    std::unique_ptr<Complex16[]> nStateVec(new Complex16[maxQPower]);
    cl::Context context = *(clObj->GetContextPtr());
    cl::Buffer nStateBuffer =
        cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(Complex16) * maxQPower, &(nStateVec[0]));
    cl::Kernel sub = *(clObj->GetSUBPtr());
    sub.setArg(0, stateBuffer);
    sub.setArg(1, ulongBuffer);
    sub.setArg(2, nStateBuffer);
    queue.finish();

    queue.enqueueNDRangeKernel(sub, cl::NullRange, // kernel, offset
        cl::NDRange(CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE), // global number of work items
        cl::NDRange(1)); // local number (per group)

    queue.enqueueMapBuffer(nStateBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(Complex16) * maxQPower);
    ResetStateVec(nStateVec);
}

} // namespace Qrack
