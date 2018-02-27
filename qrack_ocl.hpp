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

#pragma once

#if !ENABLE_OPENCL
#error OpenCL has not been enabled
#endif

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Qrack {

class OCLSingleton;

class CoherentUnitOCL : public CoherentUnit {
public:
    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ROR(bitLenInt shift, bitLenInt start, bitLenInt length);
    virtual void ADD(const bitLenInt inOutStart, const bitLenInt inStart, const bitLenInt length);
    virtual void SUB(const bitLenInt inOutStart, const bitLenInt toSub, const bitLenInt length);

protected:
    OCLSingleton* clObj;
    cl::CommandQueue queue;
    cl::Buffer stateBuffer;
    cl::Buffer cmplxBuffer;
    cl::Buffer ulongBuffer;
    cl::Buffer nrmBuffer;
    cl::Buffer maxBuffer;

    virtual void InitOCL();
    virtual void ReInitOCL();
    virtual void ResetStateVec(std::unique_ptr<Complex16[]> &nStateVec);

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const Complex16* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doApplyNorm, bool doCalcNorm);

};

/** "Qrack::OCLSingleton" manages the single OpenCL context. */
class OCLSingleton {
public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLSingleton* Instance();
    /// If this is the first time instantiating the OpenCL context, you may specify platform number and device number.
    static OCLSingleton* Instance(int plat, int dev);
    /// Get a pointer to the OpenCL context
    cl::Context* GetContextPtr();
    /// Get a pointer to the OpenCL queue
    cl::CommandQueue* GetQueuePtr();
    /// Get a pointer to the Apply2x2 function kernel
    cl::Kernel* GetApply2x2Ptr();
    /// Get a pointer to the ROL function kernel
    cl::Kernel* GetROLPtr();
    /// Get a pointer to the ROR function kernel
    cl::Kernel* GetRORPtr();
    /// Get a pointer to the ADD function kernel
    cl::Kernel* GetADDPtr();
    /// Get a pointer to the SUB function kernel
    cl::Kernel* GetSUBPtr();
    /// Get a pointer to the ADDBCD function kernel
    cl::Kernel* GetADDBCDPtr();
    /// Get a pointer to the SUBBCD function kernel
    cl::Kernel* GetSUBBCDPtr();
    /// Get a pointer to the ADDC function kernel
    cl::Kernel* GetADDCPtr();
    /// Get a pointer to the SUBC function kernel
    cl::Kernel* GetSUBCPtr();
    /// Get a pointer to the ADDBCDC function kernel
    cl::Kernel* GetADDBCDCPtr();
    /// Get a pointer to the SUBBCDC function kernel
    cl::Kernel* GetSUBBCDCPtr();

private:
    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform;
    std::vector<cl::Device> all_devices;
    cl::Device default_device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
    cl::Kernel apply2x2;
    cl::Kernel rol;
    cl::Kernel ror;
    cl::Kernel add;
    cl::Kernel sub;
    cl::Kernel addbcd;
    cl::Kernel subbcd;
    cl::Kernel addc;
    cl::Kernel subc;
    cl::Kernel addbcdc;
    cl::Kernel subbcdc;

    OCLSingleton(); // Private so that it can  not be called
    OCLSingleton(int plat, int dev); // Private so that it can  not be called
    OCLSingleton(OCLSingleton const&); // copy constructor is private
    OCLSingleton& operator=(OCLSingleton const& rhs); // assignment operator is private
    static OCLSingleton* m_pInstance;

    void InitOCL(int plat, int dev);
};
}
