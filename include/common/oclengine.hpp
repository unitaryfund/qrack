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

/** "Qrack::OCLEngine" manages the single OpenCL context. */
class OCLEngine {
public:
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static OCLEngine* Instance();
    /// If this is the first time instantiating the OpenCL context, you may specify platform number and device number.
    static OCLEngine* Instance(int plat, int dev);
    /// Get a pointer to the OpenCL context
    cl::Context* GetContextPtr();
    /// Get a pointer to the OpenCL queue
    cl::CommandQueue* GetQueuePtr(const int& dev = -1);
    /// Get a pointer to the Apply2x2 function kernel
    cl::Kernel* GetApply2x2Ptr();
    /// Get a pointer to the Apply2x2Norm function kernel
    cl::Kernel* GetApply2x2NormPtr();
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetCoherePtr();
    /// Get a pointer to the Decohere probability/angle decompose function kernel
    cl::Kernel* GetDecohereProbPtr();
    /// Get a pointer to the Decohere amplitude compose function kernel
    cl::Kernel* GetDecohereAmpPtr();
    /// Get a pointer to the Dispose probability/angle decompose function kernel
    cl::Kernel* GetDisposeProbPtr();
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetProbPtr();
    /// Get a pointer to the X function kernel
    cl::Kernel* GetXPtr();
    /// Get a pointer to the Swap function kernel
    cl::Kernel* GetSwapPtr();
    /// Get a pointer to the ROL function kernel
    cl::Kernel* GetROLPtr();
    /// Get a pointer to the ROR function kernel
    cl::Kernel* GetRORPtr();
    /// Get a pointer to the INC function kernel
    cl::Kernel* GetINCPtr();
    /// Get a pointer to the DEC function kernel
    cl::Kernel* GetDECPtr();
    /// Get a pointer to the INCC function kernel
    cl::Kernel* GetINCCPtr();
    /// Get a pointer to the DECC function kernel
    cl::Kernel* GetDECCPtr();
    /// Get a pointer to the IndexedLDA function kernel
    cl::Kernel* GetLDAPtr();
    /// Get a pointer to the IndexedADC function kernel
    cl::Kernel* GetADCPtr();
    /// Get a pointer to the IndexedSBC function kernel
    cl::Kernel* GetSBCPtr();
    
private:
    int nodeCount;
    int defaultDevIndex;

    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform;
    std::vector<cl::Device> all_devices;
    cl::Device default_device;
    std::vector<cl::Device> cluster_devices;
    cl::Context context;
    cl::Program program;
    std::vector<cl::CommandQueue> queue;
    cl::Kernel apply2x2;
    cl::Kernel apply2x2norm;
    cl::Kernel cohere;
    cl::Kernel decohereprob;
    cl::Kernel decohereamp;
    cl::Kernel disposeprob;
    cl::Kernel prob;
    cl::Kernel x;
    cl::Kernel swap;
    cl::Kernel rol;
    cl::Kernel ror;
    cl::Kernel inc;
    cl::Kernel dec;
    cl::Kernel incc;
    cl::Kernel decc;
    cl::Kernel indexedLda;
    cl::Kernel indexedAdc;
    cl::Kernel indexedSbc;

    OCLEngine(); // Private so that it can  not be called
    OCLEngine(int plat, int dev); // Private so that it can  not be called
    OCLEngine(OCLEngine const&); // copy constructor is private
    OCLEngine& operator=(OCLEngine const& rhs); // assignment operator is private
    static OCLEngine* m_pInstance;

    void InitOCL(int plat, int dev);

    unsigned long PowerOf2LessThan(unsigned long number);
    int PickIndex(const int& arg);
};

} // namespace Qrack
