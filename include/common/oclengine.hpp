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
    cl::Kernel* GetApply2x2Ptr(const int& dev = -1);
    /// Get a pointer to the Apply2x2Norm function kernel
    cl::Kernel* GetApply2x2NormPtr(const int& dev = -1);
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetCoherePtr(const int& dev = -1);
    /// Get a pointer to the Decohere probability/angle decompose function kernel
    cl::Kernel* GetDecohereProbPtr(const int& dev = -1);
    /// Get a pointer to the Decohere amplitude compose function kernel
    cl::Kernel* GetDecohereAmpPtr(const int& dev = -1);
    /// Get a pointer to the Dispose probability/angle decompose function kernel
    cl::Kernel* GetDisposeProbPtr(const int& dev = -1);
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetProbPtr(const int& dev = -1);
    /// Get a pointer to the X function kernel
    cl::Kernel* GetXPtr(const int& dev = -1);
    /// Get a pointer to the Swap function kernel
    cl::Kernel* GetSwapPtr(const int& dev = -1);
    /// Get a pointer to the ROL function kernel
    cl::Kernel* GetROLPtr(const int& dev = -1);
    /// Get a pointer to the ROR function kernel
    cl::Kernel* GetRORPtr(const int& dev = -1);
    /// Get a pointer to the INC function kernel
    cl::Kernel* GetINCPtr(const int& dev = -1);
    /// Get a pointer to the DEC function kernel
    cl::Kernel* GetDECPtr(const int& dev = -1);
    /// Get a pointer to the INCC function kernel
    cl::Kernel* GetINCCPtr(const int& dev = -1);
    /// Get a pointer to the DECC function kernel
    cl::Kernel* GetDECCPtr(const int& dev = -1);
    /// Get a pointer to the IndexedLDA function kernel
    cl::Kernel* GetLDAPtr(const int& dev = -1);
    /// Get a pointer to the IndexedADC function kernel
    cl::Kernel* GetADCPtr(const int& dev = -1);
    /// Get a pointer to the IndexedSBC function kernel
    cl::Kernel* GetSBCPtr(const int& dev = -1);

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
    std::vector<cl::Kernel> apply2x2;
    std::vector<cl::Kernel> apply2x2norm;
    std::vector<cl::Kernel> cohere;
    std::vector<cl::Kernel> decohereprob;
    std::vector<cl::Kernel> decohereamp;
    std::vector<cl::Kernel> disposeprob;
    std::vector<cl::Kernel> prob;
    std::vector<cl::Kernel> x;
    std::vector<cl::Kernel> swap;
    std::vector<cl::Kernel> rol;
    std::vector<cl::Kernel> ror;
    std::vector<cl::Kernel> inc;
    std::vector<cl::Kernel> dec;
    std::vector<cl::Kernel> incc;
    std::vector<cl::Kernel> decc;
    std::vector<cl::Kernel> indexedLda;
    std::vector<cl::Kernel> indexedAdc;
    std::vector<cl::Kernel> indexedSbc;

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
