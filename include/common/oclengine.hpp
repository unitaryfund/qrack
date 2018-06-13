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

#include <map>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Qrack {
    
    typedef std::shared_ptr<cl::CommandQueue> CommandQueuePtr;

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
    cl::Kernel* GetApply2x2Ptr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Apply2x2Norm function kernel
    cl::Kernel* GetApply2x2NormPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetCoherePtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Decohere probability/angle decompose function kernel
    cl::Kernel* GetDecohereProbPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Decohere amplitude compose function kernel
    cl::Kernel* GetDecohereAmpPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Dispose probability/angle decompose function kernel
    cl::Kernel* GetDisposeProbPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Cohere function kernel
    cl::Kernel* GetProbPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the X function kernel
    cl::Kernel* GetXPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the Swap function kernel
    cl::Kernel* GetSwapPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the ROL function kernel
    cl::Kernel* GetROLPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the ROR function kernel
    cl::Kernel* GetRORPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the INC function kernel
    cl::Kernel* GetINCPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the DEC function kernel
    cl::Kernel* GetDECPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the INCC function kernel
    cl::Kernel* GetINCCPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the DECC function kernel
    cl::Kernel* GetDECCPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the IndexedLDA function kernel
    cl::Kernel* GetLDAPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the IndexedADC function kernel
    cl::Kernel* GetADCPtr(CommandQueuePtr cqp = nullptr);
    /// Get a pointer to the IndexedSBC function kernel
    cl::Kernel* GetSBCPtr(CommandQueuePtr cqp = nullptr);
    
private:
    int nodeCount;
    int default_device_id;
    CommandQueuePtr defaultQueue;

    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform;
    std::vector<cl::Device> all_devices;
    cl::Device default_device;
    std::vector<cl::Device> cluster_devices;
    cl::Context context;
    std::map<CommandQueuePtr, cl::Program> programs;
    std::vector<CommandQueuePtr> queue;
    std::map<CommandQueuePtr, cl::Kernel> apply2x2;
    std::map<CommandQueuePtr, cl::Kernel> apply2x2norm;
    std::map<CommandQueuePtr, cl::Kernel> cohere;
    std::map<CommandQueuePtr, cl::Kernel> decohereprob;
    std::map<CommandQueuePtr, cl::Kernel> decohereamp;
    std::map<CommandQueuePtr, cl::Kernel> disposeprob;
    std::map<CommandQueuePtr, cl::Kernel> prob;
    std::map<CommandQueuePtr, cl::Kernel> x;
    std::map<CommandQueuePtr, cl::Kernel> swap;
    std::map<CommandQueuePtr, cl::Kernel> rol;
    std::map<CommandQueuePtr, cl::Kernel> ror;
    std::map<CommandQueuePtr, cl::Kernel> inc;
    std::map<CommandQueuePtr, cl::Kernel> dec;
    std::map<CommandQueuePtr, cl::Kernel> incc;
    std::map<CommandQueuePtr, cl::Kernel> decc;
    std::map<CommandQueuePtr, cl::Kernel> indexedLda;
    std::map<CommandQueuePtr, cl::Kernel> indexedAdc;
    std::map<CommandQueuePtr, cl::Kernel> indexedSbc;

    OCLEngine(); // Private so that it can  not be called
    OCLEngine(int plat, int dev); // Private so that it can  not be called
    OCLEngine(OCLEngine const&); // copy constructor is private
    OCLEngine& operator=(OCLEngine const& rhs); // assignment operator is private
    static OCLEngine* m_pInstance;

    void InitOCL(int plat, int dev);

    unsigned long PowerOf2LessThan(unsigned long number);
    CommandQueuePtr PickQueue(CommandQueuePtr);
};

} // namespace Qrack
