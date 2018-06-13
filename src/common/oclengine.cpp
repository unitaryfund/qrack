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

#include <iostream>

#include "oclengine.hpp"
#include "qenginecl.hpp"

#if ENABLE_COMPLEX8
#include "qheader_floatcl.hpp"
#else
#include "qheader_doublecl.hpp"
#endif

namespace Qrack {

/// "Qrack::OCLEngine" manages the single OpenCL context

// Public singleton methods to get pointers to various methods
cl::Context* OCLEngine::GetContextPtr() { return &context; }
cl::CommandQueue* OCLEngine::GetQueuePtr(const int& dev) { return &(queue[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetApply2x2Ptr(const int& dev) { return &(apply2x2[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetApply2x2NormPtr(const int& dev) { return &(apply2x2norm[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetCoherePtr(const int& dev) { return &(cohere[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetDecohereProbPtr(const int& dev) { return &(decohereprob[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetDisposeProbPtr(const int& dev) { return &(disposeprob[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetDecohereAmpPtr(const int& dev) { return &(decohereamp[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetProbPtr(const int& dev) { return &(prob[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetXPtr(const int& dev) { return &(x[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetSwapPtr(const int& dev) { return &(swap[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetROLPtr(const int& dev) { return &(rol[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetRORPtr(const int& dev) { return &(ror[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetINCPtr(const int& dev) { return &(inc[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetDECPtr(const int& dev) { return &(dec[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetINCCPtr(const int& dev) { return &(incc[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetDECCPtr(const int& dev) { return &(decc[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetLDAPtr(const int& dev) { return &(indexedLda[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetADCPtr(const int& dev) { return &(indexedAdc[PickIndex(dev)]); }
cl::Kernel* OCLEngine::GetSBCPtr(const int& dev) { return &(indexedSbc[PickIndex(dev)]); }

OCLEngine::OCLEngine() { InitOCL(0, -1); }
OCLEngine::OCLEngine(int plat, int dev) { InitOCL(plat, dev); }
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

int OCLEngine::PickIndex(const int& arg) {
    if (arg < 0) {
        return defaultDevIndex;
    } else {
        return arg;
    }
}

void OCLEngine::InitOCL(int plat, int dev)
{
    // get all platforms (drivers), e.g. NVIDIA

    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    default_platform = all_platforms[plat];
    std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

    // get default device (CPUs, GPUs) of the default platform
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // our algorithm relies on the device count being a power of 2
    nodeCount = PowerOf2LessThan(all_devices.size());

    if ((dev < 0) || (dev >= nodeCount)) {
        // prefer device[1] because that's usually a GPU or accelerator; device[0] is usually the CPU
        // also make sure that the default device is in our node list
        dev = nodeCount - 1;
    }
    defaultDevIndex = dev;
    default_device = all_devices[dev];
    std::cout << "Default device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    for (int i = 0; i < nodeCount; i++) {
        cluster_devices.push_back(all_devices[i]);
        std::cout << "Cluster device #"<<i<<": "<< all_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    context = cl::Context(cluster_devices);

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

#if ENABLE_COMPLEX8
    sources.push_back({ (const char*)qheader_float_cl, (long unsigned int)qheader_float_cl_len });
#else
    sources.push_back({ (const char*)qheader_double_cl, (long unsigned int)qheader_double_cl_len });
#endif
    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

    for (int i = 0; i < nodeCount; i++) {
        programs.push_back(cl::Program(context, sources));
    
        if (programs[i].build({cluster_devices[i]}) != CL_SUCCESS) {
            std::cout << "Error building for device #" << i <<": " << programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(cluster_devices[i]) << std::endl;
            exit(1);
        }

        queue.push_back(cl::CommandQueue(context, cluster_devices[i]));
    
        apply2x2.push_back(cl::Kernel(programs[i], "apply2x2"));
        apply2x2norm.push_back(cl::Kernel(programs[i], "apply2x2norm"));
        x.push_back(cl::Kernel(programs[i], "x"));
        cohere.push_back(cl::Kernel(programs[i], "cohere"));
        decohereprob.push_back(cl::Kernel(programs[i], "decohereprob"));
        decohereamp.push_back(cl::Kernel(programs[i], "decohereamp"));
        disposeprob.push_back(cl::Kernel(programs[i], "disposeprob"));
        prob.push_back(cl::Kernel(programs[i], "prob"));
        swap.push_back(cl::Kernel(programs[i], "swap"));
        rol.push_back(cl::Kernel(programs[i], "rol"));
        ror.push_back(cl::Kernel(programs[i], "ror"));
        inc.push_back(cl::Kernel(programs[i], "inc"));
        dec.push_back(cl::Kernel(programs[i], "dec"));
        incc.push_back(cl::Kernel(programs[i], "incc"));
        decc.push_back(cl::Kernel(programs[i], "decc"));
        indexedLda.push_back(cl::Kernel(programs[i], "indexedLda"));
        indexedAdc.push_back(cl::Kernel(programs[i], "indexedAdc"));
        indexedSbc.push_back(cl::Kernel(programs[i], "indexedSbc"));
    }
}

OCLEngine* OCLEngine::m_pInstance = NULL;
OCLEngine* OCLEngine::Instance()
{
    if (!m_pInstance)
        m_pInstance = new OCLEngine();
    return m_pInstance;
}

OCLEngine* OCLEngine::Instance(int plat, int dev)
{
    if (!m_pInstance) {
        m_pInstance = new OCLEngine(plat, dev);
    } else {
        std::cout << "Warning: Tried to reinitialize OpenCL environment with platform and device." << std::endl;
    }
    return m_pInstance;
}

unsigned long OCLEngine::PowerOf2LessThan(unsigned long number)
{
    unsigned long count = 0;

    if (number <= 1) return number;

    while(number != 0)
    {
        number >>= 1;
        count++;
    }

    return (1 << (count - 1));
}

} // namespace Qrack
