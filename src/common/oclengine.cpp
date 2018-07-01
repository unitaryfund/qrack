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
#include <memory>

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
cl::Context* OCLEngine::GetContextPtr(CommandQueuePtr cqp) { return &(all_contexts[PickQueue(cqp)]); }
CommandQueuePtr OCLEngine::GetQueuePtr(const int& dev) { return queue[(dev < 0) ? default_device_id : dev]; }
MutexPtr OCLEngine::GetMutexPtr(CommandQueuePtr cqp) { return (overload ? all_mutexes[PickQueue(cqp)] : nullptr); }
cl::Kernel* OCLEngine::GetApply2x2Ptr(CommandQueuePtr cqp) { return &(apply2x2[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetApply2x2NormPtr(CommandQueuePtr cqp) { return &(apply2x2norm[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetCoherePtr(CommandQueuePtr cqp) { return &(cohere[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetDecohereProbPtr(CommandQueuePtr cqp) { return &(decohereprob[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetDisposeProbPtr(CommandQueuePtr cqp) { return &(disposeprob[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetDecohereAmpPtr(CommandQueuePtr cqp) { return &(decohereamp[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetProbPtr(CommandQueuePtr cqp) { return &(prob[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetXPtr(CommandQueuePtr cqp) { return &(x[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetSwapPtr(CommandQueuePtr cqp) { return &(swap[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetROLPtr(CommandQueuePtr cqp) { return &(rol[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetRORPtr(CommandQueuePtr cqp) { return &(ror[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetINCPtr(CommandQueuePtr cqp) { return &(inc[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetDECPtr(CommandQueuePtr cqp) { return &(dec[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetINCCPtr(CommandQueuePtr cqp) { return &(incc[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetDECCPtr(CommandQueuePtr cqp) { return &(decc[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetLDAPtr(CommandQueuePtr cqp) { return &(indexedLda[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetADCPtr(CommandQueuePtr cqp) { return &(indexedAdc[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetSBCPtr(CommandQueuePtr cqp) { return &(indexedSbc[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetNormalizePtr(CommandQueuePtr cqp) { return &(normalize[PickQueue(cqp)]); }
cl::Kernel* OCLEngine::GetUpdateNormPtr(CommandQueuePtr cqp) { return &(updatenorm[PickQueue(cqp)]); }

OCLEngine::OCLEngine() { InitOCL(0, -1); }
OCLEngine::OCLEngine(int plat, int dev, bool singleNodeOverload) { overload = singleNodeOverload; InitOCL(plat, dev); }
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

CommandQueuePtr OCLEngine::PickQueue(CommandQueuePtr cqp)
{
    if (cqp == nullptr) {
        return defaultQueue;
    } else {
        return cqp;
    }
}

void OCLEngine::InitOCL(int plat, int dev)
{
    int i;
    // get all platforms (drivers), e.g. NVIDIA

    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    default_platform = all_platforms[plat];

    // get default device (CPUs, GPUs) of the default platform
    for (int i = 0; i < (int)all_platforms.size(); i++) {
        std::vector<cl::Device> platform_devices;
        all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
        all_devices.insert(all_devices.end(), platform_devices.begin(), platform_devices.end());
    }
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    // our algorithm relies on the device count being a power of 2
    deviceCount = PowerOf2LessThan(all_devices.size());

    if ((dev < 0) || (dev >= deviceCount)) {
        // prefer device[1] because that's usually a GPU or accelerator; device[0] is usually the CPU
        // also make sure that the default device is in our node list
        dev = deviceCount - 1;
    }
    default_device_id = dev;
    default_device = all_devices[dev];

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

#if ENABLE_COMPLEX8
    sources.push_back({ (const char*)qheader_float_cl, (long unsigned int)qheader_float_cl_len });
#else
    sources.push_back({ (const char*)qheader_double_cl, (long unsigned int)qheader_double_cl_len });
#endif
    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

    for (int i = 0; i < deviceCount; i++) {
        // a context is like a "runtime link" to the device and platform;
        // i.e. communication is possible
        cl::Context context = cl::Context(all_devices[i]);
        queue.push_back(std::make_shared<cl::CommandQueue>(cl::CommandQueue(context, all_devices[i])));
        all_contexts[queue[i]] = context;
        all_mutexes[queue[i]] = std::make_shared<std::recursive_mutex>();
        if (i == dev) {
            defaultQueue = queue[i];
        }
        programs[queue[i]] = cl::Program(context, sources);
        cl::Program program = programs[queue[i]];

        if (program.build({ all_devices[i] }) != CL_SUCCESS) {
            std::cout << "Error building for device #" << i << ": "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[i]) << std::endl;
            exit(1);
        }

        apply2x2[queue[i]] = cl::Kernel(program, "apply2x2");
        apply2x2norm[queue[i]] = cl::Kernel(program, "apply2x2norm");
        x[queue[i]] = cl::Kernel(program, "x");
        cohere[queue[i]] = cl::Kernel(program, "cohere");
        decohereprob[queue[i]] = cl::Kernel(program, "decohereprob");
        decohereamp[queue[i]] = cl::Kernel(program, "decohereamp");
        disposeprob[queue[i]] = cl::Kernel(program, "disposeprob");
        prob[queue[i]] = cl::Kernel(program, "prob");
        swap[queue[i]] = cl::Kernel(program, "swap");
        rol[queue[i]] = cl::Kernel(program, "rol");
        ror[queue[i]] = cl::Kernel(program, "ror");
        inc[queue[i]] = cl::Kernel(program, "inc");
        dec[queue[i]] = cl::Kernel(program, "dec");
        incc[queue[i]] = cl::Kernel(program, "incc");
        decc[queue[i]] = cl::Kernel(program, "decc");
        indexedLda[queue[i]] = cl::Kernel(program, "indexedLda");
        indexedAdc[queue[i]] = cl::Kernel(program, "indexedAdc");
        indexedSbc[queue[i]] = cl::Kernel(program, "indexedSbc");
        normalize[queue[i]] = cl::Kernel(program, "nrmlze");
        updatenorm[queue[i]] = cl::Kernel(program, "updatenorm");
    }

    // For VirtualCL support, the device info can only be accessed AFTER all contexts are created.
    std::cout << "Default platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Default device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    for (i = 0; i < deviceCount; i++) {
        std::cout << "OpenCL device #" << i << ": " << all_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
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

    if (number <= 1)
        return number;

    while (number != 0) {
        number >>= 1;
        count++;
    }

    return (1 << (count - 1));
}

} // namespace Qrack
