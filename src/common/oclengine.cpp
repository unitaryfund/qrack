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
DeviceContextPtr OCLEngine::GetDeviceContextPtr(const int& dev)
{
    if ((dev >= GetDeviceCount()) || (dev < -1)) {
        throw "Invalid OpenCL device selection";
    } else if (dev == -1) {
        return default_device_context;
    } else {
        return all_device_contexts[dev];
    }
}

std::vector<DeviceContextPtr> OCLEngine::GetDeviceContextPtrVector() { return all_device_contexts; }

void OCLEngine::SetDefaultDeviceContext(DeviceContextPtr dcp) { default_device_context = dcp; }

OCLEngine::OCLEngine() { InitOCL(); }
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

void OCLEngine::InitOCL()
{
    int i;
    // get all platforms (drivers), e.g. NVIDIA

    std::vector<cl::Platform> all_platforms;
    std::vector<cl::Device> all_devices;
    cl::Platform default_platform;
    cl::Device default_device;

    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    // get all devices
    std::vector<cl::Platform> devPlatVec;
    for (size_t i = 0; i < all_platforms.size(); i++) {
        std::vector<cl::Device> platform_devices;
        all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
        for (size_t j = 0; j < platform_devices.size(); j++) {
            devPlatVec.push_back(all_platforms[i]);
        }
        all_devices.insert(all_devices.end(), platform_devices.begin(), platform_devices.end());
    }
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    deviceCount = all_devices.size();

    // prefer the last device because that's usually a GPU or accelerator; device[0] is usually the CPU
    int dev = deviceCount - 1;

    // create the programs that we want to execute on the devices
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
        all_device_contexts.push_back(std::make_shared<OCLDeviceContext>(devPlatVec[i], all_devices[i]));
        all_device_contexts[i]->context = cl::Context(all_devices[i]);
        all_device_contexts[i]->queue = cl::CommandQueue(all_device_contexts[i]->context, all_devices[i]);

        cl::Program program = cl::Program(all_device_contexts[i]->context, sources);

        if (program.build({ all_devices[i] }) != CL_SUCCESS) {
            std::cout << "Error building for device #" << i << ": "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[i]) << std::endl;
            exit(1);
        }

        all_device_contexts[i]->calls[OCL_API_APPLY2X2] = cl::Kernel(program, "apply2x2");
        all_device_contexts[i]->calls[OCL_API_APPLY2X2_NORM] = cl::Kernel(program, "apply2x2norm");
        all_device_contexts[i]->calls[OCL_API_X] = cl::Kernel(program, "x");
        all_device_contexts[i]->calls[OCL_API_COHERE] = cl::Kernel(program, "cohere");
        all_device_contexts[i]->calls[OCL_API_DECOHEREPROB] = cl::Kernel(program, "decohereprob");
        all_device_contexts[i]->calls[OCL_API_DECOHEREAMP] = cl::Kernel(program, "decohereamp");
        all_device_contexts[i]->calls[OCL_API_DISPOSEPROB] = cl::Kernel(program, "disposeprob");
        all_device_contexts[i]->calls[OCL_API_PROB] = cl::Kernel(program, "prob");
        all_device_contexts[i]->calls[OCL_API_SWAP] = cl::Kernel(program, "swap");
        all_device_contexts[i]->calls[OCL_API_ROL] = cl::Kernel(program, "rol");
        all_device_contexts[i]->calls[OCL_API_ROR] = cl::Kernel(program, "ror");
        all_device_contexts[i]->calls[OCL_API_INC] = cl::Kernel(program, "inc");
        all_device_contexts[i]->calls[OCL_API_DEC] = cl::Kernel(program, "dec");
        all_device_contexts[i]->calls[OCL_API_INCC] = cl::Kernel(program, "incc");
        all_device_contexts[i]->calls[OCL_API_DECC] = cl::Kernel(program, "decc");
        all_device_contexts[i]->calls[OCL_API_INDEXEDLDA] = cl::Kernel(program, "indexedLda");
        all_device_contexts[i]->calls[OCL_API_INDEXEDADC] = cl::Kernel(program, "indexedAdc");
        all_device_contexts[i]->calls[OCL_API_INDEXEDSBC] = cl::Kernel(program, "indexedSbc");
        all_device_contexts[i]->calls[OCL_API_NORMALIZE] = cl::Kernel(program, "nrmlze");
        all_device_contexts[i]->calls[OCL_API_UPDATENORM] = cl::Kernel(program, "updatenorm");

        if (i == dev) {
            default_device_context = all_device_contexts[i];
            default_platform = all_platforms[i];
            default_device = all_devices[i];
        }
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

} // namespace Qrack
