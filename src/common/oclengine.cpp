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
void OCLEngine::SetDeviceContextPtrVector(std::vector<DeviceContextPtr> vec, DeviceContextPtr dcp)
{
    all_device_contexts = vec;
    if (dcp != nullptr) {
        default_device_context = dcp;
    }
}

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
    std::vector<int> device_platform_id;
    cl::Platform default_platform;
    cl::Device default_device;

    cl::Platform::get(&all_platforms);

    if (all_platforms.size() == 0) {
        std::cout << " No platforms found. Check OpenCL installation!\n";
        exit(1);
    }

    // get all devices
    std::vector<cl::Platform> devPlatVec;
    std::vector<std::vector<cl::Device>> all_platforms_devices;
    for (size_t i = 0; i < all_platforms.size(); i++) {
        all_platforms_devices.push_back(std::vector<cl::Device>());
        all_platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &(all_platforms_devices[i]));
        for (size_t j = 0; j < all_platforms_devices[i].size(); j++) {
            // VirtualCL seems to break if the assignment constructor of cl::Platform is used here from the original
            // list. Assigning the object from a new query is always fine, though. (They carry the same underlying
            // platform IDs.)
            std::vector<cl::Platform> temp_platforms;
            cl::Platform::get(&temp_platforms);
            devPlatVec.push_back(temp_platforms[i]);
            device_platform_id.push_back(i);
        }
        all_devices.insert(all_devices.end(), all_platforms_devices[i].begin(), all_platforms_devices[i].end());
    }
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }

    int deviceCount = all_devices.size();

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

    int plat_id = -1;
    std::vector<cl::Context> all_contexts;
    for (int i = 0; i < deviceCount; i++) {
        // a context is like a "runtime link" to the device and platform;
        // i.e. communication is possible
        if (device_platform_id[i] != plat_id) {
            plat_id = device_platform_id[i];
            all_contexts.push_back(cl::Context(all_platforms_devices[plat_id]));
        }
        std::shared_ptr<OCLDeviceContext> devCntxt = std::make_shared<OCLDeviceContext>(
            devPlatVec[i], all_devices[i], all_contexts[all_contexts.size() - 1], plat_id);

        cl::Program program = cl::Program(devCntxt->context, sources);

        cl_int buildError = program.build({ all_devices[i] });
        if (buildError != CL_SUCCESS) {
            std::cout << "Error building for device #" << i << ": " << buildError << ", "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(all_devices[i])
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[i]) << std::endl;

            // The default device was set above to be the last device in the list. If we can't compile for it, we use
            // the first device. If the default is the first device, and we can't compile for it, then we don't have any
            // devices that can compile at all, and the environment needs to be fixed by the user.
            if (i == dev) {
                default_device_context = all_device_contexts[0];
                default_platform = all_platforms[0];
                default_device = all_devices[0];
            }

            continue;
        }

        all_device_contexts.push_back(devCntxt);

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
        all_device_contexts[i]->calls[OCL_API_INCS] = cl::Kernel(program, "incs");
        all_device_contexts[i]->calls[OCL_API_DECS] = cl::Kernel(program, "decs");
        all_device_contexts[i]->calls[OCL_API_INCSC_1] = cl::Kernel(program, "incsc1");
        all_device_contexts[i]->calls[OCL_API_DECSC_1] = cl::Kernel(program, "decsc1");
        all_device_contexts[i]->calls[OCL_API_INCSC_2] = cl::Kernel(program, "incsc2");
        all_device_contexts[i]->calls[OCL_API_DECSC_2] = cl::Kernel(program, "decsc2");
        all_device_contexts[i]->calls[OCL_API_INCBCD] = cl::Kernel(program, "incbcd");
        all_device_contexts[i]->calls[OCL_API_DECBCD] = cl::Kernel(program, "decbcd");
        all_device_contexts[i]->calls[OCL_API_INCBCDC] = cl::Kernel(program, "incbcdc");
        all_device_contexts[i]->calls[OCL_API_DECBCDC] = cl::Kernel(program, "decbcdc");
        all_device_contexts[i]->calls[OCL_API_INDEXEDLDA] = cl::Kernel(program, "indexedLda");
        all_device_contexts[i]->calls[OCL_API_INDEXEDADC] = cl::Kernel(program, "indexedAdc");
        all_device_contexts[i]->calls[OCL_API_INDEXEDSBC] = cl::Kernel(program, "indexedSbc");
        all_device_contexts[i]->calls[OCL_API_NORMALIZE] = cl::Kernel(program, "nrmlze");
        all_device_contexts[i]->calls[OCL_API_UPDATENORM] = cl::Kernel(program, "updatenorm");
        all_device_contexts[i]->calls[OCL_API_APPLYM] = cl::Kernel(program, "applym");
        all_device_contexts[i]->calls[OCL_API_PHASEFLIP] = cl::Kernel(program, "phaseflip");
        all_device_contexts[i]->calls[OCL_API_ZEROPHASEFLIP] = cl::Kernel(program, "zerophaseflip");
        all_device_contexts[i]->calls[OCL_API_CPHASEFLIPIFLESS] = cl::Kernel(program, "cphaseflipifless");
        all_device_contexts[i]->calls[OCL_API_MUL] = cl::Kernel(program, "mul");
        all_device_contexts[i]->calls[OCL_API_DIV] = cl::Kernel(program, "div");
        all_device_contexts[i]->calls[OCL_API_CMUL] = cl::Kernel(program, "cmul");
        all_device_contexts[i]->calls[OCL_API_CDIV] = cl::Kernel(program, "cdiv");

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
