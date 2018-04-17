//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
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

namespace Qrack {

/// "Qrack::OCLEngine" manages the single OpenCL context

// Public singleton methods to get pointers to various methods
cl::Context* OCLEngine::GetContextPtr() { return &context; }
cl::CommandQueue* OCLEngine::GetQueuePtr() { return &queue; }
cl::Kernel* OCLEngine::GetApply2x2Ptr() { return &apply2x2; }
cl::Kernel* OCLEngine::GetROLPtr() { return &rol; }
cl::Kernel* OCLEngine::GetRORPtr() { return &ror; }
cl::Kernel* OCLEngine::GetINCCPtr() { return &incc; }
cl::Kernel* OCLEngine::GetDECCPtr() { return &decc; }
cl::Kernel* OCLEngine::GetSR8Ptr() { return &superposeReg8; }
cl::Kernel* OCLEngine::GetADC8Ptr() { return &adcReg8; }
cl::Kernel* OCLEngine::GetSBC8Ptr() { return &sbcReg8; }

OCLEngine::OCLEngine() { InitOCL(0, 0); }
OCLEngine::OCLEngine(int plat, int dev) { InitOCL(plat, dev); }
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

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

    // use device[1] because that's a GPU; device[0] is the CPU
    default_device = all_devices[dev];
    std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

    // a context is like a "runtime link" to the device and platform;
    // i.e. communication is possible
    context = cl::Context({ default_device });

    // create the program that we want to execute on the device
    cl::Program::Sources sources;

    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

    program = cl::Program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
        exit(1);
    }

    queue = cl::CommandQueue(context, default_device);
    apply2x2 = cl::Kernel(program, "apply2x2");
    rol = cl::Kernel(program, "rol");
    ror = cl::Kernel(program, "ror");
    incc = cl::Kernel(program, "incc");
    decc = cl::Kernel(program, "decc");
    superposeReg8 = cl::Kernel(program, "superposeReg8");
    adcReg8 = cl::Kernel(program, "adcReg8");
    sbcReg8 = cl::Kernel(program, "sbcReg8");
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

} // namespace Qrack
