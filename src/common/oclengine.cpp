//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "oclengine.hpp"

#if ENABLE_PURE32
#include "qheader32cl.hpp"
#elif ENABLE_COMPLEX8
#include "qheader_floatcl.hpp"
#else
#include "qheader_doublecl.hpp"
#endif

#include "qenginecl.hpp"

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

OCLEngine::OCLEngine()
{
    OCLInitResult res = InitOCL(false);
    all_device_contexts = res.all_device_contexts;
    default_device_context = res.default_device_context;
}
OCLEngine::OCLEngine(OCLEngine const&) {}
OCLEngine& OCLEngine::operator=(OCLEngine const& rhs) { return *this; }

OCLInitResult OCLEngine::InitOCL(bool saveBinaries)
{
#if defined(_WIN32) && !defined(__CYGWIN__)
    char* homeDrive = getenv("HOMEDRIVE");
    char* homePath = getenv("HOMEPATH");
    int newSize = strlen(homeDrive) + strlen(homePath) + 1;
    char* home = new char[newSize];
    strcpy(home, homeDrive);
    strcat(home, homePath);
    std::string homeStr(home);
    homeStr += "\\.qrack\\";
    delete[] home;
#else
    char* home = getenv("HOME");
    std::string homeStr(home);
    homeStr += "/.qrack/";
#endif

    OCLInitResult toRet;
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

#if ENABLE_PURE32
    sources.push_back({ (const char*)qheader32_cl, (long unsigned int)qheader32_cl_len });
#elif ENABLE_COMPLEX8
    sources.push_back({ (const char*)qheader_float_cl, (long unsigned int)qheader_float_cl_len });
#else
    sources.push_back({ (const char*)qheader_double_cl, (long unsigned int)qheader_double_cl_len });
#endif
    sources.push_back({ (const char*)qengine_cl, (long unsigned int)qengine_cl_len });

    int plat_id = -1;
    std::vector<cl::Context> all_contexts;
    std::vector<int> binaryStatus;
    cl_int buildError;
    for (int i = 0; i < deviceCount; i++) {
        // a context is like a "runtime link" to the device and platform;
        // i.e. communication is possible
        if (device_platform_id[i] != plat_id) {
            plat_id = device_platform_id[i];
            all_contexts.push_back(cl::Context(all_platforms_devices[plat_id]));
        }
        std::shared_ptr<OCLDeviceContext> devCntxt = std::make_shared<OCLDeviceContext>(
            devPlatVec[i], all_devices[i], all_contexts[all_contexts.size() - 1], plat_id);

        FILE* clBinFile;
        std::string clBinName = homeStr + "qrack_ocl_dev_" + std::to_string(i) + ".ir";
        cl::Program program;
        if (!saveBinaries && (clBinFile = fopen(clBinName.c_str(), "r"))) {
            long lSize;

            fseek(clBinFile, 0L, SEEK_END);
            lSize = ftell(clBinFile);
            rewind(clBinFile);

            std::vector<unsigned char> buffer(lSize);
            lSize = fread(&buffer[0], sizeof(unsigned char), lSize, clBinFile);
            fclose(clBinFile);

            program = cl::Program(devCntxt->context, { all_devices[i] }, { buffer }, &binaryStatus, &buildError);

            if ((buildError != CL_SUCCESS) || (binaryStatus[0] != CL_SUCCESS)) {
                std::cout << "Binary error for device #" << i << ": " << buildError << ", " << binaryStatus[0]
                          << std::endl;
            }
        } else {
            program = cl::Program(devCntxt->context, sources);
        }

        buildError = program.build({ all_devices[i] }, "-cl-denorms-are-zero -cl-fast-relaxed-math");
        if (buildError != CL_SUCCESS) {
            std::cout << "Error building for device #" << i << ": " << buildError << ", "
                      << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(all_devices[i])
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(all_devices[i]) << std::endl;

            // The default device was set above to be the last device in the list. If we can't compile for it, we
            // use the first device. If the default is the first device, and we can't compile for it, then we don't
            // have any devices that can compile at all, and the environment needs to be fixed by the user.
            if (i == dev) {
                toRet.default_device_context = toRet.all_device_contexts[0];
                default_platform = all_platforms[0];
                default_device = all_devices[0];
            }

            continue;
        }

        toRet.all_device_contexts.push_back(devCntxt);

        toRet.all_device_contexts[i]->calls[OCL_API_APPLY2X2] = cl::Kernel(program, "apply2x2");
        toRet.all_device_contexts[i]->calls[OCL_API_APPLY2X2_UNIT] = cl::Kernel(program, "apply2x2unit");
        toRet.all_device_contexts[i]->calls[OCL_API_APPLY2X2_NORM] = cl::Kernel(program, "apply2x2norm");
        toRet.all_device_contexts[i]->calls[OCL_API_NORMSUM] = cl::Kernel(program, "normsum");
        toRet.all_device_contexts[i]->calls[OCL_API_UNIFORMLYCONTROLLED] = cl::Kernel(program, "uniformlycontrolled");
        toRet.all_device_contexts[i]->calls[OCL_API_X] = cl::Kernel(program, "x");
        toRet.all_device_contexts[i]->calls[OCL_API_COMPOSE] = cl::Kernel(program, "compose");
        toRet.all_device_contexts[i]->calls[OCL_API_COMPOSE_MID] = cl::Kernel(program, "composemid");
        toRet.all_device_contexts[i]->calls[OCL_API_DECOMPOSEPROB] = cl::Kernel(program, "decomposeprob");
        toRet.all_device_contexts[i]->calls[OCL_API_DECOMPOSEAMP] = cl::Kernel(program, "decomposeamp");
        toRet.all_device_contexts[i]->calls[OCL_API_PROB] = cl::Kernel(program, "prob");
        toRet.all_device_contexts[i]->calls[OCL_API_PROBREG] = cl::Kernel(program, "probreg");
        toRet.all_device_contexts[i]->calls[OCL_API_PROBREGALL] = cl::Kernel(program, "probregall");
        toRet.all_device_contexts[i]->calls[OCL_API_PROBMASK] = cl::Kernel(program, "probmask");
        toRet.all_device_contexts[i]->calls[OCL_API_PROBMASKALL] = cl::Kernel(program, "probmaskall");
        toRet.all_device_contexts[i]->calls[OCL_API_SWAP] = cl::Kernel(program, "swap");
        toRet.all_device_contexts[i]->calls[OCL_API_ROL] = cl::Kernel(program, "rol");
        toRet.all_device_contexts[i]->calls[OCL_API_ROR] = cl::Kernel(program, "ror");
        toRet.all_device_contexts[i]->calls[OCL_API_INC] = cl::Kernel(program, "inc");
        toRet.all_device_contexts[i]->calls[OCL_API_CINC] = cl::Kernel(program, "cinc");
        toRet.all_device_contexts[i]->calls[OCL_API_DEC] = cl::Kernel(program, "dec");
        toRet.all_device_contexts[i]->calls[OCL_API_CDEC] = cl::Kernel(program, "cdec");
        toRet.all_device_contexts[i]->calls[OCL_API_INCC] = cl::Kernel(program, "incc");
        toRet.all_device_contexts[i]->calls[OCL_API_DECC] = cl::Kernel(program, "decc");
        toRet.all_device_contexts[i]->calls[OCL_API_INCS] = cl::Kernel(program, "incs");
        toRet.all_device_contexts[i]->calls[OCL_API_DECS] = cl::Kernel(program, "decs");
        toRet.all_device_contexts[i]->calls[OCL_API_INCSC_1] = cl::Kernel(program, "incsc1");
        toRet.all_device_contexts[i]->calls[OCL_API_DECSC_1] = cl::Kernel(program, "decsc1");
        toRet.all_device_contexts[i]->calls[OCL_API_INCSC_2] = cl::Kernel(program, "incsc2");
        toRet.all_device_contexts[i]->calls[OCL_API_DECSC_2] = cl::Kernel(program, "decsc2");
        toRet.all_device_contexts[i]->calls[OCL_API_INCBCD] = cl::Kernel(program, "incbcd");
        toRet.all_device_contexts[i]->calls[OCL_API_DECBCD] = cl::Kernel(program, "decbcd");
        toRet.all_device_contexts[i]->calls[OCL_API_INCBCDC] = cl::Kernel(program, "incbcdc");
        toRet.all_device_contexts[i]->calls[OCL_API_DECBCDC] = cl::Kernel(program, "decbcdc");
        toRet.all_device_contexts[i]->calls[OCL_API_INDEXEDLDA] = cl::Kernel(program, "indexedLda");
        toRet.all_device_contexts[i]->calls[OCL_API_INDEXEDADC] = cl::Kernel(program, "indexedAdc");
        toRet.all_device_contexts[i]->calls[OCL_API_INDEXEDSBC] = cl::Kernel(program, "indexedSbc");
        toRet.all_device_contexts[i]->calls[OCL_API_APPROXCOMPARE] = cl::Kernel(program, "approxcompare");
        toRet.all_device_contexts[i]->calls[OCL_API_NORMALIZE] = cl::Kernel(program, "nrmlze");
        toRet.all_device_contexts[i]->calls[OCL_API_UPDATENORM] = cl::Kernel(program, "updatenorm");
        toRet.all_device_contexts[i]->calls[OCL_API_APPLYM] = cl::Kernel(program, "applym");
        toRet.all_device_contexts[i]->calls[OCL_API_APPLYMREG] = cl::Kernel(program, "applymreg");
        toRet.all_device_contexts[i]->calls[OCL_API_PHASEFLIP] = cl::Kernel(program, "phaseflip");
        toRet.all_device_contexts[i]->calls[OCL_API_ZEROPHASEFLIP] = cl::Kernel(program, "zerophaseflip");
        toRet.all_device_contexts[i]->calls[OCL_API_CPHASEFLIPIFLESS] = cl::Kernel(program, "cphaseflipifless");
        toRet.all_device_contexts[i]->calls[OCL_API_PHASEFLIPIFLESS] = cl::Kernel(program, "phaseflipifless");
        toRet.all_device_contexts[i]->calls[OCL_API_MUL] = cl::Kernel(program, "mul");
        toRet.all_device_contexts[i]->calls[OCL_API_DIV] = cl::Kernel(program, "div");
        toRet.all_device_contexts[i]->calls[OCL_API_CMUL] = cl::Kernel(program, "cmul");
        toRet.all_device_contexts[i]->calls[OCL_API_CDIV] = cl::Kernel(program, "cdiv");

        if (saveBinaries) {
            size_t clBinSizes;
            program.getInfo(CL_PROGRAM_BINARY_SIZES, &clBinSizes);
            std::cout << "OpenCL #" << i << " Binary size:" << clBinSizes << std::endl;

            unsigned char* clBinary = new unsigned char[clBinSizes];
            program.getInfo(CL_PROGRAM_BINARIES, &clBinary);

            buildError = mkdir(homeStr.c_str(), 0700);

            FILE* clBinFile = fopen(clBinName.c_str(), "w");
            fwrite(clBinary, clBinSizes, sizeof(unsigned char), clBinFile);
            fclose(clBinFile);
            delete[] clBinary;
        }

        if (i == dev) {
            toRet.default_device_context = toRet.all_device_contexts[i];
            default_platform = all_platforms[plat_id];
            default_device = all_devices[i];
        }
    }

    // For VirtualCL support, the device info can only be accessed AFTER all contexts are created.
    std::cout << "Default platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "Default device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
    for (i = 0; i < deviceCount; i++) {
        std::cout << "OpenCL device #" << i << ": " << all_devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }

    return toRet;
}

OCLEngine* OCLEngine::m_pInstance = NULL;
OCLEngine* OCLEngine::Instance()
{
    if (!m_pInstance)
        m_pInstance = new OCLEngine();
    return m_pInstance;
}

} // namespace Qrack
