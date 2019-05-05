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

#include "oclengine.hpp"

#if ENABLE_PURE32
#include "qheader32cl.hpp"
#elif ENABLE_COMPLEX8
#include "qheader_floatcl.hpp"
#else
#include "qheader_doublecl.hpp"
#endif

#include "qenginecl.hpp"

#define OCL_CREATE_CALL(ENUMNAME, FUNCNAMESTR)                                                                         \
    toRet.all_device_contexts[i]->calls[ENUMNAME] = cl::Kernel(program, FUNCNAMESTR.c_str())

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

const std::vector<OCLKernelHandle> OCLEngine::kernelHandles = { OCLKernelHandle(OCL_API_APPLY2X2, "apply2x2"),
    OCLKernelHandle(OCL_API_APPLY2X2_UNIT, "apply2x2unit"), OCLKernelHandle(OCL_API_APPLY2X2_NORM, "apply2x2norm"),
    OCLKernelHandle(OCL_API_NORMSUM, "normsum"), OCLKernelHandle(OCL_API_UNIFORMLYCONTROLLED, "uniformlycontrolled"),
    OCLKernelHandle(OCL_API_X, "x"), OCLKernelHandle(OCL_API_COMPOSE, "compose"),
    OCLKernelHandle(OCL_API_COMPOSE_MID, "composemid"), OCLKernelHandle(OCL_API_DECOMPOSEPROB, "decomposeprob"),
    OCLKernelHandle(OCL_API_DECOMPOSEAMP, "decomposeamp"), OCLKernelHandle(OCL_API_PROB, "prob"),
    OCLKernelHandle(OCL_API_PROBREG, "probreg"), OCLKernelHandle(OCL_API_PROBREGALL, "probregall"),
    OCLKernelHandle(OCL_API_PROBMASK, "probmask"), OCLKernelHandle(OCL_API_PROBMASKALL, "probmaskall"),
    OCLKernelHandle(OCL_API_SWAP, "swap"), OCLKernelHandle(OCL_API_ROL, "rol"), OCLKernelHandle(OCL_API_ROR, "ror"),
    OCLKernelHandle(OCL_API_INC, "inc"), OCLKernelHandle(OCL_API_CINC, "cinc"), OCLKernelHandle(OCL_API_DEC, "dec"),
    OCLKernelHandle(OCL_API_CDEC, "cdec"), OCLKernelHandle(OCL_API_INCC, "incc"), OCLKernelHandle(OCL_API_DECC, "decc"),
    OCLKernelHandle(OCL_API_INCS, "incs"), OCLKernelHandle(OCL_API_DECS, "decs"),
    OCLKernelHandle(OCL_API_INCSC_1, "incsc1"), OCLKernelHandle(OCL_API_DECSC_1, "decsc1"),
    OCLKernelHandle(OCL_API_INCSC_2, "incsc2"), OCLKernelHandle(OCL_API_DECSC_2, "decsc2"),
    OCLKernelHandle(OCL_API_INCBCD, "incbcd"), OCLKernelHandle(OCL_API_DECBCD, "decbcd"),
    OCLKernelHandle(OCL_API_INCBCDC, "incbcdc"), OCLKernelHandle(OCL_API_DECBCDC, "decbcdc"),
    OCLKernelHandle(OCL_API_INDEXEDLDA, "indexedLda"), OCLKernelHandle(OCL_API_INDEXEDADC, "indexedAdc"),
    OCLKernelHandle(OCL_API_INDEXEDSBC, "indexedSbc"), OCLKernelHandle(OCL_API_APPROXCOMPARE, "approxcompare"),
    OCLKernelHandle(OCL_API_NORMALIZE, "nrmlze"), OCLKernelHandle(OCL_API_UPDATENORM, "updatenorm"),
    OCLKernelHandle(OCL_API_APPLYM, "applym"), OCLKernelHandle(OCL_API_APPLYMREG, "applymreg"),
    OCLKernelHandle(OCL_API_PHASEFLIP, "phaseflip"), OCLKernelHandle(OCL_API_ZEROPHASEFLIP, "zerophaseflip"),
    OCLKernelHandle(OCL_API_CPHASEFLIPIFLESS, "cphaseflipifless"),
    OCLKernelHandle(OCL_API_PHASEFLIPIFLESS, "phaseflipifless"), OCLKernelHandle(OCL_API_MUL, "mul"),
    OCLKernelHandle(OCL_API_DIV, "div"), OCLKernelHandle(OCL_API_CMUL, "cmul"), OCLKernelHandle(OCL_API_CDIV, "cdiv") };

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

cl::Program OCLEngine::MakeProgram(
    bool buildFromSource, cl::Program::Sources sources, std::string path, std::shared_ptr<OCLDeviceContext> devCntxt)
{
    FILE* clBinFile;
    cl::Program program;
    cl_int buildError = -1;
    std::vector<int> binaryStatus;
    if (!buildFromSource && (clBinFile = fopen(path.c_str(), "r"))) {
        long lSize;

        fseek(clBinFile, 0L, SEEK_END);
        lSize = ftell(clBinFile);
        rewind(clBinFile);

        std::vector<unsigned char> buffer(lSize);
        lSize = fread(&buffer[0], sizeof(unsigned char), lSize, clBinFile);
        fclose(clBinFile);

#if defined(__APPLE__)
        program = cl::Program(devCntxt->context, { devCntxt->device },
            { std::pair<const void*, unsigned long>(&buffer[0], buffer.size()) }, &binaryStatus, &buildError);
#else
        program = cl::Program(devCntxt->context, { devCntxt->device }, { buffer }, &binaryStatus, &buildError);
#endif

        if ((buildError != CL_SUCCESS) || (binaryStatus[0] != CL_SUCCESS)) {
            std::cout << "Binary error: " << buildError << ", " << binaryStatus[0] << " (Falling back to JIT.)"
                      << std::endl;
        } else {
            std::cout << "Loaded binary." << std::endl;
        }
    }

    // If, either, there are no cached binaries, or binary loading failed, then fall back to JIT.
    if (buildError != CL_SUCCESS) {
        program = cl::Program(devCntxt->context, sources);
        std::cout << "Built JIT." << std::endl;
    }

    return program;
}

void OCLEngine::SaveBinary(cl::Program program, std::string path, std::string fileName)
{
    std::vector<size_t> clBinSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
    size_t clBinSize = 0;
    int clBinIndex = 0;
    for (unsigned int i = 0; i < clBinSizes.size(); i++) {
        if (clBinSizes[i] > 0) {
            clBinSize = clBinSizes[i];
            clBinIndex = i;
            break;
        }
    }
    std::cout << "Binary size:" << clBinSize << std::endl;

    int err = mkdir(path.c_str(), 0700);
    if (err != -1) {
        std::cout << "Making directory: " << path << std::endl;
    }

    FILE* clBinFile = fopen((path + fileName).c_str(), "w");
#if defined(__APPLE__)
    std::vector<char*> clBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
    char* clBinary = clBinaries[clBinIndex];
    fwrite(clBinary, clBinSize, sizeof(char), clBinFile);
#else
    std::vector<std::vector<unsigned char>> clBinaries = program.getInfo<CL_PROGRAM_BINARIES>();
    std::vector<unsigned char> clBinary = clBinaries[clBinIndex];
    fwrite(&clBinary[0], clBinSize, sizeof(unsigned char), clBinFile);
#endif
    fclose(clBinFile);
}

OCLInitResult OCLEngine::InitOCL(bool buildFromSource, bool saveBinaries, std::string home)
{

    if (home == "*") {
        home = GetDefaultBinaryPath();
    }

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
    for (int i = 0; i < deviceCount; i++) {
        // a context is like a "runtime link" to the device and platform;
        // i.e. communication is possible
        if (device_platform_id[i] != plat_id) {
            plat_id = device_platform_id[i];
            all_contexts.push_back(cl::Context(all_platforms_devices[plat_id]));
        }
        std::shared_ptr<OCLDeviceContext> devCntxt = std::make_shared<OCLDeviceContext>(
            devPlatVec[i], all_devices[i], all_contexts[all_contexts.size() - 1], plat_id);

        std::string fileName = "qrack_ocl_dev_" + std::to_string(i) + ".ir";
        std::string clBinName = home + fileName;

        std::cout << "Device #" << i << ", ";
        cl::Program program = MakeProgram(buildFromSource, sources, clBinName, devCntxt);

        cl_int buildError = program.build({ all_devices[i] }, "-cl-denorms-are-zero -cl-fast-relaxed-math");
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

        for (unsigned int j = 0; j < kernelHandles.size(); j++) {
            OCL_CREATE_CALL(kernelHandles[j].oclapi, kernelHandles[j].kernelname);
        }

        if (saveBinaries) {
            std::cout << "OpenCL program #" << i << ", ";
            SaveBinary(program, home, fileName);
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
