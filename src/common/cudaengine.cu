//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "cudaengine.cuh"

#include <algorithm>
#include <iostream>
#include <regex>
#include <sstream>

namespace Qrack {

/// "Qrack::CUDAEngine" manages the single CUDA context

// Public singleton methods to get pointers to various methods
DeviceContextPtr CUDAEngine::GetDeviceContextPtr(const int64_t& dev)
{
    if ((dev >= GetDeviceCount()) || (dev < -1) || (dev > ((int64_t)all_device_contexts.size()))) {
        throw std::runtime_error("Invalid CUDA device selection");
    } else if (dev == -1) {
        return default_device_context;
    } else {
        return all_device_contexts[dev];
    }
}

std::vector<DeviceContextPtr> CUDAEngine::GetDeviceContextPtrVector() { return all_device_contexts; }
void CUDAEngine::SetDeviceContextPtrVector(std::vector<DeviceContextPtr> vec, DeviceContextPtr dcp)
{
    all_device_contexts = vec;
    if (dcp != nullptr) {
        default_device_context = dcp;
    }
}

void CUDAEngine::SetDefaultDeviceContext(DeviceContextPtr dcp) { default_device_context = dcp; }

InitCUDAResult CUDAEngine::InitCUDA(std::vector<int64_t> maxAllocVec)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (!deviceCount) {
        std::cout << " No devices found. Check CUDA installation!\n";
        return InitCUDAResult();
    }

    // Prefer the last device, for intuitiveness compared to OpenCL Qrack.
    int64_t dev = deviceCount - 1;
    if (getenv("QRACK_OCL_DEFAULT_DEVICE")) {
        dev = std::stoi(std::string(getenv("QRACK_OCL_DEFAULT_DEVICE")));
        if ((dev < 0) || (dev > (deviceCount - 1))) {
            std::cout << "WARNING: Invalid QRACK_OCL_DEFAULT_DEVICE selection. (Falling back to highest index device "
                         "as default.)"
                      << std::endl;
            dev = deviceCount - 1;
        }
    }

    std::vector<cudaDeviceProp> deviceProps;
    std::vector<DeviceContextPtr> all_dev_contexts;
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        deviceProps.push_back(prop);
        all_dev_contexts.push_back(std::make_shared<CUDADeviceContext>(i, maxAllocVec[i % maxAllocVec.size()]));
    }

    // For VirtualCL support, the device info can only be accessed AFTER all contexts are created.
    std::cout << "Default device: #" << dev << ", " << deviceProps[dev].name << "\n";
    for (int64_t i = 0; i < deviceCount; ++i) {
        std::cout << "CUDA device #" << i << ": " << deviceProps[i].name << "\n";
    }

    return InitCUDAResult(all_dev_contexts, all_dev_contexts[dev]);
}

CUDAEngine::CUDAEngine()
    : maxActiveAllocSizes(1U, -1)
{
    if (getenv("QRACK_MAX_ALLOC_MB")) {
        std::string devListStr = std::string(getenv("QRACK_MAX_ALLOC_MB"));
        maxActiveAllocSizes.clear();
        if (devListStr.compare("")) {
            std::stringstream devListStr_stream(devListStr);
            // See
            // https://stackoverflow.com/questions/7621727/split-a-string-into-words-by-multiple-delimiters#answer-58164098
            std::regex re("[.]");
            while (devListStr_stream.good()) {
                std::string term;
                getline(devListStr_stream, term, ',');
                // the '-1' is what makes the regex split (-1 := what was not matched)
                std::sregex_token_iterator first{ term.begin(), term.end(), re, -1 }, last;
                std::vector<std::string> tokens{ first, last };
                if (tokens.size() == 1U) {
                    maxActiveAllocSizes.push_back(stoi(term));
                    if (maxActiveAllocSizes.back() >= 0) {
                        maxActiveAllocSizes.back() = maxActiveAllocSizes.back() << 20U;
                    }
                    continue;
                }
                const unsigned maxI = stoi(tokens[0U]);
                std::vector<int64_t> limits(tokens.size() - 1U);
                for (unsigned i = 1U; i < tokens.size(); ++i) {
                    limits[i - 1U] = stoi(tokens[i]);
                }
                for (unsigned i = 0U; i < maxI; ++i) {
                    for (unsigned j = 0U; j < limits.size(); ++j) {
                        maxActiveAllocSizes.push_back(limits[j]);
                        if (maxActiveAllocSizes.back() >= 0) {
                            maxActiveAllocSizes.back() = maxActiveAllocSizes.back() << 20U;
                        }
                    }
                }
            }
        }
    }

    InitCUDAResult initResult = InitCUDA(maxActiveAllocSizes);
    SetDeviceContextPtrVector(initResult.all_dev_contexts, initResult.default_dev_context);
    activeAllocSizes = std::vector<size_t>(initResult.all_dev_contexts.size());
}

} // namespace Qrack
