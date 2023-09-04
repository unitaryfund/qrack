//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This utility precompiles all the OpenCL kernels that Qrack needs.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "config.h"
#if ENABLE_OPENCL
#include "common/oclengine.hpp"
#endif

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
#if ENABLE_OPENCL
    // Precompile, if OpenCL is available.
    std::cout << "Precompiling OCL kernels..." << std::endl;
    if (argc < 2) {
        std::cout << "Will save to: " << Qrack::OCLEngine::GetDefaultBinaryPath() << std::endl;
        Qrack::OCLEngine::InitOCL(true, true, Qrack::OCLEngine::GetDefaultBinaryPath());
    } else {
        std::cout << "Will save to: " << std::string(argv[1]) << std::endl;
        Qrack::OCLEngine::InitOCL(true, true, std::string(argv[1]) + "/");
    }
    std::cout << "Done precompiling OCL kernels." << std::endl;
#else
    std::cout << "OCL not available; nothing to precompile." << std::endl;
#endif
}
