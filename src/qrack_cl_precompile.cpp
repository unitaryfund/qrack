//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This utility precompiles all the OpenCL kernels that Qrack needs.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include <iostream>

#if ENABLE_OPENCL
#include "oclengine.hpp"
#endif

int main()
{
#if ENABLE_OPENCL
    // OpenCL type, if available.
    std::cout << "Precompiling OCL kernels..." << std::endl;
    Qrack::OCLEngine::InitOCL(true);
    std::cout << "Done precompiling OCL kernels." << std::endl;
#else
    std::cout << "OCL not available; nothing to precompile." << std::endl;
#endif
}
