//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This class allows access to on-chip RNG capabilities. The class is adapted from these two sources:
// https://codereview.stackexchange.com/questions/147656/checking-if-cpu-supports-rdrand/150230
// https://stackoverflow.com/questions/45460146/how-to-use-intels-rdrand-using-inline-assembly-with-net
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#if ENABLE_RNDFILE
#include <future>
#include <string>
#include <vector>
#endif

#if ENABLE_RDRAND
#if _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <immintrin.h>
#endif

#include "qrack_types.hpp"

namespace Qrack {

bool getRdRand(unsigned int* pv);

class RdRandom {
public:
    bool SupportsRDRAND();
    real1 Next();
#if ENABLE_RNDFILE
    std::vector<std::string> ReadDirectory(const std::string& path = std::string());

private:
    bool didInit;
    bool isPageTwo;
    std::vector<char> data1;
    std::vector<char> data2;
    std::future<void> future1;
    std::future<void> future2;
    int dataOffset;
#endif
};
} // namespace Qrack
