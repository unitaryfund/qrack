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

#if ENABLE_RDRAND
#include <cpuid.h>
#include <immintrin.h>
#endif

#pragma once

namespace RdRandWrapper {

bool getRdRand(unsigned int* pv);

class RdRandom {
public:
    bool SupportsRDRAND();

    double Next();
};
} // namespace RdRandWrapper
