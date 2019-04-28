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

#include "rdrandwrapper.hpp"

namespace RdRandWrapper {

bool getRdRand(unsigned int* pv)
{
#if ENABLE_RDRAND
    const int max_rdrand_tries = 10;
    for (int i = 0; i < max_rdrand_tries; ++i) {
        if (_rdrand32_step(pv))
            return true;
    }
#endif
    return false;
}

bool RdRandom::SupportsRDRAND()
{
#if ENABLE_RDRAND
    const unsigned int flag_RDRAND = (1 << 30);

    unsigned int eax, ebx, ecx, edx;
    ecx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return ((ecx & flag_RDRAND) == flag_RDRAND);
#else
    return false;
#endif
}

double RdRandom::Next()
{
    unsigned int v;
    double res = 0;
    double part = 1;
    if (!getRdRand(&v)) {
        throw "Failed to get hardware RNG number.";
    }
    v &= 0x7fffffff;
    for (int i = 0; i < 31; i++) {
        part /= 2;
        if (v & (1U << i)) {
            res += part;
        }
    }
    return res;
}

} // namespace RdRandWrapper
