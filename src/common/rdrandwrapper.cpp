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

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <sys/types.h>

#include "rdrandwrapper.hpp"

namespace Qrack {

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

#if _MSC_VER
    int ex[4];
    __cpuid(ex, 1);

    return ((ex[2] & flag_RDRAND) == flag_RDRAND);
#else
    unsigned int eax, ebx, ecx, edx;
    ecx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return ((ecx & flag_RDRAND) == flag_RDRAND);
#endif

#else
    return false;
#endif
}

#if ENABLE_RNDFILE
// From http://www.cplusplus.com/forum/unices/3548/
std::vector<std::string> RdRandom::ReadDirectory(const std::string& path)
{
    std::vector<std::string> result;
    dirent* de;
    DIR* dp;
    errno = 0;
    dp = opendir(path.empty() ? "." : path.c_str());
    if (dp) {
        while (true) {
            errno = 0;
            de = readdir(dp);
            if (de == NULL) {
                break;
            }
            result.push_back(path + std::string(de->d_name));
        }
        closedir(dp);
        std::sort(result.begin(), result.end());
    }
    return result;
}
#endif

real1 RdRandom::Next()
{
    real1 res = 0;
    real1 part = 1;
#if ENABLE_RNDFILE
    if (!didInit) {
        std::vector<std::string> fileNames = {};

        while (fileNames.size() == 0) {
            fileNames = ReadDirectory("~/.qrack/rng");
        }

        std::ifstream in1(fileNames[0]);
        std::string contents1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
        remove(fileNames[0].c_str());
        fileNames.erase(fileNames.begin());
        data1.resize(contents1.size());
        std::copy(contents1.begin(), contents1.end(), data1.begin());

        future2 = std::async(std::launch::async, [&]() {
            while (fileNames.size() == 0) {
                fileNames = ReadDirectory("~/.qrack/rng");
            }

            std::ifstream in2(fileNames[0]);
            std::string contents2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
            remove(fileNames[0].c_str());
            data2.resize(contents2.size());
            std::copy(contents2.begin(), contents2.end(), data2.begin());
        });

        didInit = true;
    }
    if ((isPageTwo && (data2.size() - dataOffset) < 4) || (!isPageTwo && (data1.size() - dataOffset) < 4)) {
        if (isPageTwo) {
            future1.get();
            future2 = std::async(std::launch::async, [&]() {
                std::vector<std::string> fileNames;

                while (fileNames.size() == 0) {
                    fileNames = ReadDirectory("~/.qrack/rng");
                }

                std::ifstream in2(fileNames[0]);
                std::string contents2((std::istreambuf_iterator<char>(in2)), std::istreambuf_iterator<char>());
                remove(fileNames[0].c_str());
                data2.resize(contents2.size());
                std::copy(contents2.begin(), contents2.end(), data2.begin());
            });
        } else {
            future2.get();
            future1 = std::async(std::launch::async, [&]() {
                std::vector<std::string> fileNames = {};

                while (fileNames.size() == 0) {
                    fileNames = ReadDirectory("~/.qrack/rng");
                }

                std::ifstream in1(fileNames[0]);
                std::string contents1((std::istreambuf_iterator<char>(in1)), std::istreambuf_iterator<char>());
                remove(fileNames[0].c_str());
                data1.resize(contents1.size());
                std::copy(contents1.begin(), contents1.end(), data1.begin());
            });
        }
        isPageTwo = !isPageTwo;
        dataOffset = 0;
    }
    for (int i = 0; i < 4; i++) {
        part /= 256;
        res += part * ((isPageTwo ? data2[dataOffset + i] : data1[dataOffset + i]) + 128);
    }
    dataOffset += 4;
    return res;
#else
    unsigned int v;
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
#endif
    return res;
}

} // namespace Qrack
