//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This class allows access to on-chip RNG capabilities. The class is adapted from these two sources:
// https://codereview.stackexchange.com/questions/147656/checking-if-cpu-supports-rdrand/150230
// https://stackoverflow.com/questions/45460146/how-to-use-intels-rdrand-using-inline-assembly-with-net
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "rdrandwrapper.hpp"

#if ENABLE_RNDFILE
#include <chrono>
#include <thread>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <sys/types.h>
#endif

namespace Qrack {

bool getRdRand(unsigned* pv)
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

#if ENABLE_RNDFILE
// From http://www.cplusplus.com/forum/unices/3548/
std::vector<std::string> _readDirectoryFileNames(const std::string& path)
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
            if (std::string(de->d_name) != "." && std::string(de->d_name) != "..") {
                result.push_back(path + "/" + std::string(de->d_name));
            }
        }
        closedir(dp);
        if (result.size() > 0) {
            std::sort(result.begin(), result.end());
        }
    }
    return result;
}

std::string _getDefaultRandomNumberFilePath()
{
    if (getenv("QRACK_RNG_PATH")) {
        std::string toRet = std::string(getenv("QRACK_RNG_PATH"));
        if ((toRet.back() != '/') && (toRet.back() != '\\')) {
#if defined(_WIN32) && !defined(__CYGWIN__)
            toRet += "\\";
#else
            toRet += "/";
#endif
        }
        return toRet;
    }
#if defined(_WIN32) && !defined(__CYGWIN__)
    return std::string(getenv("HOMEDRIVE") ? getenv("HOMEDRIVE") : "") +
        std::string(getenv("HOMEPATH") ? getenv("HOMEPATH") : "") + "\\.qrack\\rng\\";
#else
    return std::string(getenv("HOME") ? getenv("HOME") : "") + "/.qrack/rng/";
#endif
}

bool _readNextRandDataFile(size_t fileOffset, std::vector<char>& data)
{
    std::vector<std::string> fileNames = {};

    fileNames = _readDirectoryFileNames(_getDefaultRandomNumberFilePath());
    while (fileNames.size() <= fileOffset) {
        fileNames = _readDirectoryFileNames(_getDefaultRandomNumberFilePath());
    }

    FILE* dataFile;
    while (!(dataFile = fopen(fileNames[fileOffset].c_str(), "r"))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    fseek(dataFile, 0L, SEEK_END);
    size_t fSize = ftell(dataFile);

    if (fSize == 0) {
        fclose(dataFile);
        return false;
    }

    rewind(dataFile);

    data.resize(fSize);
    fSize = fread(&data[0], sizeof(unsigned char), fSize, dataFile);
    fclose(dataFile);

    return true;
}
#endif

bool RdRandom::SupportsRDRAND()
{
#if ENABLE_RDRAND
    const unsigned flag_RDRAND = (1 << 30);

#if _MSC_VER
    int ex[4];
    __cpuid(ex, 1);

    return ((ex[2] & flag_RDRAND) == flag_RDRAND);
#else
    unsigned eax, ebx, ecx, edx;
    ecx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    return ((ecx & flag_RDRAND) == flag_RDRAND);
#endif

#else
    return false;
#endif
}

#if ENABLE_RNDFILE
unsigned RdRandom::NextRaw()
{
    if (!didInit) {
        while ((data1.size() - dataOffset) < 4) {
            if (_readNextRandDataFile(fileOffset, data1)) {
                fileOffset++;
                dataOffset = 0;
            }
        }
        readFuture = std::async(std::launch::async, [&]() {
            while (!_readNextRandDataFile(fileOffset, data2)) {
            }
            fileOffset++;
        });
        didInit = true;
    } else if ((isPageTwo && ((data2.size() - dataOffset) < 4)) || (!isPageTwo && ((data1.size() - dataOffset) < 4))) {
        readFuture.get();
        dataOffset = 0;
        if (isPageTwo) {
            readFuture = std::async(std::launch::async, [&]() {
                while (!_readNextRandDataFile(fileOffset, data1)) {
                }
                fileOffset++;
            });
        } else {
            readFuture = std::async(std::launch::async, [&]() {
                while (!_readNextRandDataFile(fileOffset, data2)) {
                }
                fileOffset++;
            });
        }
        isPageTwo = !isPageTwo;
    }

    unsigned v = 0UL;
    for (int i = 0; i < 4; i++) {
        v |= ((unsigned char)(data1[dataOffset] + 128)) << (i * bitsInByte);
        dataOffset++;
    }

    return v;
}
#else
unsigned RdRandom::NextRaw()
{
    unsigned v;
    if (!getRdRand(&v)) {
        throw std::runtime_error("Failed to get hardware RNG number.");
    }

    return v;
}
#endif

real1_f RdRandom::Next()
{
    unsigned v = NextRaw();

    real1_f res = ZERO_R1;
    real1_f part = ONE_R1;
    for (unsigned i = 0U; i < 32U; i++) {
        part /= 2;
        if ((v >> i) & 1U) {
            res += part;
        }
    }

#if FPPOW > 5
    v = NextRaw();

    for (unsigned i = 0U; i < 32U; i++) {
        part /= 2;
        if ((v >> i) & 1U) {
            res += part;
        }
    }
#endif

    return res;
}
} // namespace Qrack
