//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This class allows access to on-chip RNG capabilities. The class is adapted from these two sources:
// https://codereview.stackexchange.com/questions/147656/checking-if-cpu-supports-rdrand/150230
// https://stackoverflow.com/questions/45460146/how-to-use-intels-rdrand-using-inline-assembly-with-net
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "rdrandwrapper.hpp"

#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <string>
#include <sys/types.h>
#include <thread>

namespace Qrack {
// From http://www.cplusplus.com/forum/unices/3548/
std::vector<std::string> _readDirectoryFileNames(const std::string& path)
{
    std::vector<std::string> result;
    errno = 0;
    DIR* dp = opendir(path.empty() ? "." : path.c_str());
    if (dp) {
        while (true) {
            errno = 0;
            dirent* de = readdir(dp);
            if (de == NULL) {
                break;
            }
            if (std::string(de->d_name) != "." && std::string(de->d_name) != "..") {
                result.push_back(path + "/" + std::string(de->d_name));
            }
        }
        closedir(dp);
        std::sort(result.begin(), result.end());
    }
    return result;
}

std::string _getDefaultRandomNumberFilePath()
{
#if ENABLE_ENV_VARS
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
#endif
#if defined(_WIN32) && !defined(__CYGWIN__)
    return std::string(getenv("HOMEDRIVE") ? getenv("HOMEDRIVE") : "") +
        std::string(getenv("HOMEPATH") ? getenv("HOMEPATH") : "") + "\\.qrack\\rng\\";
#else
    return std::string(getenv("HOME") ? getenv("HOME") : "") + "/.qrack/rng/";
#endif
}

void RandFile::_readNextRandDataFile()
{
    if (dataFile) {
        fclose(dataFile);
    }

    std::string path = _getDefaultRandomNumberFilePath();
    std::vector<std::string> fileNames = _readDirectoryFileNames(path);
    if (fileNames.size() <= fileOffset) {
        throw std::runtime_error("Out of RNG files!");
    }

    while (!(dataFile = fopen(fileNames[fileOffset].c_str(), "r"))) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ++fileOffset;
}
} // namespace Qrack
