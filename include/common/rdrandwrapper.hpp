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

#pragma once

#include "qrack_types.hpp"

#if ENABLE_DEVRAND
#include <sys/random.h>
#endif

#if ENABLE_RDRAND
#if _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <immintrin.h>
#endif

namespace Qrack {

class RandSupportSingleton {
private:
    bool isSupported;

    RandSupportSingleton()
        : isSupported(CheckHardwareRDRANDSupport())
    {
        // Intentionally left blank
    }

    static bool CheckHardwareRDRANDSupport()
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

public:
    // See https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
    /// Get a pointer to the Instance of the singleton. (The instance will be instantiated, if it does not exist yet.)
    static RandSupportSingleton& Instance()
    {
        static RandSupportSingleton instance;
        return instance;
    }
    RandSupportSingleton(RandSupportSingleton const&) = delete;
    void operator=(RandSupportSingleton const&) = delete;

    bool SupportsRDRAND() { return isSupported; }
};

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
// See https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
class RandFile {
public:
    static RandFile& getInstance()
    {
        static RandFile instance;
        return instance;
    }

    unsigned NextRaw()
    {
        size_t fSize = 0;
        unsigned v;
        while (fSize < 1) {
            fSize = fread(&v, sizeof(unsigned), 1, dataFile);
            if (fSize < 1) {
                _readNextRandDataFile();
            }
        }

        return v;
    }

private:
    RandFile() { _readNextRandDataFile(); }
    ~RandFile()
    {
        if (dataFile) {
            fclose(dataFile);
        }
    }

    size_t fileOffset;
    FILE* dataFile;
    void _readNextRandDataFile();

public:
    RandFile(RandFile const&) = delete;
    void operator=(RandFile const&) = delete;
};
#endif

class RdRandom {
private:
    bool getRdRand(unsigned* pv)
    {
#if ENABLE_RDRAND || ENABLE_DEVRAND
        constexpr int max_rdrand_tries = 10;
        for (int i = 0; i < max_rdrand_tries; ++i) {
#if ENABLE_DEVRAND
            if (sizeof(unsigned) == getrandom(reinterpret_cast<char*>(pv), sizeof(unsigned), 0))
#else
            if (_rdrand32_step(pv))
#endif
                return true;
        }
#endif
        return false;
    }

public:
    bool SupportsRDRAND() { return RandSupportSingleton::Instance().SupportsRDRAND(); }

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
    unsigned NextRaw() { return RandFile::getInstance().NextRaw(); }
#else
    unsigned NextRaw()
    {
        unsigned v;
        if (!getRdRand(&v)) {
            throw std::runtime_error("Random number generator failed up to retry limit.");
        }

        return v;
    }
#endif

    real1_f Next()
    {
        unsigned v = NextRaw();

        real1_f res = ZERO_R1_F;
        real1_f part = ONE_R1_F;
        for (unsigned i = 0U; i < 32U; ++i) {
            part /= 2;
            if ((v >> i) & 1U) {
                res += part;
            }
        }

#if FPPOW > 5
        v = NextRaw();

        for (unsigned i = 0U; i < 32U; ++i) {
            part /= 2;
            if ((v >> i) & 1U) {
                res += part;
            }
        }
#endif

        return res;
    }
};
} // namespace Qrack
