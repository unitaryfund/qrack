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

#pragma once

#include "qrack_types.hpp"

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
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

namespace Qrack {

#if ENABLE_RNDFILE && !ENABLE_DEVRAND
// See https://stackoverflow.com/questions/1008019/c-singleton-design-pattern
class RandFile {
public:
    static RandFile& getInstance()
    {
        static RandFile instance;
        return instance;
    }

    unsigned NextRaw();

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
public:
    bool SupportsRDRAND();
    unsigned NextRaw();
    real1_f Next();
};
} // namespace Qrack
