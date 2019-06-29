//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#define _USE_MATH_DEFINES

#include <atomic>
#include <future>
#include <math.h>

#include "common/parallel_for.hpp"

namespace Qrack {

/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const bitCapInt begin, const bitCapInt itemCount, IncrementFunc inc, ParallelFunc fn)
{
    if (itemCount <= (bitCapInt)numCores) {
        std::vector<std::future<void>> futures(itemCount);
        bitCapInt j;
        uint32_t cpu;
        for (cpu = 0; cpu < itemCount; cpu++) {
            j = begin + cpu;
            futures[cpu] = std::async(std::launch::async, [j, cpu, inc, fn]() { fn(inc(j, cpu), cpu); });
        }
        for (cpu = 0; cpu < itemCount; cpu++) {
            futures[cpu].get();
        }
    } else if ((itemCount / PSTRIDE) < (bitCapInt)numCores) {
        bitCapInt parStride = itemCount / (bitCapInt)numCores;
        bitCapInt remainder = itemCount - (parStride * numCores);
        std::vector<std::future<void>> futures(numCores);
        int32_t cpu, count;
        bitCapInt offset = begin;
        for (cpu = 0; cpu < numCores; cpu++) {
            bitCapInt workUnit = parStride;
            if (remainder > 0) {
                workUnit++;
                remainder--;
            }
            futures[cpu] = std::async(std::launch::async, [cpu, workUnit, offset, inc, fn]() {
                for (bitCapInt j = 0; j < workUnit; j++) {
                    fn(inc(offset + j, cpu), cpu);
                }
            });
            offset += workUnit;
        }
        count = cpu;
        for (cpu = 0; cpu < count; cpu++) {
            futures[cpu].get();
        }
    } else {
        std::atomic<bitCapInt> idx;
        idx = 0;
        std::vector<std::future<void>> futures(numCores);
        for (int cpu = 0; cpu < numCores; cpu++) {
            futures[cpu] = std::async(std::launch::async, [cpu, &idx, begin, itemCount, inc, fn]() {
                bitCapInt j, k, l;
                for (bitCapInt i = idx++; true; i = idx++) {
                    l = i * PSTRIDE;
                    for (j = 0; j < PSTRIDE; j++) {
                        k = j + l;
                        /* Easiest to clamp on end. */
                        if (k >= itemCount) {
                            break;
                        }
                        fn(inc(begin + k, cpu), cpu);
                    }
                    if (k >= itemCount) {
                        break;
                    }
                }
            });
        }

        for (int cpu = 0; cpu < numCores; cpu++) {
            futures[cpu].get();
        }
    }
}

void ParallelFor::par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn)
{
    par_for_inc(begin, end - begin, [](const bitCapInt i, int cpu) { return i; }, fn);
}

void ParallelFor::par_for_set(const std::set<bitCapInt>& sparseSet, ParallelFunc fn)
{
    par_for_inc(0, sparseSet.size(),
        [&sparseSet](const bitCapInt i, int cpu) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_skip(
    const bitCapInt begin, const bitCapInt end, const bitCapInt skipMask, const bitLenInt maskWidth, ParallelFunc fn)
{
    /*
     * Add maskWidth bits by shifting the incrementor up that number of
     * bits, filling with 0's.
     *
     * For example, if the skipMask is 0x8, then the lowMask will be 0x7
     * and the high mask will be ~(0x7 + 0x8) ==> ~0xf, shifted by the
     * number of extra bits to add.
     */

    if ((skipMask << maskWidth) >= end) {
        // If we're skipping trailing bits, this is much cheaper:
        par_for(begin, skipMask, fn);
        return;
    }

    bitCapInt lowMask = skipMask - 1;
    bitCapInt highMask = ~lowMask;

    IncrementFunc incFn;
    if (lowMask == 0) {
        // If we're skipping leading bits, this is much cheaper:
        incFn = [maskWidth](bitCapInt i, int cpu) { return (i << maskWidth); };
    } else {
        incFn = [lowMask, highMask, maskWidth](
                    bitCapInt i, int cpu) { return ((i & lowMask) | ((i & highMask) << maskWidth)); };
    }

    par_for_inc(begin, (end - begin) >> maskWidth, incFn, fn);
}

void ParallelFor::par_for_mask(
    const bitCapInt begin, const bitCapInt end, const bitCapInt* maskArray, const bitLenInt maskLen, ParallelFunc fn)
{
    for (int i = 1; i < maskLen; i++) {
        if (maskArray[i] < maskArray[i - 1]) {
            throw std::invalid_argument("Masks must be ordered by size");
        }
    }

    /* Pre-calculate the masks to simplify the increment function later. */
    bitCapInt** masks = new bitCapInt*[maskLen];
    for (int i = 0; i < maskLen; i++) {
        masks[i] = new bitCapInt[2];
    }

    bool onlyLow = true;
    for (int i = 0; i < maskLen; i++) {
        masks[i][0] = maskArray[i] - 1; // low mask
        masks[i][1] = (~(masks[i][0] + maskArray[i])); // high mask
        if (maskArray[maskLen - i - 1] != (end >> (i + 1))) {
            onlyLow = false;
        }
    }

    IncrementFunc incFn;
    if (onlyLow) {
        par_for(begin, end >> maskLen, fn);
    } else {
        incFn = [&masks, maskLen](bitCapInt i, int cpu) {
            /* Push i apart, one mask at a time. */
            for (int m = 0; m < maskLen; m++) {
                i = ((i << 1) & masks[m][1]) | (i & masks[m][0]);
            }
            return i;
        };

        par_for_inc(begin, (end - begin) >> maskLen, incFn, fn);
    }

    for (int i = 0; i < maskLen; i++) {
        delete[] masks[i];
    }
    delete[] masks;
}

real1 ParallelFor::par_norm(const bitCapInt maxQPower, const StateVectorPtr stateArray)
{
    real1 nrmSqr = 0;
    if (maxQPower <= (bitCapInt)numCores) {
        std::vector<std::future<real1>> futures(maxQPower);
        bitCapInt j;
        uint32_t cpu;
        for (cpu = 0; cpu < maxQPower; cpu++) {
            j = cpu;
            futures[cpu] = std::async(std::launch::async, [j, stateArray]() { return norm(stateArray->read(j)); });
        }
        for (cpu = 0; cpu < maxQPower; cpu++) {
            nrmSqr += futures[cpu].get();
        }
    } else if ((maxQPower / PSTRIDE) < (bitCapInt)numCores) {
        bitCapInt parStride = maxQPower / numCores;
        bitCapInt remainder = maxQPower - (parStride * numCores);
        std::vector<std::future<real1>> futures(numCores);
        int32_t cpu, count;
        bitCapInt offset = 0;
        for (cpu = 0; cpu < numCores; cpu++) {
            bitCapInt workUnit = parStride;
            if (remainder > 0) {
                workUnit++;
                remainder--;
            }
            futures[cpu] = std::async(std::launch::async, [workUnit, offset, stateArray]() {
                real1 result = 0.0;
                for (bitCapInt j = 0; j < workUnit; j++) {
                    result += norm(stateArray->read(offset + j));
                }
                return result;
            });
            offset += workUnit;
        }
        count = cpu;
        for (cpu = 0; cpu < count; cpu++) {
            nrmSqr += futures[cpu].get();
        }
    } else {
        std::atomic<bitCapInt> idx;
        idx = 0;
        std::vector<std::future<real1>> futures(numCores);
        for (int cpu = 0; cpu != numCores; ++cpu) {
            futures[cpu] = std::async(std::launch::async, [&idx, maxQPower, stateArray]() {
                real1 sqrNorm = 0.0;
                bitCapInt i, j;
                bitCapInt k = 0;
                for (;;) {
                    i = idx++;
                    for (j = 0; j < PSTRIDE; j++) {
                        k = i * PSTRIDE + j;
                        if (k >= maxQPower)
                            break;
                        sqrNorm += norm(stateArray->read(k));
                    }
                    if (k >= maxQPower)
                        break;
                }
                return sqrNorm;
            });
        }

        for (int32_t cpu = 0; cpu != numCores; ++cpu) {
            nrmSqr += futures[cpu].get();
        }
    }

    return nrmSqr;
}
} // namespace Qrack
