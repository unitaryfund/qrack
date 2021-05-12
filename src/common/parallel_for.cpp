//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#define _USE_MATH_DEFINES

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#include <atomic>
#include <future>
#include <math.h>

#if ENABLE_UINT128
#include <mutex>
#endif

#include "common/parallel_for.hpp"

#if ENABLE_UINT128
#define DECLARE_ATOMIC_BITCAPINT()                                                                                     \
    std::mutex idxLock;                                                                                                \
    bitCapInt idx;
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__, &idxLock]()
#define ATOMIC_INC()                                                                                                   \
    idxLock.lock();                                                                                                    \
    i = idx++;                                                                                                         \
    idxLock.unlock();
#else
#define DECLARE_ATOMIC_BITCAPINT() std::atomic<bitCapIntOcl> idx;
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Qrack {

ParallelFor::ParallelFor()
    : numCores(1)
{
    if (getenv("QRACK_PSTRIDEPOW")) {
        pStride = (ONE_BCI << (bitCapIntOcl)std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))));
    } else {
        pStride = (ONE_BCI << (bitCapIntOcl)PSTRIDEPOW);
    }
}

/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const bitCapInt begin, const bitCapInt itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const bitCapIntOcl Stride = pStride;

    if ((itemCount / Stride) < (bitCapInt)numCores) {
        bitCapInt maxLcv = begin + itemCount;
        for (bitCapInt j = begin; j < maxLcv; j++) {
            fn(inc(j, 0), 0);
        }
        return;
    }

    DECLARE_ATOMIC_BITCAPINT();
    idx = 0;
    std::vector<std::future<void>> futures(numCores);
    for (int cpu = 0; cpu < numCores; cpu++) {
        futures[cpu] = ATOMIC_ASYNC(cpu, &idx, begin, itemCount, Stride, inc, fn)
        {
            bitCapIntOcl i, j, l;
            bitCapIntOcl k = 0;
            for (;;) {
                ATOMIC_INC();
                l = i * Stride;
                for (j = 0; j < Stride; j++) {
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

void ParallelFor::par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn)
{
    par_for_inc(
        begin, end - begin, [](const bitCapInt i, int cpu) { return i; }, fn);
}

void ParallelFor::par_for_set(const std::set<bitCapInt>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0, (bitCapInt)sparseSet.size(),
        [&sparseSet](const bitCapInt i, int cpu) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_set(const std::vector<bitCapInt>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0, (bitCapInt)sparseSet.size(),
        [&sparseSet](const bitCapInt i, int cpu) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_sparse_compose(const std::vector<bitCapInt>& lowSet, const std::vector<bitCapInt>& highSet,
    const bitLenInt& highStart, ParallelFunc fn)
{
    bitCapInt lowSize = (bitCapInt)lowSet.size();
    par_for_inc(
        0, lowSize * (bitCapInt)highSet.size(),
        [&lowSize, &highStart, &lowSet, &highSet](const bitCapInt i, int cpu) {
            bitCapInt lowPerm = i % lowSize;
            bitCapInt highPerm = (i - lowPerm) / lowSize;
            auto it = lowSet.begin();
            std::advance(it, lowPerm);
            bitCapInt perm = *it;
            it = highSet.begin();
            std::advance(it, highPerm);
            perm |= (*it) << highStart;
            return perm;
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

    bitCapIntOcl lowMask = skipMask - ONE_BCI;
    bitCapIntOcl highMask = ~lowMask;

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
    /* Pre-calculate the masks to simplify the increment function later. */
    bitCapInt** masks = new bitCapInt*[maskLen];
    for (bitLenInt i = 0; i < maskLen; i++) {
        masks[i] = new bitCapInt[2];
    }

    bool onlyLow = true;
    for (bitLenInt i = 0; i < maskLen; i++) {
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
            for (bitLenInt m = 0; m < maskLen; m++) {
                i = ((i << ONE_BCI) & masks[m][1]) | (i & masks[m][0]);
            }
            return i;
        };

        par_for_inc(begin, (end - begin) >> maskLen, incFn, fn);
    }

    for (bitLenInt i = 0; i < maskLen; i++) {
        delete[] masks[i];
    }
    delete[] masks;
}

real1_f ParallelFor::par_norm(const bitCapInt maxQPower, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(maxQPower, stateArray);
    }

    const bitCapIntOcl Stride = pStride;
    const bitCapIntOcl itemCount = maxQPower;

    real1_f nrmSqr = ZERO_R1;
    if ((itemCount / Stride) < (bitCapIntOcl)numCores) {
        real1_f nrm;
        for (bitCapIntOcl j = 0; j < itemCount; j++) {
            nrm = norm(stateArray->read(j));
            if (nrm >= norm_thresh) {
                nrmSqr += nrm;
            }
        }
    } else {
        DECLARE_ATOMIC_BITCAPINT();
        idx = 0;
        std::vector<std::future<real1_f>> futures(numCores);
        for (int cpu = 0; cpu != numCores; ++cpu) {
            futures[cpu] = ATOMIC_ASYNC(&idx, itemCount, stateArray, Stride, &norm_thresh)
            {
                real1_f sqrNorm = ZERO_R1;
                real1_f nrm;
                bitCapIntOcl i, j;
                bitCapIntOcl k = 0;
                for (;;) {
                    ATOMIC_INC();
                    for (j = 0; j < Stride; j++) {
                        k = i * Stride + j;
                        if (k >= itemCount)
                            break;

                        nrm = norm(stateArray->read(k));
                        if (nrm >= norm_thresh) {
                            sqrNorm += nrm;
                        }
                    }
                    if (k >= itemCount)
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

real1_f ParallelFor::par_norm_exact(const bitCapInt maxQPower, const StateVectorPtr stateArray)
{
    const bitCapIntOcl Stride = pStride;
    const bitCapIntOcl itemCount = maxQPower;

    real1_f nrmSqr = ZERO_R1;
    if ((itemCount / Stride) < (bitCapInt)numCores) {
        for (bitCapIntOcl j = 0; j < maxQPower; j++) {
            nrmSqr += norm(stateArray->read(j));
        }

        return nrmSqr;
    }
    DECLARE_ATOMIC_BITCAPINT();
    idx = 0;
    std::vector<std::future<real1_f>> futures(numCores);
    for (int cpu = 0; cpu != numCores; ++cpu) {
        futures[cpu] = ATOMIC_ASYNC(&idx, itemCount, Stride, stateArray)
        {
            real1_f sqrNorm = ZERO_R1;
            bitCapIntOcl i, j;
            bitCapIntOcl k = 0;
            for (;;) {
                ATOMIC_INC();
                for (j = 0; j < Stride; j++) {
                    k = i * Stride + j;
                    if (k >= itemCount)
                        break;

                    sqrNorm += norm(stateArray->read(k));
                }
                if (k >= itemCount)
                    break;
            }
            return sqrNorm;
        });
    }

    for (int32_t cpu = 0; cpu != numCores; ++cpu) {
        nrmSqr += futures[cpu].get();
    }

    return nrmSqr;
}
} // namespace Qrack
