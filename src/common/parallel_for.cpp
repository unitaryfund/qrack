//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <atomic>
#include <future>
#include <math.h>

#include "common/parallel_for.hpp"

namespace Qrack {

/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const bitCapInt begin, const bitCapInt end, IncrementFunc inc, ParallelFunc fn)
{
    std::atomic<bitCapInt> idx;
    idx = begin;

    std::vector<std::future<void>> futures(numCores);

    for (int cpu = 0; cpu < numCores; cpu++) {
        futures[cpu] = std::async(std::launch::async, [&]() {
            for (bitCapInt i = idx++; i < end; i = idx++) {
                i = inc(i);
                /* Easiest to clamp on end. */
                if (i >= end) {
                    break;
                }
                fn(i);
            }
        });
    }

    for (int cpu = 0; cpu < numCores; cpu++) {
        futures[cpu].get();
    }
}

void ParallelFor::par_for(const bitCapInt begin, const bitCapInt end, ParallelFunc fn)
{
    par_for_inc(begin, end, [](const bitCapInt i) { return i; }, fn);
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
    bitCapInt lowMask = skipMask - 1;
    bitCapInt highMask = (~(lowMask + skipMask)) << (maskWidth - 1);

    IncrementFunc incFn = [lowMask, highMask, maskWidth](
                              bitCapInt i) { return ((i << maskWidth) & highMask) | (i & lowMask); };

    par_for_inc(begin, end, incFn, fn);
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
    bitCapInt masks[maskLen][2];

    for (int i = 0; i < maskLen; i++) {
        masks[i][0] = maskArray[i] - 1; // low mask
        masks[i][1] = (~(masks[i][0] + maskArray[i])); // high mask
    }

    IncrementFunc incFn = [&masks, maskLen](bitCapInt i) {
        /* Push i apart, one mask at a time. */
        for (int m = 0; m < maskLen; m++) {
            i = ((i << 1) & masks[m][1]) | (i & masks[m][0]);
        }
        return i;
    };

    par_for_inc(begin, end, incFn, fn);
}

double ParallelFor::par_norm(const bitCapInt maxQPower, const Complex16* stateArray)
{
    // const double* sAD = reinterpret_cast<const double*>(stateArray);
    // double* sSAD = new double[maxQPower * 2];
    // std::partial_sort_copy(sAD, sAD + (maxQPower * 2), sSAD, sSAD + (maxQPower * 2));
    // Complex16* sorted = reinterpret_cast<Complex16*>(sSAD);

    std::atomic<bitCapInt> idx;
    idx = 0;
    double* nrmPart = new double[numCores];
    std::vector<std::future<void>> futures(numCores);
    for (int cpu = 0; cpu != numCores; ++cpu) {
        futures[cpu] = std::async(std::launch::async, [cpu, &idx, maxQPower, stateArray, nrmPart]() {
            double sqrNorm = 0.0;
            // double smallSqrNorm = 0.0;
            bitCapInt i;
            for (;;) {
                i = idx++;
                // if (i >= maxQPower) {
                //	sqrNorm += smallSqrNorm;
                //	break;
                //}
                // smallSqrNorm += norm(sorted[i]);
                // if (smallSqrNorm > sqrNorm) {
                //	sqrNorm += smallSqrNorm;
                //	smallSqrNorm = 0;
                //}
                if (i >= maxQPower)
                    break;
                sqrNorm += norm(stateArray[i]);
            }
            nrmPart[cpu] = sqrNorm;
        });
    }

    double nrmSqr = 0;
    for (int cpu = 0; cpu != numCores; ++cpu) {
        futures[cpu].get();
        nrmSqr += nrmPart[cpu];
    }
    return sqrt(nrmSqr);
}

}
