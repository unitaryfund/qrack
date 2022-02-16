//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QBinaryDecision tree is an alternative approach to quantum state representation, as
// opposed to state vector representation. This is a compressed form that can be
// operated directly on while compressed. Inspiration for the Qrack implementation was
// taken from JKQ DDSIM, maintained by the Institute for Integrated Circuits at the
// Johannes Kepler University Linz:
//
// https://github.com/iic-jku/ddsim
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbdt_node_interface.hpp"

#if ENABLE_PTHREAD
#include <future>
#endif
#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

bool operator==(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs)
{
    if (!lhs) {
        return !rhs;
    }

    if (!rhs) {
        return false;
    }

    return lhs->isEqual(rhs);
}

bool operator!=(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs) { return !(lhs == rhs); }

#if ENABLE_PTHREAD && (UINTPOW > 3)
// TODO: Find some way to abstract this with ParallelFor. (It's a duplicate method.)
// The reason for this design choice is that the memory per node for "Stride" and "numCores" attributes are on order of
// all other RAM per node in total. Remember that trees are recursively constructed with exponential scaling, and the
// memory per node should be thought of as akin to the memory per Schrödinger amplitude.
void QBdtNodeInterface::_par_for_qbdt(const bitCapIntOcl begin, const bitCapIntOcl end, IncrementFunc fn)
{
    const bitCapIntOcl itemCount = end - begin;

    const bitCapIntOcl Stride = (bitCapIntOcl)ONE_BCI << PSTRIDEPOW;
    if (itemCount < Stride) {
        const bitCapIntOcl maxLcv = begin + itemCount;
        for (bitCapIntOcl j = begin; j < maxLcv; j++) {
            j |= fn(j, 0);
        }
        return;
    }

    const unsigned numCores = std::thread::hardware_concurrency();
    unsigned threads = (unsigned)(itemCount / Stride);
    if (threads > numCores) {
        threads = numCores;
    }

    bitCapIntOcl idx = 0;
    std::vector<std::future<void>> futures(threads);
    std::mutex updateMutex;
    for (unsigned cpu = 0; cpu != threads; ++cpu) {
        futures[cpu] = std::async(std::launch::async, [cpu, &idx, &begin, &itemCount, &updateMutex, fn]() {
            const bitCapIntOcl Stride = (bitCapIntOcl)(ONE_BCI << (bitCapIntOcl)14U);
            for (;;) {
                bitCapIntOcl i;
                if (true) {
                    std::lock_guard<std::mutex> updateLock(updateMutex);
                    i = idx++;
                }
                const bitCapIntOcl l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const bitCapIntOcl maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                bitCapIntOcl k = 0;
                for (bitCapIntOcl j = 0; j < maxJ; j++) {
                    k = j + l;
                    k |= fn(begin + k, cpu);
                    j = k - l;
                }
                i = k / Stride;
                if (i > idx) {
                    std::lock_guard<std::mutex> updateLock(updateMutex);
                    if (i > idx) {
                        idx = i;
                    }
                }
            }
        });
    }

    for (unsigned cpu = 0; cpu != threads; ++cpu) {
        futures[cpu].get();
    }
}
#else
void QBdtNodeInterface::_par_for_qbdt(const bitCapIntOcl begin, const bitCapIntOcl end, IncrementFunc fn)
{
    const bitCapIntOcl itemCount = end - begin;
    const bitCapIntOcl maxLcv = begin + itemCount;
    for (bitCapIntOcl j = begin; j < maxLcv; j++) {
        j |= fn(j, 0);
    }
}
#endif
} // namespace Qrack
