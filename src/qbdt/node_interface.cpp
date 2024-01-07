//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#include <future>
#include <thread>
#endif

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_SAME_AMP(a, b) (norm((a) - (b)) <= _qrack_qbdt_sep_thresh)
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#endif

namespace Qrack {

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
const unsigned numThreads = std::thread::hardware_concurrency() << 1U;
#if ENABLE_ENV_VARS
const bitLenInt pStridePow =
    (((bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW)) +
        1U) >>
    1U;
#else
const bitLenInt pStridePow = (PSTRIDEPOW + 1U) >> 1U;
#endif
const bitCapInt pStride = pow2(pStridePow);
#endif

bool operator==(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs)
{
    if (!lhs) {
        return !rhs;
    }

    return lhs->isEqual(rhs);
}

bool operator!=(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs) { return !(lhs == rhs); }

bool QBdtNodeInterface::isEqual(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (!IS_SAME_AMP(scale, r->scale)) {
        return false;
    }

    return isEqualUnder(r);
}

bool QBdtNodeInterface::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (this == r.get()) {
        return true;
    }

    return isEqualBranch(r, 0U) && isEqualBranch(r, 1U);
}

bool QBdtNodeInterface::isEqualBranch(QBdtNodeInterfacePtr r, const bool& b)
{
    const size_t _b = b ? 1U : 0U;

    if (!branches[_b] || !r->branches[_b]) {
        return !branches[_b] == !r->branches[_b];
    }

    QBdtNodeInterfacePtr& lLeaf = branches[_b];
    QBdtNodeInterfacePtr& rLeaf = r->branches[_b];

    if (lLeaf.get() == rLeaf.get()) {
        return true;
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(lLeaf->mtx, rLeaf->mtx);
    std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);
#endif

    if (lLeaf != rLeaf) {
        return false;
    }

    // These lLeaf and rLeaf are deemed equal.
    // Since we allow approximation in determining equality,
    // amortize error by (L2 or L1) averaging scale.
    // (All other update operations on the branches are blocked by the mutexes.)
    // We can weight by square use_count() of each leaf, which should roughly
    // correspond to the number of branches that point to each node.

    const real1 lWeight = (real1)(lLeaf.use_count() * lLeaf.use_count());
    const real1 rWeight = (real1)(rLeaf.use_count() * rLeaf.use_count());
    const complex nScale = (lWeight * lLeaf->scale + rWeight * rLeaf->scale) / (lWeight + rWeight);

    if (IS_NODE_0(nScale)) {
        lLeaf->SetZero();
        rLeaf->SetZero();
    } else {
        lLeaf->scale = nScale;
        rLeaf->scale = nScale;
    }

    // Set the branches equal.
    rLeaf = lLeaf;

    return true;
}

QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(
    bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (IS_NODE_0(scale)) {
        SetZero();
        return NULL;
    }

    Branch();

    if (depth) {
        --depth;

        QBdtNodeInterfacePtr toRet1, toRet2;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if ((depth >= pStridePow) && (pow2Ocl(parDepth) <= numThreads)) {
            ++parDepth;
            std::future<QBdtNodeInterfacePtr> future0 = std::async(std::launch::async, [&] {
                std::lock_guard<std::mutex> lock(branches[0U]->mtx);
                return branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
            });
            if (true) {
                std::lock_guard<std::mutex> lock(branches[1U]->mtx);
                toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
            }
            toRet1 = future0.get();
        } else {
            if (true) {
                std::lock_guard<std::mutex> lock(branches[0U]->mtx);
                toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
            }
            if (true) {
                std::lock_guard<std::mutex> lock(branches[1U]->mtx);
                toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
            }
        }
#else
        toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
        toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
#endif

        return !toRet1
            ? toRet2
            : (!toRet2 ? toRet1 : ((norm(branches[1U]->scale) > norm(branches[0U]->scale)) ? toRet2 : toRet1));
    }

    QBdtNodeInterfacePtr toRet = ShallowClone();
    toRet->scale /= abs(toRet->scale);

    if (!size) {
        branches[0U] = NULL;
        branches[1U] = NULL;

        return toRet;
    }

    QBdtNodeInterfacePtr temp = toRet->RemoveSeparableAtDepth(size, 0);
    branches[0U] = temp->branches[0U];
    branches[1U] = temp->branches[1U];

    return toRet;
}

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
void QBdtNodeInterface::_par_for_qbdt(const bitCapInt end, BdtFunc fn)
{
    const bitCapInt Stride = pStride;
    bitCapInt _t;
    bi_div_mod(end, pStride, &_t, NULL);
    unsigned threads = (bitCapIntOcl)_t;
    if (threads > numThreads) {
        threads = numThreads;
    }

    if (threads <= 1U) {
        for (bitCapInt j = ZERO_BCI; bi_compare(j, end) < 0; bi_increment(&j, 1U)) {
            bi_or_ip(&j, fn(j));
        }
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = ZERO_BCI;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&myMutex, &idx, &end, &Stride, fn) {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx;
                    bi_increment(&idx, 1U);
                }
                const bitCapInt l = i * Stride;
                if (bi_compare(l, end) >= 0) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                bitCapInt j;
                for (j = ZERO_BCI; bi_compare(j, maxJ) < 0; bi_increment(&j, 1U)) {
                    bitCapInt k = j + l;
                    bi_or_ip(&k, fn(k));
                    j = k - l;
                    if (bi_compare(j, maxJ) >= 0) {
                        std::lock_guard<std::mutex> lock(myMutex);
                        bitCapInt _j;
                        bi_div_mod(j, Stride, &_j, NULL);
                        bi_or_ip(&idx, _j);
                        break;
                    }
                }
            }
        }));
    }

    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures[cpu].get();
    }
}
#else
void QBdtNodeInterface::_par_for_qbdt(const bitCapInt end, BdtFunc fn)
{
    for (bitCapInt j = 0U; bi_compare(j, end) < 0; bi_increment(&j, 1U)) {
        bi_or_ip(&j, fn(j));
    }
}
#endif
} // namespace Qrack
