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
    if (!r || !IS_SAME_AMP(scale, r->scale)) {
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
    const double lSqr = ((double)lLeaf.use_count()) * ((double)lLeaf.use_count());
    const double rSqr = ((double)rLeaf.use_count()) * ((double)rLeaf.use_count());
    const double denom = lSqr + rSqr;
    const complex nScale = (real1)(lSqr / denom) * lLeaf->scale + (real1)(rSqr / denom) * rLeaf->scale;

    lLeaf->scale = nScale;

    // Set the branches equal.
    rLeaf = lLeaf;

    return true;
}

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(
    bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
#else
QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size)
#endif
{
    if (IS_NODE_0(scale)) {
        SetZero();
        return nullptr;
    }

    Branch();

    if (depth) {
        --depth;

        QBdtNodeInterfacePtr toRet1, toRet2;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (pow2Ocl(parDepth) <= numThreads) {
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
        toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size);
        toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size);
#endif

        return !toRet1
            ? toRet2
            : (!toRet2 ? toRet1 : ((norm(branches[1U]->scale) > norm(branches[0U]->scale)) ? toRet2 : toRet1));
    }

    QBdtNodeInterfacePtr toRet = ShallowClone();
    toRet->scale /= abs(toRet->scale);

    if (!size) {
        branches[0U] = nullptr;
        branches[1U] = nullptr;

        return toRet;
    }

    QBdtNodeInterfacePtr temp = toRet->RemoveSeparableAtDepth(size, 0);
    branches[0U] = temp->branches[0U];
    branches[1U] = temp->branches[1U];

    return toRet;
}

void QBdtNodeInterface::_par_for_qbdt(const bitCapInt& end, BdtFunc fn)
{
    for (bitCapInt j = 0U; bi_compare(j, end) < 0; bi_increment(&j, 1U)) {
        bi_or_ip(&j, fn(j));
    }
}
} // namespace Qrack
