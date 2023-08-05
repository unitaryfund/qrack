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

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#include <future>
#include <thread>
#endif

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_SAME_AMP(a, b) (abs((a) - (b)) <= REAL1_EPSILON)
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

    if (this == r.get()) {
        return true;
    }

    if (!IS_SAME_AMP(scale, r->scale)) {
        return false;
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[0U].get() != r->branches[0U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[0U];
        QBdtNodeInterfacePtr rLeaf = r->branches[0U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[0U] = r->branches[0U];
    }

    if ((!branches[1U]) != (!r->branches[1U])) {
        return false;
    }

    if (branches[1U].get() != r->branches[1U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[1U];
        QBdtNodeInterfacePtr rLeaf = r->branches[1U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[1U] = r->branches[1U];
    }

    return true;
}

bool QBdtNodeInterface::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (IS_NODE_0(scale)) {
        return IS_NODE_0(r->scale);
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[0U].get() != r->branches[0U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[0U];
        QBdtNodeInterfacePtr rLeaf = r->branches[0U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> rLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[0U] = r->branches[0U];
    }

    if ((!branches[0U]) != (!r->branches[0U])) {
        return false;
    }

    if (branches[1U].get() != r->branches[1U].get()) {
        QBdtNodeInterfacePtr lLeaf = branches[1U];
        QBdtNodeInterfacePtr rLeaf = r->branches[1U];
        std::lock(lLeaf->mtx, rLeaf->mtx);
        std::lock_guard<std::mutex> lLock(lLeaf->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> rLock(rLeaf->mtx, std::adopt_lock);

        if (lLeaf != rLeaf) {
            return false;
        }

        branches[1U] = r->branches[1U];
    }

    return true;
}

QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(
    bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (IS_NODE_0(scale)) {
        return NULL;
    }

    Branch();

    if (depth) {
        --depth;

        QBdtNodeInterfacePtr toRet1, toRet2;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
            ++parDepth;
            std::future<QBdtNodeInterfacePtr> future0 = std::async(
                std::launch::async, [&] { return branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth); });
            toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
            toRet1 = future0.get();
        } else {
            toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
            toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
        }
#else
        toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size, parDepth);
        toRet2 = branches[1U]->RemoveSeparableAtDepth(depth, size, parDepth);
#endif

        return (norm(branches[0U]->scale) > norm(branches[1U]->scale)) ? toRet1 : toRet2;
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

#if ENABLE_COMPLEX_X2
void QBdtNodeInterface::PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2,
    const complex2& mtrxColShuff1, const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
    bitLenInt depth, bitLenInt parDepth)
{
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

    const bool isB0Zero = IS_NODE_0(b0->scale);
    const bool isB1Zero = IS_NODE_0(b1->scale);

    if (isB0Zero && isB1Zero) {
        b0->SetZero();
        b1->SetZero();

        return;
    }

    if (isB0Zero) {
        b0 = b1->ShallowClone();
        b0->scale = ZERO_CMPLX;
    }

    if (isB1Zero) {
        b1 = b0->ShallowClone();
        b1->scale = ZERO_CMPLX;
    }

    // TODO: If the shuffled columns are passed in, much work can be avoided.

    if (isB0Zero || isB1Zero) {
        complex2 qubit(b0->scale, b1->scale);
        qubit = matrixMul(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, qubit);
        b0->scale = qubit.c(0U);
        b1->scale = qubit.c(1U);

        return;
    }

    if (b0->isEqualUnder(b1)) {
        complex2 qubit(b0->scale, b1->scale);
        qubit = matrixMul(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, qubit);
        b0->scale = qubit.c(0U);
        b1->scale = qubit.c(1U);

        return;
    }

    if (!depth) {
        throw std::out_of_range("QBdtNode::PushStateVector() not implemented at depth=0! (You didn't push to root "
                                "depth, or root depth lacks method implementation.)");
    }

    b0 = b0->PopSpecial();
    b1 = b1->PopSpecial();

    b0->Branch();
    b1->Branch();

    // For parallelism, keep shared_ptr from deallocating.
    QBdtNodeInterfacePtr b00 = b0->branches[0U];
    QBdtNodeInterfacePtr b01 = b0->branches[1U];
    QBdtNodeInterfacePtr b10 = b1->branches[0U];
    QBdtNodeInterfacePtr b11 = b1->branches[1U];

    if (!b00) {
        b0->PushSpecial(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

    if (true) {
        std::lock(b00->mtx, b01->mtx);
        std::lock_guard<std::mutex> lock0(b00->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b01->mtx, std::adopt_lock);
        b00->scale *= b0->scale;
        b01->scale *= b0->scale;
    }
    b0->scale = SQRT1_2_R1;

    if (true) {
        std::lock(b10->mtx, b11->mtx);
        std::lock_guard<std::mutex> lock0(b10->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b11->mtx, std::adopt_lock);
        b10->scale *= b1->scale;
        b11->scale *= b1->scale;
    }
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async, [&] {
            b0->PushStateVector(
                mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[0U], b1->branches[0U], depth, parDepth);
        });
        b1->PushStateVector(
            mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[1U], b1->branches[1U], depth, parDepth);

        future0.get();
    } else {
        b0->PushStateVector(
            mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[0U], b1->branches[0U], depth, parDepth);
        b1->PushStateVector(
            mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[1U], b1->branches[1U], depth, parDepth);
    }
#else
    b0->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[0U], b1->branches[0U], depth);
    b1->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b0->branches[1U], b1->branches[1U], depth);
#endif

    b0->PopStateVector();
    b1->PopStateVector();
}
#else
void QBdtNodeInterface::PushStateVector(
    complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth, bitLenInt parDepth)
{
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

    const bool isB0Zero = IS_NODE_0(b0->scale);
    const bool isB1Zero = IS_NODE_0(b1->scale);

    if (isB0Zero && isB1Zero) {
        b0->SetZero();
        b1->SetZero();

        return;
    }

    if (isB0Zero) {
        b0 = b1->ShallowClone();
        b0->scale = ZERO_CMPLX;
    }

    if (isB1Zero) {
        b1 = b0->ShallowClone();
        b1->scale = ZERO_CMPLX;
    }

    if (isB0Zero || isB1Zero) {
        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0U] * Y0 + mtrx[1U] * Y1;
        b1->scale = mtrx[2U] * Y0 + mtrx[3U] * Y1;

        return;
    }

    if (b0->isEqualUnder(b1)) {
        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0U] * Y0 + mtrx[1U] * Y1;
        b1->scale = mtrx[2U] * Y0 + mtrx[3U] * Y1;

        return;
    }

    if (!depth) {
        throw std::out_of_range("QBdtNode::PushStateVector() not implemented at depth=0! (You didn't push to root "
                                "depth, or root depth lacks method implementation.)");
    }

    b0->Branch();
    b1->Branch();

    // For parallelism, keep shared_ptr from deallocating.
    QBdtNodeInterfacePtr b00 = b0->branches[0U];
    QBdtNodeInterfacePtr b01 = b0->branches[1U];
    QBdtNodeInterfacePtr b10 = b1->branches[0U];
    QBdtNodeInterfacePtr b11 = b1->branches[1U];

    if (!b00) {
        b0->PushSpecial(mtrx, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

    if (true) {
        std::lock(b00->mtx, b01->mtx);
        std::lock_guard<std::mutex> lock0(b00->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b01->mtx, std::adopt_lock);
        b00->scale *= b0->scale;
        b01->scale *= b0->scale;
    }
    b0->scale = SQRT1_2_R1;

    if (true) {
        std::lock(b10->mtx, b11->mtx);
        std::lock_guard<std::mutex> lock0(b10->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b11->mtx, std::adopt_lock);
        b10->scale *= b1->scale;
        b11->scale *= b1->scale;
    }
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async,
            [&] { b0->PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth, parDepth); });
        b1->PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth, parDepth);

        future0.get();
    } else {
        b0->PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth, parDepth);
        b1->PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth, parDepth);
    }
#else
    b0->PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth);
    b1->PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth);
#endif

    b0->PopStateVector();
    b1->PopStateVector();
}
#endif

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
void QBdtNodeInterface::_par_for_qbdt(const bitCapInt end, BdtFunc fn)
{
    const bitCapInt Stride = pStride;
    unsigned threads = (unsigned)(end / pStride);
    if (threads > numThreads) {
        threads = numThreads;
    }

    if (threads <= 1U) {
        for (bitCapInt j = 0U; j < end; ++j) {
            j |= fn(j);
        }
        return;
    }

    std::mutex myMutex;
    bitCapInt idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&myMutex, &idx, &end, &Stride, fn) {
            for (;;) {
                bitCapInt i;
                if (true) {
                    std::lock_guard<std::mutex> lock(myMutex);
                    i = idx++;
                }
                const bitCapInt l = i * Stride;
                if (l >= end) {
                    break;
                }
                const bitCapInt maxJ = ((l + Stride) < end) ? Stride : (end - l);
                bitCapInt j;
                for (j = 0U; j < maxJ; ++j) {
                    bitCapInt k = j + l;
                    k |= fn(k);
                    j = k - l;
                    if (j >= maxJ) {
                        std::lock_guard<std::mutex> lock(myMutex);
                        idx |= j / Stride;
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
    for (bitCapInt j = 0U; j < end; ++j) {
        j |= fn(j);
    }
}
#endif
} // namespace Qrack
