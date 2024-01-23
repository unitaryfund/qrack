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

#include "qbdt_node.hpp"

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#include <future>
#include <thread>
#endif

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
const unsigned numThreads = std::thread::hardware_concurrency() << 1U;
#if ENABLE_ENV_VARS
const bitLenInt pStridePow =
    (((bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW)) +
        7U) >> 1U;
#else
const bitLenInt pStridePow = (PSTRIDEPOW + 7U) >> 1U;
#endif
const bitCapInt pStride = pow2(pStridePow);
#endif

void QBdtNode::Prune(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr b0 = branches[0U];
    if (!b0) {
        SetZero();
        return;
    }
    QBdtNodeInterfacePtr b1 = branches[1U];

    // Prune recursively to depth.
    --depth;

    if (b0.get() == b1.get()) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(b0->mtx);
#endif
        b0->Prune(depth, parDepth);
    } else {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock(b0->mtx, b1->mtx);
        std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

        bitCapInt _t;
        bi_div_mod(pow2(depth), pStride, &_t, NULL);
        unsigned underThreads = (bitCapIntOcl)_t;

        if (underThreads == 1U) {
            underThreads = 0U;
        }
        if ((depth >= pStridePow) && (bi_compare((pow2(parDepth) * (underThreads + 1U)), numThreads) <= 0)) {
            ++parDepth;

            std::future<void> future0 = std::async(std::launch::async, [&] { b0->Prune(depth, parDepth); });
            b1->Prune(depth, parDepth);

            future0.get();
        } else {
            b0->Prune(depth, parDepth);
            b1->Prune(depth, parDepth);
        }
#else
        b0->Prune(depth, parDepth);
        b1->Prune(depth, parDepth);
#endif
    }

    Normalize();

    QBdtNodeInterfacePtr b0Ref = b0;
    QBdtNodeInterfacePtr b1Ref = b1;

    b0 = branches[0U];
    b1 = branches[1U];

    // When we lock a peer pair of (distinct) nodes, deadlock can arise from not locking both at once.
    // However, we can't assume that peer pairs of nodes don't point to the same memory (and mutex).
    // As we're locked on the node above already, our shared_ptr copies are safe.
    if (b0.get() == b1.get()) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(b0->mtx);
#endif

        const complex phaseFac = std::polar(ONE_R1, (real1)(std::arg(b0->scale)));
        scale *= phaseFac;
        b0->scale /= phaseFac;

        // Phase factor applied, and branches point to same object.
        return;
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

    if (IS_NODE_0(b0->scale)) {
        b0->SetZero();
        b1->scale /= abs(b1->scale);
    } else if (IS_NODE_0(b1->scale)) {
        b1->SetZero();
        b0->scale /= abs(b0->scale);
    }

    const complex phaseFac =
        std::polar(ONE_R1, (real1)((b0->scale == ZERO_CMPLX) ? std::arg(b1->scale) : std::arg(b0->scale)));
    scale *= phaseFac;

    b0->scale /= phaseFac;
    b1->scale /= phaseFac;

    // Now, we try to combine pointers to equivalent branches.
    const bitCapInt depthPow = pow2(depth);
    // Combine single elements at bottom of full depth, up to where branches are equal below:
    _par_for_qbdt(depthPow, [&](const bitCapInt& i) {
        const size_t topBit = SelectBit(i, depth - 1U);
        QBdtNodeInterfacePtr leaf0 = b0->branches[topBit];
        QBdtNodeInterfacePtr leaf1 = b1->branches[topBit];

        if (!leaf0 || !leaf1 || (leaf0.get() == leaf1.get())) {
            // WARNING: Mutates loop control variable!
            return (bitCapInt)(pow2(depth) - ONE_BCI);
        }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            std::lock(leaf0->mtx, leaf1->mtx);
            std::lock_guard<std::mutex> lock00(leaf0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock11(leaf1->mtx, std::adopt_lock);

            if (leaf0->isEqual(leaf1)) {
                b1->branches[topBit] = b0->branches[topBit];

                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(depth) - ONE_BCI);
            }
        }
#else
        if (leaf0->isEqual(leaf1)) {
            b1->branches[topBit] = b0->branches[topBit];

            // WARNING: Mutates loop control variable!
            return (bitCapInt)(pow2(depth) - ONE_BCI);
        }
#endif

        for (bitLenInt j = 1U; j < depth; ++j) {
            size_t bit = SelectBit(i, depth - (j + 1U));

            const QBdtNodeInterfacePtr& lRef = leaf0;
            const QBdtNodeInterfacePtr& rRef = leaf1;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock(lRef->mtx, rRef->mtx);
            std::lock_guard<std::mutex> lock0(lRef->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock1(rRef->mtx, std::adopt_lock);
#endif

            leaf0 = lRef->branches[bit];
            leaf1 = rRef->branches[bit];

            if (!leaf0 || !leaf1 || (leaf0.get() == leaf1.get())) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(depth - j) - ONE_BCI);
            }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock(leaf0->mtx, leaf1->mtx);
            std::lock_guard<std::mutex> lock00(leaf0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock11(leaf1->mtx, std::adopt_lock);
#endif

            if (leaf0->isEqual(leaf1)) {
                lRef->branches[bit] = rRef->branches[bit];

                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(depth - j) - ONE_BCI);
            }
        }

        return ZERO_BCI;
    });

    if (b0 == b1) {
        branches[1U] = branches[0U];
    }
}

void QBdtNode::Branch(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    if (!branches[0U] || !branches[1U]) {
        branches[0U] = std::make_shared<QBdtNode>(SQRT1_2_R1);
        branches[1U] = std::make_shared<QBdtNode>(SQRT1_2_R1);
    } else {
        // Split all clones.
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            QBdtNodeInterfacePtr b0 = branches[0U];
            std::lock_guard<std::mutex> lock(b0->mtx);
            branches[0U] = b0->ShallowClone();
        }
        if (true) {
            QBdtNodeInterfacePtr b1 = branches[1U];
            std::lock_guard<std::mutex> lock(b1->mtx);
            branches[1U] = b1->ShallowClone();
        }
#else
        branches[0U] = branches[0U]->ShallowClone();
        branches[1U] = branches[1U]->ShallowClone();
#endif
    }

    --depth;

    QBdtNodeInterfacePtr& b0 = branches[0U];
    QBdtNodeInterfacePtr& b1 = branches[1U];

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth <= pStridePow) || (bi_compare(pow2(parDepth), numThreads) > 0)) {
        if (true) {
            std::lock_guard<std::mutex> lock(b0->mtx);
            b0->Branch(depth, parDepth);
        }
        if (true) {
            std::lock_guard<std::mutex> lock(b1->mtx);
            b1->Branch(depth, parDepth);
        }
        return;
    }

    ++parDepth;

    std::future<void> future0 = std::async(std::launch::async, [&] {
        std::lock_guard<std::mutex> lock(b0->mtx);
        b0->Branch(depth, parDepth);
    });
    if (true) {
        std::lock_guard<std::mutex> lock(b1->mtx);
        b1->Branch(depth, parDepth);
    }
    future0.get();
#else
    b0->Branch(depth, parDepth);
    b1->Branch(depth, parDepth);
#endif
}

void QBdtNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr b0 = branches[0U];
    if (!b0) {
        SetZero();
        return;
    }
    QBdtNodeInterfacePtr b1 = branches[1U];

    --depth;
    if (b0.get() == b1.get()) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(b0->mtx);
#endif

        const real1 nrm = (real1)sqrt(2 * norm(b0->scale));

        b0->Normalize(depth);
        b0->scale *= ONE_R1 / nrm;
    } else {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock(b0->mtx, b1->mtx);
        std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

        const real1 nrm = sqrt(norm(b0->scale) + norm(b1->scale));

        b0->Normalize(depth);
        b1->Normalize(depth);

        b0->scale *= ONE_R1 / nrm;
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBdtNode::PopStateVector(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr b0 = branches[0U];
    if (!b0) {
        SetZero();
        return;
    }
    QBdtNodeInterfacePtr b1 = branches[1U];

    // Depth-first
    --depth;
    if (b0.get() == b1.get()) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lock(b0->mtx);
#endif
        b0->PopStateVector(depth);

        const real1 nrm = (real1)(2 * norm(b0->scale));

        if (nrm <= _qrack_qbdt_sep_thresh) {
            scale = ZERO_CMPLX;
            branches[0U] = NULL;
            branches[1U] = NULL;

            return;
        }

        scale = std::polar((real1)sqrt(nrm), (real1)std::arg(b0->scale));
        b0->scale /= scale;

        return;
    }

    ++parDepth;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

    b0->PopStateVector(depth);
    b1->PopStateVector(depth);

    const real1 nrm0 = norm(b0->scale);
    const real1 nrm1 = norm(b1->scale);

    if ((nrm0 + nrm1) <= _qrack_qbdt_sep_thresh) {
        scale = ZERO_CMPLX;
        branches[0U] = NULL;
        branches[1U] = NULL;

        return;
    }

    if (nrm0 <= _qrack_qbdt_sep_thresh) {
        scale = b1->scale;
        b0->SetZero();
        b1->scale = ONE_CMPLX;
        return;
    }

    if (nrm1 <= _qrack_qbdt_sep_thresh) {
        scale = b0->scale;
        b0->scale = ONE_CMPLX;
        b1->SetZero();
        return;
    }

    scale = std::polar((real1)sqrt(nrm0 + nrm1), (real1)std::arg(b0->scale));
    b0->scale /= scale;
    b1->scale /= scale;
}

void QBdtNode::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (!b) {
        return;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr b0 = branches[0U];
    QBdtNodeInterfacePtr b1 = branches[1U];

    if (!depth) {
        if (!size) {
            return;
        }

        QBdtNodeInterfacePtr c = ShallowClone();

        if (true) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> lock(b->mtx);
#endif
            scale = b->scale;
            branches[0U] = b->branches[0U]->ShallowClone();
            branches[1U] = b->branches[1U]->ShallowClone();
        }

        InsertAtDepth(c, size, 0U, parDepth);

        return;
    }
    --depth;

    if (b0.get() == b1.get()) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lockb(b->mtx);
        std::lock_guard<std::mutex> lock(b0->mtx);
#endif

        if (!depth && size) {
            QBdtNodeInterfacePtr n0 = std::make_shared<QBdtNode>(b0->scale, b->branches);
            branches[0U] = n0;
            branches[1U] = n0;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> nLock(n0->mtx);
#endif
            n0->InsertAtDepth(b, size, 0U, parDepth);

            return;
        }

        b0->InsertAtDepth(b, depth, size, parDepth);

        return;
    }

    if (!depth && size) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        std::lock_guard<std::mutex> lockb(b->mtx);
        std::lock(b0->mtx, b1->mtx);
        std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

        if (IS_NODE_0(b0->scale)) {
            branches[1U] = std::make_shared<QBdtNode>(b1->scale, b->branches);
            QBdtNodeInterfacePtr n1 = branches[1U];
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> nLock(n1->mtx);
#endif
            n1->InsertAtDepth(b, size, 0U, parDepth);
        } else if (IS_NODE_0(b0->scale)) {
            branches[0U] = std::make_shared<QBdtNode>(b0->scale, b->branches);
            QBdtNodeInterfacePtr n0 = branches[0U];
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            std::lock_guard<std::mutex> nLock(n0->mtx);
#endif
            n0->InsertAtDepth(b, size, 0U, parDepth);
        } else {
            branches[0U] = std::make_shared<QBdtNode>(b0->scale, b->branches);
            branches[1U] = std::make_shared<QBdtNode>(b1->scale, b->branches);
            QBdtNodeInterfacePtr n0 = branches[0U];
            QBdtNodeInterfacePtr n1 = branches[1U];

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
            // These were just created, so there's no chance of deadlock in separate locks.
            std::lock_guard<std::mutex> nLock0(n0->mtx);
            std::lock_guard<std::mutex> nLock1(n1->mtx);

            if ((depth >= pStridePow) || (bi_compare(pow2(parDepth), numThreads) <= 0)) {
                ++parDepth;

                std::future<void> future0 =
                    std::async(std::launch::async, [&] { n0->InsertAtDepth(b, size, 0U, parDepth); });
                n1->InsertAtDepth(b, size, 0U, parDepth);

                future0.get();
            } else {
                n0->InsertAtDepth(b, size, 0U, parDepth);
                n1->InsertAtDepth(b, size, 0U, parDepth);
            }
#else
            n0->InsertAtDepth(b, size, 0U, parDepth);
            n1->InsertAtDepth(b, size, 0U, parDepth);
#endif
        }

        return;
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

    if ((depth >= pStridePow) || (bi_compare(pow2(parDepth), numThreads) <= 0)) {
        ++parDepth;

        std::future<void> future0 =
            std::async(std::launch::async, [&] { b0->InsertAtDepth(b, depth, size, parDepth); });
        b1->InsertAtDepth(b, depth, size, parDepth);

        future0.get();
    } else {
        b0->InsertAtDepth(b, depth, size, parDepth);
        b1->InsertAtDepth(b, depth, size, parDepth);
    }
#else
    b0->InsertAtDepth(b, depth, size, parDepth);
    b1->InsertAtDepth(b, depth, size, parDepth);
#endif
}

#ifdef ENABLE_COMPLEX_X2
void QBdtNode::Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
    const complex2& mtrxColShuff2, bitLenInt depth)
{
    if (!depth) {
        return;
    }

    Branch();
    QBdtNodeInterfacePtr b0 = branches[0U];
    QBdtNodeInterfacePtr b1 = branches[1U];

    if (IS_NORM_0(mtrxCol2.c(0U)) && IS_NORM_0(mtrxCol1.c(1U))) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            std::lock(b0->mtx, b1->mtx);
            std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

            b0->scale *= mtrxCol1.c(0U);
            b1->scale *= mtrxCol2.c(1U);
        }
#else
        b0->scale *= mtrxCol1.c(0U);
        b1->scale *= mtrxCol2.c(1U);
#endif
        Prune();

        return;
    }

    if (IS_NORM_0(mtrxCol1.c(0U)) && IS_NORM_0(mtrxCol2.c(1U))) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            std::lock(b0->mtx, b1->mtx);
            std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

            branches[0U].swap(branches[1U]);
            b1->scale *= mtrxCol2.c(0U);
            b0->scale *= mtrxCol1.c(1U);
        }
#else
        branches[0U].swap(branches[1U]);
        b1->scale *= mtrxCol2.c(0U);
        b0->scale *= mtrxCol1.c(1U);
#endif
        Prune();

        return;
    }

    PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, branches[0U], branches[1U], depth);
    Prune(depth);
}

void QBdtNode::PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
    const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
    bitLenInt parDepth)
{
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

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

    b0->Branch();
    b1->Branch();

    // For parallelism, keep shared_ptr from deallocating.
    QBdtNodeInterfacePtr& b00 = b0->branches[0U];
    QBdtNodeInterfacePtr& b01 = b0->branches[1U];
    QBdtNodeInterfacePtr& b10 = b1->branches[0U];
    QBdtNodeInterfacePtr& b11 = b1->branches[1U];

    if (!b00) {
        b0->PushSpecial(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (true) {
        std::lock(b00->mtx, b01->mtx);
        std::lock_guard<std::mutex> lock0(b00->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b01->mtx, std::adopt_lock);
        b00->scale *= b0->scale;
        b01->scale *= b0->scale;
    }
#else
    b00->scale *= b0->scale;
    b01->scale *= b0->scale;
#endif
    b0->scale = SQRT1_2_R1;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (true) {
        std::lock(b10->mtx, b11->mtx);
        std::lock_guard<std::mutex> lock0(b10->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b11->mtx, std::adopt_lock);
        b10->scale *= b1->scale;
        b11->scale *= b1->scale;
    }
#else
    b10->scale *= b1->scale;
    b11->scale *= b1->scale;
#endif
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) || (bi_compare(pow2(parDepth), numThreads) <= 0)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async,
            [&] { b0->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b00, b10, depth, parDepth); });
        b1->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b01, b11, depth, parDepth);

        future0.get();
    } else {
        b0->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b00, b10, depth, parDepth);
        b1->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b01, b11, depth, parDepth);
    }
#else
    b0->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b00, b10, depth);
    b1->PushStateVector(mtrxCol1, mtrxCol2, mtrxColShuff1, mtrxColShuff2, b01, b11, depth);
#endif

    b0->PopStateVector();
    b1->PopStateVector();
}
#else
void QBdtNode::Apply2x2(complex const* mtrx, bitLenInt depth)
{
    if (!depth) {
        return;
    }

    Branch();
    QBdtNodeInterfacePtr b0 = branches[0U];
    QBdtNodeInterfacePtr b1 = branches[1U];

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            std::lock(b0->mtx, b1->mtx);
            std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

            b0->scale *= mtrx[0U];
            b1->scale *= mtrx[3U];
        }
#else
        b0->scale *= mtrx[0U];
        b1->scale *= mtrx[3U];
#endif
        Prune();

        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
        if (true) {
            std::lock(b0->mtx, b1->mtx);
            std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
            std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);

            branches[0U].swap(branches[1U]);
            b1->scale *= mtrx[1U];
            b0->scale *= mtrx[2U];
        }
#else
        branches[0U].swap(branches[1U]);
        b1->scale *= mtrx[1U];
        b0->scale *= mtrx[2U];
#endif
        Prune();

        return;
    }

    PushStateVector(mtrx, branches[0U], branches[1U], depth);
    Prune(depth);
}

void QBdtNode::PushStateVector(
    complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth, bitLenInt parDepth)
{
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    std::lock(b0->mtx, b1->mtx);
    std::lock_guard<std::mutex> lock0(b0->mtx, std::adopt_lock);
    std::lock_guard<std::mutex> lock1(b1->mtx, std::adopt_lock);
#endif

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
    QBdtNodeInterfacePtr& b00 = b0->branches[0U];
    QBdtNodeInterfacePtr& b01 = b0->branches[1U];
    QBdtNodeInterfacePtr& b10 = b1->branches[0U];
    QBdtNodeInterfacePtr& b11 = b1->branches[1U];

    if (!b00) {
        b0->PushSpecial(mtrx, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (true) {
        std::lock(b00->mtx, b01->mtx);
        std::lock_guard<std::mutex> lock0(b00->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b01->mtx, std::adopt_lock);
        b00->scale *= b0->scale;
        b01->scale *= b0->scale;
    }
#else
    b00->scale *= b0->scale;
    b01->scale *= b0->scale;
#endif
    b0->scale = SQRT1_2_R1;

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (true) {
        std::lock(b10->mtx, b11->mtx);
        std::lock_guard<std::mutex> lock0(b10->mtx, std::adopt_lock);
        std::lock_guard<std::mutex> lock1(b11->mtx, std::adopt_lock);
        b10->scale *= b1->scale;
        b11->scale *= b1->scale;
    }
#else
    b10->scale *= b1->scale;
    b11->scale *= b1->scale;
#endif
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) && (bi_compare(pow2(parDepth), numThreads) <= 0)) {
        ++parDepth;

        std::future<void> future0 =
            std::async(std::launch::async, [&] { b0->PushStateVector(mtrx, b00, b10, depth, parDepth); });
        b1->PushStateVector(mtrx, b01, b11, depth, parDepth);

        future0.get();
    } else {
        b0->PushStateVector(mtrx, b00, b10, depth, parDepth);
        b1->PushStateVector(mtrx, b01, b11, depth, parDepth);
    }
#else
    b0->PushStateVector(mtrx, b00, b10, depth);
    b1->PushStateVector(mtrx, b01, b11, depth);
#endif

    b0->PopStateVector();
    b1->PopStateVector();
}
#endif
} // namespace Qrack
