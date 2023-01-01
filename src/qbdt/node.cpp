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

#include "qbdt_node.hpp"

#if ENABLE_PTHREAD
#include <future>
#include <thread>
#endif
#include <set>

#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

const unsigned numThreads = std::thread::hardware_concurrency() << 1U;
#if ENABLE_ENV_VARS
const bitLenInt pStridePow =
    (bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW);
#else
const bitLenInt pStridePow = PSTRIDEPOW;
#endif

void QBdtNode::Prune(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NODE_0(scale)) {
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0U];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1U];

    // Prune recursively to depth.
    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (b0.get() == b1.get()) {
        b0->Prune(depth, parDepth);
    } else if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async, [&] { b0->Prune(depth, parDepth); });
        b1->Prune(depth, parDepth);

        future0.get();
    } else {
        b0->Prune(depth, parDepth);
        b1->Prune(depth, parDepth);
    }
#else
    b0->Prune(depth);
    if (b0.get() != b1.get()) {
        b1->Prune(depth);
    }
#endif

    if (IS_NODE_0(b0->scale)) {
        b0->SetZero();
        b1->scale /= abs(b1->scale);
    }
    if (IS_NODE_0(b1->scale)) {
        b0->scale /= abs(b0->scale);
        b1->SetZero();
    }

    const complex phaseFac =
        std::polar(ONE_R1, (real1)((b0->scale == ZERO_CMPLX) ? std::arg(b1->scale) : std::arg(b0->scale)));
    scale *= phaseFac;
    b0->scale /= phaseFac;
    if (b0.get() == b1.get()) {
        // Phase factor already applied, and branches point to same object.
        return;
    }
    b1->scale /= phaseFac;

    // Now, we try to combine pointers to equivalent branches.
    const bitCapInt depthPow = pow2(depth);
    // Combine single elements at bottom of full depth, up to where branches are equal below:
    _par_for_qbdt(depthPow, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf0 = b0;
        QBdtNodeInterfacePtr leaf1 = b1;

        for (bitLenInt j = 0U; j < depth; ++j) {
            size_t bit = SelectBit(i, depth - (j + 1U));

            if (!leaf0 || !leaf1 || (leaf0->branches[bit] == leaf1->branches[bit])) {
                // WARNING: Mutates loop control variable!
                return (bitCapInt)(pow2(depth - j) - ONE_BCI);
            }

            leaf0 = leaf0->branches[bit];
            leaf1 = leaf1->branches[bit];
        }

        return (bitCapInt)0U;
    });

    if (b0 == b1) {
        b1 = b0;
    }
}

void QBdtNode::Branch(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0U];
    QBdtNodeInterfacePtr& b1 = branches[1U];
    if (!b0) {
        b0 = std::make_shared<QBdtNode>(SQRT1_2_R1);
        b1 = std::make_shared<QBdtNode>(SQRT1_2_R1);
    } else {
        // Split all clones.
        b0 = b0->ShallowClone();
        b1 = b1->ShallowClone();
    }

    --depth;
    b0->Branch(depth);
    b1->Branch(depth);
}

void QBdtNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0U];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1U];

    --depth;
    const real1 nrm = (real1)sqrt(norm(b0->scale) + norm(b1->scale));
    b0->Normalize(depth);
    b0->scale *= ONE_R1 / nrm;
    if (b0.get() != b1.get()) {
        b1->Normalize(depth);
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBdtNode::PopStateVector(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0U];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1U];

    // Depth-first
    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if (b0.get() == b1.get()) {
        b0->PopStateVector(depth, parDepth);
    } else if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async, [&] { b0->PopStateVector(depth, parDepth); });
        b1->PopStateVector(depth, parDepth);

        future0.get();
    } else {
        b0->PopStateVector(depth, parDepth);
        b1->PopStateVector(depth, parDepth);
    }
#else
    b0->PopStateVector(depth);
    if (b0.get() != b1.get()) {
        b1->PopStateVector(depth);
    }
#endif

    const real1 nrm0 = norm(b0->scale);
    const real1 nrm1 = norm(b1->scale);

    if ((nrm0 + nrm1) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (nrm0 <= FP_NORM_EPSILON) {
        scale = b1->scale;
        b0->SetZero();
        b1->scale = ONE_CMPLX;
        return;
    }

    if (nrm1 <= FP_NORM_EPSILON) {
        scale = b0->scale;
        b0->scale = ONE_CMPLX;
        b1->SetZero();
        return;
    }

    scale = std::polar((real1)sqrt(nrm0 + nrm1), (real1)std::arg(b0->scale));
    b0->scale /= scale;
    b1->scale /= scale;
}

void QBdtNode::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size)
{
    if (norm(scale) == ZERO_R1) {
        return;
    }

    if (!depth) {
        if (!size) {
            return;
        }

        QBdtNodeInterfacePtr c = ShallowClone();
        scale = b->scale;
        branches[0U] = b->branches[0U]->ShallowClone();
        branches[1U] = b->branches[1U]->ShallowClone();

        InsertAtDepth(c, size, 0U);

        return;
    }
    --depth;

    if (!depth && size) {
        QBdtNodeInterfacePtr c = branches[0U];

        if (!IS_NODE_0(c->scale)) {
            branches[0U] = std::make_shared<QBdtNode>(c->scale, b->branches);
            branches[0U]->InsertAtDepth(c, size, 0);

            if (c.get() == branches[1U].get()) {
                branches[1U] = branches[0U];
                return;
            }
        }

        if (IS_NODE_0(branches[1U]->scale)) {
            return;
        }

        c = branches[1U];
        branches[1U] = std::make_shared<QBdtNode>(branches[1U]->scale, b->branches);
        branches[1U]->InsertAtDepth(c, size, 0U);

        return;
    }

    if (!branches[0U]) {
        return;
    }

    branches[0U]->InsertAtDepth(b, depth, size);
    if (branches[0U].get() != branches[1U].get()) {
        branches[1U]->InsertAtDepth(b, depth, size);
    }
}

#if ENABLE_COMPLEX_X2
void QBdtNode::Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, bitLenInt depth)
{
    if (!depth) {
        return;
    }

    Branch();
    QBdtNodeInterfacePtr& b0 = branches[0U];
    QBdtNodeInterfacePtr& b1 = branches[1U];

    if (IS_NORM_0(mtrxCol2.c[0U]) && IS_NORM_0(mtrxCol1.c[1U])) {
        b0->scale *= mtrxCol1.c[0U];
        b1->scale *= mtrxCol2.c[1U];
        Prune();

        return;
    }

    if (IS_NORM_0(mtrxCol1.c[0U]) && IS_NORM_0(mtrxCol2.c[1U])) {
        b0.swap(b1);
        b0->scale *= mtrxCol2.c[0U];
        b1->scale *= mtrxCol1.c[1U];
        Prune();

        return;
    }

    PushStateVector(mtrxCol1, mtrxCol2, b0, b1, depth);
    Prune(depth);
}

void QBdtNode::PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b0,
    QBdtNodeInterfacePtr& b1, bitLenInt depth, bitLenInt parDepth)
{
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
        qubit.c2 = matrixMul(mtrxCol1.c2, mtrxCol2.c2, qubit.c2);
        b0->scale = qubit.c[0U];
        b1->scale = qubit.c[1U];

        return;
    }

    if (b0->isEqualUnder(b1)) {
        complex2 qubit(b0->scale, b1->scale);
        qubit.c2 = matrixMul(mtrxCol1.c2, mtrxCol2.c2, qubit.c2);
        b0->scale = qubit.c[0U];
        b1->scale = qubit.c[1U];

        return;
    }

    if (!depth) {
        throw std::out_of_range("QBdtNode::PushStateVector() not implemented at depth=0! (You didn't push to root "
                                "depth, or root depth lacks method implementation.)");
    }

    b0->Branch();
    b1->Branch();

    if (!b0->branches[0U]) {
        b0->PushSpecial(mtrxCol1, mtrxCol2, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

    b0->branches[0U]->scale *= b0->scale;
    b0->branches[1U]->scale *= b0->scale;
    b0->scale = SQRT1_2_R1;

    b1->branches[0U]->scale *= b1->scale;
    b1->branches[1U]->scale *= b1->scale;
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(std::launch::async,
            [&] { PushStateVector(mtrxCol1, mtrxCol2, b0->branches[0U], b1->branches[0U], depth, parDepth); });
        PushStateVector(mtrxCol1, mtrxCol2, b0->branches[1U], b1->branches[1U], depth, parDepth);

        future0.get();
    } else {
        PushStateVector(mtrxCol1, mtrxCol2, b0->branches[0U], b1->branches[0U], depth, parDepth);
        PushStateVector(mtrxCol1, mtrxCol2, b0->branches[1U], b1->branches[1U], depth, parDepth);
    }
#else
    PushStateVector(mtrxCol1, mtrxCol2, b0->branches[0U], b1->branches[0U], depth);
    PushStateVector(mtrxCol1, mtrxCol2, b0->branches[1U], b1->branches[1U], depth);
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
    QBdtNodeInterfacePtr& b0 = branches[0U];
    QBdtNodeInterfacePtr& b1 = branches[1U];

    if (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U])) {
        b0->scale *= mtrx[0U];
        b1->scale *= mtrx[3U];
        Prune();

        return;
    }

    if (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U])) {
        b0.swap(b1);
        b0->scale *= mtrx[1U];
        b1->scale *= mtrx[2U];
        Prune();

        return;
    }

    PushStateVector(mtrx, b0, b1, depth);
    Prune(depth);
}

void QBdtNode::PushStateVector(
    complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth, bitLenInt parDepth)
{
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

    if (!b0->branches[0U]) {
        b0->PushSpecial(mtrx, b1);

        b0->PopStateVector();
        b1->PopStateVector();

        return;
    }

    b0->branches[0U]->scale *= b0->scale;
    b0->branches[1U]->scale *= b0->scale;
    b0->scale = SQRT1_2_R1;

    b1->branches[0U]->scale *= b1->scale;
    b1->branches[1U]->scale *= b1->scale;
    b1->scale = SQRT1_2_R1;

    --depth;
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    if ((depth >= pStridePow) && (pow2(parDepth) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(
            std::launch::async, [&] { PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth, parDepth); });
        PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth, parDepth);

        future0.get();
    } else {
        PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth, parDepth);
        PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth, parDepth);
    }
#else
    PushStateVector(mtrx, b0->branches[0U], b1->branches[0U], depth);
    PushStateVector(mtrx, b0->branches[1U], b1->branches[1U], depth);
#endif

    b0->PopStateVector();
    b1->PopStateVector();
}
#endif
} // namespace Qrack
