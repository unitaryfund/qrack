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
#endif
#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

void QBdtNode::Prune(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1];

    // Prune recursively to depth.
    depth--;
    branches[0]->Prune(depth);
    if (b0.get() != b1.get()) {
        branches[1]->Prune(depth);
    }

    const complex phaseFac =
        std::polar(ONE_R1, (real1)(IS_NORM_0(b0->scale) ? std::arg(b1->scale) : std::arg(b0->scale)));
    scale *= phaseFac;
    b0->scale /= phaseFac;
    if (b0.get() == b1.get()) {
        // Phase factor already applied, and branches point to same object.
        return;
    }
    b1->scale /= phaseFac;

    // Now, we try to combine pointers to equivalent branches.
    const bitCapIntOcl depthPow = (bitCapIntOcl)ONE_BCI << depth;
    // Combine single elements at bottom of full depth, up to where branches are equal below:
    _par_for_qbdt(0, depthPow, [&](const bitCapInt& i, const unsigned& cpu) {
        QBdtNodeInterfacePtr leaf0 = b0;
        QBdtNodeInterfacePtr leaf1 = b1;

        for (bitLenInt j = 0; j < depth; j++) {
            size_t bit = SelectBit(i, depth - (j + 1U));

            if (!leaf0 || !leaf1) {
                break;
            }

            if (leaf0->branches[bit] == leaf1->branches[bit]) {
                leaf0->branches[bit] = leaf1->branches[bit];
                // WARNING: Mutates loop control variable!
                return (bitCapIntOcl)(((bitCapIntOcl)ONE_BCI << (depth - j)) - ONE_BCI);
            }

            leaf0 = leaf0->branches[bit];
            leaf1 = leaf1->branches[bit];
        }

        return (bitCapIntOcl)0U;
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
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0];
    QBdtNodeInterfacePtr& b1 = branches[1];
    if (!b0) {
        b0 = std::make_shared<QBdtNode>(SQRT1_2_R1);
        b1 = std::make_shared<QBdtNode>(SQRT1_2_R1);
    } else {
        // Split all clones.
        b0 = b0->ShallowClone();
        b1 = b1->ShallowClone();
    }

    depth--;
    b0->Branch(depth);
    b1->Branch(depth);
}

void QBdtNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1];

    const real1 nrm = (real1)sqrt(norm(b0->scale) + norm(b1->scale));
    b0->Normalize(depth - 1U);
    b0->scale *= ONE_R1 / nrm;
    if (b0.get() != b1.get()) {
        b1->Normalize(depth - 1U);
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBdtNode::PopStateVector(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBdtNodeInterfacePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBdtNodeInterfacePtr& b1 = branches[1];

    // Depth-first
    depth--;
    b0->PopStateVector(depth);
    if (b0.get() != b1.get()) {
        b1->PopStateVector(depth);
    }

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

void QBdtNode::Apply2x2(const complex* mtrx, bitLenInt depth)
{
    if (!depth) {
        return;
    }

    Branch();
    QBdtNodeInterfacePtr& b0 = branches[0];
    QBdtNodeInterfacePtr& b1 = branches[1];

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        b0->scale *= mtrx[0];
        b1->scale *= mtrx[3];
        Prune();

        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        b0.swap(b1);
        b0->scale *= mtrx[1];
        b1->scale *= mtrx[2];
        Prune();

        return;
    }

    PushStateVector(mtrx, b0, b1, depth);
    Prune(depth);
}

void QBdtNode::PushStateVector(const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
{
    const bool isB0Zero = IS_NORM_0(b0->scale);
    const bool isB1Zero = IS_NORM_0(b1->scale);

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
        b0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        b1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return;
    }

    const bool isSame = ((b0->branches[0] == b1->branches[0]) && (b0->branches[1] == b1->branches[1]));
    if (isSame) {
        b1->branches[0] = b0->branches[0];
        b1->branches[1] = b0->branches[1];

        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        b1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return;
    }

    if (!depth) {
        throw std::out_of_range("QBdtNode::PushStateVector() not implemented at depth=0! (You didn't push to root "
                                "depth, or root depth lacks method implementation.)");
    }
    depth--;

    b0->Branch();
    b1->Branch();

    b0->branches[0]->scale *= b0->scale;
    b0->branches[1]->scale *= b0->scale;
    b0->scale = SQRT1_2_R1;

    b1->branches[0]->scale *= b1->scale;
    b1->branches[1]->scale *= b1->scale;
    b1->scale = SQRT1_2_R1;

    PushStateVector(mtrx, b0->branches[0], b1->branches[0], depth);
    PushStateVector(mtrx, b0->branches[1], b1->branches[1], depth);

    b0->PopStateVector();
    b1->PopStateVector();
}

void QBdtNode::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, bitLenInt size)
{
    if (norm(scale) <= FP_NORM_EPSILON) {
        return;
    }

    if (!depth) {
        QBdtNodeInterfacePtr c = ShallowClone();
        branches[0] = b->branches[0];
        branches[1] = b->branches[1];

        if (!size || !c->branches[0]) {
            return;
        }

        InsertAtDepth(c, size, 0);

        return;
    }
    depth--;

    if (!depth && size) {
        QBdtNodeInterfacePtr c = branches[0];

        if (norm(branches[0]->scale) > FP_NORM_EPSILON) {
            branches[0] = std::make_shared<QBdtNode>(branches[0]->scale, b->branches);
            branches[0]->Branch();
            branches[0]->InsertAtDepth(c, size, 0);

            if (c.get() == branches[1].get()) {
                branches[1] = branches[0];
                return;
            }
        }

        if (norm(branches[1]->scale) <= FP_NORM_EPSILON) {
            return;
        }

        c = branches[1];
        branches[1] = std::make_shared<QBdtNode>(branches[1]->scale, b->branches);
        branches[1]->Branch();
        branches[1]->InsertAtDepth(c, size, 0);

        return;
    }

    if (!branches[0]) {
        return;
    }

    branches[0]->InsertAtDepth(b, depth, size);
    if (branches[0].get() != branches[1].get()) {
        branches[1]->InsertAtDepth(b, depth, size);
    }
}
} // namespace Qrack
