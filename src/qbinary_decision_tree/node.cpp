//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// QEngineShard is the atomic qubit unit of the QUnit mapper. "PhaseShard" optimizations are basically just a very
// specific "gate fusion" type optimization, where multiple gates are composed into single product gates before
// application to the state vector, to reduce the total number of gates that need to be applied. Rather than handling
// this as a "QFusion" layer optimization, which will typically sit BETWEEN a base QEngine set of "shards" and a QUnit
// that owns them, this particular gate fusion optimization can be avoid representational entanglement in QUnit in the
// first place, which QFusion would not help with. Alternatively, another QFusion would have to be in place ABOVE the
// QUnit layer, (with QEngine "below,") for this to work. Additionally, QFusion is designed to handle more general gate
// fusion, not specifically controlled phase gates, which are entirely commuting among each other and possibly a
// jumping-off point for further general "Fourier basis" optimizations which should probably reside in QUnit, analogous
// to the |+>/|-> basis changes QUnit takes advantage of for "H" gates.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qbinary_decision_tree_node.hpp"

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

void QBinaryDecisionTreeNode::PruneNarrowOrWide(bitLenInt depth, bool isNarrow, bitCapInt perm)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Prune recursively to depth.
    size_t bit = perm & 1U;
    perm >>= 1U;

    if (isNarrow) {
        // Either we're narrow, or else there's no point in pruning same pointer branch twice.
        branches[bit]->PruneNarrowOrWide(depth - 1U, isNarrow, perm);
    } else {
        size_t maxLcv = (b0 == b1) ? 1 : 2;
        for (size_t i = 0; i < maxLcv; i++) {
            branches[i]->PruneNarrowOrWide(depth - 1U, false, perm);
        }
    }

    if (!IS_NORM_0(b0->scale - b1->scale)) {
        return;
    }

    if (b0 == b1) {
        // Combining branches is the only other thing we try, below.
        return;
    }
    // Now, we try to combine pointers to equivalent branches.

    bitCapInt depthPow = ONE_BCI << depth;
    complex scale0, scale1;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf0, leaf1;
    for (bitCapInt i = 0; i < depthPow; i++) {
        leaf0 = b0;
        leaf1 = b1;

        scale0 = leaf0->scale;
        scale1 = leaf1->scale;

        for (j = 0; j < depth; j++) {
            bit = (i >> j) & 1U;

            if (leaf0 == leaf1) {
                break;
            }

            if (leaf0) {
                scale0 *= leaf0->scale;
                leaf0 = leaf0->branches[bit];
            }
            if (leaf1) {
                scale1 *= leaf1->scale;
                leaf1 = leaf1->branches[bit];
            }
        }

        if ((leaf0 != leaf1) || !IS_NORM_0(scale0 - scale1)) {
            // We can't combine our immediate children within depth.
            return;
        }
    }

    // The branches terminate equal, within depth.
    b1 = b0;
}

void QBinaryDecisionTreeNode::Branch(bitLenInt depth)
{
    if (!depth) {
        return;
    }
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    if (!b0) {
        b0 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
        b1 = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
    }

    // Split all clones.
    b0 = b0->ShallowClone();
    b1 = b1->ShallowClone();

    b0->Branch(depth - 1U);
    b1->Branch(depth - 1U);
}

void QBinaryDecisionTreeNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    b0->Normalize(depth - 1U);
    if (b0 != b1) {
        b1->Normalize(depth - 1U);
    }

    real1 nrm = (real1)(norm(b0->scale) + norm(b1->scale));
    if (nrm <= FP_NORM_EPSILON) {
        throw std::runtime_error("QBinaryDecisionTree: Tried to normalize 0.");
    }
    nrm = sqrt(nrm);

    b0->scale *= ONE_R1 / nrm;
    if (b0 != b1) {
        b1->scale *= ONE_R1 / nrm;
    }
}

void QBinaryDecisionTreeNode::ConvertStateVector(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    if (!b0) {
        return;
    }
    QBinaryDecisionTreeNodePtr& b1 = branches[1];

    // Depth-first
    b0->ConvertStateVector(depth);
    if (b0 != b1) {
        b1->ConvertStateVector(depth);
    }

    real1 nrm0 = norm(b0->scale);
    real1 nrm1 = norm(b1->scale);

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
    if (b0 != b1) {
        b1->scale /= scale;
    }

    CorrectPhase();
}

void QBinaryDecisionTreeNode::CorrectPhase()
{
    QBinaryDecisionTreeNodePtr& b0 = branches[0];
    QBinaryDecisionTreeNodePtr& b0b0 = b0->branches[0];
    QBinaryDecisionTreeNodePtr& b1 = branches[1];
    QBinaryDecisionTreeNodePtr& b1b0 = b1->branches[0];

    if (!b0 || (b0 == b1) || !b0b0 || !b1b0) {
        // Combining branches UP TO OVERALL PHASE is the only other thing we try, below.
        return;
    }

    QBinaryDecisionTreeNodePtr& b0b1 = b0->branches[1];
    QBinaryDecisionTreeNodePtr& b1b1 = b1->branches[1];

    // First, if our 2 sets of 2 children differ only by an OVERALL PHASE factor, we pull this factor up into the two
    // parents, equally.

    if (IS_NORM_0(b0->scale) || IS_NORM_0(b1->scale)) {
        return;
    }

    if (IS_NORM_0(b0b0->scale) != IS_NORM_0(b1b0->scale)) {
        return;
    }

    complex offsetFactor;
    if (IS_NORM_0(b0b0->scale)) {
        // Avoid division by 0.
        offsetFactor = (b1->scale * b1b1->scale) / (b0->scale * b0b1->scale);
    } else {
        offsetFactor = (b1->scale * b1b0->scale) / (b0->scale * b0b0->scale);
    }

    if (IS_NORM_0(ONE_CMPLX - offsetFactor) || (abs(ONE_R1 - norm(offsetFactor)) > FP_NORM_EPSILON) ||
        !IS_NORM_0(offsetFactor * b0->scale * b0b1->scale - b1->scale * b1b1->scale)) {
        return;
    }

    // We want to preserve the original numerical ket representation while handling states like |+> and |->.

    b0->scale *= offsetFactor;
    b1->scale /= offsetFactor;

    b0b0->scale /= offsetFactor;
    b0b1->scale /= offsetFactor;
    b1b0->scale *= offsetFactor;
    b1b1->scale *= offsetFactor;

    // Notice that the overall effect on tree to traversal, to produce a ket amplitude, is totally cancelled.
}

} // namespace Qrack
