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
    // If scale of this node is zero, nothing under it makes a difference.
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (!depth || !branches[0]) {
        return;
    }

    // Prune recursively to depth.
    depth--;
    size_t bit = perm & 1U;
    perm >>= 1U;

    if (isNarrow) {
        // Either we're narrow, or else there's no point in pruning same pointer branch twice.
        branches[bit]->PruneNarrowOrWide(depth, isNarrow, perm);
    } else {
        int maxLcv = (branches[0] == branches[1]) ? 1 : 2;
        for (int i = 0; i < maxLcv; i++) {
            branches[i]->PruneNarrowOrWide(depth, false, perm);
        }
    }

    if (branches[0] == branches[1]) {
        // Combining branches is the only other thing we try, below.
        return;
    }
    // Now, we try to combine pointers to equivalent branches.

    bitCapInt depthPow = ONE_BCI << depth;
    complex scale1, scale2;
    bitCapInt i;
    bitLenInt j;
    QBinaryDecisionTreeNodePtr leaf1, leaf2;
    for (i = 0; i < depthPow; i++) {
        leaf1 = branches[0];
        leaf2 = branches[1];

        scale1 = ONE_CMPLX;
        scale2 = ONE_CMPLX;

        for (j = 0; j < depth; j++) {
            bit = (i >> j) & 1U;

            if (leaf1) {
                scale1 *= leaf1->scale;
                leaf1 = leaf1->branches[bit];
            }

            if (leaf2) {
                scale2 *= leaf2->scale;
                leaf2 = leaf2->branches[bit];
            }
        }

        if (leaf1 || leaf2 || !IS_NORM_0(scale1 - scale2)) {
            // We can't combine our immediate children within depth.
            return;
        }
    }

    // The branches terminate equal, within depth.
    branches[1] = branches[0];

    return;
}

void QBinaryDecisionTreeNode::Branch(bitLenInt depth)
{
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (!depth) {
        return;
    }

    if (!branches[0]) {
        branches[0] = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
        branches[1] = std::make_shared<QBinaryDecisionTreeNode>(SQRT1_2_R1);
    }

    // Split all clones.
    branches[0] = branches[0]->ShallowClone();
    branches[1] = branches[1]->ShallowClone();

    branches[0]->Branch(depth - 1U);
    branches[1]->Branch(depth - 1U);
}

void QBinaryDecisionTreeNode::Normalize(bitLenInt depth)
{
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (!depth || !branches[0]) {
        return;
    }

    branches[0]->Normalize(depth - 1U);
    if (branches[0] != branches[1]) {
        branches[1]->Normalize(depth - 1U);
    }

    real1 nrm = (real1)(norm(branches[0]->scale) + norm(branches[1]->scale));
    if (nrm <= FP_NORM_EPSILON) {
        throw std::runtime_error("QBinaryDecisionTree: Tried to normalize 0.");
    }
    nrm = sqrt(nrm);

    branches[0]->scale *= ONE_R1 / nrm;
    if (branches[0] != branches[1]) {
        branches[1]->scale *= ONE_R1 / nrm;
    }
}

void QBinaryDecisionTreeNode::ConvertStateVector(bitLenInt depth)
{
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (!depth || !branches[0]) {
        return;
    }

    depth--;

    // Depth-first
    branches[0]->ConvertStateVector(depth);
    if (branches[0] != branches[1]) {
        branches[1]->ConvertStateVector(depth);
    }

    real1 nrm0 = norm(branches[0]->scale);
    real1 nrm1 = norm(branches[1]->scale);

    if ((nrm0 + nrm1) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (nrm0 <= FP_NORM_EPSILON) {
        scale = branches[1]->scale;
        branches[0]->SetZero();
        branches[1]->scale = ONE_CMPLX;
        branches[1]->Prune(depth);
        return;
    }

    if (nrm1 <= FP_NORM_EPSILON) {
        scale = branches[0]->scale;
        branches[0]->scale = ONE_CMPLX;
        branches[1]->SetZero();
        branches[0]->Prune(depth);
        return;
    }

    scale = complex(sqrt(nrm0 + nrm1), ZERO_R1);
    branches[0]->scale /= scale;
    if (branches[0] != branches[1]) {
        branches[1]->scale /= scale;
    }

    int maxLcv = (branches[0] == branches[1]) ? 1 : 2;
    for (int i = 0; i < maxLcv; i++) {
        branches[i]->Prune(depth);
    }
}

} // namespace Qrack
