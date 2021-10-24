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

void QBinaryDecisionTreeNode::PruneShallowOrDeep(bitLenInt depth, bool isShallow, bitCapInt perm)
{
    if (!depth) {
        return;
    }

    // If scale of this node is zero, nothing under it makes a difference.
    // To contract a tree of all zero scales, as this is depth first, takes 2 orders, of identity and direct descendent.
    if (IS_NORM_0(scale) ||
        ((!branches[0] || IS_NORM_0(branches[0]->scale)) && (!branches[1] || IS_NORM_0(branches[1]->scale)))) {
        scale = ZERO_CMPLX;
        branches[0] = NULL;
        branches[1] = NULL;
        return;
    }

    // Prune recursively to depth.
    depth--;
    // (If perm == 0, then bit == 0.)
    size_t bit = perm & 1U;
    perm >>= 1U;

    if (isShallow || (branches[0] == branches[1])) {
        // Either we're shallow, or else there's no point in pruning same pointer branch twice.
        branches[bit]->PruneShallowOrDeep(depth, isShallow, perm);
    } else {
        for (bitLenInt i = 0U; i < 2U; i++) {
            if (branches[i]) {
                branches[i]->PruneShallowOrDeep(depth, isShallow, perm);
            }
        }
    }

    // We're going to try to combine 0 and 1 branches, now.
    if (!branches[0] || !branches[1]) {
        return;
    }

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

        if (!IS_NORM_0(scale1 - scale2) || leaf1 || leaf2) {
            break;
        }
    }

    if (i < depthPow) {
        // The branches are not equal to depth, or they might be unequal past depth.
        return;
    }

    // Otherwise, the branches are equal.
    branches[1] = branches[0];

    // If all descendent pairs are the same, contract the scale multiple into this as a terminal leaf.
    leaf1 = branches[0];
    leaf2 = leaf1;
    scale1 = scale;
    while (leaf1 && (leaf1 == leaf2)) {
        scale1 *= leaf1->scale;
        leaf1 = leaf2->branches[0];
        leaf2 = leaf2->branches[1];
    }

    if (leaf1) {
        return;
    }

    // Same branch, all the way down.
    scale = scale1;
    branches[0] = NULL;
    branches[1] = NULL;
}

void QBinaryDecisionTreeNode::Branch(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (!branches[0]) {
        branches[0] = std::make_shared<QBinaryDecisionTreeNode>();
    }
    if (!branches[1]) {
        branches[1] = std::make_shared<QBinaryDecisionTreeNode>();
    }

    // If we push zero into a lower branch, we'll prune upwards.
    // However, we'll prune OVER nonzero scales, if we don't push down.
    branches[0]->scale *= scale;
    if (branches[0] == branches[1]) {
        // Deeply split all clones.
        branches[1] = branches[0]->ShallowClone();
    } else {
        // If branches are not the same, we need to scale both.
        branches[1]->scale *= scale;
    }
    // We rescaled, by pushing our scale into (both) lower branches.
    scale = ONE_CMPLX;

    branches[0]->Branch(depth - 1U);
    branches[1]->Branch(depth - 1U);
}

} // namespace Qrack
