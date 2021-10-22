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

#pragma once

#include <memory>

#include "common/qrack_types.hpp"

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

struct QBinaryDecisionTreeNode;
typedef std::shared_ptr<QBinaryDecisionTreeNode> QBinaryDecisionTreeNodePtr;

class QBinaryDecisionTreeNode {
protected:
    void PruneShallowOrDeep(bitLenInt depth, bool isShallow = false, bitCapInt perm = 0)
    {
        if (!depth) {
            return;
        }
        depth--;

        // If perm == 0, then bit == 0.
        size_t bit = perm & 1U;
        perm >>= 1U;

        bitLenInt maxLcv = isShallow ? (bit + 1U) : 2;
        for (bitLenInt i = bit; i < maxLcv; i++) {
            if (branches[i]) {
                if (IS_NORM_0(branches[i]->scale)) {
                    branches[i] = NULL;
                } else {
                    branches[i]->Prune(depth);
                }
            }
        }

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
                    leaf1 = leaf1->branches[bit];
                    scale1 *= leaf1->scale;
                }
                if (leaf2) {
                    leaf2 = leaf2->branches[bit];
                    scale2 *= leaf2->scale;
                }
            }

            if (!IS_NORM_0(scale1 - scale2)) {
                break;
            }
        }

        if (i != depthPow) {
            return;
        }

        branches[0] = branches[1];

        // If all descendent pairs are the same, contract the scale multiple into this is a terminal leaf.
        leaf1 = branches[0];
        leaf2 = branches[1];
        scale1 = scale;
        while (leaf1 == leaf2) {
            if (!leaf1) {
                break;
            }

            scale1 *= leaf1->scale;

            leaf1 = leaf1->branches[0];
            leaf2 = leaf1->branches[1];
        }

        if (!leaf1) {
            scale = scale1;
        }
    }

public:
    complex scale;
    QBinaryDecisionTreeNodePtr branches[2];

    QBinaryDecisionTreeNode()
        : scale(ONE_CMPLX)
        , branches({ NULL, NULL })
    {
    }

    QBinaryDecisionTreeNode(complex scl)
        : scale(scl)
        , branches({ NULL, NULL })
    {
    }

    QBinaryDecisionTreeNode(complex scl, QBinaryDecisionTreeNodePtr brnchs[2])
        : scale(scl)
    {
        std::copy(brnchs, brnchs + 2, branches);
    }

    QBinaryDecisionTreeNodePtr ShallowClone() { return std::make_shared<QBinaryDecisionTreeNode>(scale, branches); }

    // tMG SP1
    bool isNoChildren() { return !branches[0] && !branches[1]; }

    void Branch(bitLenInt depth = 1U, complex val = ONE_CMPLX)
    {
        if (!depth) {
            return;
        }

        if (!branches[0]) {
            branches[0] = std::make_shared<QBinaryDecisionTreeNode>(val);
        }
        if (!branches[1]) {
            branches[1] = std::make_shared<QBinaryDecisionTreeNode>(val);
        }

        if (branches[0] == branches[1]) {
            branches[1] = ShallowClone();
        }

        branches[0]->Branch(depth - 1U, val);
        branches[1]->Branch(depth - 1U, val);
    }

    void Prune(bitLenInt depth = bitsInCap) { PruneShallowOrDeep(depth, false); }

    void Prune(bitCapInt perm, bitLenInt depth = bitsInCap) { PruneShallowOrDeep(depth, true, perm); }
};

} // namespace Qrack
