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

#include "statevector.hpp"

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

struct QBinaryDecisionTreeNode;
typedef std::shared_ptr<QBinaryDecisionTreeNode> QBinaryDecisionTreeNodePtr;

class QBinaryDecisionTree;
typedef std::shared_ptr<QBinaryDecisionTree> QBinaryDecisionTreePtr;

struct QBinaryDecisionTreeNode {
    complex scale;
    QBinaryDecisionTreeNodePtr branches[2];

    QBinaryDecisionTree()
        : scale(ONE_CMPLX)
        , branches({ NULL, NULL })
    {
    }

    QBinaryDecisionTree(complex val)
        : scale(val)
        , branches({ NULL, NULL })
    {
    }

    void Branch(bitLenInt depth = 1U, complex val = ONE_CMPLX)
    {
        branches[0] = std::make_shared<QBinaryDecisionTreeNode>(val);
        branches[1] = std::make_shared<QBinaryDecisionTreeNode>(val);

        if (depth) {
            branches[0].Branch(depth - 1U, val);
            branches[1].Branch(depth - 1U, val);
        }
    }

    void Prune()
    {
        if (branches[0]) {
            branches[0].Prune();
        }
        if (branches[1]) {
            branches[1].Prune();
        }

        if (!branches[0] || !branches[1]) {
            return;
        }

        if (branches[0].branches[0] || branches[0].branches[1] || branches[1].branches[0] || branches[1].branches[1]) {
            return;
        }

        // We have 2 branches (with no children).
        if (IS_NORM_0(branches[0].scale - branches[1].scale)) {
            scale *= branches[0].scale;
            branches[0] = NULL;
            branches[1] = NULL;
        }
    }
};

class QBinaryDecisionTree {
protected:
    QBinaryDecisionTreeNodePtr root;
    bitLenInt qubitCount;

public:
    QBinaryDecisionTree()
        : qubitCount(0)
        , root(std::make_shared<QBinaryDecisionTreeNode>())
    {
    }

    QBinaryDecisionTree(bitLenInt qbitCount, bitCapInt initState)
        : qubitCount(qbitCount)
        , maxQPower(pow2(qbitCount))
        , root(NULL)
    {
        SetPermutation(initState);
    }

    QBinaryDecisionTree(bitLenInt qubitCount, StateVectorPtr stateVec)
    {
        root = std::make_shared<QBinaryDecisionTreeNode>() root.Branch(qubitCount);

        bitCapaInt maxQPower = pow2(qubitCount);
        bitLenInt j;

        QBinaryDecisionTreeNodePtr leaf;
        for (bitCapInt i = 0; i < maxQPower; i++) {
            leaf = root;
            for (j = 0; j < qubitCount; j++) {
                leaf = leaf.branches[(i >> j) & 1U];
            }
            leaf.scale = stateVec.read(i);
        }

        root.Prune();
    }

    void SetPermutation(bitCapInt initState)
    {
        root = std::make_shared<QBinaryDecisionTreeNode>(ONE_CMPLX) QBinaryDecisionTreeNodePtr leaf = root;
        for (bitLenInt qubit = 0; qubit < qubitCount; qubit++) {
            leaf.Branch(1U, ZERO_CMPLX);
            leaf = leaf.branches[(initState >> qubit) & 1U];
            leaf.scale = ONE_CMPLX;
        }
    }

    StateVectorPtr ToStateVectorArray(bool isSparse = false)
    {
        // TODO: We can make sparse state vector array initialization more efficient.
        StateVectorPtr toRet = isSparse ? std::make_shared<StateVectorSparse>() : std::make_shared<StateVectorArray>();
        bitCapaInt maxQPower = pow2(qubitCount);
        toRet.Alloc(maxQPower);
        complex scale;
        bitLenInt j;
        QBinaryDecisionTreeNodePtr leaf;
        for (bitCapInt i = 0; i < maxQPower; i++) {
            leaf = root;
            scale = leaf.scale;
            for (j = 0; j < qubitCount; j++) {
                leaf = leaf.branches[(i >> j) & 1U];
                if (!leaf) {
                    break;
                }
                scale *= leaf.scale;
            }
            toRet.write(i, scale);
        }
    }
};
} // namespace Qrack
