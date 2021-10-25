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

#include "common/qrack_types.hpp"

namespace Qrack {

class QBinaryDecisionTreeNode;
typedef std::shared_ptr<QBinaryDecisionTreeNode> QBinaryDecisionTreeNodePtr;

class QBinaryDecisionTreeNode {
protected:
    void PruneNarrowOrWide(bitLenInt depth, bool isNarrow = false, bitCapInt perm = 0);

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
        branches[0] = brnchs[0] ? brnchs[0]->DeepClone() : NULL;
        if (brnchs[0] == brnchs[1]) {
            branches[1] = branches[0];
        } else {
            branches[1] = brnchs[1] ? brnchs[1]->DeepClone() : NULL;
        }
    }

    QBinaryDecisionTreeNodePtr DeepClone() { return std::make_shared<QBinaryDecisionTreeNode>(scale, branches); }

    QBinaryDecisionTreeNodePtr ShallowClone()
    {
        QBinaryDecisionTreeNodePtr toRet = std::make_shared<QBinaryDecisionTreeNode>(scale);
        toRet->branches[0] = branches[0];
        toRet->branches[1] = branches[1];

        return toRet;
    }

    bool isNoChildren() { return !branches[0] && !branches[1]; }

    void Branch(bitLenInt depth = 1U);

    void Prune(bitLenInt depth) { PruneNarrowOrWide(depth, false); }

    void Prune(bitLenInt depth, bitCapInt perm) { PruneNarrowOrWide(depth, true, perm); }

    bool Normalize(bitLenInt depth);
};

} // namespace Qrack
