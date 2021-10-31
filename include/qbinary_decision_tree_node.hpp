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

#pragma once

#include "common/qrack_types.hpp"

namespace Qrack {

struct QBinaryDecisionTreeNode;
typedef std::shared_ptr<QBinaryDecisionTreeNode> QBinaryDecisionTreeNodePtr;

struct QBinaryDecisionTreeNode {
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

    QBinaryDecisionTreeNodePtr ShallowClone()
    {
        QBinaryDecisionTreeNodePtr toRet = std::make_shared<QBinaryDecisionTreeNode>(scale);
        toRet->branches[0] = branches[0];
        toRet->branches[1] = branches[1];

        return toRet;
    }

    void SetZero()
    {
        scale = ZERO_CMPLX;
        branches[0] = NULL;
        branches[1] = NULL;
    }

    void Branch(bitLenInt depth = 1U);

    void Prune(bitLenInt depth);

    void Normalize(bitLenInt depth);

    void ConvertStateVector(bitLenInt depth);

    void CorrectPhase();
};

} // namespace Qrack
