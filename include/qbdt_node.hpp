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

#include "qbdt_node_interface.hpp"

namespace Qrack {

class QBdtNode;
typedef std::shared_ptr<QBdtNode> QBdtNodePtr;

class QBdtNode : public QBdtNodeInterface {
public:
    QBdtNodePtr branches[2];

    QBdtNode()
        : QBdtNodeInterface()
    {
        branches[0] = NULL;
        branches[1] = NULL;
    }

    QBdtNode(complex scl)
        : QBdtNodeInterface(scl)
    {
        branches[0] = NULL;
        branches[1] = NULL;
    }

    virtual QBdtNodeInterfacePtr ShallowClone()
    {
        QBdtNodePtr toRet = std::make_shared<QBdtNode>(scale);
        toRet->branches[0] = branches[0];
        toRet->branches[1] = branches[1];

        return toRet;
    }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        branches[0] = NULL;
        branches[1] = NULL;
    }

    virtual void Branch(bitLenInt depth = 1U, bool isZeroBranch = false);

    virtual void Prune(bitLenInt depth = 1U);

    virtual void Normalize(bitLenInt depth);

    virtual void ConvertStateVector(bitLenInt depth);
};

} // namespace Qrack
