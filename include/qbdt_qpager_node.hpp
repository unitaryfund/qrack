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

#include "qbinary_decision_tree_node_interface.hpp"

namespace Qrack {

class QPagerNode;
typedef std::shared_ptr<QPagerNode> QPagerNodePtr;

class QPagerNode : QBdtNodeInterface {
public:
    QInterfacePtr base;

    QPagerNode()
        : QBdtNodeInterface()
        , base(NULL)
    {
        // Intentionally left blank
    }

    QPagerNode(complex scl)
        : QBdtNodeInterface(scl)
        , base(NULL)
    {
        // Intentionally left blank
    }

    QPagerNode(complex scl, QInterfacePtr b)
        : QBdtNodeInterface(scl)
        , base(b)
    {
        // Intentionally left blank
    }

    QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QPagerNode>(scale, base); }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        base = NULL;
    }

    virtual void Branch(bitLenInt depth = 1U, bool isZeroBranch = false);

    virtual void Prune(bitLenInt depth = 1U);

    virtual void Normalize(bitLenInt depth);

    virtual void ConvertStateVector(bitLenInt depth);
};

} // namespace Qrack
