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
    QBdtNode()
        : QBdtNodeInterface()
    {
        // Intentionally left blank
    }

    QBdtNode(complex scl)
        : QBdtNodeInterface(scl)
    {
        // Intentionally left blank
    }

    QBdtNode(complex scl, QBdtNodeInterfacePtr* b)
        : QBdtNodeInterface(scl, b)
    {
        if (norm(scale) <= FP_NORM_EPSILON) {
            SetZero();
        }
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtNode>(scale, branches); }

    virtual bool isEqual(QBdtNodeInterfacePtr r) { return this == r.get(); }

    virtual void Branch(bitLenInt depth = 1U);

    virtual void Prune(bitLenInt depth = 1U);

    virtual void Normalize(bitLenInt depth);

    virtual void ConvertStateVector(bitLenInt depth);
};

} // namespace Qrack
