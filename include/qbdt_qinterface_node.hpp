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
#include "qinterface.hpp"

namespace Qrack {

class QBdtQInterfaceNode;
typedef std::shared_ptr<QBdtQInterfaceNode> QBdtQInterfaceNodePtr;

class QBdtQInterfaceNode : public QBdtNodeInterface {
public:
    QInterfacePtr qReg;

    QBdtQInterfaceNode()
        : QBdtNodeInterface(ZERO_CMPLX)
        , qReg(NULL)
    {
        // Intentionally left blank.
    }

    QBdtQInterfaceNode(complex scl, QInterfacePtr q)
        : QBdtNodeInterface(scl)
        , qReg(q)
    {
        if (norm(scale) <= FP_NORM_EPSILON) {
            SetZero();
        }
    }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        qReg = NULL;
    }

    virtual QBdtNodeInterfacePtr ShallowClone()
    {
        return std::make_shared<QBdtQInterfaceNode>(scale, qReg ? qReg->Clone() : NULL);
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r)
    {
        return (this == r.get()) ||
            ((norm(scale - r->scale) <= FP_NORM_EPSILON) &&
                ((norm(scale) <= FP_NORM_EPSILON) ||
                    qReg->ApproxCompare(std::dynamic_pointer_cast<QBdtQInterfaceNode>(r)->qReg)));
    }

    virtual void Normalize(bitLenInt depth)
    {
        if (qReg) {
            qReg->NormalizeState();
        }
    }

    virtual void Branch(bitLenInt depth = 1U)
    {
        throw std::out_of_range("QBdtQInterfaceNode::Branch() not implemented! (Recursion went too deep.)");
    }

    virtual void ConvertStateVector(bitLenInt depth)
    {
        throw std::out_of_range(
            "QBdtQInterfaceNode::ConvertStateVector() not implemented! (Don't set/get state vector amplitudes.)");
    }
};

} // namespace Qrack
