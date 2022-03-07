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

class QBdtQEngineNode;
typedef std::shared_ptr<QBdtQEngineNode> QBdtQEngineNodePtr;

class QBdtQInterfaceNode : public QBdtNodeInterface {
protected:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b0,
        QBdtNodeInterfacePtr& b1, bitLenInt depth)
#else
    virtual void PushStateVector(
        const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtQInterfaceNode::PushStateVector() not implemented!");
    }

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
        // Intentionally left blank.
    }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        qReg = NULL;
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtQInterfaceNode>(scale, qReg); }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual void Normalize(bitLenInt depth);

    virtual void Branch(bitLenInt depth = 1U);

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size);

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size);

    virtual void PopStateVector(bitLenInt depth = 1U) {}

    virtual void Prune(bitLenInt depth = 1U) {}

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, bitLenInt depth)
#else
    virtual void Apply2x2(const complex* mtrx, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtQInterfaceNode::Apply2x2() not implemented!");
    }
};

} // namespace Qrack
