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
#include "qengine.hpp"

namespace Qrack {

class QBdtQEngineNode;
typedef std::shared_ptr<QBdtQEngineNode> QBdtQEngineNodePtr;

class QBdtQEngineNode : public QBdtNodeInterface {
protected:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b0,
        QBdtNodeInterfacePtr& b1, bitLenInt depth)
#else
    virtual void PushStateVector(
        const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtQEngineNode::PushStateVector() not implemented!");
    }

public:
    QEnginePtr qReg;

    QBdtQEngineNode()
        : QBdtNodeInterface(ZERO_CMPLX)
        , qReg(NULL)
    {
        // Intentionally left blank.
    }

    QBdtQEngineNode(complex scl, QEnginePtr q)
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

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtQEngineNode>(scale, qReg); }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual bool isEqualUnder(QBdtNodeInterfacePtr r);

    virtual void Normalize(bitLenInt depth);

    virtual void Branch(bitLenInt depth = 1U);

    virtual void Prune(bitLenInt depth = 1U);

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size);

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size);

    virtual void PopStateVector(bitLenInt depth = 1U) { Prune(); }

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, bitLenInt depth)
#else
    virtual void Apply2x2(const complex* mtrx, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtQEngineNode::Apply2x2() not implemented!");
    }
#if ENABLE_COMPLEX_X2
    virtual void PushSpecial(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b1);
#else
    virtual void PushSpecial(const complex* mtrx, QBdtNodeInterfacePtr& b1);
#endif
};

} // namespace Qrack
