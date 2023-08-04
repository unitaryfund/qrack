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
#include "qstabilizer.hpp"

namespace Qrack {

class QBdtQStabilizerNode;
typedef std::shared_ptr<QBdtQStabilizerNode> QBdtQStabilizerNodePtr;

class QBdtQStabilizerNode : public QBdtNodeInterface {
protected:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
        bitLenInt parDepth = 1U)
#else
    virtual void PushStateVector(complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U)
#endif
    {
        throw std::out_of_range("QBdtQStabilizerNode::PushStateVector() not implemented!");
    }

public:
    QStabilizerPtr qReg;

    QBdtQStabilizerNode()
        : QBdtNodeInterface(ZERO_CMPLX)
        , qReg(NULL)
    {
        // Intentionally left blank.
    }

    QBdtQStabilizerNode(complex scl, QStabilizerPtr q)
        : QBdtNodeInterface(scl)
        , qReg(q)
    {
        // Intentionally left blank
    }

    virtual ~QBdtQStabilizerNode()
    {
        // Virtual destructor for inheritance
    }

    virtual bool IsStabilizer() { return true; }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        qReg = NULL;
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtQStabilizerNode>(scale, qReg); }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual bool isEqualUnder(QBdtNodeInterfacePtr r);

    virtual void Normalize(bitLenInt depth = 1U)
    {
        // Intentionally left blank
    }

    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDepth = 1U);

    virtual void Prune(bitLenInt depth = 1U, bitLenInt parDepth = 1U);

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(
        bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        // Intentionally left blank
    }

#if ENABLE_COMPLEX_X2
    virtual void Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, bitLenInt depth)
#else
    virtual void Apply2x2(complex const* mtrx, bitLenInt depth)
#endif
    {
        throw std::out_of_range("QBdtQStabilizerNode::Apply2x2() not implemented!");
    }

    virtual QBdtNodeInterfacePtr PopSpecial(bitLenInt depth = 1U);
};

} // namespace Qrack
