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
#include "qunitclifford.hpp"

namespace Qrack {

class QBdtQStabilizerNode;
typedef std::shared_ptr<QBdtQStabilizerNode> QBdtQStabilizerNodePtr;

class QBdtQStabilizerNode : public QBdtNodeInterface {
public:
    QUnitCliffordPtr qReg;

    QBdtQStabilizerNode()
        : QBdtNodeInterface(ZERO_CMPLX)
        , qReg(NULL)
    {
        // Intentionally left blank.
    }

    QBdtQStabilizerNode(complex scl, QUnitCliffordPtr q)
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

    virtual bool isEqualBranch(QBdtNodeInterfacePtr r, const bool& b);

    virtual void Normalize(bitLenInt depth = 1U)
    {
        // Intentionally left blank
    }

    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDepth = 1U);

    virtual QBdtNodeInterfacePtr Prune(
        bitLenInt depth = 1U, bitLenInt parDepth = 1U, const bool& isCliffordBlocked = false);

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(
        bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        // Intentionally left blank
    }

    virtual QBdtNodeInterfacePtr PopSpecial(bitLenInt depth = 1U);
};

} // namespace Qrack
