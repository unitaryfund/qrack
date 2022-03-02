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

#include "qbdt_qinterface_node.hpp"
#include "qengine.hpp"

namespace Qrack {
bool QBdtQInterfaceNode::isEqual(QBdtNodeInterfacePtr r)
{
    if (this == r.get()) {
        return true;
    }

    if (norm(scale - r->scale) > FP_NORM_EPSILON) {
        return false;
    }

    if (norm(scale) <= FP_NORM_EPSILON) {
        return true;
    }

    QInterfacePtr rReg = std::dynamic_pointer_cast<QBdtQInterfaceNode>(r)->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    if (qReg->ApproxCompare(rReg)) {
        qReg = rReg;
        return true;
    }

    return false;
}

void QBdtQInterfaceNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (norm(scale) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (qReg) {
        qReg->NormalizeState();
    }
}

void QBdtQInterfaceNode::Branch(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (norm(scale) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (qReg) {
        qReg = qReg->Clone();
    }
}

void QBdtQInterfaceNode::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, bitLenInt size)
{
    if (norm(scale) <= FP_NORM_EPSILON) {
        return;
    }

    if (depth) {
        throw std::runtime_error("QBdtQInterfaceNode::InsertAtDepth() not implemented for nonzero depth!");
    }

    QBdtQInterfaceNodePtr bEng = std::dynamic_pointer_cast<QBdtQInterfaceNode>(b);
    qReg->Compose(bEng->qReg);
}

QBdtNodeInterfacePtr QBdtQEngineNode::RemoveSeparableAtDepth(bitLenInt depth, bitLenInt size)
{
    if (!size || (norm(scale) <= FP_NORM_EPSILON)) {
        return NULL;
    }

    QBdtQEngineNodePtr toRet = std::dynamic_pointer_cast<QBdtQEngineNode>(ShallowClone());
    toRet->scale /= abs(toRet->scale);

    if (!qReg) {
        return toRet;
    }

    toRet->qReg = std::dynamic_pointer_cast<QEngine>(qReg)->CloneEmpty();
    std::dynamic_pointer_cast<QEngine>(toRet->qReg)->SetQubitCount(size);

    qReg->Decompose(depth, toRet->qReg);

    return toRet;
}
} // namespace Qrack
