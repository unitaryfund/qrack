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

#define IS_SAME_AMP(a, b) (norm((a) - (b)) <= (REAL1_EPSILON * REAL1_EPSILON))

namespace Qrack {
bool QBdtQInterfaceNode::isEqual(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (!IS_SAME_AMP(scale, r->scale)) {
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
    qReg->Compose(bEng->qReg, 0U);
}

QBdtNodeInterfacePtr QBdtQInterfaceNode::RemoveSeparableAtDepth(bitLenInt depth, bitLenInt size)
{
    if (!size || (norm(scale) <= FP_NORM_EPSILON)) {
        return NULL;
    }

    QBdtQInterfaceNodePtr toRet = std::dynamic_pointer_cast<QBdtQInterfaceNode>(ShallowClone());
    toRet->scale /= abs(toRet->scale);

    if (!qReg) {
        return toRet;
    }

    toRet->qReg = qReg->Decompose(depth, size);

    return toRet;
}
} // namespace Qrack
