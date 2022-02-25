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

#include "qbdt_node_interface.hpp"

namespace Qrack {

bool operator==(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs)
{
    if (!lhs) {
        return !rhs;
    }

    if (!rhs) {
        return false;
    }

    return lhs->isEqual(rhs);
}

bool operator!=(const QBdtNodeInterfacePtr& lhs, const QBdtNodeInterfacePtr& rhs) { return !(lhs == rhs); }

QBdtNodeInterfacePtr operator-(const QBdtNodeInterfacePtr& t)
{
    QBdtNodeInterfacePtr m = t->ShallowClone();
    m->scale *= -ONE_CMPLX;

    return m;
}

void QBdtNodeInterface::_par_for_qbdt(const bitCapInt begin, const bitCapInt end, BdtFunc fn)
{
    const bitCapInt itemCount = end - begin;
    const bitCapInt maxLcv = begin + itemCount;
    for (bitCapInt j = begin; j < maxLcv; j++) {
        j |= fn(j, 0);
    }
}

void QBdtNodeInterface::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, bitLenInt size)
{
    if (norm(scale) <= FP_NORM_EPSILON) {
        return;
    }

    if (depth) {
        depth--;
        if (branches[0]) {
            branches[0]->InsertAtDepth(b, depth, size);
            if (branches[0].get() != branches[1].get()) {
                branches[1]->InsertAtDepth(b, depth, size);
            }
        }

        return;
    }

    QBdtNodeInterfacePtr c = ShallowClone();
    branches[0] = b->branches[0];
    branches[1] = b->branches[1];

    if (!size || !c->branches[0]) {
        return;
    }

    InsertAtDepth(c, size, 0);
}

QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(bitLenInt depth, bitLenInt size)
{
    if (norm(scale) <= FP_NORM_EPSILON) {
        return NULL;
    }

    Branch();

    if (depth) {
        depth--;
        if (!branches[0]) {
            return NULL;
        }
        QBdtNodeInterfacePtr toRet = branches[0]->RemoveSeparableAtDepth(depth, size);
        if (branches[0].get() != branches[1].get()) {
            if (toRet) {
                branches[1]->RemoveSeparableAtDepth(depth, size);
            } else {
                toRet = branches[1]->RemoveSeparableAtDepth(depth, size);
            }
        }

        return toRet;
    }

    QBdtNodeInterfacePtr toRet = ShallowClone();
    toRet->scale /= abs(toRet->scale);

    if (!size) {
        branches[0] = NULL;
        branches[1] = NULL;

        return toRet;
    }

    QBdtNodeInterfacePtr temp = toRet->RemoveSeparableAtDepth(size, 0);
    branches[0] = temp->branches[0];
    branches[1] = temp->branches[1];

    return toRet;
}
} // namespace Qrack
