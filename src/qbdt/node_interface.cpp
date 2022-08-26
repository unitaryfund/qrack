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

#define IS_SAME_AMP(a, b) (norm((a) - (b)) <= (REAL1_EPSILON * REAL1_EPSILON))

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

bool QBdtNodeInterface::isEqual(QBdtNodeInterfacePtr r)
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

    if (branches[0U] != r->branches[0U]) {
        return false;
    }

    branches[0U] = r->branches[0U];

    if (branches[1U] != r->branches[1U]) {
        return false;
    }

    branches[1U] = r->branches[1U];

    return true;
}

bool QBdtNodeInterface::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (norm(scale) <= FP_NORM_EPSILON) {
        return norm(r->scale) <= FP_NORM_EPSILON;
    }

    if (branches[0U] != r->branches[0U]) {
        return false;
    }

    branches[0U] = r->branches[0U];

    if (branches[1U] != r->branches[1U]) {
        return false;
    }

    branches[1U] = r->branches[1U];

    return true;
}

void QBdtNodeInterface::_par_for_qbdt(const bitCapInt begin, const bitCapInt end, BdtFunc fn)
{
    const bitCapInt itemCount = end - begin;
    const bitCapInt maxLcv = begin + itemCount;
    for (bitCapInt j = begin; j < maxLcv; ++j) {
        j |= fn(j, 0U);
    }
}

QBdtNodeInterfacePtr QBdtNodeInterface::RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size)
{
    if (norm(scale) <= FP_NORM_EPSILON) {
        return NULL;
    }

    Branch();

    if (depth) {
        --depth;
        if (!branches[0U]) {
            return NULL;
        }

        QBdtNodeInterfacePtr toRet1 = branches[0U]->RemoveSeparableAtDepth(depth, size);
        QBdtNodeInterfacePtr toRet2 =
            (branches[0U].get() == branches[1U].get()) ? toRet1 : branches[1U]->RemoveSeparableAtDepth(depth, size);

        return (norm(toRet1->scale) > norm(toRet2->scale)) ? toRet1 : toRet2;
    }

    if (!size) {
        return NULL;
    }

    QBdtNodeInterfacePtr toRet = ShallowClone();
    toRet->scale /= abs(toRet->scale);

    QBdtNodeInterfacePtr temp = toRet->RemoveSeparableAtDepth(size, 0);

    branches[0U] = temp->branches[0U];
    branches[1U] = temp->branches[1U];

    return toRet;
}
} // namespace Qrack
