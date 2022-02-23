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

#if ENABLE_PTHREAD
#include <future>
#endif
#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

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

void PushStateVector(const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
{
    const bool isB0Zero = IS_NORM_0(b0->scale);
    const bool isB1Zero = IS_NORM_0(b1->scale);

    if (isB0Zero && isB1Zero) {
        b0->SetZero();
        b1->SetZero();

        return;
    }

    if (isB0Zero) {
        b0 = b1->ShallowClone();
        b0->scale = ZERO_CMPLX;
    }

    if (isB1Zero) {
        b1 = b0->ShallowClone();
        b1->scale = ZERO_CMPLX;
    }

    if (isB0Zero || isB1Zero) {
        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        b1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return;
    }

    const bool isSame = !depth || ((b0->branches[0] == b1->branches[0]) && (b0->branches[1] == b1->branches[1]));
    if (isSame) {
        b1->branches[0] = b0->branches[0];
        b1->branches[1] = b0->branches[1];

        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        b1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return;
    }
    depth--;

    b0->Branch();
    b1->Branch();

    b0->branches[0]->scale *= b0->scale;
    b0->branches[1]->scale *= b0->scale;
    b0->scale = SQRT1_2_R1;

    b1->branches[0]->scale *= b1->scale;
    b1->branches[1]->scale *= b1->scale;
    b1->scale = SQRT1_2_R1;

    PushStateVector(mtrx, b0->branches[0], b1->branches[0], depth);
    PushStateVector(mtrx, b0->branches[1], b1->branches[1], depth);

    b0->ConvertStateVector(1U);
    b1->ConvertStateVector(1U);
}

void QBdtNodeInterface::Apply2x2(const complex* mtrx, bitLenInt depth)
{
    if (!depth) {
        return;
    }

    Branch();
    QBdtNodeInterfacePtr& b0 = branches[0];
    QBdtNodeInterfacePtr& b1 = branches[1];

    if (IS_NORM_0(mtrx[1]) && IS_NORM_0(mtrx[2])) {
        b0->scale *= mtrx[0];
        b1->scale *= mtrx[3];
        Prune();

        return;
    }

    if (IS_NORM_0(mtrx[0]) && IS_NORM_0(mtrx[3])) {
        b0.swap(b1);
        b0->scale *= mtrx[1];
        b1->scale *= mtrx[2];
        Prune();

        return;
    }

    PushStateVector(mtrx, b0, b1, depth);
    Prune(depth);
}

} // namespace Qrack
