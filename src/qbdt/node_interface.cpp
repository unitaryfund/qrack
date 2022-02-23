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

void QBdtNodeInterface::Apply2x2(const complex* mtrx, bitLenInt depth)
{
    if (!depth) {
        return;
    }
    depth--;

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

    const bool isB0Zero = IS_NORM_0(b0->scale);
    const bool isB1Zero = IS_NORM_0(b1->scale);

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
        Prune();

        return;
    }

    const bool isSame = (b0->branches[0] == b1->branches[0]) && (b0->branches[1] == b1->branches[1]);

    if (isSame) {
        b1->branches[0] = b0->branches[0];
        b1->branches[1] = b0->branches[1];

        const complex Y0 = b0->scale;
        const complex Y1 = b1->scale;
        b0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        b1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;
        Prune();

        return;
    }

    const bitCapInt depthPow = ONE_BCI << depth;

    b0->Branch(depth, true);
    b1->Branch(depth, true);

    _par_for_qbdt(0, depthPow, [&](const bitCapInt& i, const int& cpu) {
        QBdtNodeInterfacePtr leaf0 = b0;
        QBdtNodeInterfacePtr leaf1 = b1;

        complex scale0 = b0->scale;
        complex scale1 = b1->scale;

        // b0 and b1 can't both be 0.
        bool isZero = false;

        bitLenInt j;
        for (j = 0; j < depth; j++) {
            const size_t bit = SelectBit(i, depth - (j + 1U));

            leaf0 = leaf0->branches[bit];
            scale0 *= leaf0->scale;

            leaf1 = leaf1->branches[bit];
            scale1 *= leaf1->scale;

            isZero = IS_NORM_0(scale0) && IS_NORM_0(scale1);

            if (isZero) {
                break;
            }
        }

        if (isZero) {
            leaf0->SetZero();
            leaf1->SetZero();

            // WARNING: Mutates loop control variable!
            return (bitCapInt)((ONE_BCI << (depth - (j + 1U))) - ONE_BCI);
        }

        const complex Y0 = scale0;
        const complex Y1 = scale1;
        leaf0->scale = mtrx[0] * Y0 + mtrx[1] * Y1;
        leaf1->scale = mtrx[2] * Y0 + mtrx[3] * Y1;

        return (bitCapInt)0U;
    });

    b0->ConvertStateVector(depth);
    b1->ConvertStateVector(depth);
    Prune(depth + 1U);
}

} // namespace Qrack
