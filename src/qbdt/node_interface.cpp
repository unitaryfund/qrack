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

} // namespace Qrack
