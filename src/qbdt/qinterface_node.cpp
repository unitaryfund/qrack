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

#include "qinterface_qbdt_node.hpp"

#if ENABLE_PTHREAD
#include <future>
#endif
#include <set>

#define IS_NORM_0(c) (norm(c) <= FP_NORM_EPSILON)

namespace Qrack {

void QInterfaceQbdtNode::Prune(bitLenInt depth)
{
    // TODO
}

void QInterfaceQbdtNode::Branch(bitLenInt depth, bool isZeroBranch)
{
    // TODO
}

void QInterfaceQbdtNode::Normalize(bitLenInt depth)
{
    // TODO
}

void QInterfaceQbdtNode::ConvertStateVector(bitLenInt depth)
{
    // TODO
}
} // namespace Qrack
