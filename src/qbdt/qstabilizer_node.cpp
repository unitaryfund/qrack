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

#include "qbdt_node.hpp"
#include "qbdt_qstabilizer_node.hpp"

#define IS_0_PROB(p) (p < (ONE_R1 / 4))
#define IS_1_PROB(p) (p > (3 * ONE_R1 / 4))
#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_SAME_AMP(a, b) (abs((a) - (b)) <= REAL1_EPSILON)

namespace Qrack {
bool QBdtQStabilizerNode::isEqual(QBdtNodeInterfacePtr r)
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

    if (IS_NODE_0(scale)) {
        return true;
    }

    QStabilizerPtr rReg = std::dynamic_pointer_cast<QBdtQStabilizerNode>(r)->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    if (qReg->ApproxCompare(rReg)) {
        qReg = rReg;
        return true;
    }

    return false;
}

bool QBdtQStabilizerNode::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (IS_NODE_0(scale)) {
        return IS_NODE_0(r->scale);
    }

    QStabilizerPtr rReg = std::dynamic_pointer_cast<QBdtQStabilizerNode>(r)->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    if (qReg->ApproxCompare(rReg)) {
        qReg = rReg;
        return true;
    }

    return false;
}

void QBdtQStabilizerNode::Branch(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth) {
        return;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    if (qReg) {
        qReg = std::dynamic_pointer_cast<QStabilizer>(qReg->Clone());
    }
}

void QBdtQStabilizerNode::Prune(bitLenInt depth, bitLenInt unused)
{
    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    const real1_f phaseArg = qReg->FirstNonzeroPhase();
    qReg->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, -phaseArg);
    scale *= std::polar(ONE_R1, (real1)phaseArg);
}

void QBdtQStabilizerNode::InsertAtDepth(
    QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (IS_NODE_0(scale)) {
        SetZero();
        return;
    }

    QBdtQStabilizerNodePtr bEng = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b);
    qReg->Compose(bEng->qReg, depth);
}

QBdtNodeInterfacePtr QBdtQStabilizerNode::RemoveSeparableAtDepth(
    bitLenInt depth, const bitLenInt& size, bitLenInt parDepth)
{
    if (!size) {
        return NULL;
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return NULL;
    }

    QBdtQStabilizerNodePtr toRet = std::dynamic_pointer_cast<QBdtQStabilizerNode>(ShallowClone());
    toRet->scale /= abs(toRet->scale);

    if (!qReg) {
        return toRet;
    }

    toRet->qReg = std::dynamic_pointer_cast<QStabilizer>(qReg->Decompose(depth, size));

    return toRet;
}

QBdtNodeInterfacePtr QBdtQStabilizerNode::PopSpecial(bitLenInt depth)
{
    if (!depth || IS_NODE_0(scale)) {
        return shared_from_this();
    }

    --depth;

    // Quantum teleportation algorithm:

    QBdtNodeInterfacePtr nRoot = std::make_shared<QBdtNode>(scale);

    // We need a Bell pair for teleportation, with one end on each side of the QBDT/stabilizer domain wall.
    // We allocate one end in the stabilizer simulator.
    qReg = std::dynamic_pointer_cast<QStabilizer>(qReg->Clone());
    const bitLenInt aliceBellBit = qReg->GetQubitCount();
    qReg->Allocate(1U);

    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |+>, (or effectively |0> followed by H gate).
    const QStabilizerPtr qRegB1 = std::dynamic_pointer_cast<QStabilizer>(qReg->Clone());
    nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, qReg);
    nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, qRegB1);
    QBdtNodeInterfacePtr& b0 = nRoot->branches[0U];
    QBdtNodeInterfacePtr& b1 = nRoot->branches[1U];

    // We act CNOT from |+> control to |0> target.
    // (Notice, we act X gate in |1> branch and no gate in |0> branch.)
    qRegB1->X(aliceBellBit);
    nRoot->Prune(2U);
    // This is the Bell pair "Eve" creates to distribute to "Alice" and "Bob," in quantum teleportation.

    // "Alice" prepares and sends a qubit to "Bob"; stabilizer prepares and sends a qubit to QBDT.
    // Alice's "prepared state" to teleport is the 0U index stabilizer qubit, (same in both branches).
    // Alice uses the "prepared state" qubit as the control of a CNOT on the Bell pair.
    qReg->CNOT(0U, aliceBellBit);
    qRegB1->CNOT(0U, aliceBellBit);
    nRoot->Prune(2U);

    // Alice acts H on her "prepared state":
    qReg->H(0U);
    qRegB1->H(0U);
    nRoot->Prune(2U);

    // Alice now measures both of her bits, and records the results.

    // First, measure Alice's Bell pair bit.
    const real1 p01 = qReg->Prob(aliceBellBit);
    const real1 p11 = qRegB1->Prob(aliceBellBit);
    const bool q1 = qReg->Rand() < ((p01 + p11) / 2);

    bool isB0 = !((q1 && IS_0_PROB(p01)) || (!q1 && IS_1_PROB(p01)));
    if (isB0) {
        qReg->ForceM(aliceBellBit, q1);
        qReg->Dispose(aliceBellBit, 1U);
    } else {
        b0->SetZero();
    }

    bool isB1 = !((q1 && IS_0_PROB(p11)) || (!q1 && IS_1_PROB(p11)));
    if (isB1) {
        qRegB1->ForceM(aliceBellBit, q1);
        qRegB1->Dispose(aliceBellBit, 1U);
    } else {
        b1->SetZero();
    }

    nRoot->Prune(2U);
    nRoot->Normalize(2U);

    // Next, measure Alice's "prepared state" bit.
    const real1 p00 = isB0 ? qReg->Prob(0U) : qRegB1->Prob(0U);
    const real1 p10 = isB1 ? qRegB1->Prob(0U) : qReg->Prob(0U);
    const bool q0 = qReg->Rand() < ((p00 + p10) / 2);

    if (isB0) {
        if ((q0 && IS_0_PROB(p00)) || (!q0 && IS_1_PROB(p00))) {
            b0->SetZero();
        } else {
            qReg->ForceM(0U, q0);
            qReg->Dispose(0U, 1U);
        }
    }

    if (isB1) {
        if ((q0 && IS_0_PROB(p10)) || (!q0 && IS_1_PROB(p10))) {
            b1->SetZero();
        } else {
            qRegB1->ForceM(0U, q0);
            qRegB1->Dispose(0U, 1U);
        }
    }

    nRoot->Prune(2U);
    nRoot->Normalize(2U);

    // Bob acts 0 to 2 corrective gates based upon Alice's measured bits.
    if (q0) {
        b1->scale = -b1->scale;
        nRoot->Prune(2U);
    }
    if (q1) {
        std::swap(b0, b1);
        nRoot->Prune(2U);
    }

    // This process might need to be repeated, recursively.
    if (!IS_NORM_0(b0->scale)) {
        b0 = b0->PopSpecial(depth);
    }
    if (!IS_NORM_0(b1->scale)) {
        b1 = b1->PopSpecial(depth);
    }
    nRoot->Prune(2U);

    // We're done! Just return the replacement for "this" pointer.
    return nRoot;
}

} // namespace Qrack
