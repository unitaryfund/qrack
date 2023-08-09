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

    if (!r->IsStabilizer()) {
        return false;
    }

    QUnitCliffordPtr rReg = std::dynamic_pointer_cast<QBdtQStabilizerNode>(r)->qReg;

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

    if (IS_NODE_0(r->scale)) {
        return false;
    }

    if (!r->IsStabilizer()) {
        return false;
    }

    QUnitCliffordPtr rReg = std::dynamic_pointer_cast<QBdtQStabilizerNode>(r)->qReg;

    if (!rReg) {
        return !qReg;
    }

    if (!qReg) {
        return false;
    }

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
        qReg = std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone());
    }
}

QBdtNodeInterfacePtr QBdtQStabilizerNode::Prune(bitLenInt depth, bitLenInt unused, const bool& unused2)
{
    if (IS_NODE_0(scale)) {
        SetZero();
        return shared_from_this();
    }

    const real1_f phaseArg = qReg->FirstNonzeroPhase();
    qReg->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, -phaseArg);
    scale *= std::polar(ONE_R1, (real1)phaseArg);

    return shared_from_this();
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

    toRet->qReg = std::dynamic_pointer_cast<QUnitClifford>(qReg->Decompose(depth, size));

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
    const bitLenInt aliceBellBit = qReg->GetQubitCount();
    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |+>, (or effectively |0> followed by H gate).
    QBdtNodeInterfacePtr& b0 = nRoot->branches[0U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
    QBdtNodeInterfacePtr& b1 = nRoot->branches[1U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));

    QUnitCliffordPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    QUnitCliffordPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;
    // We allocate the other Bell pair end in the stabilizer simulator.
    qReg0->Allocate(1U);
    qReg1->Allocate(1U);

    // We act CNOT from |+> control to |0> target.
    // (Notice, we act X gate in |1> branch and no gate in |0> branch.)
    qReg1->X(aliceBellBit);
    nRoot = nRoot->Prune(2U, 1U, true);
    // This is the Bell pair "Eve" creates to distribute to "Alice" and "Bob," in quantum teleportation.

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // "Alice" prepares and sends a qubit to "Bob"; stabilizer prepares and sends a qubit to QBDT.
    // Alice's "prepared state" to teleport is the 0U index stabilizer qubit, (same in both branches).
    // Alice uses the "prepared state" qubit as the control of a CNOT on the Bell pair.
    qReg0->CNOT(0U, aliceBellBit);
    qReg1->CNOT(0U, aliceBellBit);
    nRoot = nRoot->Prune(2U, 1U, true);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // Alice acts H on her "prepared state":
    qReg0->H(0U);
    qReg1->H(0U);
    nRoot = nRoot->Prune(2U, 1U, true);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // Alice now measures both of her bits, and records the results.

    // First, measure Alice's Bell pair bit.
    const real1 p01 = qReg0->Prob(aliceBellBit);
    const real1 p11 = qReg1->Prob(aliceBellBit);
    const bool q1 = qReg->Rand() < ((p01 + p11) / 2);

    bool isB0 = !((q1 && IS_0_PROB(p01)) || (!q1 && IS_1_PROB(p01)));
    if (isB0) {
        qReg0->ForceM(aliceBellBit, q1);
        qReg0->Dispose(aliceBellBit, 1U);
    } else {
        b0->SetZero();
    }

    bool isB1 = !((q1 && IS_0_PROB(p11)) || (!q1 && IS_1_PROB(p11)));
    if (isB1) {
        qReg1->ForceM(aliceBellBit, q1);
        qReg1->Dispose(aliceBellBit, 1U);
    } else {
        b1->SetZero();
    }

    nRoot = nRoot->Prune(2U, 1U, true);
    nRoot->Normalize(2U);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // Next, measure Alice's "prepared state" bit.
    const real1 p00 = isB0 ? qReg0->Prob(0U) : qReg1->Prob(0U);
    const real1 p10 = isB1 ? qReg1->Prob(0U) : qReg0->Prob(0U);
    const bool q0 = qReg->Rand() < ((p00 + p10) / 2);

    if (isB0) {
        if ((q0 && IS_0_PROB(p00)) || (!q0 && IS_1_PROB(p00))) {
            b0->SetZero();
        } else {
            qReg0->ForceM(0U, q0);
            qReg0->Dispose(0U, 1U);
        }
    }

    if (isB1) {
        if ((q0 && IS_0_PROB(p10)) || (!q0 && IS_1_PROB(p10))) {
            b1->SetZero();
        } else {
            qReg1->ForceM(0U, q0);
            qReg1->Dispose(0U, 1U);
        }
    }

    nRoot = nRoot->Prune(2U, 1U, true);
    nRoot->Normalize(2U);

    // Bob acts 0 to 2 corrective gates based upon Alice's measured bits.
    if (q0) {
        b1->scale = -b1->scale;
        nRoot = nRoot->Prune(1U, 1U, true);
    }
    if (q1) {
        std::swap(b0, b1);
        nRoot = nRoot->Prune(1U, 1U, true);
    }

    // This process might need to be repeated, recursively.
    if (!IS_NORM_0(b0->scale)) {
        b0 = b0->PopSpecial(depth);
    }
    if (!IS_NORM_0(b1->scale)) {
        b1 = b1->PopSpecial(depth);
    }
    nRoot = nRoot->Prune(1U, 1U, true);

    // We're done! Just return the replacement for "this" pointer.
    return nRoot;
}

} // namespace Qrack
