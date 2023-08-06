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
    bitLenInt aliceBellBit = qReg->GetQubitCount();
    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |+>, (or effectively |0> followed by H gate).
    QBdtNodeInterfacePtr& b0 = nRoot->branches[0U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QStabilizer>(qReg->Clone()));
    QBdtNodeInterfacePtr& b1 = nRoot->branches[1U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QStabilizer>(qReg->Clone()));

    QStabilizerPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    QStabilizerPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;
    // We allocate the other Bell pair end in the stabilizer simulator.
    qReg0->Allocate(1U);
    qReg1->Allocate(1U);

    // We act CNOT from |+> control to |0> target.
    // (Notice, we act X gate in |1> branch and no gate in |0> branch.)
    qReg1->X(aliceBellBit);
    nRoot->Prune(2U);
    // This is the Bell pair "Eve" creates to distribute to "Alice" and "Bob," in quantum teleportation.

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // "Alice" prepares and sends a qubit to "Bob"; stabilizer prepares and sends a qubit to QBDT.
    // Alice's "prepared state" to teleport is the 0U index stabilizer qubit, (same in both branches).
    // Alice uses the "prepared state" qubit as the control of a CNOT on the Bell pair.
    qReg0->CNOT(0U, aliceBellBit);
    qReg1->CNOT(0U, aliceBellBit);
    nRoot->Prune(2U);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // Alice acts H on her "prepared state":
    qReg0->H(0U);
    qReg1->H(0U);
    nRoot->Prune(2U);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // Alice would now measure both of her bits, and record the results.
    // Bob would act 0 to 2 corrective gates based upon Alice's measured bits.
    // However, we want to avoid measurement. (Our stabilizer simulation isn't aware of phase with mid-circuit
    // measurement.) Instead, we can just do the entire algorithm in a unitary manner!

    // Bob controls his (C)Z correction with Alice's "prepared state" bit.
    qReg1->Z(0U);
    qReg0->Dispose(0U, 1U);
    qReg1->Dispose(0U, 1U);
    --aliceBellBit;
    nRoot->Prune(2U);

    qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // TODO: The CX gate is backwards.

    // Bob controls his (C)X correction with Alice's "prepared state" bit.
    // qReg1->X(aliceBellBit);
    // nRoot->Prune(2U);
    // nRoot->Normalize(2U);

    const real1 p01 = qReg0->Prob(aliceBellBit);
    const real1 p11 = qReg1->Prob(aliceBellBit);
    const bool q1 = qReg->Rand() < ((p01 + p11) / 2);

    if (!((q1 && IS_0_PROB(p01)) || (!q1 && IS_1_PROB(p01)))) {
        qReg0->ForceM(aliceBellBit, q1);
        qReg0->Dispose(aliceBellBit, 1U);
    } else {
        b0->SetZero();
        nRoot->Normalize();
    }

    if (!((q1 && IS_0_PROB(p11)) || (!q1 && IS_1_PROB(p11)))) {
        qReg1->ForceM(aliceBellBit, q1);
        qReg1->Dispose(aliceBellBit, 1U);
    } else {
        b1->SetZero();
        nRoot->Normalize();
    }

    if (q1) {
        std::swap(b0, b1);
        nRoot->Prune();
    }

    // This process might need to be repeated, recursively.
    if (!IS_NORM_0(b0->scale)) {
        b0 = b0->PopSpecial(depth);
    }
    if (!IS_NORM_0(b1->scale)) {
        b1 = b1->PopSpecial(depth);
    }
    nRoot->Prune();

    // We're done! Just return the replacement for "this" pointer.
    return nRoot;
}

} // namespace Qrack
