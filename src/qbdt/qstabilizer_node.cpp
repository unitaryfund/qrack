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
bool QBdtQStabilizerNode::isEqualBranch(QBdtNodeInterfacePtr r, const bool& b)
{
    QBdtQStabilizerNodePtr rStab = std::dynamic_pointer_cast<QBdtQStabilizerNode>(r);
    QUnitCliffordPtr rReg = rStab->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    QUnitCliffordPtr lReg = qReg;

    if (ancillaCount < rStab->ancillaCount) {
        lReg = std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone());
        lReg->Allocate(rStab->ancillaCount - ancillaCount);
    }

    if (ancillaCount > rStab->ancillaCount) {
        rReg = std::dynamic_pointer_cast<QUnitClifford>(rReg->Clone());
        rReg->Allocate(ancillaCount - rStab->ancillaCount);
    }

    if (!lReg->ApproxCompare(rReg)) {
        return false;
    }

    if (ancillaCount > rStab->ancillaCount) {
        qReg = rStab->qReg;
    } else {
        rStab->qReg = qReg;
    }

    return true;
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
    if (!depth) {
        return shared_from_this();
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return shared_from_this();
    }

    --depth;

    QBdtNodeInterfacePtr nRoot = std::make_shared<QBdtNode>(scale);

    // If the stabilizer qubit to "pop" satisfies the separability condition and other assumptions of "Decompose(),"
    // then we can completely avoid the quantum teleportation algorithm, and just duplicate the stabilizer qubit as a
    // QBdtNode branch pair qubit, by direct query and preparation of state.

    if (qReg->CanDecomposeDispose(0U, 1U)) {
        QUnitCliffordPtr clone = std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone());
        QInterfacePtr qubit = clone->Decompose(0U, 1U);
        complex stateVec[2U];
        qubit->GetQuantumState(stateVec);

        if (IS_NORM_0(stateVec[0U])) {
            nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>();
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(ONE_CMPLX, clone);

            return nRoot;
        }

        nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(stateVec[0U], clone);
        if (IS_NORM_0(stateVec[1U])) {
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>();

            return nRoot;
        }

        clone = std::dynamic_pointer_cast<QUnitClifford>(clone->Clone());
        nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(stateVec[1U], clone);

        return nRoot;
    }

    // Quantum teleportation algorithm:

    // We need a Bell pair for teleportation, with one end on each side of the QBDT/stabilizer domain wall.
    const bitLenInt aliceBellBit = qReg->GetQubitCount() - ancillaCount;
    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |+>, (or effectively |0> followed by H gate).
    nRoot->branches[0U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
    nRoot->branches[1U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
    QBdtQStabilizerNodePtr b0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[0U]);
    QBdtQStabilizerNodePtr b1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[1U]);
    QUnitCliffordPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    QUnitCliffordPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    // We allocate the other Bell pair end in the stabilizer simulator.
    // We're trying to minimize the use of auxiliary qubits, and reuse them.
    if (!ancillaCount) {
        qReg0->Allocate(1U);
        qReg1->Allocate(1U);
        ++(b0->ancillaCount);
        ++(b1->ancillaCount);
    }

    // We act CNOT from |+> control to |0> target.
    // (Notice, we act X gate in |1> branch and no gate in |0> branch.)
    qReg1->X(aliceBellBit);

    // This is the Bell pair "Eve" creates to distribute to "Alice" and "Bob," in quantum teleportation.

    // "Alice" prepares and sends a qubit to "Bob"; stabilizer prepares and sends a qubit to QBDT.
    // Alice's "prepared state" to teleport is the 0U index stabilizer qubit, (same in both branches).
    // Alice uses the "prepared state" qubit as the control of a CNOT on the Bell pair.
    qReg0->CNOT(0U, aliceBellBit);
    qReg1->CNOT(0U, aliceBellBit);

    // Alice acts H on her "prepared state":
    qReg0->H(0U);
    qReg1->H(0U);

    // Alice now measures both of her bits, and records the results.

    // First, measure Alice's Bell pair bit.
    const real1 p01 = qReg0->Prob(aliceBellBit);
    const real1 p11 = qReg1->Prob(aliceBellBit);
    const bool q1 = (7 * ONE_R1 / 8) > ((p01 + p11) / 2);

    const bool isB0 = !((q1 && IS_0_PROB(p01)) || (!q1 && IS_1_PROB(p01)));
    if (isB0) {
        qReg0->ForceM(aliceBellBit, q1);
        // This is now considered an auxiliary qubit, at back of index order.
        // We reset it to |0>, always, when done with it.
        if (q1) {
            qReg0->X(aliceBellBit);
        }
    } else {
        b0->SetZero();
    }

    const bool isB1 = !((q1 && IS_0_PROB(p11)) || (!q1 && IS_1_PROB(p11)));
    if (isB1) {
        qReg1->ForceM(aliceBellBit, q1);
        // This is now considered an auxiliary qubit, at back of index order.
        // We reset it to |0>, always, when done with it.
        if (q1) {
            qReg1->X(aliceBellBit);
        }
    } else {
        b1->SetZero();
    }

    // Next, measure Alice's "prepared state" bit.
    const real1 p00 = isB0 ? qReg0->Prob(0U) : qReg1->Prob(0U);
    const real1 p10 = isB1 ? qReg1->Prob(0U) : qReg0->Prob(0U);
    const bool q0 = (7 * ONE_R1 / 8) > ((p00 + p10) / 2);

    if (isB0) {
        if ((q0 && IS_0_PROB(p00)) || (!q0 && IS_1_PROB(p00))) {
            b0->SetZero();
        } else {
            qReg0->ForceM(0U, q0);
            // This is now considered an (additional) auxiliary qubit.
            ++(b0->ancillaCount);
            // We reset it to |0>, always, when done with it.
            if (q0) {
                qReg0->X(0U);
            }
            // To place it at back of index order, we use a logical shift:
            qReg0->ROR(1U, 0U, qReg0->GetQubitCount());
        }
    }

    if (isB1) {
        if ((q0 && IS_0_PROB(p10)) || (!q0 && IS_1_PROB(p10))) {
            b1->SetZero();
        } else {
            qReg1->ForceM(0U, q0);
            // This is now considered an (additional) auxiliary qubit.
            ++(b1->ancillaCount);
            // We reset it to |0>, always, when done with it.
            if (q0) {
                qReg1->X(0U);
            }
            // To place it at back of index order, we use a logical shift:
            qReg1->ROR(1U, 0U, qReg1->GetQubitCount());
        }
    }

    // Bob acts 0 to 2 corrective gates based upon Alice's measured bits.
    if (q0) {
        b1->scale = -b1->scale;
    }
    if (q1) {
        nRoot->branches[0U].swap(nRoot->branches[1U]);
    }

    // This process might need to be repeated, recursively.
    nRoot->branches[0U] = b0->PopSpecial(depth);
    nRoot->branches[1U] = b1->PopSpecial(depth);

    nRoot->Prune(2U, 1U, true);

    // We're done! Just return the replacement for "this" pointer.
    return nRoot;
}

} // namespace Qrack
