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

#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
#include <future>
#include <thread>
#endif

#define IS_0_PROB(p) (p < (ONE_R1 / 4))
#define IS_1_PROB(p) (p > (3 * ONE_R1 / 4))
#define IS_NODE_0(c) (norm(c) <= _qrack_qbdt_sep_thresh)
#define IS_SAME_AMP(a, b) (abs((a) - (b)) <= REAL1_EPSILON)

namespace Qrack {
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
const unsigned numThreads = std::thread::hardware_concurrency() << 1U;
#if ENABLE_ENV_VARS
const bitLenInt pStridePow =
    (((bitLenInt)(getenv("QRACK_PSTRIDEPOW") ? std::stoi(std::string(getenv("QRACK_PSTRIDEPOW"))) : PSTRIDEPOW)) +
        1U) >>
    1U;
#else
const bitLenInt pStridePow = (PSTRIDEPOW + 1U) >> 1U;
#endif
const bitCapInt pStride = pow2(pStridePow);
#endif

bool QBdtQStabilizerNode::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (this == r.get()) {
        return true;
    }

    QBdtQStabilizerNodePtr rStab = r->IsStabilizer() ? std::dynamic_pointer_cast<QBdtQStabilizerNode>(r) : NULL;
    const bool isRTerminal = rStab ? (rStab->ancillaCount >= rStab->qReg->GetQubitCount()) : !r->branches[0U];

    if (ancillaCount >= qReg->GetQubitCount()) {
        return isRTerminal;
    }

    if (isRTerminal) {
        return false;
    }

    if (rStab) {
        return isEqualBranch(r, 0U);
    }

    QBdtNodeInterfacePtr ths = PopSpecial();

    return ths->isEqualBranch(r, 0U) && ths->isEqualBranch(r, 1U);
}

bool QBdtQStabilizerNode::isEqualBranch(QBdtNodeInterfacePtr r, const bool& unused)
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
    } else if (ancillaCount > rStab->ancillaCount) {
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
    if (!size) {
        return;
    }

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

QBdtNodeInterfacePtr QBdtQStabilizerNode::PopSpecial(bitLenInt depth, bitLenInt parDepth)
{
    if (!depth || (ancillaCount >= qReg->GetQubitCount())) {
        return shared_from_this();
    }

    if (IS_NODE_0(scale)) {
        SetZero();
        return shared_from_this();
    }

    --depth;

    QBdtNodeInterfacePtr nRoot = std::make_shared<QBdtNode>(scale);
    nRoot->mtx = mtx;

    // Swap gate decomposition:

    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |0>.

    if (qReg->IsSeparableZ(0U)) {
        // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
        if (qReg->M(0U)) {
            // |0>
            nRoot->branches[0U] =
                std::make_shared<QBdtQStabilizerNode>(ONE_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[0U] =
                std::make_shared<QBdtQStabilizerNode>(ZERO_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        } else {
            // |1>
            qReg->X(0U);
            nRoot->branches[0U] =
                std::make_shared<QBdtQStabilizerNode>(ZERO_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[1U] =
                std::make_shared<QBdtQStabilizerNode>(ONE_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        }
    } else if (qReg->IsSeparableX(0U)) {
        // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
        qReg->H(0U);
        if (qReg->M(0U)) {
            // |+>
            nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(
                SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(
                SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        } else {
            // |->
            qReg->X(0U);
            nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(
                SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(
                -SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        }
    } else if (qReg->IsSeparableY(0U)) {
        // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
        qReg->IS(0U);
        qReg->H(0U);
        if (qReg->M(0U)) {
            // |"left">
            nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(
                SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(
                complex(ZERO_R1, SQRT1_2_R1), std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        } else {
            // |"right">
            qReg->X(0U);
            nRoot->branches[0U] = std::make_shared<QBdtQStabilizerNode>(
                SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
            nRoot->branches[1U] = std::make_shared<QBdtQStabilizerNode>(
                complex(ZERO_R1, -SQRT1_2_R1), std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        }
    } else {
        // CNOT from QBdt qubit to stabilizer qubit...
        // (We act X gate in nRoot |1> branch and no gate in |0> branch.)
        // Since there's no probability of |1> branch, yet, we simply skip this.

        // CNOT from stabilizer qubit to QBdt qubit...

        // H QBdt
        nRoot->branches[0U] =
            std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        nRoot->branches[1U] =
            std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
        QBdtQStabilizerNodePtr b0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[0U]);
        QBdtQStabilizerNodePtr b1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[1U]);
        QUnitCliffordPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
        QUnitCliffordPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

        // H-Z-H is X
        qReg1->Z(0U);

        // TODO: H QBdt
        throw std::runtime_error("Not implemented!");

        // CNOT from QBdt qubit to stabilizer qubit...
        // (Notice, we act X gate in nRoot |1> branch and no gate in |0> branch.)
        qReg1->X(0U);
    }

    QBdtQStabilizerNodePtr b0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[0U]);
    QBdtQStabilizerNodePtr b1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[1U]);
    QUnitCliffordPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    QUnitCliffordPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    if ((ancillaCount + 1U) >= qReg->GetQubitCount()) {
        // If this is the last stabilizer qubit, clear the container and reset the ancilla count.
        qReg0->Clear();
        qReg1->Clear();
        b0->ancillaCount = 0U;
        b1->ancillaCount = 0U;
    } else if (qReg->CanDecomposeDispose(0U, 1U)) {
        // If the stabilizer qubit can be disposed, avoid an ancilla.
        qReg0->Dispose(0U, 1U);
        qReg1->Dispose(0U, 1U);
    } else {
        // Otherwise, the stabilizer qubit becomes an ancilla.
        qReg0->ROR(1U, 0U, qReg0->GetQubitCount());
        qReg1->ROR(1U, 0U, qReg1->GetQubitCount());
        ++(b0->ancillaCount);
        ++(b1->ancillaCount);
    }

    nRoot = nRoot->Prune(2U, 1U, true);

    // This process might need to be repeated, recursively.
#if ENABLE_QBDT_CPU_PARALLEL && ENABLE_PTHREAD
    unsigned underThreads = (unsigned)(pow2(depth) / pStride);
    if (underThreads == 1U) {
        underThreads = 0U;
    }
    if ((depth >= pStridePow) && ((pow2(parDepth) * (underThreads + 1U)) <= numThreads)) {
        ++parDepth;

        std::future<void> future0 = std::async(
            std::launch::async, [&] { nRoot->branches[0U] = nRoot->branches[0U]->PopSpecial(depth, parDepth); });
        nRoot->branches[1U] = nRoot->branches[1U]->PopSpecial(depth, parDepth);

        future0.get();
    } else {
        nRoot->branches[0U] = nRoot->branches[0U]->PopSpecial(depth, parDepth);
        nRoot->branches[1U] = nRoot->branches[1U]->PopSpecial(depth, parDepth);
    }
#else
    nRoot->branches[0U] = nRoot->branches[0U]->PopSpecial(depth, parDepth);
    nRoot->branches[1U] = nRoot->branches[1U]->PopSpecial(depth, parDepth);
#endif

    nRoot = nRoot->Prune(2U, 1U, true);

    // We're done! Just return the replacement for "this" pointer.
    return nRoot;
}

} // namespace Qrack
