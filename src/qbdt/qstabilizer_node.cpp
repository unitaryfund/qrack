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

    QBdtNodeInterfacePtr lhs = PopSpecial();

    return lhs->isEqualUnder(r);
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
    if (!depth) {
        return shared_from_this();
    }

    if (IS_NODE_0(scale)) {
        SetZero();
    }

    --depth;

    // Creating a "new root" (to replace keyword "this" class instance node, on return) effectively allocates a new
    // qubit reset to |0>.
    QBdtNodeInterfacePtr nRoot = std::make_shared<QBdtNode>(scale);
    nRoot->mtx = mtx;
    nRoot->branches[0U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
    nRoot->branches[1U] =
        std::make_shared<QBdtQStabilizerNode>(SQRT1_2_R1, std::dynamic_pointer_cast<QUnitClifford>(qReg->Clone()));
    QBdtQStabilizerNodePtr b0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[0U]);
    QBdtQStabilizerNodePtr b1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(nRoot->branches[1U]);
    QUnitCliffordPtr qReg0 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b0)->qReg;
    QUnitCliffordPtr qReg1 = std::dynamic_pointer_cast<QBdtQStabilizerNode>(b1)->qReg;

    if (!IS_NODE_0(scale)) {
        // Swap gate decomposition:
        if (qReg->IsSeparableZ(0U)) {
            // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
            if (!qReg->M(0U)) {
                // |0>
                nRoot->branches[0U]->scale = ONE_CMPLX;
                nRoot->branches[1U]->scale = ZERO_CMPLX;
            } else {
                // |1>
                nRoot->branches[0U]->scale = ZERO_CMPLX;
                nRoot->branches[1U]->scale = ONE_CMPLX;

                qReg0->X(0U);
                qReg1->X(0U);
            }
        } else if (qReg->IsSeparableX(0U)) {
            // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
            qReg0->H(0U);
            qReg1->H(0U);
            if (!qReg0->M(0U)) {
                // |+>
                nRoot->branches[0U]->scale = complex(SQRT1_2_R1, ZERO_R1);
                nRoot->branches[1U]->scale = complex(SQRT1_2_R1, ZERO_R1);
            } else {
                // |->
                nRoot->branches[0U]->scale = complex(SQRT1_2_R1, ZERO_R1);
                nRoot->branches[1U]->scale = complex(-SQRT1_2_R1, ZERO_R1);

                qReg0->X(0U);
                qReg1->X(0U);
            }
        } else if (qReg->IsSeparableY(0U)) {
            // If the stabilizer qubit is separable, we just prepare the QBDD qubit in the same state.
            qReg0->IS(0U);
            qReg0->H(0U);
            qReg1->IS(0U);
            qReg1->H(0U);
            if (!qReg0->M(0U)) {
                // |"left">
                nRoot->branches[0U]->scale = complex(SQRT1_2_R1, ZERO_R1);
                nRoot->branches[1U]->scale = complex(ZERO_R1, SQRT1_2_R1);
            } else {
                // |"right">
                nRoot->branches[0U]->scale = complex(SQRT1_2_R1, ZERO_R1);
                nRoot->branches[1U]->scale = complex(ZERO_R1, -SQRT1_2_R1);

                qReg0->X(0U);
                qReg1->X(0U);
            }
        } else {
            // SWAP gate from QBdt qubit to stabilizer qubit with 3 CNOTs...

            // We initialize the QBdt qubit to |0>.

            // We act X gate in nRoot |1> branch and no gate in |0> branch.
            // Since there's no probability of |1> branch, yet, we simply skip this.

            // H on QBdt qubit - we initialized the QBDD qubit at this point.

            // H-Z-H is X
            qReg1->Z(0U);

            // Second H on QBdt qubit...
            // Think about it: the QBDD qubit has equal scale factors in both branches.
            // The stabilizer subsystems in each QBDD branch only differ by that last Z.
            // Then, acting H is like applying 1/sqrt(2) to each branch and summing...
            // If QBDD |0> branch is "A" and |1> branch is "B,"
            // then "A" becomes "A+B," and "B" becomes "A-B". For "A+B,"
            // this doubles the amplitudes where the stabilizer qubit is |0>, and
            // this exactly cancels the amplitudes where the stabilizer qubit is |1>.
            // For "A-B," the situation is exactly opposite.
            // If the stabilizer qubit were separable, QBDD would end up a Z eigenstate.
            // However, we've handled that case, so "stabilizer rank" is 2.
            // ...This ends up being just post selection in both stabilizers!
            qReg0->ForceM(0U, false);
            qReg1->ForceM(0U, true);
            // This doesn't change logical state, but the stabilizer qubit is an "ancilla,"
            // so we reset it to exactly |0>, across branches.
            qReg1->X(0U);

            // CNOT from QBdt qubit to stabilizer qubit...
            // (Notice, we act X gate in nRoot |1> branch and no gate in |0> branch.)
            qReg1->X(0U);
        }
    }

    // The stabilizer qubit becomes an ancilla.
    qReg0->ROR(1U, 0U, qReg0->GetQubitCount());
    qReg1->ROR(1U, 0U, qReg1->GetQubitCount());
    ++(b0->ancillaCount);
    ++(b1->ancillaCount);

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
