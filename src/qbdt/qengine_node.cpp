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

#include "qbdt_qengine_node.hpp"

#define IS_SAME_AMP(a, b) (abs((a) - (b)) <= REAL1_EPSILON)

namespace Qrack {
bool QBdtQEngineNode::isEqual(QBdtNodeInterfacePtr r)
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

    if (IS_NORM_0(scale)) {
        return true;
    }

    QEnginePtr rReg = std::dynamic_pointer_cast<QBdtQEngineNode>(r)->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    if (qReg->ApproxCompare(rReg)) {
        qReg = rReg;
        return true;
    }

    return false;
}

bool QBdtQEngineNode::isEqualUnder(QBdtNodeInterfacePtr r)
{
    if (!r) {
        return false;
    }

    if (this == r.get()) {
        return true;
    }

    if (IS_NORM_0(scale)) {
        return IS_NORM_0(r->scale);
    }

    QEnginePtr rReg = std::dynamic_pointer_cast<QBdtQEngineNode>(r)->qReg;

    if (qReg.get() == rReg.get()) {
        return true;
    }

    if (qReg->ApproxCompare(rReg)) {
        qReg = rReg;
        return true;
    }

    return false;
}

void QBdtQEngineNode::Normalize(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (qReg) {
        qReg->UpdateRunningNorm();
        qReg->NormalizeState();
    }
}

void QBdtQEngineNode::Branch(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    if (qReg) {
        qReg = std::dynamic_pointer_cast<QEngine>(qReg->Clone());
    }
}

void QBdtQEngineNode::Prune(bitLenInt depth, bitLenInt unused)
{
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    const real1_f phaseArg = qReg->FirstNonzeroPhase();
    qReg->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, -phaseArg);
    scale *= std::polar(ONE_R1, (real1)phaseArg);
}

void QBdtQEngineNode::InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size)
{
    if (IS_NORM_0(scale)) {
        return;
    }

    QBdtQEngineNodePtr bEng = std::dynamic_pointer_cast<QBdtQEngineNode>(b);
    qReg->Compose(bEng->qReg, depth);
}

QBdtNodeInterfacePtr QBdtQEngineNode::RemoveSeparableAtDepth(bitLenInt depth, const bitLenInt& size)
{
    if (!size || IS_NORM_0(scale)) {
        return NULL;
    }

    QBdtQEngineNodePtr toRet = std::dynamic_pointer_cast<QBdtQEngineNode>(ShallowClone());
    toRet->scale /= abs(toRet->scale);

    if (!qReg) {
        return toRet;
    }

    toRet->qReg = std::dynamic_pointer_cast<QEngine>(qReg->Decompose(depth, size));

    return toRet;
}

#if ENABLE_COMPLEX_X2
void QBdtQEngineNode::PushSpecial(const complex2& mtrxCol1, const complex2& mtrxCol2, QBdtNodeInterfacePtr& b1)
{
    const complex mtrx[4U]{ mtrxCol1.c[0U], mtrxCol2.c[0U], mtrxCol1.c[1U], mtrxCol2.c[1U] };
#else
void QBdtQEngineNode::PushSpecial(complex const* mtrx, QBdtNodeInterfacePtr& b1)
{
#endif
    const bool is0Zero = IS_NORM_0(scale);
    const bool is1Zero = IS_NORM_0(b1->scale);

    if (is0Zero && is1Zero) {
        SetZero();
        b1->SetZero();

        return;
    }

    QEnginePtr qReg0 = qReg;
    QEnginePtr qReg1 = std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg;

    if (is0Zero) {
        qReg0 = std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg->CloneEmpty();
    } else if (is1Zero) {
        qReg1 = qReg->CloneEmpty();
    }

    qReg0->NormalizeState((real1_f)(ONE_R1 / norm(scale)), REAL1_DEFAULT_ARG, (real1_f)std::arg(scale));
    qReg1->NormalizeState((real1_f)(ONE_R1 / norm(b1->scale)), REAL1_DEFAULT_ARG, (real1_f)std::arg(b1->scale));

    scale = SQRT1_2_R1;
    b1->scale = SQRT1_2_R1;

    qReg0->ShuffleBuffers(qReg1);

    qReg0->Mtrx(mtrx, qReg0->GetQubitCount() - 1U);
    qReg1->Mtrx(mtrx, qReg1->GetQubitCount() - 1U);

    qReg0->ShuffleBuffers(qReg1);

    qReg = qReg0;
    std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg = qReg1;
}

void QBdtQEngineNode::PopStateVector(bitLenInt depth)
{
    if (IS_NORM_0(scale)) {
        SetZero();
        return;
    }

    qReg->UpdateRunningNorm();
    const real1_f nrm = qReg->GetRunningNorm();

    if (nrm <= _qrack_qbdt_sep_thresh) {
        SetZero();
        return;
    }

    qReg->NormalizeState();
    scale = std::polar((real1)sqrt(nrm), ZERO_R1);
}
} // namespace Qrack
