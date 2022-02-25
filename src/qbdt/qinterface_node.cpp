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

#include "qbdt_qinterface_node.hpp"
#include "qengine.hpp"

namespace Qrack {
void QBdtQEngineNode::PushStateVector(
    const complex* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth)
{
    QEnginePtr qReg0 = std::dynamic_pointer_cast<QEngine>(std::dynamic_pointer_cast<QBdtQEngineNode>(b0)->qReg);
    QEnginePtr qReg1 = std::dynamic_pointer_cast<QEngine>(std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg);

    const bool is0Zero = IS_NORM_0(b0->scale);
    const bool is1Zero = IS_NORM_0(b1->scale);

    if (is0Zero && is1Zero) {
        b0->SetZero();
        b1->SetZero();

        return;
    }

    if (is0Zero) {
        qReg0 = std::dynamic_pointer_cast<QEngine>(std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg)->CloneEmpty();
        std::dynamic_pointer_cast<QBdtQEngineNode>(b0)->qReg = qReg0;
    } else if (is1Zero) {
        qReg1 = std::dynamic_pointer_cast<QEngine>(std::dynamic_pointer_cast<QBdtQEngineNode>(b0)->qReg)->CloneEmpty();
        std::dynamic_pointer_cast<QBdtQEngineNode>(b1)->qReg = qReg1;
    }

    if (!is0Zero) {
        qReg0->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, std::arg(b0->scale));
    }
    if (!is1Zero) {
        qReg1->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, std::arg(b1->scale));
    }

    b0->scale = SQRT1_2_R1;
    b1->scale = SQRT1_2_R1;

    qReg0->ShuffleBuffers(qReg1);

    qReg0->Mtrx(mtrx, qReg0->GetQubitCount() - 1U);
    qReg1->Mtrx(mtrx, qReg1->GetQubitCount() - 1U);

    qReg0->ShuffleBuffers(qReg1);
}

QBdtNodeInterfacePtr QBdtQEngineNode::RemoveSeparableAtDepth(bitLenInt depth, bitLenInt size)
{
    if (!size || !depth || (norm(scale) <= FP_NORM_EPSILON)) {
        return NULL;
    }
    depth--;

    QBdtQEngineNodePtr toRet = std::dynamic_pointer_cast<QBdtQEngineNode>(ShallowClone());
    toRet->scale /= abs(toRet->scale);

    if (!qReg) {
        return toRet;
    }

    toRet->qReg = std::dynamic_pointer_cast<QEngine>(qReg)->CloneEmpty();
    std::dynamic_pointer_cast<QEngine>(toRet->qReg)->SetQubitCount(size);

    qReg->Decompose(depth, toRet->qReg);

    return toRet;
}

void QBdtQEngineNode::Prune(bitLenInt depth)
{
    if (!depth) {
        return;
    }

    if (norm(scale) <= FP_NORM_EPSILON) {
        SetZero();
        return;
    }

    if (!qReg) {
        return;
    }

    real1_f phaseArg = qReg->FirstNonzeroPhase();
    qReg->UpdateRunningNorm();
    qReg->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, -phaseArg);
    scale *= (complex)std::polar((real1_f)ONE_R1, (real1_f)phaseArg);
}
} // namespace Qrack
