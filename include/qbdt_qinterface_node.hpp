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

#pragma once

#include "qbdt_node_interface.hpp"
#include "qengine.hpp"

namespace Qrack {

class QBdtQInterfaceNode;
typedef std::shared_ptr<QBdtQInterfaceNode> QBdtQInterfaceNodePtr;

class QBdtQInterfaceNode : public QBdtNodeInterface {
public:
    QEnginePtr qReg;

    QBdtQInterfaceNode()
        : QBdtNodeInterface(ZERO_CMPLX)
        , qReg(NULL)
    {
        // Intentionally left blank.
    }

    QBdtQInterfaceNode(complex scl, QEnginePtr q)
        : QBdtNodeInterface(scl)
        , qReg(q)
    {
        if (norm(scale) <= FP_NORM_EPSILON) {
            SetZero();
        }
    }

    virtual void SetZero()
    {
        QBdtNodeInterface::SetZero();
        qReg = NULL;
    }

    virtual QBdtNodeInterfacePtr ShallowClone() { return std::make_shared<QBdtQInterfaceNode>(scale, qReg); }

    virtual bool isEqual(QBdtNodeInterfacePtr r)
    {
        return (this == r.get()) ||
            ((norm(scale - r->scale) <= FP_NORM_EPSILON) &&
                ((norm(scale) <= FP_NORM_EPSILON) ||
                    qReg->ApproxCompare(std::dynamic_pointer_cast<QBdtQInterfaceNode>(r)->qReg)));
    }

    virtual void Normalize(bitLenInt depth)
    {
        if (qReg) {
            qReg->NormalizeState();
        }
    }

    virtual void ConvertStateVector(bitLenInt depth)
    {
        if (!qReg) {
            return;
        }

        // TODO: This isn't valid for stabilizer
        qReg->UpdateRunningNorm();
        real1_f nrm = qReg->GetRunningNorm();

        if (nrm <= FP_NORM_EPSILON) {
            SetZero();
            return;
        }

        real1_f phaseArg = qReg->FirstNonzeroPhase();
        qReg->NormalizeState(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG, -phaseArg);
        scale *= std::polar(nrm, phaseArg);
    }
};

} // namespace Qrack
