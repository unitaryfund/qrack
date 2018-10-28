//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"
#include <algorithm>

namespace Qrack {

class QEngine;
typedef std::shared_ptr<QEngine> QEnginePtr;

/**
 * Abstract QEngine implementation, for all "Schroedinger method" engines
 */
class QEngine : public QInterface {

public:
    QEngine(bitLenInt n, std::shared_ptr<std::default_random_engine> rgp = nullptr, bool doNorm = true)
        : QInterface(n, rgp, doNorm){};

    /** Destructor of QInterface */
    virtual ~QEngine(){};

    virtual bool ForceM(bitLenInt qubitIndex, bool result, bool doForce = true, real1 nrmlzr = 1.0);
    using QInterface::M;
    virtual bool M(bitLenInt qubit);
    virtual bitCapInt M(const bitLenInt* bits, const bitLenInt& length);
    // virtual bitCapInt MReg(bitLenInt start, bitLenInt length);

    virtual void ApplyM(bitCapInt regMask, bool result, complex nrm);
    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm) = 0;
};
} // namespace Qrack
