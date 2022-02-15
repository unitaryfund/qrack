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

#include "common/qrack_types.hpp"

namespace Qrack {

class QBdtNodeInterface;
typedef std::shared_ptr<QBdtNodeInterface> QBdtNodeInterfacePtr;

class QBdtNodeInterface {
protected:
    size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }
    void par_for_qbdt(const bitCapIntOcl begin, const bitCapIntOcl end, IncrementFunc fn);

public:
    complex scale;

    QBdtNodeInterface()
        : scale(ONE_CMPLX)
    {
        // Intentionally left blank
    }

    QBdtNodeInterface(complex scl)
        : scale(scl)
    {
        // Intentionally left blank
    }

    virtual QBdtNodeInterfacePtr ShallowClone() = 0;

    virtual void SetZero() { scale = ZERO_CMPLX; }

    virtual void Branch(bitLenInt depth = 1U, bool isZeroBranch = false) = 0;

    virtual void Prune(bitLenInt depth = 1U) = 0;

    virtual void Normalize(bitLenInt depth) = 0;

    virtual void ConvertStateVector(bitLenInt depth) = 0;
};

} // namespace Qrack
