//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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

#include "common/qrack_functions.hpp"

#include <mutex>

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

namespace Qrack {

class QBdtNodeInterface;
typedef std::shared_ptr<QBdtNodeInterface> QBdtNodeInterfacePtr;

class QBdtNodeInterface : public std::enable_shared_from_this<QBdtNodeInterface> {
protected:
    static size_t SelectBit(bitCapInt perm, bitLenInt bit) { return (size_t)((perm >> bit) & 1U); }
    static void _par_for_qbdt(const bitCapInt end, BdtFunc fn);

public:
#if ENABLE_COMPLEX_X2
    virtual void PushStateVector(const complex2& mtrxCol1, const complex2& mtrxCol2, const complex2& mtrxColShuff1,
        const complex2& mtrxColShuff2, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1, bitLenInt depth,
        bitLenInt parDepth = 1U)
#else
    virtual void PushStateVector(complex const* mtrx, QBdtNodeInterfacePtr& b0, QBdtNodeInterfacePtr& b1,
        bitLenInt depth, bitLenInt parDepth = 1U)
#endif
    {
        throw std::out_of_range("QBdtNodeInterface::PushStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    complex scale;
    QBdtNodeInterfacePtr branches[2U];
    std::shared_ptr<std::mutex> mtx;

    QBdtNodeInterface()
        : scale(ONE_CMPLX)
    {
        branches[0U] = NULL;
        branches[1U] = NULL;
        mtx = std::make_shared<std::mutex>();
    }

    QBdtNodeInterface(complex scl)
        : scale(scl)
    {
        branches[0U] = NULL;
        branches[1U] = NULL;
        mtx = std::make_shared<std::mutex>();
    }

    QBdtNodeInterface(complex scl, QBdtNodeInterfacePtr* b)
        : scale(scl)
    {
        branches[0U] = b[0U];
        branches[1U] = b[1U];
        mtx = std::make_shared<std::mutex>();
    }

    virtual ~QBdtNodeInterface()
    {
        // Virtual destructor for inheritance
    }

    virtual bool IsStabilizer() { return false; }

    virtual void InsertAtDepth(QBdtNodeInterfacePtr b, bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::InsertAtDepth() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual QBdtNodeInterfacePtr RemoveSeparableAtDepth(
        bitLenInt depth, const bitLenInt& size, bitLenInt parDepth = 1U);

    virtual void SetZero()
    {
        scale = ZERO_CMPLX;

        if (!branches[0U]) {
            return;
        }

        if (true) {
            const QBdtNodeInterfacePtr b0 = branches[0U];
            std::lock_guard<std::mutex> lock(*(b0->mtx.get()));
            branches[0U] = NULL;
        }

        if (true) {
            const QBdtNodeInterfacePtr b1 = branches[1U];
            std::lock_guard<std::mutex> lock(*(b1->mtx.get()));
            branches[1U] = NULL;
        }
    }

    virtual bool isEqual(QBdtNodeInterfacePtr r);

    virtual bool isEqualUnder(QBdtNodeInterfacePtr r);

    virtual bool isEqualBranch(QBdtNodeInterfacePtr r, const bool& b);

    virtual QBdtNodeInterfacePtr ShallowClone()
    {
        throw std::out_of_range("QBdtNodeInterface::ShallowClone() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void PopStateVector(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::PopStateVector() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Branch(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::Branch() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual QBdtNodeInterfacePtr Prune(
        bitLenInt depth = 1U, bitLenInt parDepth = 1U, const bool& isCliffordBlocked = false)
    {
        throw std::out_of_range("QBdtNodeInterface::Prune() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual void Normalize(bitLenInt depth = 1U)
    {
        throw std::out_of_range("QBdtNodeInterface::Normalize() not implemented! (You probably set "
                                "QRACK_QBDT_SEPARABILITY_THRESHOLD too high.)");
    }

    virtual QBdtNodeInterfacePtr Apply2x2(const complex2& mtrxCol1, const complex2& mtrxCol2,
        const complex2& mtrxColShuff1, const complex2& mtrxColShuff2, bitLenInt depth)
    {
        throw std::out_of_range("QBdtQStabilizerNode::Apply2x2() not implemented!");
    }
    virtual QBdtNodeInterfacePtr Apply2x2(complex const* mtrx, bitLenInt depth)
    {
        throw std::out_of_range("QBdtQStabilizerNode::Apply2x2() not implemented!");
    }

    virtual QBdtNodeInterfacePtr PopSpecial(bitLenInt depth = 1U, bitLenInt parDepth = 1U)
    {
        throw std::out_of_range(
            "QBdtNodeInterface::PopSpecial() not implemented! (Check IsStabilizer() before PopSpecial().)");
    }
};

bool operator==(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs);
bool operator!=(QBdtNodeInterfacePtr lhs, QBdtNodeInterfacePtr rhs);
} // namespace Qrack
