//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

namespace Qrack {

class QFusion;
typedef std::shared_ptr<QFusion> QFusionPtr;

class QFusion : public QInterface {
protected:
    QInterfacePtr qReg;
    QInterfaceEngine engineType;
    std::shared_ptr<std::default_random_engine> rand_generator;

    std::vector<std::shared_ptr<complex[4]>> bitBuffers;

public:
    QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState = 0,
        std::shared_ptr<std::default_random_engine> rgp = nullptr);

    void ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex);

protected:
    inline void FlushBit(bitLenInt qubitIndex) { if (bitBuffers[qubitIndex]) qReg->ApplySingleBit(bitBuffers[qubitIndex].get(), true, qubitIndex); }
    inline void FlushReg(const bitLenInt& start, const bitLenInt& length);
};
}
