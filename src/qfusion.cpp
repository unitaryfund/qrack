//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qfactory.hpp"
#include "qfusion.hpp"

namespace Qrack {

QFusion::QFusion(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
    , engineType(eng)
{
    if (rgp == nullptr) {
        /* Used to control the random seed for all allocated interfaces. */
        rand_generator = std::make_shared<std::default_random_engine>();
        rand_generator->seed(std::time(0));
    } else {
        rand_generator = rgp;
    }

    qReg = CreateQuantumInterface(engineType, qBitCount, initState, rand_generator);

    bitBuffers.resize(qBitCount);
}

void QFusion::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubitIndex) {
     if (qubitCount < 3U) {
         qReg->ApplySingleBit(mtrx, true, qubitIndex);
         return;
     }

     std::shared_ptr<complex[4]> outBuffer;
     if (bitBuffers[qubitIndex]) {
         std::shared_ptr<complex[4]> inBuffer = bitBuffers[qubitIndex];

         outBuffer[0] = (mtrx[0] * inBuffer[0]) + (mtrx[1] * inBuffer[2]);
         outBuffer[1] = (mtrx[0] * inBuffer[1]) + (mtrx[1] * inBuffer[3]);
         outBuffer[2] = (mtrx[2] * inBuffer[0]) + (mtrx[3] * inBuffer[2]);
         outBuffer[3] = (mtrx[2] * inBuffer[1]) + (mtrx[3] * inBuffer[3]);

         bitBuffers[qubitIndex] = outBuffer;
     } else {
         std::copy(mtrx, mtrx + 4, outBuffer.get());
         bitBuffers[qubitIndex] = outBuffer;
     }
}

void QFusion::FlushReg(const bitLenInt& start, const bitLenInt& length) {
    for (bitLenInt i = 0U; i < length; i++) {
        FlushBit(start + i);
    }
}
}
