//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

namespace Qrack {

struct QUnitStateVector;
typedef std::shared_ptr<QUnitStateVector> QUnitStateVectorPtr;

struct QUnitStateVector {
    complex phaseOffset;
    std::map<bitLenInt, bitLenInt> idMap;
    std::map<bitLenInt, bitLenInt> offsetMap;
    std::vector<std::map<bitCapInt, complex>> amps;

    QUnitStateVector()
        : phaseOffset(ONE_CMPLX)
    {
        // Intentionally left blank
    }

    QUnitStateVector(const complex& p, const std::map<bitLenInt, bitLenInt>& i, const std::vector<std::map<bitCapInt, complex>>& a)
        : phaseOffset(p)
        , idMap(i)
        , amps(a)
    {
        bitLenInt totQubits = 0U;
        for (size_t i = 0U; i < amps.size(); ++i) {
             const bitLenInt lastQubits = log2(amps[i].size());
             for (size_t j = 0U; j < l; ++j) {
                  offsetMap[totQubits + j] = totQubits;
             }
             totQubits += lastQubits;
        }
    }

    complex operator[](size_t p) const
    {
        complex amp = ONE_CMPLX;
        
    }
};

} // namespace Qrack
