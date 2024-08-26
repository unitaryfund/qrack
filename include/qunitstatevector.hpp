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
    bitCapInt maxQPower;
    complex phaseOffset;
    std::map<bitLenInt, bitLenInt> idMap;
    std::map<bitLenInt, bitLenInt> offsetMap;
    std::vector<std::map<bitCapInt, complex>> amps;

    QUnitStateVector()
        : phaseOffset(ONE_CMPLX)
    {
        // Intentionally left blank
    }

    QUnitStateVector(const bitCapInt& m, const complex& p, const std::map<bitLenInt, bitLenInt>& i, const std::vector<std::map<bitCapInt, complex>>& a)
        : maxQPower(m)
        , phaseOffset(p)
        , idMap(i)
        , amps(a)
    {
        bitLenInt totQubits = 0U;
        for (const auto& a : amps) {
             const bitLenInt lastQubits = log2(a.size());
             for (size_t j = 0U; j < lastQubits; ++j) {
                  offsetMap[totQubits + j] = totQubits;
                  idMap[totQubits + j] += totQubits;
             }
             totQubits += lastQubits;
        }
    }

    complex operator[](size_t p)
    {
        if (p >= maxQPower) {
            throw std::invalid_argument("QUnit::GetAmplitudeOrProb argument out-of-bounds!");
        }

        std::map<bitLenInt, bitCapInt> perms;
        for (auto qid = idMap.begin(); qid != idMap.end(); ++qid) {
            const size_t i = std::distance(idMap.begin(), qid);
            const bitLenInt& m = offsetMap[i];
            if (bi_and_1(p >> i)) {
                bi_or_ip(&(perms[m]), pow2(qid->first - m));
            }
        }

        complex result(ONE_R1, ZERO_R1);
        for (auto qi = perms.begin(); qi != perms.end(); ++qi) {
            const size_t i = std::distance(perms.begin(), qi);
            result *= amps[i][qi->second];
            if ((2 * norm(result)) <= FP_NORM_EPSILON) {
                break;
            }
        }

        return result;
    }
};

} // namespace Qrack
