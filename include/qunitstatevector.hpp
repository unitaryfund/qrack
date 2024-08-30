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

struct IdOffset {
    bitLenInt id;
    bitLenInt offset;

    IdOffset()
    {
        // Intentionally left blank
    }

    IdOffset(bitLenInt i, bitLenInt o)
        : id(i)
        , offset(o)
    {
        // Intentionally left blank
    }
};

struct QUnitStateVector {
    complex phaseOffset;
    std::map<bitLenInt, IdOffset> idMap;
    std::vector<std::map<bitCapInt, complex>> amps;

    QUnitStateVector(
        complex p, const std::map<bitLenInt, IdOffset>& i, const std::vector<std::map<bitCapInt, complex>>& a)
        : phaseOffset(p)
        , idMap(i)
        , amps(a)
    {
        // Intentionally left blank
    }

    complex get(const bitCapInt& p)
    {
        if (p >= pow2(idMap.size())) {
            throw std::invalid_argument("QUnitStateVector::get() argument out-of-bounds!");
        }

        std::map<bitLenInt, bitCapInt> perms;
        for (auto qid = idMap.begin(); qid != idMap.end(); ++qid) {
            const size_t i = std::distance(idMap.begin(), qid);
            if (bi_and_1(p >> i)) {
                bi_or_ip(&(perms[qid->second.offset]), pow2(qid->second.id));
            }
        }

        complex result = phaseOffset;
        for (size_t i = 0U; i < amps.size(); ++i) {
            const auto& found = amps[i].find(perms[i]);
            if (found == amps[i].end()) {
                result = ZERO_CMPLX;
                break;
            }
            result *= found->second;
            if ((2 * norm(result)) <= REAL1_EPSILON) {
                result = ZERO_CMPLX;
                break;
            }
        }

        return result;
    }
};

} // namespace Qrack
