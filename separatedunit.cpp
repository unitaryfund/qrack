//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2018. All rights reserved.
//
// This is an abstraction on "CoherentUnit" per https://arxiv.org/abs/1710.05867
// 
// "SeparatedUnit" keeps representation of qubit states separated until explicitly
// entangled. This makes for large gains in memory and speed optimization in the
// best case scenario. "CoherentUnit" has been optimized for the worst case scenario.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include "separatedunit.hpp"

#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {

SeparatedUnit::SeparatedUnit(bitLenInt qBitCount) {
    bitLenInt i;
    std::unique_ptr<QbLookup[]> ql(new QbLookup[qBitCount]);
    qubitLookup.reset();
    qubitLookup = std::move(ql);
    for (i = 0; i < qBitCount; i++) {
        qubitLookup[i].cu = i;
        qubitLookup[i].qb = 0;
        coherentUnits.push_back(CoherentUnit(1));
    }
}

/// Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state
SeparatedUnit::SeparatedUnit(bitLenInt qBitCount, bitCapInt initState) {
    bitLenInt i;
    std::unique_ptr<QbLookup[]> ql(new QbLookup[qBitCount]);
    qubitLookup.reset();
    qubitLookup = std::move(ql);
    for (i = 0; i < qBitCount; i++) {
        qubitLookup[i].cu = i;
        qubitLookup[i].qb = 0;
        coherentUnits.push_back(CoherentUnit(1));
    }
}

bool SeparatedUnit::M(bitLenInt qubitIndex) {
    bool result;
    QbLookup qbl = qubitLookup[qubitIndex];
    CoherentUnit cu = coherentUnits[qbl.cu];
    result = cu.M(qbl.qb);

    CoherentUnit ncu = CoherentUnit(1);
    cu.Decohere(qbl.qb, 1, ncu);
   
    qbl.cu = coherentUnits.size();
    qbl.qb = 0;
    coherentUnits.push_back(ncu);

    return result;
}

}
