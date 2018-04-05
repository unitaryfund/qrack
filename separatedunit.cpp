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

bool compare(QbListEntry i, QbListEntry j) {
    bool lessThan;
    if (i.cu == j.cu) {
        lessThan = (i.start < j.start);
    }
    else {
        lessThan = (i.cu < j.cu);
    }
    return lessThan;
}

/// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
SeparatedUnit::SeparatedUnit(bitLenInt qBitCount)
{
    qubitCount = qBitCount;

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
SeparatedUnit::SeparatedUnit(bitLenInt qBitCount, bitCapInt initState)
{
    qubitCount = qBitCount;

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

/// Get a count of qubits in the SeparatedUnit
bitLenInt SeparatedUnit::GetQubitCount() {
    return qubitCount;
}

/// Measure a bit
bool SeparatedUnit::M(bitLenInt qubitIndex)
{
    bool result;
    QbLookup qbl = qubitLookup[qubitIndex];
    CoherentUnit cu = coherentUnits[qbl.cu];
    result = cu.M(qbl.qb);

    if (cu.GetQubitCount() > 1) {
        CoherentUnit ncu = CoherentUnit(1);
        cu.Decohere(qbl.qb, 1, ncu);

        qbl.cu = coherentUnits.size();
        qbl.qb = 0;
        coherentUnits.push_back(ncu);
    }

    return result;
}

/// Measure permutation state of a register
bitCapInt SeparatedUnit::MReg(bitLenInt start, bitLenInt length)
{
    bitCapInt result = 0;
    bitLenInt i, j, k;
    QbListEntry qbe;
    QbLookup qbl;

    std::vector<QbListEntry> qbList;
    GetOrderedBitList(start, length, qbList);

    j = 0;
    for (i = 0; i < qbList.size(); i++) {
        qbe = qbList[i];
        qbl = qubitLookup[j];
        CoherentUnit cu = coherentUnits[qbe.cu];
        result |= (cu.MReg(qbe.start, qbe.length)) << j;
        j += qbe.length;
        for (k = 0; k < qbe.length; k++) {
            if (cu.GetQubitCount() > 1) {
                CoherentUnit ncu = CoherentUnit(1);
                cu.Decohere(qbe.start + k, 1, ncu);

                qbl.cu = coherentUnits.size();
                qbl.qb = 0;
                coherentUnits.push_back(ncu);
            }
        }
    }

    return result;
}

void SeparatedUnit::GetOrderedBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry> qbList) {
    bitLenInt i, j;
    QbLookup qbl;
    QbListEntry qbe;
    for (i = 0; i < length; i++) {
        qbl = qubitLookup[start + i];
        qbe.cu = qbl.cu;
        qbe.start = qbl.qb;
        qbe.length = 1;
        qbList.push_back(qbe);
    }

    j = 0;
    for (i = 0; i < length; i++) {
        if ((qbList[j].cu == qbList[j + 1].cu) && ((qbList[j].start + qbList[j].length) == qbList[j + 1].start)) {
            qbList[j].length++;
            qbList.erase(qbList.begin() + j + 1);
        }
        else {
            j++;
        }
    }
}

void SeparatedUnit::GetParallelBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry> qbList) {
    bitLenInt i, j;
    QbLookup qbl;
    QbListEntry qbe;
    for (i = 0; i < length; i++) {
        qbl = qubitLookup[start + i];
        qbe.cu = qbl.cu;
        qbe.start = qbl.qb;
        qbe.length = 1;
        qbList.push_back(qbe);
    }
    std::sort(qbList.begin(), qbList.end(), compare);
    j = 0;
    for (i = 0; i < length; i++) {
        if ((qbList[j].cu == qbList[j + 1].cu) && ((qbList[j].start + qbList[j].length) == qbList[j + 1].start)) {
            qbList[j].length++;
            qbList.erase(qbList.begin() + j + 1);
        }
        else {
            j++;
        }
    }
}

} // namespace Qrack
