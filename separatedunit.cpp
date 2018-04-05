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

/// Compile an order-preserving list of CoherentUnit bit strings for applying an register-wise operation
/**
  * This operation optimizes compiling a list out of qubit pile when bit order is important. We apply register-wise operations over a pile of arbitrarily entangled and separated qubits. Entangled qubits are stored together in single CoherentUnit objects, but their mapping to SeparatedUnit bit indices can be generally random. Sometimes, we must preserve bit order to correctly carry out the operation, whereas sometimes our operation is bitwise parallel and does not depend on the ordering of bits in the list.
  */
void SeparatedUnit::GetOrderedBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry> qbList) {
    //Start by getting a list (of sublists) of all the bits we need, with bit sublist length of 1.
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

    //If contiguous sublists in the list we just made are also contiguous in the same coherent unit, we can combine them to optimize with register-wise gate methods.
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

/// Compile a list of CoherentUnit bit strings for applying a bitwise-parallel operation
/**
  * This operation optimizes compiling a list out of qubit pile when bit order is not important. We apply register-wise operations over a pile of arbitrarily entangled and separated qubits. Entangled qubits are stored together in single CoherentUnit objects, but their mapping to SeparatedUnit bit indices can be generally random. Sometimes, we must preserve bit order to correctly carry out the operation, whereas sometimes our operation is bitwise parallel and does not depend on the ordering of bits in the list.
  */
void SeparatedUnit::GetParallelBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry> qbList) {
    //Start by getting a list (of sublists) of all the bits we need, with bit sublist length of 1.
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
    //The ordering of bits returned is unimportant, so we can better optimize by sorting this list by CoherentUnit index and qubit index, to maximize the reduction of the list.
    std::sort(qbList.begin(), qbList.end(), compare);
    //If contiguous sublists in the list we just sorted are also contiguous in the same coherent unit, we can combine them to optimize with register-wise gate methods.
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
