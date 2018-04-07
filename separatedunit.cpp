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
#include <iostream>

#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {

bool compare(QbListEntry i, QbListEntry j)
{
    bool lessThan;
    if (i.cu == j.cu) {
        lessThan = (i.start < j.start);
    } else {
        lessThan = (i.cu < j.cu);
    }
    return lessThan;
}

/// Initialize a coherent unit with qBitCount number of bits, to initState unsigned integer permutation state
SeparatedUnit::SeparatedUnit(bitLenInt qBitCount, bitCapInt initState)
{
    rand_generator_ptr[0] = std::default_random_engine();
    randomSeed = std::time(0);
    SetRandomSeed(randomSeed);
    qubitCount = qBitCount;

    bool setBit;
    bitLenInt i;
    std::unique_ptr<QbLookup[]> ql(new QbLookup[qBitCount]);
    std::unique_ptr<bitLenInt[]> qil(new bitLenInt[qBitCount * qBitCount]());
    qubitLookup = std::move(ql);
    qubitInverseLookup = std::move(qil);
    for (i = 0; i < qBitCount; i++) {
        setBit = (initState & (1 << i)) > 0;
        qubitLookup[i].cu = i;
        qubitLookup[i].qb = 0;
        qubitInverseLookup[i * qBitCount] = i;
        coherentUnits.push_back(
            std::shared_ptr<CoherentUnit>(new CoherentUnit(1, (setBit ? 1 : 0), rand_generator_ptr)));
    }
}

/// Initialize a coherent unit with qBitCount number of bits, all to |0> state.
SeparatedUnit::SeparatedUnit(bitLenInt qBitCount)
    : SeparatedUnit(qBitCount, 0)
{
}

/// PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
double SeparatedUnit::Prob(bitLenInt qubitIndex)
{
    QbLookup qbl = qubitLookup[qubitIndex];
    return coherentUnits[qbl.cu]->Prob(qbl.qb);
}

/// Measure a bit
bool SeparatedUnit::M(bitLenInt qubitIndex)
{
    bool result;
    QbLookup qbl = qubitLookup[qubitIndex];
    result = coherentUnits[qbl.cu]->M(qbl.qb);

    if (coherentUnits[qbl.cu]->GetQubitCount() > 1) {
        std::shared_ptr<CoherentUnit> ncu(new CoherentUnit(1, 0, rand_generator_ptr));
        coherentUnits[qbl.cu]->Decohere(qbl.qb, 1, *ncu);

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
    bitLenInt i, j;
    QbListEntry qbe;
    QbLookup qbl;

    std::vector<QbListEntry> qbList(length);
    GetOrderedBitList(start, length, &qbList);

    j = 0;
    for (i = 0; i < qbList.size(); i++) {
        qbe = qbList[i];
        result |= (coherentUnits[qbe.cu]->MReg(qbe.start, qbe.length)) << j;
        j += qbe.length;
    }

    for (i = 0; i < length; i++) {
        qbl = qubitLookup[start + i];
        if (coherentUnits[qbl.cu]->GetQubitCount() > 1) {
            std::shared_ptr<CoherentUnit> ncu(new CoherentUnit(1, 0, rand_generator_ptr));
            coherentUnits[qbl.cu]->Decohere(qbl.qb, 1, *ncu);

            qbl.cu = coherentUnits.size();
            qbl.qb = 0;
            coherentUnits.push_back(ncu);
        }
    }

    return result;
}

/// Set individual bit to pure |0> (false) or |1> (true) state
/**
 * To set a bit, the bit is first measured. If the result of measurement matches "value," the bit is considered set.
 * If the result of measurement is the opposite of "value," an X gate is applied to the bit. The state ends up
 * entirely in the "value" state, with a random phase factor.
 */
void SeparatedUnit::SetBit(bitLenInt qubitIndex, bool value)
{
    QbLookup qbl = qubitLookup[qubitIndex];
    coherentUnits[qbl.cu]->SetBit(qbl.qb, value);
}

/// Set entire SeparatedUnit to given permutation
void SeparatedUnit::SetPermutation(bitCapInt value)
{
    SetReg(0, qubitCount, value);
}


/// Set register bits to given permutation
void SeparatedUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    bitLenInt i;

    MReg(start, length);

    for (i = 0; i < length; i++) {
        coherentUnits[qubitLookup[start + i].cu]->SetPermutation(((value & (1 << i)) > 0) ? 1 : 0);
    }
}

/**
 * Set 8 bit register bits by a superposed index-offset-based read from
 * classical memory
 *
 * "inputStart" is the start index of 8 qubits that act as an index into
 * the 256 byte "values" array. The "outputStart" bits are first cleared,
 * then the separable |input, 00000000> permutation state is mapped to
 * |input, values[input]>, with "values[input]" placed in the "outputStart"
 * register.
 *
 * While a CoherentUnit represents an interacting set of qubit-based
 * registers, or a virtual quantum chip, the registers need to interact in
 * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
 * method similar to the X addressing mode of the MOS 6502 chip, if the X
 * register can be in a state of coherent superposition when it loads from
 * RAM.
 *
 * The physical motivation for this addressing mode can be explained as
 * follows: say that we have a superconducting quantum interface device
 * (SQUID) based chip. SQUIDs have already been demonstrated passing
 * coherently superposed electrical currents. In a sufficiently
 * quantum-mechanically isolated qubit chip with a classical cache, with
 * both classical RAM and registers likely cryogenically isolated from the
 * environment, SQUIDs could (hopefully) pass coherently superposed
 * electrical currents into the classical RAM cache to load values into a
 * qubit register. The state loaded would be a superposition of the values
 * of all RAM to which coherently superposed electrical currents were
 * passed.
 *
 * In qubit system similar to the MOS 6502, say we have qubit-based
 * "accumulator" and "X index" registers, and say that we start with a
 * superposed X index register. In (classical) X addressing mode, the X
 * index register value acts an offset into RAM from a specified starting
 * address. The X addressing mode of a LoaD Accumulator (LDA) instruction,
 * by the physical mechanism described above, should load the accumulator
 * in quantum parallel with the values of every different address of RAM
 * pointed to in superposition by the X index register. The superposed
 * values in the accumulator are entangled with those in the X index
 * register, by way of whatever values the classical RAM pointed to by X
 * held at the time of the load. (If the RAM at index "36" held an unsigned
 * char value of "27," then the value "36" in the X index register becomes
 * entangled with the value "27" in the accumulator, and so on in quantum
 * parallel for all superposed values of the X index register, at once.) If
 * the X index register or accumulator are then measured, the two registers
 * will both always collapse into a random but valid key-value pair of X
 * index offset and value at that classical RAM address.
 *
 * Note that a "superposed store operation in classical RAM" is not
 * possible by analagous reasoning. Classical RAM would become entangled
 * with both the accumulator and the X register. When the state of the
 * registers was collapsed, we would find that only one "store" operation
 * to a single memory address had actually been carried out, consistent
 * with the address offset in the collapsed X register and the byte value
 * in the collapsed accumulator. It would not be possible by this model to
 * write in quantum parallel to more than one address of classical memory
 * at a time.
 */

unsigned char SeparatedUnit::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    std::vector<QbListEntry> qbList(8);
    GetParallelBitList(inputStart, 8, &qbList);
    std::vector<QbListEntry> qbListOutput(8);
    GetParallelBitList(outputStart, 8, &qbListOutput);
    qbList.insert(qbList.end(), qbListOutput.begin(), qbListOutput.end());
    OptimizeParallelBitList(&qbList);

    EntangleBitList(qbList);

    return coherentUnits[qubitLookup[inputStart].cu]->SuperposeReg8(qubitLookup[inputStart].qb, qubitLookup[outputStart].qb, values);
}

/**
 * Add to entangled 8 bit register state with a superposed
 * index-offset-based read from classical memory
 *
 * inputStart" is the start index of 8 qubits that act as an index into the
 * 256 byte "values" array. The "outputStart" bits would usually already be
 * entangled with the "inputStart" bits via a SuperposeReg8() operation.
 * With the "inputStart" bits being a "key" and the "outputStart" bits
 * being a value, the permutation state |key, value> is mapped to |key,
 * value + values[key]>. This is similar to classical parallel addition of
 * two arrays.  However, when either of the registers are measured, both
 * registers will collapse into one random VALID key-value pair, with any
 * addition or subtraction done to the "value." See SuperposeReg8() for
 * context.
 *
 * While a CoherentUnit represents an interacting set of qubit-based
 * registers, or a virtual quantum chip, the registers need to interact in
 * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
 * method similar to the X addressing mode of the MOS 6502 chip, if the X
 * register can be in a state of coherent superposition when it loads from
 * RAM. "AdcSuperposReg8" and "SbcSuperposeReg8" perform add and subtract
 * (with carry) operations on a state usually initially prepared with
 * SuperposeReg8().
 */
unsigned char SeparatedUnit::AdcSuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    QbListEntry carryQbe;
    std::vector<QbListEntry> qbList(8);
    GetParallelBitList(inputStart, 8, &qbList);
    std::vector<QbListEntry> qbListOutput(8);
    GetParallelBitList(outputStart, 8, &qbListOutput);
    qbList.insert(qbList.end(), qbListOutput.begin(), qbListOutput.end());
    carryQbe.cu = qubitLookup[carryIndex].cu;
    carryQbe.start = qubitLookup[carryIndex].qb;
    carryQbe.length = 1;
    qbList.push_back(carryQbe);
    OptimizeParallelBitList(&qbList);

    EntangleBitList(qbList);

    return coherentUnits[qubitLookup[inputStart].cu]->AdcSuperposeReg8(qubitLookup[inputStart].qb, qubitLookup[outputStart].qb, qubitLookup[carryIndex].qb, values);
}

/**
 * Subtract from an entangled 8 bit register state with a superposed
 * index-offset-based read from classical memory
 *
 * "inputStart" is the start index of 8 qubits that act as an index into
 * the 256 byte "values" array. The "outputStart" bits would usually
 * already be entangled with the "inputStart" bits via a SuperposeReg8()
 * operation.  With the "inputStart" bits being a "key" and the
 * "outputStart" bits being a value, the permutation state |key, value> is
 * mapped to |key, value - values[key]>. This is similar to classical
 * parallel addition of two arrays.  However, when either of the registers
 * are measured, both registers will collapse into one random VALID
 * key-value pair, with any addition or subtraction done to the "value."
 * See CoherentUnit::SuperposeReg8 for context.
 *
 * While a CoherentUnit represents an interacting set of qubit-based
 * registers, or a virtual quantum chip, the registers need to interact in
 * some way with (classical or quantum) RAM. SuperposeReg8 is a RAM access
 * method similar to the X addressing mode of the MOS 6502 chip, if the X
 * register can be in a state of coherent superposition when it loads from
 * RAM. "AdcSuperposReg8" and "SbcSuperposeReg8" perform add and subtract
 * (with carry) operations on a state usually initially prepared with
 * SuperposeReg8().
 */
unsigned char SeparatedUnit::SbcSuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    QbListEntry carryQbe;
    std::vector<QbListEntry> qbList(8);
    GetParallelBitList(inputStart, 8, &qbList);
    std::vector<QbListEntry> qbListOutput(8);
    GetParallelBitList(outputStart, 8, &qbListOutput);
    qbList.insert(qbList.end(), qbListOutput.begin(), qbListOutput.end());
    carryQbe.cu = qubitLookup[carryIndex].cu;
    carryQbe.start = qubitLookup[carryIndex].qb;
    carryQbe.length = 1;
    qbList.push_back(carryQbe);
    OptimizeParallelBitList(&qbList);

    EntangleBitList(qbList);

    return coherentUnits[qubitLookup[inputStart].cu]->SbcSuperposeReg8(qubitLookup[inputStart].qb, qubitLookup[outputStart].qb, qubitLookup[carryIndex].qb, values);
}

void SeparatedUnit::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit) {
    QbListEntry qbe;
    std::vector<QbListEntry> qbList(3);
    qbe.cu = qubitLookup[inputBit1].cu;
    qbe.start = qubitLookup[inputBit1].qb;
    qbe.length = 1;
    qbList[0] = qbe;
    qbe.cu = qubitLookup[inputBit2].cu;
    qbe.start = qubitLookup[inputBit2].qb;
    qbe.length = 1;
    qbList[1] = qbe;
    qbe.cu = qubitLookup[outputBit].cu;
    qbe.start = qubitLookup[outputBit].qb;
    qbe.length = 1;
    qbList[2] = qbe;
    OptimizeParallelBitList(&qbList);

    EntangleBitList(qbList);

    coherentUnits[qubitLookup[inputBit1].cu]->AND(qubitLookup[inputBit1].qb, qubitLookup[inputBit2].qb, qubitLookup[outputBit].qb);
}

void SeparatedUnit::AND(bitLenInt inputStart1, bitLenInt inputStart2, bitLenInt outputStart, bitLenInt length) {
    bitLenInt i;
    for (i = 0; i < length; i++) {
        AND(inputStart1 + i, inputStart2 + i, outputStart + i);
    }
}

/**
 * Compile an order-preserving list of CoherentUnit bit strings for applying an register-wise operation
 *
 * This operation optimizes compiling a list out of qubit pile when bit order is important. We apply register-wise
 * operations over a pile of arbitrarily entangled and separated qubits. Entangled qubits are stored together in single
 * CoherentUnit objects, but their mapping to SeparatedUnit bit indices can be generally random. Sometimes, we must
 * preserve bit order to correctly carry out the operation, whereas sometimes our operation is bitwise parallel and does
 * not depend on the ordering of bits in the list.
 */
void SeparatedUnit::GetOrderedBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry>* qbList)
{
    // Start by getting a list (of sublists) of all the bits we need, with bit sublist length of 1.
    bitLenInt i, j;
    QbLookup qbl;
    QbListEntry qbe;
    for (i = 0; i < length; i++) {
        qbl = qubitLookup[start + i];
        qbe.cu = qbl.cu;
        qbe.start = qbl.qb;
        qbe.length = 1;
        (*qbList)[i] = qbe;
    }

    // If contiguous sublists in the list we just made are also contiguous in the same coherent unit, we can combine
    // them to optimize with register-wise gate methods.
    j = 0;
    for (i = 0; i < length; i++) {
        if (((*qbList)[j].cu == (*qbList)[j + 1].cu) &&
            (((*qbList)[j].start + (*qbList)[j].length) == (*qbList)[j + 1].start)) {
            (*qbList)[j].length++;
            qbList->erase(qbList->begin() + j + 1);
        } else {
            j++;
        }
    }
}

/// Compile a list of CoherentUnit bit strings for applying a bitwise-parallel operation
/**
 * This operation optimizes compiling a list out of qubit pile when bit order is not important. We apply register-wise
 * operations over a pile of arbitrarily entangled and separated qubits. Entangled qubits are stored together in single
 * CoherentUnit objects, but their mapping to SeparatedUnit bit indices can be generally random. Sometimes, we must
 * preserve bit order to correctly carry out the operation, whereas sometimes our operation is bitwise parallel and does
 * not depend on the ordering of bits in the list.
 */
void SeparatedUnit::GetParallelBitList(bitLenInt start, bitLenInt length, std::vector<QbListEntry>* qbList)
{
    // Start by getting a list (of sublists) of all the bits we need, with bit sublist length of 1.
    bitLenInt i, j;
    QbLookup qbl;
    QbListEntry qbe;
    for (i = 0; i < length; i++) {
        qbl = qubitLookup[start + i];
        qbe.cu = qbl.cu;
        qbe.start = qbl.qb;
        qbe.length = 1;
        (*qbList)[i] = qbe;
    }
    // The ordering of bits returned is unimportant, so we can better optimize by sorting this list by CoherentUnit
    // index and qubit index, to maximize the reduction of the list.
    std::sort(qbList->begin(), qbList->end(), compare);
    // If contiguous sublists in the list we just sorted are also contiguous in the same coherent unit, we can combine
    // them to optimize with register-wise gate methods.
    j = 0;
    for (i = 0; i < length; i++) {
        if (((*qbList)[j].cu == (*qbList)[j + 1].cu) &&
            (((*qbList)[j].start + (*qbList)[j].length) == (*qbList)[j + 1].start)) {
            (*qbList)[j].length++;
            qbList->erase(qbList->begin() + j + 1);
        } else {
            j++;
        }
    }
}

/// Combines two lists returned by GetParallelBitList() by the same logic as that algorithm
void SeparatedUnit::OptimizeParallelBitList(std::vector<QbListEntry>* qbList)
{
    bitLenInt i, j;
    bitLenInt length = qbList->size();
    // The ordering of bits returned is unimportant, so we can better optimize by sorting this list by CoherentUnit
    // index and qubit index, to maximize the reduction of the list.
    std::sort(qbList->begin(), qbList->end(), compare);
    // If contiguous sublists in the list we just sorted are also contiguous in the same coherent unit, we can combine
    // them to optimize with register-wise gate methods.
    j = 0;
    for (i = 0; i < length; i++) {
        if (((*qbList)[j].cu == (*qbList)[j + 1].cu) &&
            (((*qbList)[j].start + (*qbList)[j].length) == (*qbList)[j + 1].start)) {
            (*qbList)[j].length++;
            qbList->erase(qbList->begin() + j + 1);
        } else {
            j++;
        }
    }
}

/// Entangle and sort the indices of a list of CoherentUnit objects
void SeparatedUnit::EntangleBitList(std::vector<QbListEntry> qbList) {
    if (qbList.size() < 2) {
        return;
    }

    bitLenInt i, j, k;
    bitLenInt firstCu, cuLen, invLookup, cuRemoved;
    QbListEntry qbe;

    firstCu = qbList[0].cu;
    k = coherentUnits[firstCu]->GetQubitCount();
    for (i = 1; i < qbList.size(); i++) {
        qbe = qbList[i];
        cuLen = coherentUnits[qbe.cu]->GetQubitCount();
        for (j = 0; j < cuLen; j++) {
            invLookup = qubitInverseLookup[qbe.cu * qubitCount + j];
            qubitLookup[invLookup].cu = firstCu;
            qubitLookup[invLookup].qb = k + j;
            qubitInverseLookup[firstCu * qubitCount + k] = invLookup;
            qubitInverseLookup[qbe.cu * qubitCount + j] = 0;
        }
        coherentUnits[firstCu]->Cohere(*(coherentUnits[qbe.cu]));
        k += cuLen;
    }

    // Swap qubits into appropriate order, then update coherentUnits list.
    cuLen = coherentUnits[firstCu]->GetQubitCount();
    QuickSortQubits(&(qubitInverseLookup[firstCu * qubitCount]), 0, cuLen - 1,
        coherentUnits[firstCu]);
    // Update lookup table
    for (i = 0; i < cuLen; i++) {
        invLookup = qubitInverseLookup[firstCu * qubitCount + i];
        qubitLookup[invLookup].cu = firstCu;
        qubitLookup[invLookup].qb = i;
    }

    // Update coherentUnit list and inverse lookup at end
    cuLen = qbList.size() - 1;
    if (cuLen > 0) {
        std::vector<bitLenInt> cuToDelete(cuLen);
        for (i = 0; i < cuLen; i++) {
            cuToDelete[i] = qbList[i + 1].cu;
        }
        std::sort(cuToDelete.begin(), cuToDelete.end());
        for (i = 0; i < cuLen; i++) {
            cuRemoved = cuToDelete[cuLen - i - 1];
            coherentUnits.erase(coherentUnits.begin() + cuRemoved);
            for (j = 0; j < qubitCount; j++) {
                if (qubitLookup[j].cu >= cuRemoved) {
                    qubitLookup[j].cu--;
                }
            }
            for (j = cuRemoved; j < (coherentUnits.size() - 1); j++) {
                for (k = 0; k < qubitCount; k++) {
                    qubitInverseLookup[j * qubitCount + k] = qubitInverseLookup[(j + 1) * qubitCount + k];
                }
            } 
        }
    }
}

// This function takes last element as pivot, places the pivot element at its correct position in sorted array, and
// places all smaller (smaller than pivot) to left of pivot and all greater elements to right of pivot.
bitLenInt SeparatedUnit::PartitionQubits(
    bitLenInt* arr, bitLenInt low, bitLenInt high, std::weak_ptr<CoherentUnit> cuWeak)
{
    std::shared_ptr<CoherentUnit> cu = cuWeak.lock();
    // pivot
    bitLenInt pivot = arr[high];
    // Index of smaller element
    bitLenInt i = (low - 1);

    for (bitLenInt j = low; j <= high - 1; j++) {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot) {
            // increment index of smaller element
            i++;
            std::swap(arr[i], arr[j]);
            cu->Swap(i, j);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    cu->Swap(i + 1, high);
    return (i + 1);
}

// The main function that implements QuickSort
// arr[] --> Array to be sorted,
// low  --> Starting index,
// high  --> Ending index
void SeparatedUnit::QuickSortQubits(bitLenInt* arr, bitLenInt low, bitLenInt high, std::weak_ptr<CoherentUnit> cuWeak)
{
    if (low < high) {
        // pi is partitioning index, arr[p] is not at right place
        bitLenInt pi = PartitionQubits(arr, low, high, cuWeak);

        // Separately sort elements before
        // partition and after partition
        QuickSortQubits(arr, low, pi - 1, cuWeak);
        QuickSortQubits(arr, pi + 1, high, cuWeak);
    }
}

} // namespace Qrack
