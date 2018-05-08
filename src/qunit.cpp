//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Copyright 2017-2018, Daniel Strano and the Qrack and VM6502Q contributors.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <ctime>
#include <map>
#include <initializer_list>

#include "qunit.hpp"
#include "qfactory.hpp"

namespace Qrack {

QUnit::QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount), engine(eng)
{
    if (rgp == nullptr) {
        /* Used to control the random seed for all allocated interfaces. */
        rand_generator = std::make_shared<std::default_random_engine>();
        rand_generator->seed(std::time(0));
    } else {
        rand_generator = rgp;
    }

    shards.resize(qBitCount);
    
    for (bitLenInt i = 0; i < qBitCount; i++) {
        shards[i].unit = CreateQuantumInterface(engine, engine, 1, ((1 << i) & initState) >> i, rand_generator);
        shards[i].mapped = 0;
    }
}

void QUnit::CopyState(QUnitPtr orig)
{
    shards.clear();
    SetQubitCount(orig->GetQubitCount());

    /* Set up the shards to refer to the new unit. */
    std::map<QInterfacePtr, QInterfacePtr> otherUnits;
    for (auto otherShard :orig->shards) {
        QEngineShard shard;
        shard.mapped = otherShard.mapped;
        if (otherUnits.find(otherShard.unit) == otherUnits.end()) {
            otherUnits[otherShard.unit] = CreateQuantumInterface(engine, engine, otherShard.unit->GetQubitCount(), 0, rand_generator);
        }
        shard.unit = otherUnits[otherShard.unit];
        shards.push_back(shard);
    }

    for (auto otherUnit : otherUnits) {
        otherUnit.second->CopyState(otherUnit.first);
    }
}

void QUnit::CopyState(QInterfacePtr orig)
{
    QInterfacePtr unit = CreateQuantumInterface(engine, engine, orig->GetQubitCount(), 0, rand_generator);
    unit->CopyState(orig);

    shards.clear();
    SetQubitCount(orig->GetQubitCount());

    /* Set up the shards to refer to the new unit. */
    for (bitLenInt i = 0; i < (orig->GetQubitCount()); i++) {
        QEngineShard shard;
        shard.unit = unit;
        shard.mapped = i;
        shards.push_back(shard);
    }
}

void QUnit::SetQuantumState(complex* inputState)
{
    auto unit = CreateQuantumInterface(engine, engine, qubitCount, 0, rand_generator);
    unit->SetQuantumState(inputState);

    int idx = 0;
    for (auto &&shard : shards) {
        shard.unit = unit;
        shard.mapped = idx++;
    }
}

/*
 * Append QInterface to the end of the unit.
 */
bitLenInt QUnit::Cohere(QInterfacePtr toCopy)
{
    bitLenInt oldCount = qubitCount;

    /* Increase the number of bits in this object. */
    SetQubitCount(qubitCount + toCopy->GetQubitCount());

    /* Create a clone of the quantum state in toCopy. */
    QInterfacePtr clone = CreateQuantumInterface(engine, engine, 1, 0,
            rand_generator);
    clone->CopyState(toCopy);

    /* Destroy the quantum state in toCopy. */
    complex emptyState[] = {complex(0, 0), complex(0, 0)};
    toCopy->SetQuantumState(emptyState);

    /* Update shards to reference the cloned state. */
    for (bitLenInt i = 0; i < clone->GetQubitCount(); i++) {
        shards[i + oldCount].unit = clone;
        shards[i + oldCount].mapped = i;
    }

    return oldCount;
}

std::map<QInterfacePtr, bitLenInt> QUnit::Cohere(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto &&q : toCopy) {
        ret[q] = Cohere(q);
    }

    return ret;
}

void QUnit::Detach(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    /* TODO: This method should compose the bits for the destination without cohering the length first */

    if (length > 1) {
        EntangleRange(start, length);
        OrderContiguous(shards[start].unit);
    }

    QInterfacePtr unit = shards[start].unit;
    bitLenInt mapped = shards[start].mapped;

    if (dest && unit->GetQubitCount() > length) {
        unit->Decohere(mapped, length, dest);
    } else if (dest) {
        dest->CopyState(unit);
    } else {
        unit->Dispose(mapped, length);
    }

    shards.erase(shards.begin() + start, shards.begin() + start + length);
    SetQubitCount(qubitCount - length);

    if (unit->GetQubitCount() == length) {
        return;
    }

    /* Find the rest of the qubits. */
    for (auto shard : shards) {
        if (shard.unit == unit && shard.mapped > (mapped + length)) {
            shard.mapped -= length;
        }
    }
}

void QUnit::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    Detach(start, length, dest);
}

void QUnit::Dispose(bitLenInt start, bitLenInt length)
{
    Detach(start, length, nullptr);
}

template <class It>
QInterfacePtr QUnit::EntangleIterator(It first, It last)
{
    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to cohere. */
    for (auto bit = first + 1; bit != last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            found[shards[**bit].unit] = true;
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    if (units.size() != 0) {
        auto &&offsets = unit1->Cohere(units);

        /* Since each unit will be collapsed in-order, one set of bits at a time. */
        for (auto &&shard : shards) {
            auto search = offsets.find(shard.unit);
            if (search != offsets.end()) {
                shard.mapped += search->second;
                shard.unit = unit1;
            }
        }
    }

    /* Change the source parameters to the correct newly mapped bit indexes. */
    for (auto bit = first; bit != last; ++bit) {
        **bit = shards[**bit].mapped;
    }

    return unit1;
}

QInterfacePtr QUnit::Entangle(std::initializer_list<bitLenInt *> bits)
{
    return EntangleIterator(bits.begin(), bits.end());
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length)
{
    if (length == 1) {
        return shards[start].unit;
    }

    std::vector<bitLenInt> bits(length);
    std::vector<bitLenInt *> ebits(length);
    for (auto i = 0; i < length; i++) {
        bits[i] = i + start;
        ebits[i] = &bits[i];
    }

    return EntangleIterator(ebits.begin(), ebits.end());
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start1, bitLenInt length1, bitLenInt start2, bitLenInt length2)
{
    std::vector<bitLenInt> bits(length1 + length2);
    std::vector<bitLenInt *> ebits(length1 + length2);

    for (auto i = 0; i < length1; i++) {
        bits[i] = i + start1;
        ebits[i] = &bits[i];
    }

    for (auto i = 0; i < length2; i++) {
        bits[i + length1] = i + start2;
        ebits[i + length1] = &bits[i + length1];
    }

    return EntangleIterator(ebits.begin(), ebits.end());
}

/*
 * Accept a variable number of bits, entangle them all into a single QInterface
 * object, and then call the supplied function on that object.
 */
template <typename F, typename ... B>
void QUnit::EntangleAndCallMember(F fn, B ... bits)
{
    auto qbits = Entangle({&bits...});
    ((*qbits).*fn)(bits...);
}

template <typename F, typename ... B>
void QUnit::EntangleAndCall(F fn, B ... bits)
{
    auto qbits = Entangle({&bits...});
    fn(qbits, bits...);
}

void QUnit::OrderContiguous(QInterfacePtr unit)
{
    if (unit->GetQubitCount() == 1) {
        return;
    }

    /* Create a sortable collection of all of the bits that are in the unit. */
    std::vector<QSortEntry> bits(unit->GetQubitCount());

    int j = 0;
    for (int i = 0; i < qubitCount; i++) {
        if (shards[i].unit == unit) {
            bits[j].mapped = shards[i].mapped;
            bits[j].bit = i;
            j++;
        }
    }

    SortUnit(unit, bits, 0, bits.size() - 1);
}

/* Sort a container of bits, calling Swap() on each. */
void QUnit::SortUnit(QInterfacePtr unit, std::vector<QSortEntry> &bits, bitLenInt low, bitLenInt high)
{
    bitLenInt i = low, j = high;
    QSortEntry pivot = bits[(low + high) / 2];

    while (i <= j) {
        while (bits[i] < pivot) {
            i++;
        }
        while (bits[j] > pivot) {
            j--;
        }
        if (i <= j) {
            /* Note: Using the length variant to avoid a likely-temporary bug in the single-bit variant. */
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped);     /* Change the global mapping. */
            std::swap(bits[i].mapped, bits[j].mapped);                /* Change the contents of the sorting array. */
            i++;
            j--;
        }
    }
    if (low < j) {
        SortUnit(unit, bits, low, j);
    }
    if (i < high) {
        SortUnit(unit, bits, i, high);
    }
}

void QUnit::DumpShards()
{
    int i = 0;
    for (auto shard : shards) {
        printf("%2d.\t%p[%d]\n", i++, shard.unit.get(), shard.mapped);
    }
}

double QUnit::Prob(bitLenInt qubit)
{
    QEngineShard &shard = shards[qubit];
    return (shard.unit->Prob)(shard.mapped);
}

double QUnit::ProbAll(bitCapInt perm)
{
    double result = 1.0;

    std::map<QInterfacePtr, bitCapInt> perms;

    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (perm & (1 << i)) {
            perms[shards[i].unit] |= 1 << shards[i].mapped;
        }
    }

    for (auto &&qi : perms) {
        result *= qi.first->ProbAll(qi.second);
    }

    return result;
}

/// Measure a bit
bool QUnit::M(bitLenInt qubit)
{
    bool result = shards[qubit].unit->M(shards[qubit].mapped);

    QInterfacePtr unit = shards[qubit].unit;
    bitLenInt mapped = shards[qubit].mapped;

    if (unit->GetQubitCount() == 1) {
        /* If we're keeping the bits, and they're already in their own unit, there's nothing to do. */
        return result;
    }

    QInterfacePtr dest = CreateQuantumInterface(engine, engine, 1, 0, rand_generator);
    unit->Decohere(mapped, 1, dest);

    /* Update the mappings. */
    shards[qubit].unit = dest;
    shards[qubit].mapped = 0;
    for (auto &&shard : shards) {
        if (shard.unit == unit && shard.mapped > mapped) {
            shard.mapped--;
        }
    }

    return result;
}

/// Measure permutation state of a register
bitCapInt QUnit::MReg(bitLenInt start, bitLenInt length)
{
    bitCapInt result = 0;

    for (bitLenInt bit = 0; bit < length; bit++) {
        if (M(bit + start)) {
            result |= 1 << bit;
        }
    }

    return result;
}

void QUnit::SetBit(bitLenInt qubit, bool value)
{
    if (M(qubit) != value) {
        shards[qubit].unit->X(shards[qubit].mapped);
    }
}

/// Set register bits to given permutation
void QUnit::SetReg(bitLenInt start, bitLenInt length, bitCapInt value)
{
    MReg(start, length);

    for (bitLenInt i = 0; i < length; i++) {
        shards[i].unit->SetPermutation((value & (1 << i)) > 0 ? 1 : 0);
    }
}

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    QEngineShard &shard1 = shards[qubit1];
    QEngineShard &shard2 = shards[qubit2];

    QEngineShard tmp;

    // Swap the bit mapping.
    tmp.mapped = shard1.mapped;
    shard1.mapped = shard2.mapped;
    shard2.mapped = tmp.mapped;

    // Swap the QInterface object.
    tmp.unit = shard1.unit;
    shard1.unit = shard2.unit;
    shard2.unit = tmp.unit;
}

/* Unfortunately, many methods are overloaded, which prevents using just the address-to-member. */
#define PTR3(OP) (void (QInterface::*)(bitLenInt, bitLenInt, bitLenInt)) &QInterface::OP
#define PTR2(OP) (void (QInterface::*)(bitLenInt, bitLenInt)) &QInterface::OP

void QUnit::AND(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCallMember(PTR3(AND), inputBit1, inputBit2, outputBit);
}

void QUnit::OR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCallMember(PTR3(OR), inputBit1, inputBit2, outputBit);
}

void QUnit::XOR(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCallMember(PTR3(XOR), inputBit1, inputBit2, outputBit);
}

void QUnit::CLAND(bitLenInt inputBit, bool inputClassicalBit, bitLenInt outputBit)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CLAND(b1, inputClassicalBit, b2);
        }, inputBit, outputBit);
}

void QUnit::CLOR(bitLenInt inputBit, bool inputClassicalBit, bitLenInt outputBit)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CLOR(b1, inputClassicalBit, b2);
        }, inputBit, outputBit);
}

void QUnit::CLXOR(bitLenInt inputBit, bool inputClassicalBit, bitLenInt outputBit)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CLXOR(b1, inputClassicalBit, b2);
        }, inputBit, outputBit);
}

void QUnit::CCNOT(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCallMember(PTR3(CCNOT), inputBit1, inputBit2, outputBit);
}

void QUnit::AntiCCNOT(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt outputBit)
{
    EntangleAndCallMember(PTR3(AntiCCNOT), inputBit1, inputBit2, outputBit);
}

void QUnit::CNOT(bitLenInt control, bitLenInt target)
{
    EntangleAndCallMember(PTR2(CNOT), control, target);
}

void QUnit::AntiCNOT(bitLenInt control, bitLenInt target)
{
    EntangleAndCallMember(PTR2(AntiCNOT), control, target);
}

void QUnit::H(bitLenInt qubit)
{
    shards[qubit].unit->H(shards[qubit].mapped);
}

void QUnit::X(bitLenInt qubit)
{
    shards[qubit].unit->X(shards[qubit].mapped);
}

void QUnit::Y(bitLenInt qubit)
{
    shards[qubit].unit->Y(shards[qubit].mapped);
}

void QUnit::Z(bitLenInt qubit)
{
    shards[qubit].unit->Z(shards[qubit].mapped);
}

void QUnit::CY(bitLenInt control, bitLenInt target)
{
    EntangleAndCallMember(PTR2(CY), control, target);
}

void QUnit::CZ(bitLenInt control, bitLenInt target)
{
    EntangleAndCallMember(PTR2(CZ), control, target);
}

void QUnit::RT(double radians, bitLenInt qubit)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt q) {
            unit->RT(radians, q);
        }, qubit);
}

void QUnit::RX(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RX(radians, shards[qubit].mapped);
}

void QUnit::RY(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RY(radians, shards[qubit].mapped);
}

void QUnit::RZ(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RZ(radians, shards[qubit].mapped);
}

void QUnit::CRT(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRT(radians, b1, b2);
        }, control, target);
}

void QUnit::CRX(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRX(radians, b1, b2);
        }, control, target);
}

void QUnit::CRY(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRY(radians, b1, b2);
        }, control, target);
}

void QUnit::CRZ(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRZ(radians, b1, b2);
        }, control, target);
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QUnit::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        Reverse(start, end);
        Reverse(start, start + shift);
        Reverse(start + shift, end);
    }
}

/// "Circular shift right" - (Uses swap-based algorithm for speed)
void QUnit::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    shift %= length;
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        Reverse(start + shift, end);
        Reverse(start, start + shift);
        Reverse(start, end);
    }
}

void QUnit::INC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->INC(toMod, shards[start].mapped, length);
}

void QUnit::INCx(INCxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    /*
     * FUTURE: If start[length] and carry are already in the same QE, then it
     * doesn't make sense to Decompose and re-entangle them.
     */
    M(flagIndex);

    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);

    /* Make sure the flag bit is entangled in the same QU. */
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            ((*unit).*fn)(toMod, b1, length, b2);
        }, start, flagIndex);
}

void QUnit::INCxx(INCxxFn fn, bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt flag1Index, bitLenInt flag2Index)
{
    /*
     * FUTURE: If start[length] and carry are already in the same QE, then it
     * doesn't make sense to Decompose and re-entangle them.
     */

    /*
     * Overflow flag should not be measured, however the carry flag still needs
     * to be measured.
     */
    M(flag2Index);

    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);

    /* Make sure the flag bit is entangled in the same QU. */
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2, bitLenInt b3) {
            ((*unit).*fn)(toMod, b1, length, b2, b3);
        }, start, flag1Index, flag2Index);
}

void QUnit::INCC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCC, toMod, start, length, carryIndex);
}

void QUnit::INCS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INCx(&QInterface::INCS, toMod, start, length, overflowIndex);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INCxx(&QInterface::INCSC, toMod, start, length, overflowIndex, carryIndex);
}

void QUnit::INCSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCSC, toMod, start, length, carryIndex);
}

void QUnit::INCBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->INCBCD(toMod, shards[start].mapped, length);
}

void QUnit::INCBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::INCBCDC, toMod, start, length, carryIndex);
}

void QUnit::DEC(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->DEC(toMod, shards[start].mapped, length);
}

void QUnit::DECC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECC, toMod, start, length, carryIndex);
}

void QUnit::DECS(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    INCx(&QInterface::DECS, toMod, start, length, overflowIndex);
}

void QUnit::DECSC(
    bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    INCxx(&QInterface::DECSC, toMod, start, length, overflowIndex, carryIndex);
}

void QUnit::DECSC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECSC, toMod, start, length, carryIndex);
}

void QUnit::DECBCD(bitCapInt toMod, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->DECBCD(toMod, shards[start].mapped, length);
}

void QUnit::DECBCDC(bitCapInt toMod, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    INCx(&QInterface::DECBCDC, toMod, start, length, carryIndex);
}

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->ZeroPhaseFlip(shards[start].mapped, length);
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    EntangleRange(start, length);
    EntangleRange(start, 1, flagIndex, 1);
    OrderContiguous(shards[start].unit);
    shards[start].unit->CPhaseFlipIfLess(greaterPerm, shards[start].mapped, length, shards[flagIndex].mapped);
}

void QUnit::PhaseFlip()
{
    for (auto &&shard : shards) {
        shard.unit->PhaseFlip();
    }
}

bitCapInt QUnit::IndexedLDA(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength);
    OrderContiguous(shards[indexStart].unit);

    return shards[indexStart].unit->IndexedLDA(shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, values);
}

bitCapInt QUnit::IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength);
    EntangleRange(indexStart, 1, carryIndex, 1);
    OrderContiguous(shards[indexStart].unit);

    return shards[indexStart].unit->IndexedADC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);
}

bitCapInt QUnit::IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
{
    EntangleRange(indexStart, indexLength, valueStart, valueLength);
    EntangleRange(indexStart, 1, carryIndex, 1);
    OrderContiguous(shards[indexStart].unit);

    return shards[indexStart].unit->IndexedSBC(shards[indexStart].mapped, indexLength, shards[valueStart].mapped, valueLength, shards[carryIndex].mapped, values);
}

} // namespace Qrack
