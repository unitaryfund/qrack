//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
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

    for (auto &&shard : shards) {
        shard.unit = CreateQuantumInterface(engine, engine, 1, 0, rand_generator);
        shard.mapped = 0;
    }
}

Complex16* QUnit::GetState() {
    EntangleRange(0, qubitCount);
    OrderContiguous(shards[0].unit);
    return shards[0].unit->GetState();
}

void QUnit::CopyState(QInterfacePtr orig) {
    EntangleRange(0, qubitCount);
    OrderContiguous(shards[0].unit);
    return shards[0].unit->CopyState(orig);
}

void QUnit::SetQuantumState(Complex16* inputState)
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
    bitLenInt ret = qubitCount;

    shards.resize(qubitCount + toCopy->GetQubitCount());
    for (bitLenInt i = 0; i < toCopy->GetQubitCount(); i++) {
        // TODO: shards[i + qubitCount].unit = CreateQuantumInterface(engine, engine, toCopy);
        shards[i + qubitCount].unit = toCopy;
        shards[i + qubitCount].mapped = i;
    }

    qubitCount = qubitCount + toCopy->GetQubitCount();
    maxQPower = 1 << qubitCount;

    return ret;
}

std::map<QInterfacePtr, bitLenInt> QUnit::Cohere(std::vector<QInterfacePtr> toCopy)
{
    std::map<QInterfacePtr, bitLenInt> ret;

    for (auto &&q : toCopy) {
        ret[q] = Cohere(q);
    }

    return ret;
}

/*
 * Normal QInterface::Decohere would remove the bits entirely and reduce the
 * qBitCount, but this resets them to 0.
 */
void QUnit::Decohere(bitLenInt start, bitLenInt length, QInterfacePtr dest)
{
    /* TODO: This method should compose the bits for the destination without cohering the length first */

    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    QInterfacePtr unit = shards[start].unit;
    bitLenInt mapped = shards[start].mapped;
    bitLenInt i = 0;
    if (unit->GetQubitCount() > length) {
        unit->Decohere(mapped, length, dest);
        while (i < shards.size()) {
            if (shards[i].unit == unit && shards[i].mapped >= (mapped + length)) {
                shards[i].mapped -= length;
                i++;
            }
            else if (shards[i].unit == unit && shards[i].mapped >= mapped) {
                shards.erase(shards.begin() + i);
            }
            else {
                i++;
            }
        }
    }
    else {
        dest->CopyState(unit);
        while (i < shards.size()) {
            if (shards[i].unit == unit) {
                shards.erase(shards.begin() + i);
            }
            else {
                i++;
            }
        }
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;
}

void QUnit::Dispose(bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    QInterfacePtr unit = shards[start].unit;
    bitLenInt mapped = shards[start].mapped;
    bitLenInt i = 0;
    if (unit->GetQubitCount() > length) {
        unit->Dispose(shards[start].mapped, length);
        while (i < shards.size()) {
            if (shards[i].unit == unit && shards[i].mapped >= (mapped + length)) {
                shards[i].mapped -= length;
                i++;
            }
            else if (shards[i].unit == unit && shards[i].mapped >= mapped) {
                shards.erase(shards.begin() + i);
            }
            else {
                i++;
            }
        }
    }
    else {
        while (i < shards.size()) {
            if (shards[i].unit == unit) {
                shards.erase(shards.begin() + i);
            }
            else {
                i++;
            }
        }
    }

    qubitCount = qubitCount - length;
    maxQPower = 1 << qubitCount;
}

void QUnit::Decompose(bitLenInt qubit)
{
    std::shared_ptr<QInterface> unit = shards[qubit].unit;
    for (auto &&shard : shards) {
        if (shard.unit == unit) {
            shard.unit = CreateQuantumInterface(engine, engine, 1, 0, rand_generator);
            shard.unit->SetBit(0, unit->M(shard.mapped));
            shard.mapped = 0;
        }
    }
}

QInterfacePtr QUnit::Entangle(std::initializer_list<bitLenInt *> bits)
{
    return EntangleIterator(bits.begin(), bits.end());
}

template <class It>
QInterfacePtr QUnit::EntangleIterator(It first, It last)
{
    std::vector<QInterfacePtr> units;
    units.reserve((int)(last - first));

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    bool areAllSameUnit = true;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to cohere. */
    for (auto bit = first + 1; bit != last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            units.push_back(shards[**bit].unit);
        }
        if (shards[**bit].unit != unit1) {
            areAllSameUnit = false;
        }
    }

    /* If the bits are already entangled, our work is done. */
    if (areAllSameUnit) return unit1;

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    auto &&offsets = unit1->Cohere(units);

    /* Since each unit will be collapsed in-order, one set of bits at a time. */
    for (auto &&shard : shards) {
        auto search = offsets.find(shard.unit);
        if (search != offsets.end()) {
            shard.mapped = search->second;
            shard.unit = unit1;
        }
    }

    /* Change the source parameters to the correct newly mapped bit indexes. */
    for (auto bit = first; bit != last; ++bit) {
        **bit = shards[**bit].mapped;
    }

    return unit1;
}

QInterfacePtr QUnit::EntangleRange(bitLenInt start, bitLenInt length)
{
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
            unit->Swap(bits[i].mapped, bits[j].mapped); /* Change the location in the QE itself. */
            std::swap(shards[bits[i].bit].mapped, shards[bits[j].bit].mapped);     /* Change the global mapping. */
            std::swap(bits[i], bits[j]);                /* Change the contents of the sorting array. */
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

void QUnit::ProbArray(double* probArray)
{
    for (size_t bit = 0; bit < shards.size(); bit++) {
        probArray[bit] = Prob(bit);
    }
}

/// Measure a bit
bool QUnit::M(bitLenInt qubit)
{
    bool result = shards[qubit].unit->M(shards[qubit].mapped);

    /*
     * Decomposes all of the bits in the shard, performing M() on each one and
     * setting each new CU to the appropriate value.
     */
    Decompose(qubit);

    return result;
}

/// Measure permutation state of a register
bitCapInt QUnit::MReg(bitLenInt start, bitLenInt length)
{
    bitCapInt result = 0;

    for (bitLenInt bit = 0; bit < length; bit++) {
        result |= M(bit + start) << (bit + start);
    }

    return result;
}

void QUnit::SetBit(bitLenInt qubit, bool value)
{
    shards[qubit].unit->SetBit(shards[qubit].mapped, value);
    Decompose(qubit);
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

void QUnit::Swap(bitLenInt qubit1, bitLenInt qubit2, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        Swap(qubit1 + i, qubit2 + i);
    }
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
    EntangleAndCallMember(PTR2(CNOT), control, target);
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

void QUnit::RTDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit].unit->RTDyad(numerator, denominator, shards[qubit].mapped);
}

void QUnit::RX(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RX(radians, shards[qubit].mapped);
}

void QUnit::RXDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit].unit->RXDyad(numerator, denominator, shards[qubit].mapped);
}

void QUnit::RY(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RY(radians, shards[qubit].mapped);
}

void QUnit::RYDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit].unit->RYDyad(numerator, denominator, shards[qubit].mapped);
}

void QUnit::RZ(double radians, bitLenInt qubit)
{
    shards[qubit].unit->RZ(radians, shards[qubit].mapped);
}

void QUnit::RZDyad(int numerator, int denominator, bitLenInt qubit)
{
    shards[qubit].unit->RZDyad(numerator, denominator, shards[qubit].mapped);
}

void QUnit::CRT(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRT(radians, b1, b2);
        }, control, target);
}

void QUnit::CRTDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRTDyad(numerator, denominator, b1, b2);
        }, control, target);
}

void QUnit::CRX(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRX(radians, b1, b2);
        }, control, target);
}

void QUnit::CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRXDyad(numerator, denominator, b1, b2);
        }, control, target);
}

void QUnit::CRY(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRY(radians, b1, b2);
        }, control, target);
}

void QUnit::CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRYDyad(numerator, denominator, b1, b2);
        }, control, target);
}

void QUnit::CRZ(double radians, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRZ(radians, b1, b2);
        }, control, target);
}

void QUnit::CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target)
{
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CRZDyad(numerator, denominator, b1, b2);
        }, control, target);
}

void QUnit::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->ROL(shift, shards[start].mapped, length);
}

void QUnit::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    OrderContiguous(shards[start].unit);
    shards[start].unit->ROR(shift, shards[start].mapped, length);
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

    // Overflow flag should not be measured:
    // M(flag1Index);

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
    OrderContiguous(shards[start].unit);

    /* Make sure the flag bit is entangled in the same QU. */
    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2) {
            unit->CPhaseFlipIfLess(greaterPerm, b1, length, b2);
        }, start, flagIndex);
}

void QUnit::PhaseFlip()
{
    for (auto &&shard : shards) {
        shard.unit->PhaseFlip();
    }
}

unsigned char QUnit::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    const bitLenInt length = 8;

    // TODO: This logic is overridden to demonstrate correct output from the lookup table search unit test. //
    EntangleRange(outputStart, 2 * length);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    OrderContiguous(shards[inputStart].unit);

    return shards[inputStart].unit->SuperposeReg8(shards[inputStart].mapped, shards[outputStart].mapped, values);
}

unsigned char QUnit::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    const bitLenInt length = 8;
    EntangleRange(inputStart, length, outputStart, length);
    OrderContiguous(shards[inputStart].unit);
    unsigned char result = 0;

    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2, bitLenInt b3) {
            result = unit->AdcSuperposeReg8(b1, b2, b3, values);
        }, inputStart, outputStart, carryIndex);

    return result;
}

unsigned char QUnit::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    const bitLenInt length = 8;
    EntangleRange(inputStart, length, outputStart, length);
    OrderContiguous(shards[inputStart].unit);
    unsigned char result = 0;

    EntangleAndCall([&](QInterfacePtr unit, bitLenInt b1, bitLenInt b2, bitLenInt b3) {
            result = unit->SbcSuperposeReg8(b1, b2, b3, values);
        }, inputStart, outputStart, carryIndex);

    return result;
}

} // namespace Qrack
