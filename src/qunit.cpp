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

QUnit::QUnit(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, uint32_t rand_seed) : QInterface(qBitCount), engine(eng)
{
    if (rand_seed == 0) {
        rand_seed = std::time(0);
    }

    /* Used to control the random seed for all allocated interfaces. */
    rand_generator = std::make_shared<std::default_random_engine>();
    rand_generator->seed(rand_seed);

    shards.resize(qBitCount);

    for (auto shard : shards) {
        shard.unit = CreateQuantumInterface(engine, 1, 0, rand_generator);
        shard.mapped = 0;
    }
}

void QUnit::Decompose(bitLenInt qubit)
{
    std::shared_ptr<QInterface> unit = shards[qubit].unit;
    for (auto shard : shards) {
        if (shard.unit == unit) {
            shard.unit = CreateQuantumInterface(engine, 1, 0, rand_generator);
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

    QInterfacePtr unit1 = shards[**first].unit;
    std::map<QInterfacePtr, bool> found;

    found[unit1] = true;

    /* Walk through all of the supplied bits and create a unique list to cohere. */
    for (auto bit = first; bit != last; ++bit) {
        if (found.find(shards[**bit].unit) == found.end()) {
            units.push_back(shards[**bit].unit);
        }
    }

    /* Collapse all of the other units into unit1, returning a map to the new bit offset. */
    auto offsets = unit1->Cohere(units);

    /* Since each unit will be collapsed in-order, one set of bits at a time. */
    for (auto shard : shards) {
        auto search = offsets.find(shard.unit);
        if (search != offsets.end()) {
            shard.mapped += search->second;
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

    for (auto qi : perms) {
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
    bitLenInt end = start + length;
    bitCapInt result = 0;

    for (bitLenInt bit = start; bit < end; bit++) {
        result |= M(bit) << bit;
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
    for (bitLenInt bit = start; bit < length; bit++) {
        shards[bit].unit->SetBit(shards[bit].mapped, value & (1 << bit));
        Decompose(bit);
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

#define PTR3(X) (void (QInterface::*)(bitLenInt, bitLenInt, bitLenInt)) &QInterface::X
#define PTR2(X) (void (QInterface::*)(bitLenInt, bitLenInt)) &QInterface::X

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

void QUnit::X(bitLenInt start, bitLenInt length)
{
    for (bitLenInt i = 0; i < length; i++) {
        X(start + i);
    }
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

/// "Circular shift right" - shift bits right, and carry first bits.
void QUnit::ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    /* SetReg and Reverse both do mapping under the hood. */
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Reverse(start, end);
            Reverse(start, start + shift);
            Reverse(start + shift, end);
        }
    }
}

/// "Circular shift right" - shift bits right, and carry first bits.
void QUnit::ROR(bitLenInt shift, bitLenInt start, bitLenInt length)
{
    if ((length > 0) && (shift > 0)) {
        bitLenInt end = start + length;
        if (shift >= length) {
            SetReg(start, length, 0);
        } else {
            Reverse(start + shift, end);
            Reverse(start, start + shift);
            Reverse(start, end);
        }
    }
}

void QUnit::INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);

    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::INCBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DEC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    EntangleRange(start, length);
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECS(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECSC(
    bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt overflowIndex, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECSC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECBCD(bitCapInt toAdd, bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::DECBCDC(bitCapInt toAdd, bitLenInt start, bitLenInt length, bitLenInt carryIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::QFT(bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::ZeroPhaseFlip(bitLenInt start, bitLenInt length)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
{
    /* XXX TODO Map arbitrary list. */
}

void QUnit::PhaseFlip()
{
    for (auto shard : shards) {
        shard.unit->PhaseFlip();
    }
}

unsigned char QUnit::SuperposeReg8(bitLenInt inputStart, bitLenInt outputStart, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
    return 0;
}

unsigned char QUnit::AdcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
    return 0;
}

unsigned char QUnit::SbcSuperposeReg8(
    bitLenInt inputStart, bitLenInt outputStart, bitLenInt carryIndex, unsigned char* values)
{
    /* XXX TODO Map arbitrary list. */
    return 0;
}

} // namespace Qrack
