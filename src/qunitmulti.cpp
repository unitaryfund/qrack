//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// QUnitMulti is a multiprocessor variant of QUnit.
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qunitmulti.hpp"

namespace Qrack {

QUnitMulti::QUnitMulti(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac, bool doNorm,
    bool randomGlobalPhase, bool useHostMem, int ignored, bool useHardwareRNG)
    : QUnit(QINTERFACE_OPENCL, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, -1,
          useHardwareRNG)
{
    // Notice that this constructor does not take an engine type parameter, and it always passes QINTERFACE_OPENCL to
    // the QUnit constructor. For QUnitMulti, the "shard" engines are therefore guaranteed to always be QEngineOCL
    // types, and it's safe to assume that they can be cast from QInterfacePtr types to QEngineOCLPtr types in this
    // class.
    deviceCount = OCLEngine::Instance()->GetDeviceCount();
    defaultDeviceID = OCLEngine::Instance()->GetDefaultDeviceID();
    RedistributeQEngines();
}

void QUnitMulti::RedistributeQEngines()
{
    // TODO: Remove:
    WaitAllBits();

    // Get shard sizes and devices
    std::vector<QInterfacePtr> qips;
    std::vector<QEngineInfo> qinfos;
    bitCapInt totSize = 0;
    bitCapInt sz;
    QEngineOCLPtr qOCL;

    for (auto&& shard : shards) {
        if (std::find(qips.begin(), qips.end(), shard.unit) == qips.end()) {
            sz = 1U << ((shard.unit)->GetQubitCount());
            totSize += sz;
            qips.push_back(shard.unit);
            qOCL = std::static_pointer_cast<QEngineOCL>(shard.unit);
            qinfos.push_back(QEngineInfo(sz, qOCL->GetDeviceID(), qOCL));
        }
    }

    std::vector<bitCapInt> devSizes(deviceCount);
    std::fill(devSizes.begin(), devSizes.end(), 0U);
    bitLenInt devID;
    bitLenInt i, j;

    // We distribute in descending size order:
    std::sort(qinfos.rbegin(), qinfos.rend());

    WaitAllBits();

    for (i = 0; i < qinfos.size(); i++) {
        devID = i;
        // If a given device has 0 load, or if the engine adds negligible load, we can let any given unit keep its
        // residency on this device.
        // if (qinfos[i].size <= 2U) {
        //    break;
        //}
        if (devSizes[qinfos[i].deviceID] != 0U) {
            // If two devices have identical load, we prefer the default OpenCL device.
            sz = devSizes[defaultDeviceID];
            devID = defaultDeviceID;

            // Find the device with the lowest load.
            for (j = 0; j < deviceCount; j++) {
                if (devSizes[j] < sz) {
                    sz = devSizes[j];
                    devID = j;
                }
            }

            // Add this unit to the device with the lowest load.
            WaitUnit(qinfos[i].unit);
            qinfos[i].unit->SetDevice(devID);
        }
        // Update the size of buffers handles by this device.
        devSizes[devID] += qinfos[i].size;
    }
}

void QUnitMulti::WaitUnit(const QInterfacePtr& unit)
{
    for (bitLenInt i = 0; i < qubitCount; i++) {
        if ((shards[i].unit == unit) && shards[i].future.valid()) {
            shards[i].future.get();
        }
    }
}

void QUnitMulti::WaitAllBits()
{
    for (bitLenInt i = 0; i < qubitCount; i++) {
        if (shards[i].future.valid()) {
            shards[i].future.get();
        }
    }
}

void QUnitMulti::Detach(bitLenInt start, bitLenInt length, QUnitMultiPtr dest)
{
    WaitAllBits();
    QUnit::Detach(start, length, dest);
    RedistributeQEngines();
}

QInterfacePtr QUnitMulti::EntangleIterator(
    std::vector<bitLenInt*>::iterator first, std::vector<bitLenInt*>::iterator last)
{
    WaitAllBits();
    QInterfacePtr toRet = QUnit::EntangleIterator(first, last);
    RedistributeQEngines();
    return toRet;
}

QInterfacePtr QUnitMulti::EntangleRange(bitLenInt start, bitLenInt length)
{
    if (length == 1) {
        WaitBit(start);
        return shards[start].unit;
    }

    return QUnit::EntangleRange(start, length);
}

QInterfacePtr QUnitMulti::EntangleAll()
{
    WaitAllBits();
    return QUnit::EntangleAll();
}

void QUnitMulti::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    complex mtrxLocal[4];
    std::copy(mtrx, mtrx + 4, mtrxLocal);
    QEngineShard shard = shards[qubit];

    shards[qubit].isProbDirty = true;

    WaitBit(qubit);
    shards[qubit].future = std::async(std::launch::async,
        [shard, mtrxLocal, doCalcNorm]() { shard.unit->ApplySingleBit(mtrxLocal, doCalcNorm, shard.mapped); });
}

void QUnitMulti::SetPermutation(bitCapInt perm, complex phaseFac)
{
    WaitAllBits();
    QUnit::SetPermutation(perm, phaseFac);
}

void QUnitMulti::CopyState(QUnitMultiPtr orig) { CopyState(orig.get()); }

// protected method
void QUnitMulti::CopyState(QUnitMulti* orig)
{
    WaitAllBits();
    orig->WaitAllBits();

    QUnit::CopyState((QUnit*)orig);
}

void QUnitMulti::CopyState(QInterfacePtr orig)
{
    WaitAllBits();
    QUnit::CopyState(orig);
}

void QUnitMulti::SetQuantumState(const complex* inputState)
{
    WaitAllBits();
    QUnit::SetQuantumState(inputState);
}

void QUnitMulti::GetQuantumState(complex* outputState)
{
    WaitAllBits();
    QUnit::GetQuantumState(outputState);
}

void QUnitMulti::GetProbs(real1* outputProbs)
{
    WaitAllBits();
    QUnit::GetProbs(outputProbs);
}

complex QUnitMulti::GetAmplitude(bitCapInt perm)
{
    WaitAllBits();
    return QUnit::GetAmplitude(perm);
}

/*
 * Append QInterface to the end of the unit.
 */
void QUnitMulti::Compose(QUnitMultiPtr toCopy, bool isMid, bitLenInt start)
{
    WaitAllBits();
    toCopy->WaitAllBits();

    QUnit::Compose(std::static_pointer_cast<QUnit>(toCopy), isMid, start);
}

bool QUnitMulti::TrySeparate(bitLenInt start, bitLenInt length)
{
    if (length == qubitCount) {
        return true;
    }

    if ((length == 1) && (shards[start].unit->GetQubitCount() == 1)) {
        return true;
    }

    if (length <= 1) {
        WaitAllBits();
    }

    return QUnit::TrySeparate(start, length);
}

void QUnitMulti::DumpShards()
{
    WaitAllBits();
    QUnit::DumpShards();
}

real1 QUnitMulti::Prob(bitLenInt qubit)
{
    WaitBit(qubit);
    return QUnit::Prob(qubit);
}

real1 QUnitMulti::ProbAll(bitCapInt perm)
{
    WaitAllBits();
    return QUnit::ProbAll(perm);
}

bool QUnitMulti::ForceM(bitLenInt qubit, bool res, bool doForce)
{
    WaitBit(qubit);
    return QUnit::ForceM(qubit, res, doForce);
}

void QUnitMulti::PhaseFlip()
{
    WaitBit(0);
    shards[0].unit->PhaseFlip();
}

void QUnitMulti::UpdateRunningNorm()
{
    WaitAllBits();
    QUnit::UpdateRunningNorm();
}

void QUnitMulti::Finish()
{
    WaitAllBits();
    QUnit::Finish();
}

bool QUnitMulti::ApproxCompare(QUnitMultiPtr toCompare)
{
    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        return false;
    }

    WaitAllBits();

    return QUnit::ApproxCompare(std::static_pointer_cast<QUnit>(toCompare));
}

QInterfacePtr QUnitMulti::Clone()
{
    WaitAllBits();
    return QUnit::Clone();
}

} // namespace Qrack
