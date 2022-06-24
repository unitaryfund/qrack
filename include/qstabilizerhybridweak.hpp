//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2022. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.
#pragma once

#include "qstabilizerhybrid.hpp"

namespace Qrack {

class QStabilizerHybridWeak;
typedef std::shared_ptr<QStabilizerHybridWeak> QStabilizerHybridWeakPtr;

/**
 * A "Qrack::QStabilizerHybrid" internally switched between Qrack::QStabilizer and Qrack::QEngine to maximize
 * performance.
 */
class QStabilizerHybridWeak : public QEngine {
protected:
    bitLenInt ancillaCount;

public:
    QStabilizerHybridWeak(std::vector<QInterfaceEngine> eng, bitLenInt qBitCount, bitCapInt initState = 0U,
        qrack_rand_gen_ptr rgp = nullptr, complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false,
        bool randomGlobalPhase = true, bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true,
        bool useSparseStateVec = false, real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {},
        bitLenInt qubitThreshold = 0U, real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QStabilizerHybrid(eng, qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, deviceId,
              useHardwareRNG, useSparseStateVec, norm_thresh, devList, qubitThreshold, separation_thresh)
        , ancillaCount(0);
    {
    }

    QStabilizerHybridWeak(bitLenInt qBitCount, bitCapInt initState = 0U, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int64_t deviceId = -1, bool useHardwareRNG = true, bool useSparseStateVec = false,
        real1_f norm_thresh = REAL1_EPSILON, std::vector<int64_t> devList = {}, bitLenInt qubitThreshold = 0U,
        real1_f separation_thresh = FP_NORM_EPSILON_F)
        : QStabilizerHybridWeak({ QINTERFACE_OPTIMAL_BASE }, qBitCount, initState, rgp, phaseFac, doNorm,
              randomGlobalPhase, useHostMem, deviceId, useHardwareRNG, useSparseStateVec, norm_thresh, devList,
              qubitThreshold, separation_thresh)
    {
    }

    void SwitchToEngine()
    {
        if (engine) {
            return;
        }

        bitLenInt i = 0;
        while (i < ancillaCount) {
            if (stabilizer->IsSeparable(qubitCount + i)) {
                stabilizer->Dispose(qubitCount + i, 1);
                shards.erase(shards.begin() + qubitCount + i);
                --ancillaCount;
            } else {
                ++i;
            }
        }

        engine = MakeEngine();
        stabilizer->GetQuantumState(engine);
        stabilizer = NULL;
        FlushBuffers();

        // When we measure, we act postselection on reverse T-gadgets.
        engine->ForceMReg(qubitCount, ancillaCount, 0, true, true);
        // Ancillae are separable after measurement.
        engine->Dispose(qubitCount, ancillaCount);
        // We have extra "gate fusion" shards leftover.
        shards.erase(shards.begin() + qubitCount, shards.end());
        // We're done with ancillae.
        ancillaCount = 0;
    }

    using QStabilizerHybrid::Compose;
    bitLenInt Compose(QStabilizerHybridPtr toCopy, bitLenInt start)
    {
        if (start == qubitCount) {
            return Compose(toCopy);
        }

        const bitLenInt origSize = qubitCount;
        ROL(origSize - start, 0, qubitCount);
        const bitLenInt result = Compose(toCopy);
        ROR(origSize - start, 0, qubitCount);

        return result;
    }
    bitLenInt Compose(QStabilizerHybridPtr toCopy)
    {
        const bitLenInt nQubits = qubitCount + toCopy->qubitCount;
        bitLenInt toRet;

        if (engine) {
            toCopy->SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else if (toCopy->engine) {
            SwitchToEngine();
            toRet = engine->Compose(toCopy->engine);
        } else {
            toRet = stabilizer->Compose(toCopy->stabilizer, qubitCount);
        }

        // Resize the shards buffer.
        shards.insert(shards.begin() + qubitCount, toCopy->shards.begin(), toCopy->shards.end());
        // Split the common shared_ptr references, with toCopy.
        for (bitLenInt i = qubitCount; i < nQubits; ++i) {
            if (shards[i]) {
                shards[i] = shards[i]->Clone();
            }
        }

        SetQubitCount(nQubits);
        ancillaCount += toCopy->ancillaCount;

        return toRet;
    }

    void Mtrx(const complex* lMtrx, bitLenInt target)
    {
        const bool wasCached = (bool)shards[target];
        complex mtrx[4U];
        if (wasCached) {
            shards[target]->Compose(lMtrx);
            std::copy(shards[target]->gate, shards[target]->gate + 4U, mtrx);
            shards[target] = NULL;
        } else {
            std::copy(lMtrx, lMtrx + 4U, mtrx);
        }

        if (engine) {
            engine->Mtrx(mtrx, target);
            return;
        }

        if (IS_CLIFFORD(mtrx) ||
            (randGlobalPhase && (IS_PHASE(mtrx) || IS_INVERT(mtrx)) && stabilizer->IsSeparableZ(target))) {
            stabilizer->Mtrx(mtrx, target);
            return;
        }

        if (IS_PHASE(mtrx)) {
            QStabilizerPtr ancilla = std::make_shared<QStabilizer>(
                1U, 0U, rand_generator, CMPLX_DEFAULT_ARG, false, randGlobalPhase, false, -1, useRDRAND);

            // Form potentially entangled representation, with this.
            bitLenInt ancillaIndex = stabilizer->Compose(ancilla);

            // Act reverse T-gadget with measurement basis preparation.
            stabilizer->CNOT(target, ancillaIndex);
            complex iMtrx[4];
            inv2x2(mtrx, iMtrx);
            shards.push_back(std::make_shared<MpsShard>(iMtrx));
            CacheEigenstate(ancillaIndex);
            stabilizer->H(ancillaIndex);

            // When we measure, we act postselection, but not yet.
            // ForceM(ancillaIndex, false, true, true);
            // Ancilla is separable after measurement.
            // Dispose(ancillaIndex, 1U);

            ++ancillaCount;
        }

        shards[target] = std::make_shared<MpsShard>(mtrx);
        if (!wasCached) {
            CacheEigenstate(target);
        }
    }

    bitCapInt MAll()
    {
        bitCapInt toRet = 0U;
        if (stabilizer) {
            for (bitLenInt i = 0; i < ancillaCount; i++) {
                // When we measure, we act postselection on reverse T-gadgets.
                ForceM(qubitCount + i, false, true, true);
            }
            // Ancillae are separable after measurement.
            stabilizer->Dispose(qubitCount, ancillaCount);
            ancillaCount = 0;

            for (bitLenInt i = 0U; i < qubitCount; ++i) {
                if (shards[i] && shards[i]->IsInvert()) {
                    InvertBuffer(i);
                }

                if (shards[i]) {
                    if (!shards[i]->IsPhase() && stabilizer->IsSeparableZ(i)) {
                        // Bit was already rotated to Z basis, if separable.
                        CollapseSeparableShard(i);
                    }

                    // Otherwise, buffer will not change the fact that state appears maximally mixed.
                    shards[i] = NULL;
                }

                if (stabilizer->M(i)) {
                    toRet |= pow2(i);
                }
            }
        } else {
            toRet = engine->MAll();
        }

        SetPermutation(toRet);

        return toRet;
    }
};
} // namespace Qrack
