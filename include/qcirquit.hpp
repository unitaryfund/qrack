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

#include "qinterface.hpp"

namespace Qrack {

struct QCircuitGate;
typedef std::shared_ptr<QCircuitGate> QCircuitGatePtr;

struct QCircuitGate {
    bitLenInt target;
    std::map<bitCapInt, std::unique_ptr<complex[]>> payloads;
    std::set<bitLenInt> controls;

    QCircuitGate(bitLenInt trgt, complex matrix[])
        : target(trgt)
    {
        payloads[0] = std::unique_ptr<complex[]>(new complex[4]);
        std::copy(matrix, matrix + 4, payloads[0].get());
    }

    QCircuitGate(bitLenInt trgt, complex matrix[], const std::set<bitLenInt>& ctrls, bitCapInt perm)
        : target(trgt)
        , controls(ctrls)
    {
        payloads[perm] = std::unique_ptr<complex[]>(new complex[4]);
        std::copy(matrix, matrix + 4, payloads[perm].get());
    }

    QCircuitGate(
        bitLenInt trgt, const std::map<bitCapInt, std::unique_ptr<complex[]>>& pylds, const std::set<bitLenInt>& ctrls)
        : target(trgt)
        , controls(ctrls)
    {
        for (const auto& payload : pylds) {
            const auto& p = payloads[payload.first] = std::unique_ptr<complex[]>(new complex[4]);
            std::copy(payload.second.get(), payload.second.get() + 4, p.get());
        }
    }

    bool CanCombine(QCircuitGatePtr other)
    {
        if (target != other->target) {
            return false;
        }

        if (controls.size() != other->controls.size()) {
            return false;
        }

        for (const bitLenInt& control : other->controls) {
            if (controls.find(control) == controls.end()) {
                return false;
            }
        }

        return true;
    }

    void Combine(QCircuitGatePtr other)
    {
        for (const auto& payload : other->payloads) {
            if (payloads.find(payload.first) == payloads.end()) {
                const auto& p = payloads[payload.first] = std::unique_ptr<complex[]>(new complex[4]);
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            const auto& p = payloads[payload.first];
            complex out[4];
            mul2x2(payload.second.get(), p.get(), out);
            std::copy(out, out + 4U, p.get());
        }
    }

    bool TryCombine(QCircuitGatePtr other)
    {
        if (!CanCombine(other)) {
            return false;
        }
        Combine(other);

        return true;
    }

    std::vector<QCircuitGatePtr> Expand()
    {
        std::vector<QCircuitGatePtr> toRet;
        toRet.reserve(payloads.size());
        for (const auto& payload : payloads) {
            toRet.emplace_back(std::make_shared<QCircuitGate>(target, payload.second.get(), controls, payload.first));
        }

        return toRet;
    }

    bool HasCommonControl(QCircuitGatePtr other)
    {
        std::set<bitLenInt>::iterator first1 = controls.begin();
        std::set<bitLenInt>::iterator last1 = controls.end();
        std::set<bitLenInt>::iterator first2 = other->controls.begin();
        std::set<bitLenInt>::iterator last2 = other->controls.begin();
        while (first1 != last1 && first2 != last2) {
            if (*first1 < *first2) {
                ++first1;
            } else if (*first2 < *first1) {
                ++first2;
            } else {
                return true;
            }
        }
        return false;
    }

    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            if ((norm(payload.second[1]) > FP_NORM_EPSILON) || (norm(payload.second[2]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    bool IsInvert()
    {
        for (const auto& payload : payloads) {
            if ((norm(payload.second[0]) > FP_NORM_EPSILON) || (norm(payload.second[3]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    bool CanPass(QCircuitGatePtr other)
    {
        if (HasCommonControl(other)) {
            return false;
        }

        if (target != other->target) {
            return true;
        }

        return IsPhase() && other->IsPhase();
    }
};

class QCircuit;
typedef std::shared_ptr<QCircuit> QCircuitPtr;

class QCircuit {
protected:
    bitLenInt maxQubit;
    std::map<bitLenInt, bitLenInt> qubitMap;
    std::vector<QCircuitGatePtr> gates;

    std::unique_ptr<complex[]> InvertPayload(const complex* m)
    {
        std::unique_ptr<complex[]> mtrx(new complex[4]);
        mtrx[0] = m[2];
        mtrx[1] = m[3];
        mtrx[2] = m[0];
        mtrx[3] = m[1];

        return mtrx;
    }

public:
    QCircuit()
        : maxQubit(0)
        , qubitMap()
        , gates()
    {
        // Intentionally left blank
    }

    bitLenInt GetQubitCount() { return maxQubit; }

    void AppendGate(QCircuitGatePtr nGate)
    {
        bitLenInt nMaxQubit = maxQubit;
        if (nGate->target > nMaxQubit) {
            nMaxQubit = nGate->target;
        }
        if (!(nGate->controls.empty())) {
            const bitLenInt q = *(nGate->controls.rbegin());
            if (q > nMaxQubit) {
                nMaxQubit = q;
            }
        }
        for (; maxQubit < nMaxQubit; ++maxQubit) {
            qubitMap[maxQubit] = maxQubit;
        }

        const bitLenInt nTarget = qubitMap[nGate->target];
        std::set<bitLenInt> nControls;
        for (const auto& control : nGate->controls) {
            nControls.insert(qubitMap[control]);
        }
        QCircuitGatePtr lGate = std::make_shared<QCircuitGate>(nTarget, nGate->payloads, nControls);

        for (QCircuitGatePtr gate : gates) {
            if (gate->TryCombine(lGate)) {
                return;
            }
            if (!gate->CanPass(lGate)) {
                break;
            }
        }
        gates.push_back(nGate);
    }

    void Swap(bitLenInt q1, bitLenInt q2)
    {
        bitLenInt nMaxQubit = maxQubit;
        if (q1 > nMaxQubit) {
            nMaxQubit = q1;
        }
        if (q2 > nMaxQubit) {
            nMaxQubit = q2;
        }
        for (; maxQubit < nMaxQubit; ++maxQubit) {
            qubitMap[maxQubit] = maxQubit;
        }

        std::swap(qubitMap[q1], qubitMap[q2]);
    }

    void Run(QInterfacePtr qsim)
    {
        if (qsim->GetQubitCount() < maxQubit) {
            qsim->Allocate(maxQubit - qsim->GetQubitCount());
        }

        std::vector<bool> controlStates(maxQubit, false);
        for (const QCircuitGatePtr& gate : gates) {
            const bitLenInt& t = gate->target;
            if (!gate->controls.size()) {
                const complex* gMtrx = gate->payloads[0].get();
                if (controlStates[t]) {
                    std::unique_ptr<complex[]> mtrx = InvertPayload(gMtrx);
                    qsim->Mtrx(mtrx.get(), t);
                } else {
                    qsim->Mtrx(gMtrx, t);
                }

                continue;
            }

            // TODO: If payload multiplicity is high, just use "uniformly controlled" gate.

            for (const auto& payload : gate->payloads) {
                std::map<bitLenInt, bool> controlMismatch;
                bitLenInt mismatchCount = 0U;
                for (const auto& c : gate->controls) {
                    controlMismatch[c] = ((bool)((payload.first >> c) & 1)) != controlStates[c];
                    if (controlMismatch[c]) {
                        ++mismatchCount;
                    }
                }

                if (((size_t)(mismatchCount << 1U)) < gate->controls.size()) {
                    for (const auto& c : controlMismatch) {
                        if (c.second && controlStates[c.first]) {
                            qsim->X(c.first);
                            controlStates[c.first] = false;
                        }
                    }

                    if (!controlStates[t]) {
                        qsim->MACMtrx(std::vector<bitLenInt>(gate->controls.begin(), gate->controls.end()),
                            payload.second.get(), t);

                        continue;
                    }

                    std::unique_ptr<complex[]> mtrx = InvertPayload(payload.second.get());
                    qsim->MACMtrx(std::vector<bitLenInt>(gate->controls.begin(), gate->controls.end()), mtrx.get(), t);

                    continue;
                }

                for (const auto& c : controlMismatch) {
                    if (!c.second && !controlStates[c.first]) {
                        qsim->X(c.first);
                        controlStates[c.first] = true;
                    }
                }

                if (!controlStates[t]) {
                    qsim->MCMtrx(
                        std::vector<bitLenInt>(gate->controls.begin(), gate->controls.end()), payload.second.get(), t);

                    continue;
                }

                std::unique_ptr<complex[]> mtrx = InvertPayload(payload.second.get());
                qsim->MCMtrx(std::vector<bitLenInt>(gate->controls.begin(), gate->controls.end()), mtrx.get(), t);

                continue;
            }
        }
    }
};
} // namespace Qrack
