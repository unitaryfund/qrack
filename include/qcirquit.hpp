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

#include "common/qrack_types.hpp"

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

struct QCircuit;
typedef std::shared_ptr<QCircuit> QCircuitPtr;

struct QCircuit {
    std::vector<QCircuitGatePtr> gates;

    void AppendGate(QCircuitGatePtr nGate)
    {
        for (QCircuitGatePtr gate : gates) {
            if (gate->TryCombine(nGate)) {
                return;
            }
            if (!gate->CanPass(nGate)) {
                break;
            }
        }
        gates.push_back(nGate);
    }
};
} // namespace Qrack
