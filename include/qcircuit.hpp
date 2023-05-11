//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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

#define amp_leq_0(x) (norm(x) <= FP_NORM_EPSILON)

namespace Qrack {

struct QCircuitGate;
typedef std::shared_ptr<QCircuitGate> QCircuitGatePtr;

struct QCircuitGate {
    bitLenInt target;
    std::map<bitCapInt, std::unique_ptr<complex[]>> payloads;
    std::vector<bitLenInt> controls;
    std::set<bitLenInt> orderedControls;

    QCircuitGate(bitLenInt q1, bitLenInt q2)
        : target(q1)
        , payloads()
        , controls({ q2 })

    {
        orderedControls.insert(q1);
        orderedControls.insert(q2);
    }

    QCircuitGate(bitLenInt trgt, complex matrix[])
        : target(trgt)
    {
        payloads[0] = std::unique_ptr<complex[]>(new complex[4]);
        std::copy(matrix, matrix + 4, payloads[0].get());
    }

    QCircuitGate(bitLenInt trgt, complex matrix[], const std::vector<bitLenInt>& ctrls, bitCapInt perm)
        : target(trgt)
        , controls(ctrls)
        , orderedControls(ctrls.begin(), ctrls.end())
    {
        payloads[perm] = std::unique_ptr<complex[]>(new complex[4]);
        std::copy(matrix, matrix + 4, payloads[perm].get());
    }

    QCircuitGate(bitLenInt trgt, const std::map<bitCapInt, std::unique_ptr<complex[]>>& pylds,
        const std::vector<bitLenInt>& ctrls)
        : target(trgt)
        , controls(ctrls)
        , orderedControls(ctrls.begin(), ctrls.end())
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

        if (orderedControls.size() != other->orderedControls.size()) {
            return false;
        }

        for (const bitLenInt& control : other->orderedControls) {
            if (orderedControls.find(control) == orderedControls.end()) {
                return false;
            }
        }

        return true;
    }

    void Combine(QCircuitGatePtr other)
    {
        for (const auto& payload : other->payloads) {
            const auto& pit = payloads.find(payload.first);
            if (pit == payloads.end()) {
                const auto& p = payloads[payload.first] = std::unique_ptr<complex[]>(new complex[4]);
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            complex* p = pit->second.get();
            complex out[4];
            mul2x2(payload.second.get(), p, out);

            if (amp_leq_0(out[1]) && amp_leq_0(out[2]) && amp_leq_0(ONE_CMPLX - out[0]) &&
                amp_leq_0(ONE_CMPLX - out[3])) {
                payloads.erase(pit);

                continue;
            }

            std::copy(out, out + 4U, p);
        }

        if (!payloads.size()) {
            controls.clear();
            orderedControls.clear();
            const auto& p = payloads[0] = std::unique_ptr<complex[]>(new complex[4]);
            p[0] = ONE_CMPLX;
            p[1] = ZERO_CMPLX;
            p[2] = ZERO_CMPLX;
            p[3] = ONE_CMPLX;
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

    bool IsIdentity()
    {
        if (controls.size()) {
            return false;
        }

        if (payloads.size() != 1U) {
            return false;
        }

        complex* p = payloads.begin()->second.get();

        if (amp_leq_0(p[1]) && amp_leq_0(p[2]) && amp_leq_0(ONE_CMPLX - p[0]) && amp_leq_0(ONE_CMPLX - p[3])) {
            return true;
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
        if ((orderedControls.find(other->target) == orderedControls.end()) &&
            (other->orderedControls.find(target) == other->orderedControls.end())) {
            return (target != other->target) || (IsPhase() && other->IsPhase());
        }

        return false;
    }

    std::unique_ptr<complex[]> MakeUniformlyControlledPayload()
    {
        const bitCapIntOcl maxQPower = (1U << controls.size());
        std::unique_ptr<complex[]> toRet(new complex[4U * maxQPower]);
        for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
            complex* mtrx = toRet.get() + (i << 2U);
            if (payloads.find(i) == payloads.end()) {
                mtrx[0] = ONE_CMPLX;
                mtrx[1] = ZERO_CMPLX;
                mtrx[2] = ZERO_CMPLX;
                mtrx[3] = ONE_CMPLX;

                continue;
            }

            const complex* oMtrx = payloads[i].get();
            std::copy(oMtrx, oMtrx + 4U, mtrx);
        }

        return toRet;
    }

    std::vector<bitLenInt> GetControlsVector() { return controls; }
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

    void Swap(bitLenInt q1, bitLenInt q2) { AppendGate(std::make_shared<QCircuitGate>(q1, q2)); }

    void AppendGate(QCircuitGatePtr nGate);
    void Run(QInterfacePtr qsim);
};
} // namespace Qrack
