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

#include <list>

#define amp_leq_0(x) (norm(x) <= FP_NORM_EPSILON)

namespace Qrack {

/**
 * Single gate in `QCircuit` definition
 */
struct QCircuitGate;
typedef std::shared_ptr<QCircuitGate> QCircuitGatePtr;

struct QCircuitGate {
    bitLenInt target;
    std::map<bitCapInt, std::shared_ptr<complex>> payloads;
    std::set<bitLenInt> controls;

    /**
     * Single-qubit gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[])
        : target(trgt)
    {
        payloads[0] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, payloads[0].get());
    }

    /**
     * Controlled gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[], const std::set<bitLenInt>& ctrls, bitCapInt perm)
        : target(trgt)
        , controls(ctrls)
    {
        const std::shared_ptr<complex>& p = payloads[perm] =
            std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, p.get());
    }

    /**
     * Uniformly controlled gate constructor (that only accepts control qubits is ascending order)
     */
    QCircuitGate(
        bitLenInt trgt, const std::map<bitCapInt, std::shared_ptr<complex>>& pylds, const std::set<bitLenInt>& ctrls)
        : target(trgt)
        , controls(ctrls)
    {
        for (const auto& payload : pylds) {
            const std::shared_ptr<complex>& p = payloads[payload.first] =
                std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4, p.get());
        }
    }

    QCircuitGatePtr Clone() { return std::make_shared<QCircuitGate>(target, payloads, controls); }

    /**
     * Can I combine myself with gate `other`?
     */
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

    /**
     * Set this gate to the identity operator.
     */
    void Clear()
    {
        controls.clear();
        payloads.clear();

        payloads[0] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        complex* p = payloads[0].get();
        p[0] = ONE_CMPLX;
        p[1] = ZERO_CMPLX;
        p[2] = ZERO_CMPLX;
        p[3] = ONE_CMPLX;
    }

    /**
     * Combine myself with gate `other`
     */
    void Combine(QCircuitGatePtr other)
    {
        for (const auto& payload : other->payloads) {
            const auto& pit = payloads.find(payload.first);
            if (pit == payloads.end()) {
                const std::shared_ptr<complex>& p = payloads[payload.first] =
                    std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
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
            Clear();
            return;
        }
    }

    /**
     * Check if I can combine with gate `other`, and do so, if possible
     */
    bool TryCombine(QCircuitGatePtr other)
    {
        if (!CanCombine(other)) {
            return false;
        }
        Combine(other);

        return true;
    }

    /**
     * Am I an identity gate?
     */
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

    /**
     * Am I a phase gate?
     */
    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1]) > FP_NORM_EPSILON) || (norm(p[2]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I Pauli X plus a phase gate?
     */
    bool IsInvert()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Do I commute with gate `other`?
     */
    bool CanPass(QCircuitGatePtr other)
    {
        const std::set<bitLenInt>::iterator c = other->controls.find(target);
        if (c != other->controls.end()) {
            if (controls.find(other->target) != controls.end()) {
                return IsPhase() && other->IsPhase();
            }
            if (IsPhase()) {
                return true;
            }
            if (controls.size() || !IsInvert()) {
                return false;
            }

            const bitCapInt p = pow2(std::distance(other->controls.begin(), c));
            std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
            for (const auto& payload : other->payloads) {
                nPayloads[payload.first ^ p] = payload.second;
            }
            other->payloads = nPayloads;

            return true;
        }

        if (controls.find(other->target) != controls.end()) {
            return other->IsPhase();
        }

        return (target != other->target) || (IsPhase() && other->IsPhase());
    }

    /**
     * To run as a uniformly controlled gate, generate my payload array.
     */
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

    /**
     * Convert my set of qubit indices to a vector
     */
    std::vector<bitLenInt> GetControlsVector() { return std::vector<bitLenInt>(controls.begin(), controls.end()); }
};

/**
 * Define and optimize a circuit, before running on a `QInterface`.
 */
class QCircuit;
typedef std::shared_ptr<QCircuit> QCircuitPtr;

class QCircuit {
protected:
    bitLenInt qubitCount;
    std::list<QCircuitGatePtr> gates;

    /**
     * Reverse truth values of 2x2 complex matrix
     */
    static std::unique_ptr<complex[]> InvertPayload(const complex* m)
    {
        std::unique_ptr<complex[]> mtrx(new complex[4]);
        mtrx[0] = m[1];
        mtrx[1] = m[0];
        mtrx[2] = m[3];
        mtrx[3] = m[2];

        return mtrx;
    }

public:
    /**
     * Default constructor
     */
    QCircuit()
        : qubitCount(0)
        , gates()
    {
        // Intentionally left blank
    }

    /**
     * Manual constructor
     */
    QCircuit(bitLenInt qbCount, std::list<QCircuitGatePtr> g)
        : qubitCount(qbCount)
        , gates(g)
    {
        // Intentionally left blank
    }

    QCircuitPtr Clone()
    {
        QCircuitPtr clone = std::make_shared<QCircuit>();
        clone->qubitCount = qubitCount;
        for (const QCircuitGatePtr& gate : gates) {
            clone->gates.push_back(gate->Clone());
        }

        return clone;
    }

    /**
     * Get the (automatically calculated) count of qubits in this circuit, so far.
     */
    bitLenInt GetQubitCount() { return qubitCount; }

    /**
     * Return the raw list of gates.
     */
    std::list<QCircuitGatePtr> GetGateList() { return gates; }

    /**
     * Add a `Swap` gate to the gate sequence.
     */
    void Swap(bitLenInt q1, bitLenInt q2)
    {
        if (q1 == q2) {
            return;
        }

        // If all swap gates are constructed in the same order, between high and low qubits, then the chances of
        // combining them might be higher.
        if (q1 > q2) {
            std::swap(q1, q2);
        }

        const complex m[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        const std::set<bitLenInt> s1 = { q1 };
        const std::set<bitLenInt> s2 = { q2 };
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, 1U));
        AppendGate(std::make_shared<QCircuitGate>(q2, m, s1, 1U));
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, 1U));
    }

    /**
     * Add a gate to the gate sequence.
     */
    void AppendGate(QCircuitGatePtr nGate);
    /**
     * Run this circuit.
     */
    void Run(QInterfacePtr qsim);
};
} // namespace Qrack
