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

#include "qcircuit.hpp"

namespace Qrack {

void QCircuit::AppendGate(QCircuitGatePtr nGate)
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

void QCircuit::Run(QInterfacePtr qsim)
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

        const bitLenInt controlCount = gate->controls.size();
        if ((gate->payloads.size() << 1U) >= (1U << controlCount)) {
            for (bitLenInt control : gate->controls) {
                if (controlStates[control]) {
                    qsim->X(control);
                    controlStates[control] = false;
                }
            }
            if (controlStates[t]) {
                qsim->X(t);
                controlStates[t] = false;
            }
            std::unique_ptr<complex[]> payload = gate->MakeUniformlyControlledPayload();
            qsim->UniformlyControlledSingleBit(gate->GetControlsVector(), gate->target, payload.get());

            continue;
        }

        for (const auto& payload : gate->payloads) {
            std::map<bitLenInt, bool> controlMismatch;
            bitLenInt mismatchCount = 0U;
            for (const auto& c : gate->controls) {
                controlMismatch[c] = ((bool)((payload.first >> c) & 1)) != controlStates[c];
                if (controlMismatch[c]) {
                    ++mismatchCount;
                }
            }

            if (((size_t)(mismatchCount << 1U)) < controlCount) {
                for (const auto& c : controlMismatch) {
                    if (c.second && controlStates[c.first]) {
                        qsim->X(c.first);
                        controlStates[c.first] = false;
                    }
                }

                if (!controlStates[t]) {
                    qsim->MACMtrx(gate->GetControlsVector(), payload.second.get(), t);

                    continue;
                }

                std::unique_ptr<complex[]> mtrx = InvertPayload(payload.second.get());
                qsim->MACMtrx(gate->GetControlsVector(), mtrx.get(), t);

                continue;
            }

            for (const auto& c : controlMismatch) {
                if (!c.second && !controlStates[c.first]) {
                    qsim->X(c.first);
                    controlStates[c.first] = true;
                }
            }

            if (!controlStates[t]) {
                qsim->MCMtrx(gate->GetControlsVector(), payload.second.get(), t);

                continue;
            }

            std::unique_ptr<complex[]> mtrx = InvertPayload(payload.second.get());
            qsim->MCMtrx(gate->GetControlsVector(), mtrx.get(), t);

            continue;
        }
    }

    for (bitLenInt i = 0U; i < controlStates.size(); ++i) {
        if (controlStates[i]) {
            qsim->X(i);
        }
    }
}
} // namespace Qrack
