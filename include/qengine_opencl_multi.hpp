//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#pragma once

#include "qengine_opencl.hpp"

namespace Qrack {
    
class QEngineOCLMulti;

/** OpenCL enhanced QEngineCPU implementation. */
class QEngineOCLMulti : public QInterface {
protected:
    bitLenInt subQubitCount;
    bitCapInt subMaxQPower;
    size_t subBufferSize;
    OCLEngine* clObj;
    std::vector<QEngineOCLPtr> substateEngines;
    std::vector<std::vector<cl::Buffer>> substateBuffers;

public:
    QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, int deviceCount = -1, std::shared_ptr<std::default_random_engine> rgp = nullptr);

private:
    template <typename F> void SingleBitGate(std::vector<F> fns, bitLenInt bits);
    
    void ShuffleBuffers(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer);
    
    inline bitCapInt log2(bitCapInt n) {
        bitLenInt pow = 0;
        bitLenInt p = n;
        while (p != 0) {
            p >>= 1;
            pow++;
        }
        return pow;
    }
};
} // namespace Qrack
