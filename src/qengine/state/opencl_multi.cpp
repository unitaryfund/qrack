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

#include "oclengine.hpp"
#include "qengine_opencl_multi.hpp"

namespace Qrack {

#define CMPLX_NORM_LEN 5

std::vector<cl::Buffer> QEngineOCLMulti::SplitBuffer(QEngineOCLPtr) {
    return std::vector<cl::Buffer>();
}

    QEngineOCLMulti::QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, int deviceCount, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
{
    maxQPower = 1 << qubitCount;
    
    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetNodeCount();
    }
    
    bitLenInt devPow = log2(deviceCount);
    
    bitLenInt subQubits = qubitCount >> (devPow - 1);
    bitCapInt subMaxQPower = 1 << subQubits;
    bool foundInitState = false;
    bitCapInt subInitVal = 0;
        
    for (int i = 0; i < deviceCount; i++) {
        if ((!foundInitState) && (subMaxQPower * i > initState)) {
            subInitVal = initState << log2(i);
            foundInitState = true;
        }
        substateEngines.push_back(std::make_shared<QEngineOCL>(QEngineOCL(subQubits, subInitVal, rgp, i)));
        subInitVal = 0;
        
        substateBuffers.push_back(SplitBuffer(substateEngines[i]));
    }
}

} // namespace Qrack
