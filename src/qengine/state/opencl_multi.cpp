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

QEngineOCLMulti::QEngineOCLMulti(bitLenInt qBitCount, bitCapInt initState, int deviceCount, std::shared_ptr<std::default_random_engine> rgp)
    : QInterface(qBitCount)
{
    maxQPower = 1 << qubitCount;
    
    clObj = OCLEngine::Instance();
    if (deviceCount == -1) {
        deviceCount = clObj->GetNodeCount();
    }
    
    bitLenInt devPow = log2(deviceCount);
    
    subQubitCount = qubitCount >> (devPow - 1);
    subMaxQPower = 1 << subQubitCount;
    subBufferSize = sizeof(complex) * subMaxQPower;
    bool foundInitState = false;
    bitCapInt subInitVal = 0;
        
    for (int i = 0; i < deviceCount; i++) {
        if ((!foundInitState) && (subMaxQPower * i > initState)) {
            subInitVal = initState - (subMaxQPower * i);
            foundInitState = true;
        }
        substateEngines.push_back(std::make_shared<QEngineOCL>(QEngineOCL(subQubitCount, subInitVal, rgp, i)));
        subInitVal = 0;
    }
}
    
void QEngineOCLMulti::ShuffleBuffers(CommandQueuePtr queue, cl::Buffer buff1, cl::Buffer buff2, cl::Buffer tempBuffer) {
    queue->enqueueCopyBuffer(buff1, tempBuffer, subBufferSize, 0, subBufferSize);
    queue->finish();
    
    queue->enqueueCopyBuffer(buff2, buff1, 0, subBufferSize, subBufferSize);
    queue->finish();
    
    queue->enqueueCopyBuffer(tempBuffer, buff2, 0, subBufferSize, subBufferSize);
    queue->finish();
}
    
template <typename F> void QEngineOCLMulti::SingleBitGate(std::vector<F> fns, bitLenInt bit) {
    // TODO: This logic only handles 2 devices, for the moment. Extend generally.
    bool isSubLocal = (bit < subQubitCount);
    bitLenInt localBit = bit;
    
    if (isSubLocal) {
        for (auto fn : fns) {
            fn(localBit);
        }
    } else {
        localBit = subQubitCount;

        cl::Context context = *(clObj->GetContextPtr());
        cl::Buffer tempBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, subBufferSize >> 1);
        
        cl::Buffer buff1 = substateEngines[0]->GetStateBuffer();
        cl::Buffer buff2 = substateEngines[1]->GetStateBuffer();
        
        CommandQueuePtr queue = substateEngines[0]->GetQueuePtr();
        
        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
        
        for (auto fn : fns) {
            fn(localBit);
        }
        
        ShuffleBuffers(queue, buff1, buff2, tempBuffer);
    }
}

} // namespace Qrack
