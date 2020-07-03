//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// QUnit maintains explicit separability of qubits as an optimization on a QEngine.
// See https://arxiv.org/abs/1710.05867
// (The makers of Qrack have no affiliation with the authors of that paper.)
//
// When we allocate a quantum register, all bits are in a (re)set state. At this point,
// we know they are separable, in the sense of full Schmidt decomposability into qubits
// in the "natural" or "permutation" basis of the register. Many operations can be
// traced in terms of fewer qubits that the full "Schr\{"o}dinger representation."
//
// Based on experimentation, QUnit is designed to avoid increasing representational
// entanglement for its primary action, and only try to decrease it when inquiries
// about probability need to be made otherwise anyway. Avoiding introducing the cost of
// basically any entanglement whatsoever, rather than exponentially costly "garbage
// collection," should be the first and ultimate concern, in the authors' experience.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qpager.hpp"

namespace Qrack {

QPager::QPager(QInterfaceEngine eng, bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp, complex phaseFac,
    bool ignored, bool randomGlobalPhase, bool useHostMem, int deviceId, bool useHardwareRNG, bool useSparseStateVec,
    real1 norm_thresh, std::vector<bitLenInt> devList)
    : QInterface(qBitCount, rgp, ignored, useHardwareRNG, randomGlobalPhase, norm_thresh)
    , engine(eng)
    , devID(deviceId)
    , phaseFactor(phaseFac)
    , useHostRam(useHostMem)
    , useRDRAND(useHardwareRNG)
    , isSparse(useSparseStateVec)
{
    if ((qPageQubitCount) > (sizeof(bitCapIntOcl) * bitsInByte)) {
        throw std::invalid_argument(
            "Cannot instantiate a register with greater capacity than native types on emulating system.");
    }

    bitCapInt pagePerm = 0;
    for (bitCapInt i = 0; i < qPageCount; i++) {
        bool isPermInPage = (initState >= pagePerm);
        pagePerm += qPageMaxQPower;
        isPermInPage &= (initState < pagePerm);
        if (isPermInPage) {
            qPages.push_back(MakeEngine(qPageQubitCount, initState - (pagePerm - qPageMaxQPower)));
        } else {
            qPages.push_back(MakeEngine(qPageQubitCount, 0));
            qPages.back()->SetAmplitude(0, ZERO_CMPLX);
        }
    }
}

} // namespace Qrack
