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

#include "config.h"

namespace Qrack {

enum OCLAPI {
    OCL_API_UNKNOWN = 0,
    OCL_API_APPLY2X2,
    OCL_API_APPLY2X2_SINGLE,
    OCL_API_APPLY2X2_NORM_SINGLE,
    OCL_API_APPLY2X2_DOUBLE,
    OCL_API_APPLY2X2_WIDE,
    OCL_API_APPLY2X2_SINGLE_WIDE,
    OCL_API_APPLY2X2_NORM_SINGLE_WIDE,
    OCL_API_APPLY2X2_DOUBLE_WIDE,
    OCL_API_PHASE_SINGLE,
    OCL_API_PHASE_SINGLE_WIDE,
    OCL_API_INVERT_SINGLE,
    OCL_API_INVERT_SINGLE_WIDE,
    OCL_API_UNIFORMLYCONTROLLED,
    OCL_API_UNIFORMPARITYRZ,
    OCL_API_UNIFORMPARITYRZ_NORM,
    OCL_API_CUNIFORMPARITYRZ,
    OCL_API_COMPOSE,
    OCL_API_COMPOSE_WIDE,
    OCL_API_COMPOSE_MID,
    OCL_API_DECOMPOSEPROB,
    OCL_API_DECOMPOSEAMP,
    OCL_API_DISPOSEPROB,
    OCL_API_DISPOSE,
    OCL_API_PROB,
    OCL_API_CPROB,
    OCL_API_PROBREG,
    OCL_API_PROBREGALL,
    OCL_API_PROBMASK,
    OCL_API_PROBMASKALL,
    OCL_API_PROBPARITY,
    OCL_API_FORCEMPARITY,
    OCL_API_EXPPERM,
    OCL_API_X_SINGLE,
    OCL_API_X_SINGLE_WIDE,
    OCL_API_X_MASK,
    OCL_API_Z_SINGLE,
    OCL_API_Z_SINGLE_WIDE,
    OCL_API_PHASE_PARITY,
    OCL_API_ROL,
    OCL_API_APPROXCOMPARE,
    OCL_API_NORMALIZE,
    OCL_API_NORMALIZE_WIDE,
    OCL_API_UPDATENORM,
    OCL_API_APPLYM,
    OCL_API_APPLYMREG,
    OCL_API_CLEARBUFFER,
    OCL_API_SHUFFLEBUFFERS,
#if ENABLE_ALU
    OCL_API_INC,
    OCL_API_CINC,
    OCL_API_INCDECC,
    OCL_API_INCS,
    OCL_API_INCDECSC_1,
    OCL_API_INCDECSC_2,
    OCL_API_MUL,
    OCL_API_DIV,
    OCL_API_MULMODN_OUT,
    OCL_API_IMULMODN_OUT,
    OCL_API_POWMODN_OUT,
    OCL_API_CMUL,
    OCL_API_CDIV,
    OCL_API_CMULMODN_OUT,
    OCL_API_CIMULMODN_OUT,
    OCL_API_CPOWMODN_OUT,
    OCL_API_FULLADD,
    OCL_API_IFULLADD,
    OCL_API_INDEXEDLDA,
    OCL_API_INDEXEDADC,
    OCL_API_INDEXEDSBC,
    OCL_API_HASH,
    OCL_API_CPHASEFLIPIFLESS,
    OCL_API_PHASEFLIPIFLESS,
#if ENABLE_BCD
    OCL_API_INCBCD,
    OCL_API_INCDECBCDC
#endif
#endif
};

} // namespace Qrack
