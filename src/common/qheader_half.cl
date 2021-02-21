//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define cmplx half2
#define cmplx2 half4
#define cmplx4 half8
#define real1 half
#define ZERO_R1 0.0h
#define ONE_R1 1.0h
#define SineShift M_PI_2_H
#define PI_R1 M_PI_H
#define REAL1_EPSILON 2e-17h
