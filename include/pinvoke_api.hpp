// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "stddef.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#define MICROSOFT_QUANTUM_DECL __declspec(dllexport)
#define MICROSOFT_QUANTUM_DECL_IMPORT __declspec(dllimport)
#else
#define MICROSOFT_QUANTUM_DECL
#define MICROSOFT_QUANTUM_DECL_IMPORT
#endif

// SAL only defined in windows.
#ifndef _In_
#define _In_
#define _In_reads_(n)
#endif

typedef void (*IdCallback)(unsigned);
typedef bool (*ProbAmpCallback)(size_t, double, double);

#if !(FPPOW < 6 && !ENABLE_COMPLEX_X2)
struct _QrackTimeEvolveOpHeader;
#endif

extern "C" {
// non-quantum
MICROSOFT_QUANTUM_DECL unsigned init();
MICROSOFT_QUANTUM_DECL unsigned init_count(_In_ unsigned q);
MICROSOFT_QUANTUM_DECL unsigned init_clone(_In_ unsigned sid);
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid);
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s);
MICROSOFT_QUANTUM_DECL void set_concurrency(_In_ unsigned sid, _In_ unsigned p);
MICROSOFT_QUANTUM_DECL void Dump(_In_ unsigned sid, _In_ ProbAmpCallback callback);

// pseudo-quantum
MICROSOFT_QUANTUM_DECL double Prob(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL double PermutationExpectation(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c);

MICROSOFT_QUANTUM_DECL void DumpIds(_In_ unsigned sid, _In_ IdCallback callback);

MICROSOFT_QUANTUM_DECL size_t random_choice(_In_ unsigned sid, _In_ size_t n, _In_reads_(n) double* p);

MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void PhaseParity(
    _In_ unsigned sid, _In_ double lambda, _In_ unsigned n, _In_reads_(n) unsigned* q);

MICROSOFT_QUANTUM_DECL void ResetAll(_In_ unsigned sid);

// allocate and release
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid);
MICROSOFT_QUANTUM_DECL bool release(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid);

// single-qubit gates
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void U(
    _In_ unsigned sid, _In_ unsigned q, _In_ double theta, _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void Mtrx(_In_ unsigned sid, _In_reads_(8) double* m, _In_ unsigned q);

// multi-controlled single-qubit gates

MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MCU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void MCMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q);

MICROSOFT_QUANTUM_DECL void MACX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL void MACU(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q,
    _In_ double theta, _In_ double phi, _In_ double lambda);
MICROSOFT_QUANTUM_DECL void MACMtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_reads_(8) double* m, _In_ unsigned q);

MICROSOFT_QUANTUM_DECL void Multiplex1Mtrx(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q, double* m);

// rotations
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q);

// multi-controlled rotations
MICROSOFT_QUANTUM_DECL void MCR(
    _In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q);

// Exponential of Pauli operators
MICROSOFT_QUANTUM_DECL void Exp(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_ double phi,
    _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q);

// measurements
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q);
MICROSOFT_QUANTUM_DECL unsigned Measure(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) int* b, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void MeasureShots(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ unsigned s, _In_reads_(s) unsigned* m);

MICROSOFT_QUANTUM_DECL void SWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2);
MICROSOFT_QUANTUM_DECL void ISWAP(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2);
MICROSOFT_QUANTUM_DECL void FSim(
    _In_ unsigned sid, _In_ double theta, _In_ double phi, _In_ unsigned qi1, _In_ unsigned qi2);
MICROSOFT_QUANTUM_DECL void CSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2);
MICROSOFT_QUANTUM_DECL void ACSWAP(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned qi1, _In_ unsigned qi2);

// Schmidt decomposition
MICROSOFT_QUANTUM_DECL void Compose(_In_ unsigned sid1, _In_ unsigned sid2, unsigned* q);
MICROSOFT_QUANTUM_DECL unsigned Decompose(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void Dispose(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q);

MICROSOFT_QUANTUM_DECL void AND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void OR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void XOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void NAND(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void NOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void XNOR(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLXOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLNAND(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);
MICROSOFT_QUANTUM_DECL void CLXNOR(_In_ unsigned sid, _In_ bool ci, _In_ unsigned qi, _In_ unsigned qo);

MICROSOFT_QUANTUM_DECL void QFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c);
MICROSOFT_QUANTUM_DECL void IQFT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c);

MICROSOFT_QUANTUM_DECL void ADD(_In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void SUB(_In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void ADDS(_In_ unsigned sid, unsigned a, unsigned s, _In_ unsigned n, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void SUBS(_In_ unsigned sid, unsigned a, unsigned s, _In_ unsigned n, _In_reads_(n) unsigned* q);
MICROSOFT_QUANTUM_DECL void MUL(
    _In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void DIV(
    _In_ unsigned sid, unsigned a, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void MULN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void DIVN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void POWN(
    _In_ unsigned sid, unsigned a, unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);

MICROSOFT_QUANTUM_DECL void MCADD(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned nq, _In_reads_(nq) unsigned* q);
MICROSOFT_QUANTUM_DECL void MCSUB(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned nq, _In_reads_(nq) unsigned* q);
MICROSOFT_QUANTUM_DECL void MCMUL(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void MCDIV(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void MCMULN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void MCDIVN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);
MICROSOFT_QUANTUM_DECL void MCPOWN(_In_ unsigned sid, unsigned a, _In_ unsigned nc, _In_reads_(nc) unsigned* c,
    unsigned m, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_reads_(n) unsigned* o);

MICROSOFT_QUANTUM_DECL void LDA(_In_ unsigned sid, _In_ unsigned ni, _In_reads_(ni) unsigned* qi, _In_ unsigned nv,
    _In_reads_(nv) unsigned* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void ADC(_In_ unsigned sid, unsigned s, _In_ unsigned ni, _In_reads_(ni) unsigned* qi,
    _In_ unsigned nv, _In_reads_(nv) unsigned* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void SBC(_In_ unsigned sid, unsigned s, _In_ unsigned ni, _In_reads_(ni) unsigned* qi,
    _In_ unsigned nv, _In_reads_(nv) unsigned* qv, unsigned char* t);
MICROSOFT_QUANTUM_DECL void Hash(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, unsigned char* t);

MICROSOFT_QUANTUM_DECL bool TrySeparate1Qb(_In_ unsigned sid, _In_ unsigned qi1);
MICROSOFT_QUANTUM_DECL bool TrySeparate2Qb(_In_ unsigned sid, _In_ unsigned qi1, _In_ unsigned qi2);
MICROSOFT_QUANTUM_DECL bool TrySeparateTol(
    _In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ double tol);
MICROSOFT_QUANTUM_DECL void SetReactiveSeparate(_In_ unsigned sid, _In_ bool irs);

#if !(FPPOW < 6 && !ENABLE_COMPLEX_X2)
MICROSOFT_QUANTUM_DECL void TimeEvolve(_In_ unsigned sid, _In_ double t, _In_ unsigned n,
    _In_reads_(n) _QrackTimeEvolveOpHeader* teos, unsigned mn, _In_reads_(mn) double* mtrx);
#endif

// permutation oracle emulation
// MICROSOFT_QUANTUM_DECL void PermuteBasis(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_
// std::size_t table_size, _In_reads_(table_size) std::size_t *permutation_table);  MICROSOFT_QUANTUM_DECL void
// AdjPermuteBasis(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* q, _In_ std::size_t table_size,
// _In_reads_(table_size) std::size_t *permutation_table);
}
