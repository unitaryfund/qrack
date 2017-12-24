//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have. (This is a sequential implementation, for reference.)
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <complex>
#include <random>
#include <stdexcept>

#define Complex16 std::complex<double>
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

namespace Qrack {
	class Register {
		public:
			
			Register(bitLenInt qBitCount) : rand_distribution(0.0, 1.0) {
				if (qBitCount > (sizeof(bitCapInt) * bitsInByte))
					throw std::invalid_argument("Cannot instantiate a register with greater capacity than native types on emulating system.");
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				stateVec = new Complex16[maxQPower];
				bitCapInt lcv;
				stateVec[0] = Complex16(cosine, sine);
				for (lcv = 1; lcv < maxQPower; lcv++) {
					stateVec[lcv] = Complex16(0.0, 0.0);
				}
			}
			Register(unsigned int qBitCount, bitCapInt initState) : rand_distribution(0.0, 1.0) {
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				runningNorm = 1.0;
				qubitCount = qBitCount;
				maxQPower = 1<<qBitCount;
				stateVec = new Complex16[maxQPower];
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if (lcv == initState) {
						stateVec[lcv] = Complex16(cosine, sine);
					}	
					else {
						stateVec[lcv] = Complex16(0.0, 0.0);
					}
				}
			}
			Register(const Register& pqs) : rand_distribution(0.0, 1.0) {
				runningNorm = pqs.runningNorm;
				qubitCount = pqs.qubitCount;
				maxQPower = pqs.maxQPower;
				stateVec = new Complex16[maxQPower];
				std::copy(pqs.stateVec, pqs.stateVec + qubitCount, stateVec);
			}
			~Register() {
				delete [] stateVec;
			}

			int GetQubitCount() {
				return qubitCount;
			}
			void CloneRawState(Complex16* output) {
				if (runningNorm != 1.0) NormalizeState();
				std::copy(stateVec, stateVec + qubitCount, output);
			}
			double Rand() {
				return rand_distribution(rand_generator);
			}
			void SetPermutation(bitCapInt perm) {
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				runningNorm = 1.0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if (lcv == perm) {
						stateVec[lcv] = Complex16(cosine, sine);
					}	
					else {
						stateVec[lcv] = Complex16(0.0, 0.0);
					}
				}
			}
			void SetQuantumState(Complex16* inputState) {
				std::copy(inputState, inputState + qubitCount, stateVec);
			}

			//Logic Gates:
			void CCNOT(unsigned int control1, unsigned int control2, unsigned int target) {
				//if ((control1 >= qubitCount) || (control2 >= qubitCount))
				//	throw std::invalid_argument("CCNOT tried to operate on bit index greater than total bits.");
				if (control1 == control2) throw std::invalid_argument("CCNOT control bits cannot be same bit.");
				if (control1 == target || control2 == target)
					throw std::invalid_argument("CCNOT control bits cannot also be target.");

				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};

				bitCapInt qPowers[4];
				qPowers[1] = 1 << control1;
				qPowers[2] = 1 << control2;
				qPowers[3] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2] + qPowers[3];
				Complex16 qubit[2];
				//Complex16 b = Complex16(0.0, 0.0);
				double sqrNorm = 0.0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPowers[0]) == 0) {
						qubit[0] = stateVec[lcv + qPowers[1] + qPowers[2] + qPowers[3]];
						qubit[1] = stateVec[lcv + qPowers[1] + qPowers[2]];						

						//cblas_zhemv(CblasRowMajor, CblasUpper, 2, &nrmlzr, pauliX, 2, qubit, 1, &b, qubit, 1);		
						zmv2x2(Complex16(1.0 / runningNorm, 0.0), pauliX, qubit);

						stateVec[lcv + qPowers[1] + qPowers[2] + qPowers[3]] = qubit[0];
						stateVec[lcv + qPowers[1] + qPowers[2]] = qubit[1];
					}
				}

				UpdateRunningNorm();
			}
			void CNOT(unsigned int control, unsigned int target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CNOT control bit cannot also be target.");

				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliX);
			}
			void H(unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("H tried to operate on bit index greater than total bits.");
				if (runningNorm != 1.0) NormalizeState();

				const Complex16 had[4] = {
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0)
				};
				Apply2x2(qubitIndex, had);
			}
			bool M(bitLenInt qubitIndex) {
				if (runningNorm != 1.0) NormalizeState();

				bool result;
				double prob = rand_distribution(rand_generator);
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				bitCapInt qPower = 1 << qubitIndex;
				double oneChance = Prob(qubitIndex);

				result = (prob < oneChance) && oneChance > 0.0;
				double nrmlzr = 1.0;
				bitCapInt lcv;
				if (result) {
					if (oneChance > 0.0) nrmlzr = oneChance;
					for (lcv = 0; lcv < maxQPower; lcv++) {
						if ((lcv & qPower) == 0) {
							stateVec[lcv] = Complex16(0.0, 0.0);
						}
						else {
							stateVec[lcv] = Complex16(
								cosine * real(stateVec[lcv]) - sine * imag(stateVec[lcv]),
								sine * real(stateVec[lcv]) + cosine * imag(stateVec[lcv])
							) / nrmlzr;
						}
					}
				}
				else {
					if (oneChance < 1.0) nrmlzr = sqrt(1.0 - oneChance);
					for (lcv = 0; lcv < maxQPower; lcv++) {
						if ((lcv & qPower) == 0) {
							stateVec[lcv] = Complex16(
								cosine * real(stateVec[lcv]) - sine * imag(stateVec[lcv]),
								sine * real(stateVec[lcv]) + cosine * imag(stateVec[lcv])
							) / nrmlzr;
						}
						else {
							stateVec[lcv] = Complex16(0.0, 0.0);
						}
					}
				}

				UpdateRunningNorm();

				return result;
			}
			bool MAll(bitCapInt fullRegister) {
				bool result;
				double prob = rand_distribution(rand_generator);
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				Complex16 toTest = stateVec[fullRegister];
				double oneChance = real(toTest) * real(toTest) + imag(toTest) * imag(toTest);
				result = (prob < oneChance);

				double nrmlzr;
				bitCapInt lcv;
				bitCapInt maxPower = 1 << qubitCount;
				if (result) {
					for (lcv = 0; lcv < maxPower; lcv++) {
						if (lcv == fullRegister) {
							stateVec[lcv] = Complex16(cosine, sine);
						}
						else {
							stateVec[lcv] = Complex16(0.0, 0.0);
						}
					}
				}
				else {
					nrmlzr = sqrt(1.0 - oneChance);
					for (lcv = 0; lcv < maxPower; lcv++) {
						if (lcv == fullRegister) {
							stateVec[lcv] = Complex16(0.0, 0.0);
						}
						else {
							stateVec[lcv] = Complex16(
								cosine * real(stateVec[lcv]) - sine * imag(stateVec[lcv]),
								sine * real(stateVec[lcv]) + cosine * imag(stateVec[lcv])
							) / nrmlzr;
						}
					}
				}

				return result;
			}
			double Prob(unsigned int qubitIndex) {
				if (runningNorm != 1.0) NormalizeState();

				bitCapInt qPower = 1 << qubitIndex;
				double oneChance = 0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPower) == qPower) {
						oneChance += normSqrd(stateVec + lcv);
					} 
				}

				return oneChance;
			}
			double ProbAll(bitCapInt fullRegister) {
				if (runningNorm != 1.0) NormalizeState();

				return normSqrd(stateVec + fullRegister);
			}
			void ProbArray(double* probArray) {
				if (runningNorm != 1.0) NormalizeState();

				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					probArray[lcv] = normSqrd(stateVec + lcv); 
				}
			}
			void R1(double radians, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians);
				double sine = sin(radians); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				Apply2x2(qubitIndex, mtrx);
			}
			void R1Dyad(int numerator, int denominator, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				R1((M_PI * numerator) / denominator, qubitIndex);
			}
			void RX(double radians, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				Apply2x2(qubitIndex, pauliRX);
			}
			void RXDyad(int numerator, int denominator, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RX((-M_PI * numerator) / denominator, qubitIndex);
			}
			void RY(double radians, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				Apply2x2(qubitIndex, pauliRY);
			}
			void RYDyad(int numerator, int denominator, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RY((-M_PI * numerator) / denominator, qubitIndex);
			}
			void RZ(double radians, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				Apply2x2(qubitIndex, pauliRZ);
			}
			void RZDyad(int numerator, int denominator, unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RZ((-M_PI * numerator) / denominator, qubitIndex);
			}
			void SetBit(bitLenInt qubitIndex1, bool value) {
				if (value != M(qubitIndex1)) {
					X(qubitIndex1);
				}
			}
			void Swap(unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if ((qubitIndex1 >= qubitCount) || (qubitIndex2 >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("Swap bits cannot be the same bit.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};

				bitCapInt qPowers[3];
				qPowers[1] = 1 << qubitIndex1;
				qPowers[2] = 1 << qubitIndex2;
				qPowers[0] = qPowers[1] + qPowers[2];
				Complex16 qubit[2];
				//Complex16 b = Complex16(0.0, 0.0);
				double sqrNorm = 0.0;
				unsigned int lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPowers[0]) == 0) { 
						qubit[0] = stateVec[lcv + qPowers[2]];
						qubit[1] = stateVec[lcv + qPowers[1]];

						//cblas_zhemv(CblasRowMajor, CblasUpper, 2, &a, pauliX, 2, qubit, 1, &b, qubit, 1);
						zmv2x2(Complex16(1.0 / runningNorm, 0.0), pauliX, qubit);

						stateVec[lcv + qPowers[2]] = qubit[0];
						stateVec[lcv + qPowers[1]] = qubit[1];
					}
				}

				UpdateRunningNorm();
			}
			void X(unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliX);
			}
			void XAll() {
				unsigned int lcv;
				for (lcv = 0; lcv < qubitCount; lcv++) {
					X(lcv);
				}

				UpdateRunningNorm();
			}
			void Y(unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliY);
			}
			void Z(unsigned int qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliZ);
			}
			void CR1(double radians, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if ((qubitIndex1 >= qubitCount) || (qubitIndex2 >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CR1 control bit cannot also be target.");
				double cosine = cos(radians);
				double sine = sin(radians); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, mtrx);
			}
			void CR1Dyad(int numerator, int denominator, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CR1Dyad control bit cannot also be target.");
				CR1((-M_PI * numerator) / denominator, qubitIndex1, qubitIndex2);
			}
			void CRX(double radians, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRX control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, pauliRX);
			}
			void CRXDyad(int numerator, int denominator, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRXDyad control bit cannot also be target.");
				CRX((-M_PI * numerator) / denominator, qubitIndex1, qubitIndex2);
			}
			void CRY(double radians, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRY control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, pauliRY);
			}
			void CRYDyad(int numerator, int denominator, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRYDyad control bit cannot also be target.");
				CRY((-M_PI * numerator) / denominator, qubitIndex1, qubitIndex2);
			}
			void CRZ(double radians, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRZ control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, pauliRZ);
			}
			void CRZDyad(int numerator, int denominator, unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CRZDyad control bit cannot also be target.");
				CRZ((-M_PI * numerator) / denominator, qubitIndex1, qubitIndex2);
			}
			void CY(unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CY control bit cannot also be target.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, pauliY);
			}
			void CZ(unsigned int qubitIndex1, unsigned int qubitIndex2) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (qubitIndex1 == qubitIndex2) throw std::invalid_argument("CZ control bit cannot also be target.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				ApplyControlled2x2(qubitIndex1, qubitIndex2, pauliZ);
			}

			//Single register instructions:

			///Arithmetic shift left, with index 0 bit as sign bit
			void ASL(bitLenInt shift) {
				if (shift > 0) {
					int i;
					if (shift >= (qubitCount - 1)) {
						for (i = 1; i < qubitCount; i++) {
							SetBit(i, 0);
						}
					}
					else {
						Reverse(1, qubitCount);
						Reverse(1, shift);
						Reverse(shift + 1, qubitCount);

						for (i = 0; i < shift; i++) {
							SetBit(i + 1, false);
						}
					}
				}
			}
			///Arithmetic shift right, with index 0 bit as sign bit
			void ASR(bitLenInt shift) {
				if (shift > 0) {
					int i;
					if (shift >= (qubitCount - 1)) {	
						for (i = 1; i < qubitCount; i++) {
							SetBit(i, 0);
						}
					}
					else {
						Reverse(shift + 1, qubitCount);
						Reverse(1, shift);
						Reverse(1, qubitCount);

						for (i = 0; i < shift; i++) {
							SetBit(qubitCount - 1 - i, false);
						}
					}
				}
			}
			///Logical shift left, filling the extra bits with |0>
			void LSL(bitLenInt shift) {
				if (shift > 0) {
					if (shift >= qubitCount) {	
						SetPermutation(0);
					}
					else {
						int i;
						ROL(shift);
						for (i = 0; i < shift; i++) {
							SetBit(i, false);
						}
					}
				}
			}
			///Logical shift right, filling the extra bits with |0>
			void LSR(bitLenInt shift) {
				if (shift > 0) {
					if (shift >= qubitCount) {	
						SetPermutation(0);
					}
					else {
						int i;
						ROR(shift);
						for (i = 0; i < shift; i++) {
							SetBit(qubitCount - 1 - i, false);
						}
					}
				}
			}
			/// "Circular shift left" - shift bits left, and carry last bits.
			void ROL(bitLenInt shift) {
				shift = shift % qubitCount;
				if (shift > 0) {
					Reverse(0, qubitCount);
					Reverse(0, shift - 1);
					Reverse(shift, qubitCount);
				}
			}
			/// "Circular shift right" - shift bits right, and carry first bits.
			void ROR(bitLenInt shift) {
				shift = shift % qubitCount;
				if (shift > 0) {
					Reverse(shift, qubitCount);
					Reverse(0, shift - 1);
					Reverse(0, qubitCount);
				}
			}
			///"Circular rotate left" permutation
			/** "Circular rotate left" permutation - Adds argument to permutation state, carrying highest permutations to lowest. */
			void RADD(bitCapInt toAdd) {
				std::rotate(stateVec, stateVec + maxQPower - toAdd, stateVec + maxQPower); 
			}
			///"Circular rotate right" permutation
			/** "Circular rotate right" permutation - Subtracts argument from permutation state, carrying lowest permutations to highest. */
			void RSUB(bitCapInt toSub) {
				std::rotate(stateVec, stateVec + toSub, stateVec + maxQPower); 
			}
			///Subtract (with sign, carry overflow to maximum positive)
			void SADD(bitCapInt toAdd) {
				RotateComplex(1, maxQPower, toAdd, false, 2, stateVec);
				RotateComplex(0, maxQPower, toAdd, true, 2, stateVec);				
			}
			///Add (with sign, carry overflow to minimum negative)
			void SSUB(bitCapInt toSub) {
				RotateComplex(0, maxQPower, toSub, false, 2, stateVec);
				RotateComplex(1, maxQPower, toSub, true, 2, stateVec);
			}
			///Add (without sign)
			void UADD(bitCapInt toAdd) {
				RADD(toAdd);

				bitCapInt lcv;
				for (lcv = 0; lcv < toAdd; lcv++) {
					runningNorm -= normSqrd(stateVec + lcv);
					stateVec[lcv] = Complex16(0.0, 0.0);
				}
			}
			///Subtract (without sign)
			void USUB(bitCapInt toSub) {
				RSUB(toSub);

				bitCapInt lcv;
				for (lcv = 0; lcv < toSub; lcv++) {
					runningNorm -= normSqrd(stateVec + maxQPower - lcv - 1);
					stateVec[maxQPower - lcv - 1] = Complex16(0.0, 0.0);
				}
			}
			/// Quantum Fourier Transform - Apply the quantum Fourier transform to the register
			void QFT() {
				int i, j;
				for (i = 0; i < qubitCount; i++) {
					H(i);
					for (j = 1; j < qubitCount - i; j++) {
						CR1Dyad(1, 1<<j, i + j, i); 
					}
				}
			}

		private:
			double runningNorm;
			unsigned int qubitCount;
			unsigned int maxQPower;
			Complex16* stateVec;

			std::default_random_engine rand_generator;
			std::uniform_real_distribution<double> rand_distribution;

			void Apply2x2(unsigned int qubitIndex, const Complex16* mtrx) {
				bitCapInt qPower = 1 << qubitIndex;
				Complex16 qubit[2];
				//Complex16 b = Complex16(0.0, 0.0);
				double sqrNorm = 0.0;
				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPower) == 0) {
						qubit[0] = stateVec[lcv + qPower];
						qubit[1] = stateVec[lcv];						

						//cblas_zhemv(CblasRowMajor, CblasUpper, 2, &a, mtrx, 2, qubit, 1, &b, qubit, 1);
						zmv2x2(Complex16(1.0 / runningNorm, 0.0), mtrx, qubit);		

						stateVec[lcv + qPower] = qubit[0];
						stateVec[lcv] = qubit[1];
					}
				}

				UpdateRunningNorm();
			}

			void ApplyControlled2x2(unsigned int qubitIndex1, unsigned int qubitIndex2, const Complex16* mtrx) {
				bitCapInt qPowers[3];
				qPowers[1] = 1 << qubitIndex1;
				qPowers[2] = 1 << qubitIndex2;
				qPowers[0] = qPowers[1] + qPowers[2];
				Complex16 qubit[2];
				//Complex16 b = Complex16(0.0, 0.0);
				double sqrNorm = 0.0;
				unsigned int lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					if ((lcv & qPowers[0]) == 0) {
						qubit[0] = stateVec[lcv + qPowers[2] + qPowers[1]];
						qubit[1] = stateVec[lcv + qPowers[1]];
						
						//cblas_zhemv(CblasRowMajor, CblasUpper, 2, &nrmlzr, pauliX, 2, qubit, 1, &b, qubit, 1);
						zmv2x2(Complex16(1.0 / runningNorm, 0.0), mtrx, qubit);

						stateVec[lcv + qPowers[2] + qPowers[1]] = qubit[0];
						stateVec[lcv + qPowers[1]] = qubit[1];						
					}
				}

				UpdateRunningNorm();
			}

			void NormalizeState() {
				long int lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					stateVec[lcv] /= runningNorm;
				}
				runningNorm = 1.0;
			}

			double normSqrd(Complex16* cmplx) {
				return real(*cmplx) * real(*cmplx) + imag(*cmplx) * imag(*cmplx);
			}

			void UpdateRunningNorm() {
				long int lcv;
				double sqrNorm = 0.0;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					sqrNorm += normSqrd(stateVec + lcv);
				}

				runningNorm = sqrt(sqrNorm);
			}

			void Reverse(bitLenInt start, bitLenInt end) {
				if (start + 1 < end) {
					end -= 1;
					bitLenInt i;
					bitLenInt iter = start + (end - start - 1) / 2;
					for (i = start; i <= iter; i++) {
						Swap(i, end - i + start);
					}
				}
			}

			void ReverseComplex(bitCapInt start, bitCapInt end, bitCapInt stride, Complex16* vec) {
				if (start + 1 < end) {
					end -= 1;
					Complex16 temp;
					bitCapInt i;
					bitCapInt iter = start + (end - start - 1) / 2;
					for (i = start; i <= iter; i+=stride) {
						temp = vec[i];
						vec[i] = vec[end - i + start];
						vec[end - i + start] = temp;
					}
				}
			}

			void RotateComplex(bitCapInt start, bitCapInt end, bitCapInt shift, bool leftRot, bitCapInt stride, Complex16* vec) {
				shift *= stride;
				if (shift > 0) {
					if (leftRot) {
						ReverseComplex(start, end, stride, vec);
						ReverseComplex(start, start + shift - stride, stride, vec);
						ReverseComplex(start + shift, end, stride, vec);
					}
					else {
						ReverseComplex(start + shift, end, stride, vec);
						ReverseComplex(start, start + shift - stride, stride, vec);
						ReverseComplex(start, end, stride, vec);
					}
				}
			}

			void zmv2x2(const Complex16 alpha, const Complex16* A, Complex16* Y) {
				Complex16 Y0 = Y[0];
				Y[0] = alpha * (A[0] * Y0 + A[1] * Y[1]);
				Y[1] = alpha * (A[2] * Y0 + A[3] * Y[1]);
			}
	};
}
