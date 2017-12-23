//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano 2017. All rights reserved.
//
// This is a header-only, quick-and-dirty, multithreaded, universal quantum register
// simulation, allowing (nonphysical) register cloning and direct measurement of
// probability and phase, to leverage what advantages classical emulation of qubits
// can have.
//
// Licensed under the GNU General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/gpl-3.0.en.html
// for details.

#include <stdint.h>
#include <math.h>
#include <complex>
#include <random>
#include <stdexcept>
#include <atomic>
#include <thread>
#include <future>

#define Complex16 std::complex<double>
#define bitLenInt uint8_t
#define bitCapInt uint64_t
#define bitsInByte 8

#include "par_for.hpp"

namespace Qrack {
	/// The "Qrack::Register" class represents an independent quantum processor register		
	/** The "Qrack::Register" class represents an independent quantum processor register, including primitive bit logic gates and single register (abstract) opcodes. */
	class Register {
		public:
			///Initialize a register with qBitCount number of bits, all to |0> state.
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
			///Initialize a register with qBitCount number pf bits, to initState unsigned integer permutation state
			Register(bitLenInt qBitCount, bitCapInt initState) : rand_distribution(0.0, 1.0) {
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
			///PSEUDO-QUANTUM Initialize a cloned register with same exact quantum state as pqs
			Register(const Register& pqs) : rand_distribution(0.0, 1.0) {
				runningNorm = pqs.runningNorm;
				qubitCount = pqs.qubitCount;
				maxQPower = pqs.maxQPower;
				stateVec = new Complex16[maxQPower];
				std::copy(pqs.stateVec, pqs.stateVec + qubitCount, stateVec);
			}
			///Delete a register, with heap objects
			~Register() {
				delete [] stateVec;
			}
			///Get the count of bits in this register
			int GetQubitCount() {
				return qubitCount;
			}
			///PSEUDO-QUANTUM Output the exact quantum state of this register as a permutation basis array of complex numbers
			void CloneRawState(Complex16* output) {
				if (runningNorm != 1.0) NormalizeState();
				std::copy(stateVec, stateVec + qubitCount, output);
			}
			///Generate a random double from 0 to 1
			double Rand() {
				return rand_distribution(rand_generator);
			}
			///Set |0>/|1> bit basis pure quantum permutation state, as an unsigned int
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
			///Set arbitrary pure quantum state, in unsigned int permutation basis
			void SetQuantumState(Complex16* inputState) {
				std::copy(inputState, inputState + qubitCount, stateVec);
			}

			//Logic Gates:
			/// Doubly-controlled not
			void CCNOT(bitLenInt control1, bitLenInt control2, bitLenInt target) {
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
				//Complex16 b = Complex16(0.0, 0.0);
				par_for (0, maxQPower, stateVec, Complex16(1.0 / runningNorm, 0.0), pauliX, qPowers,
					[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
						if ((lcv & qPowers[0]) == 0) {
							Complex16 qubit[2];

							qubit[0] = stateVec[lcv + qPowers[1] + qPowers[2] + qPowers[3]];
							qubit[1] = stateVec[lcv + qPowers[1] + qPowers[2]];						

							Complex16 Y0 = qubit[0];
							qubit[0] = nrm * (mtrx[0] * Y0 + mtrx[1] * qubit[1]);
							qubit[1] = nrm * (mtrx[2] * Y0 + mtrx[3] * qubit[1]);

							stateVec[lcv + qPowers[1] + qPowers[2] + qPowers[3]] = qubit[0];
							stateVec[lcv + qPowers[1] + qPowers[2]] = qubit[1];
						}
					}
				);

				UpdateRunningNorm();
			}

			///Controlled not
			void CNOT(bitLenInt control, bitLenInt target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CNOT control bit cannot also be target.");

				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliX);
			}
			///Hadamard gate
			void H(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("H tried to operate on bit index greater than total bits.");
				if (runningNorm != 1.0) NormalizeState();

				const Complex16 had[4] = {
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(1.0 / M_SQRT2, 0.0),
					Complex16(1.0 / M_SQRT2, 0.0), Complex16(-1.0 / M_SQRT2, 0.0)
				};
				Apply2x2(qubitIndex, had);
			}
			///Measurement gate
			bool M(bitLenInt qubitIndex) {
				if (runningNorm != 1.0) NormalizeState();

				bool result;
				double prob = rand_distribution(rand_generator);
				double angle = rand_distribution(rand_generator) * 2.0 * M_PI;
				double cosine = cos(angle);
				double sine = sin(angle);

				bitCapInt qPowers[1];
				qPowers[0] = 1 << qubitIndex;
				double oneChance = Prob(qubitIndex);

				result = (prob < oneChance) && oneChance > 0.0;
				double nrmlzr = 1.0;
				bitCapInt lcv;
				if (result) {
					if (oneChance > 0.0) nrmlzr = oneChance;
					par_for (0, maxQPower, stateVec, Complex16(cosine / nrmlzr, sine), NULL, qPowers,
						[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
							if ((lcv & qPowers[0]) == 0) {
								stateVec[lcv] = Complex16(0.0, 0.0);
							}
							else {
								stateVec[lcv] = nrm * stateVec[lcv];
							}
						}
					);
					for (lcv = 0; lcv < maxQPower; lcv++) {
						
					}
				}
				else {
					if (oneChance < 1.0) nrmlzr = sqrt(1.0 - oneChance);
					par_for (0, maxQPower, stateVec, Complex16(cosine / nrmlzr, sine), NULL, qPowers,
						[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
							if ((lcv & qPowers[0]) == 0) {
								stateVec[lcv] = nrm * stateVec[lcv];
							}
							else {
								stateVec[lcv] = Complex16(0.0, 0.0);
							}
						}
					);
				}

				UpdateRunningNorm();

				return result;
			}
			///Measure for exact permutation of all register bits
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
			///PSEUDO-QUANTUM Direct measure of bit probability to be in |1> state
			double Prob(bitLenInt qubitIndex) {
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
			///PSEUDO-QUANTUM Direct measure of full register probability to be in permutation state
			double ProbAll(bitCapInt fullRegister) {
				if (runningNorm != 1.0) NormalizeState();

				return normSqrd(stateVec + fullRegister);
			}
			///PSEUDO-QUANTUM Direct measure of all bit probabilities in register to be in |1> state
			void ProbArray(double* probArray) {
				if (runningNorm != 1.0) NormalizeState();

				bitCapInt lcv;
				for (lcv = 0; lcv < maxQPower; lcv++) {
					probArray[lcv] = normSqrd(stateVec + lcv); 
				}
			}
			///"Phase shift gate" - Rotates as e^(-i*\theta) around |1> state 
			void R1(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians);
				double sine = sin(radians); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				Apply2x2(qubitIndex, mtrx);
			}
			///Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state
			/** Dyadic fraction "phase shift gate" - Rotates as e^(i*(M_PI * numerator) / denominator) around |1> state. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void R1Dyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				R1((M_PI * numerator) / denominator, qubitIndex);
			}
			///x axis rotation gate - Rotates as e^(-i*\theta) around Pauli x axis 
			void RX(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				Apply2x2(qubitIndex, pauliRX);
			}
			///Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis
			/** Dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void RXDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RX((-M_PI * numerator) / denominator, qubitIndex);
			}
			///y axis rotation gate - Rotates as e^(-i*\theta) around Pauli y axis 
			void RY(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				Apply2x2(qubitIndex, pauliRY);
			}
			///Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void RYDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RY((-M_PI * numerator) / denominator, qubitIndex);
			}
			///z axis rotation gate - Rotates as e^(-i*\theta) around Pauli z axis 
			void RZ(double radians, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				Apply2x2(qubitIndex, pauliRZ);
			}
			///Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void RZDyad(int numerator, int denominator, bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				RZ((-M_PI * numerator) / denominator, qubitIndex);
			}
			///Set individual bit to pure |0> (false) or |1> (true) state
			void SetBit(bitLenInt qubitIndex1, bool value) {
				if (value != M(qubitIndex1)) {
					X(qubitIndex1);
				}
			}
			///Swap values of two bits in register
			void Swap(bitLenInt qubitIndex1, bitLenInt qubitIndex2) {
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
				//Complex16 b = Complex16(0.0, 0.0);
				par_for (0, maxQPower, stateVec, Complex16(1.0 / runningNorm, 0.0), pauliX, qPowers,
					[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
						if ((lcv & qPowers[0]) == 0) {
							Complex16 qubit[2];

							qubit[0] = stateVec[lcv + qPowers[2]];
							qubit[1] = stateVec[lcv + qPowers[1]];						

							Complex16 Y0 = qubit[0];
							qubit[0] = nrm * (mtrx[0] * Y0 + mtrx[1] * qubit[1]);
							qubit[1] = nrm * (mtrx[2] * Y0 + mtrx[3] * qubit[1]);

							stateVec[lcv + qPowers[2]] = qubit[0];
							stateVec[lcv + qPowers[1]] = qubit[1];
						}
					}
				);

				UpdateRunningNorm();
			}
			///NOT gate, which is also Pauli x matrix
			void X(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				const Complex16 pauliX[4] = {
					Complex16(0.0, 0.0), Complex16(1.0, 0.0),
					Complex16(1.0, 0.0), Complex16(0.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliX);
			}
			///Apply NOT gate, (which is Pauli x matrix,) to each bit in register
			void XAll() {
				bitLenInt lcv;
				for (lcv = 0; lcv < qubitCount; lcv++) {
					X(lcv);
				}

				UpdateRunningNorm();
			}
			///Apply Pauli Y matrix to bit
			void Y(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliY);
			}
			///Apply Pauli Z matrix to bit
			void Z(bitLenInt qubitIndex) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				Apply2x2(qubitIndex, pauliZ);
			}
			///Controlled "phase shift gate"
			/** Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta) around |1> state */
			void CR1(double radians, bitLenInt control, bitLenInt target) {
				//if ((control >= qubitCount) || (target >= qubitCount))
				//	throw std::invalid_argument("CNOT tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CR1 control bit cannot also be target.");
				double cosine = cos(radians);
				double sine = sin(radians); 
				const Complex16 mtrx[4] = {
					Complex16(1.0, 0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(control, target, mtrx);
			}
			///Controlled dyadic fraction "phase shift gate"
			/** Controlled "phase shift gate" - if control bit is true, rotates target bit as e^(-i*\theta) around |1> state */
			void CR1Dyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CR1Dyad control bit cannot also be target.");
				CR1((-M_PI * numerator) / denominator, control, target);
			}
			///Controlled x axis rotation
			/** Controlled x axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli x axis */
			void CRX(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("X tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRX control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRX[4] = {
					Complex16(cosine, 0.0), Complex16(0.0, -sine),
					Complex16(0.0, -sine), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(control, target, pauliRX);
			}
			///Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis
			/** Controlled dyadic fraction x axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli x axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRXDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRXDyad control bit cannot also be target.");
				CRX((-M_PI * numerator) / denominator, control, target);
			}
			///Controlled y axis rotation
			/** Controlled y axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli y axis */
			void CRY(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRY control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				Complex16 pauliRY[4] = {
					Complex16(cosine, 0.0), Complex16(-sine, 0.0),
					Complex16(sine, 0.0), Complex16(cosine, 0.0)
				};
				ApplyControlled2x2(control, target, pauliRY);
			}
			///Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis
			/** Controlled dyadic fraction y axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli y axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRYDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRYDyad control bit cannot also be target.");
				CRY((-M_PI * numerator) / denominator, control, target);
			}
			///Controlled z axis rotation
			/** Controlled z axis rotation - if control bit is true, rotates as e^(-i*\theta) around Pauli z axis */
			void CRZ(double radians, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRZ control bit cannot also be target.");
				double cosine = cos(radians / 2.0);
				double sine = sin(radians / 2.0); 
				const Complex16 pauliRZ[4] = {
					Complex16(cosine, -sine), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(cosine, sine)
				};
				ApplyControlled2x2(control, target, pauliRZ);
			}
			///Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis
			/** Controlled dyadic fraction z axis rotation gate - Rotates as e^(i*(M_PI * numerator) / denominator) around Pauli z axis. NOTE THAT DYADIC OPERATION ANGLE SIGN IS REVERSED FROM RADIAN ROTATION OPERATORS. */
			void CRZDyad(int numerator, int denominator, bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CRZDyad control bit cannot also be target.");
				CRZ((-M_PI * numerator) / denominator, control, target);
			}
			///Apply controlled Pauli Y matrix to bit
			void CY(bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Y tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CY control bit cannot also be target.");
				const Complex16 pauliY[4] = {
					Complex16(0.0, 0.0), Complex16(0.0, -1.0),
					Complex16(0.0, 1.0), Complex16(0.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliY);
			}
			///Apply controlled Pauli Z matrix to bit
			void CZ(bitLenInt control, bitLenInt target) {
				//if (qubitIndex >= qubitCount) throw std::invalid_argument("Z tried to operate on bit index greater than total bits.");
				if (control == target) throw std::invalid_argument("CZ control bit cannot also be target.");
				const Complex16 pauliZ[4] = {
					Complex16(1.0, 0.0), Complex16(0.0, 0.0),
					Complex16(0.0, 0.0), Complex16(-1.0, 0.0)
				};
				ApplyControlled2x2(control, target, pauliZ);
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
						Reverse(1, qubitCount - 1);
						Reverse(1, shift);
						Reverse(shift + 1, qubitCount - 1);

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
						Reverse(shift + 1, qubitCount - 1);
						Reverse(1, shift);
						Reverse(1, qubitCount - 1);

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
					Reverse(0, qubitCount - 1);
					Reverse(0, shift - 1);
					Reverse(shift, qubitCount - 1);
				}
			}
			/// "Circular shift right" - shift bits right, and carry first bits.
			void ROR(bitLenInt shift) {
				shift = shift % qubitCount;
				if (shift > 0) {
					Reverse(shift, qubitCount - 1);
					Reverse(0, shift - 1);
					Reverse(0, qubitCount - 1);
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
			bitLenInt qubitCount;
			bitCapInt maxQPower;
			Complex16* stateVec;

			std::default_random_engine rand_generator;
			std::uniform_real_distribution<double> rand_distribution;

			void Apply2x2(bitLenInt qubitIndex, const Complex16* mtrx) {
				bitCapInt qPowers[1];
				qPowers[0] = 1 << qubitIndex;
				//Complex16 b = Complex16(0.0, 0.0);
				par_for (0, maxQPower, stateVec, Complex16(1.0 / runningNorm, 0.0), mtrx, qPowers,
					[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
						if ((lcv & qPowers[0]) == 0) {
							Complex16 qubit[2];

							qubit[0] = stateVec[lcv + qPowers[0]];
							qubit[1] = stateVec[lcv];						

							Complex16 Y0 = qubit[0];
							qubit[0] = nrm * (mtrx[0] * Y0 + mtrx[1] * qubit[1]);
							qubit[1] = nrm * (mtrx[2] * Y0 + mtrx[3] * qubit[1]);

							stateVec[lcv + qPowers[0]] = qubit[0];
							stateVec[lcv] = qubit[1];
						}
					}
				);

				UpdateRunningNorm();
			}

			void ApplyControlled2x2(bitLenInt control, bitLenInt target, const Complex16* mtrx) {
				bitCapInt qPowers[3];
				qPowers[1] = 1 << control;
				qPowers[2] = 1 << target;
				qPowers[0] = qPowers[1] + qPowers[2];
				//Complex16 b = Complex16(0.0, 0.0);
				par_for (0, maxQPower, stateVec, Complex16(1.0 / runningNorm, 0.0), mtrx, qPowers,
					[](const int lcv, const int cpu, Complex16* stateVec, const Complex16 nrm, const Complex16* mtrx, const bitCapInt* qPowers) {
						if ((lcv & qPowers[0]) == 0) {
							Complex16 qubit[2];

							qubit[0] = stateVec[lcv + qPowers[1] + qPowers[2]];
							qubit[1] = stateVec[lcv + qPowers[1]];						

							Complex16 Y0 = qubit[0];
							qubit[0] = nrm * (mtrx[0] * Y0 + mtrx[1] * qubit[1]);
							qubit[1] = nrm * (mtrx[2] * Y0 + mtrx[3] * qubit[1]);

							stateVec[lcv + qPowers[1] + qPowers[2]] = qubit[0];
							stateVec[lcv + qPowers[1]] = qubit[1];
						}
					}
				);

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
				bitLenInt i;
				bitLenInt iter = start + (end - start - 1) / 2;
				for (i = start; i <= iter; i++) {
					Swap(i, end - i + start);
				}
			}
	};
}
