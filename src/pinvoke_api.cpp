//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2020. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "pinvoke_api.hpp"

// for details.

#include <map>
#include <vector>

// "qfactory.hpp" pulls in all headers needed to create any type of "Qrack::QInterface."
#include "qfactory.hpp"

using namespace Qrack;

enum Pauli
{
	/// Pauli Identity operator. Corresponds to Q# constant "PauliI."
	PauliI = 0U,
	/// Pauli X operator. Corresponds to Q# constant "PauliX."
	PauliX = 1U,
	/// Pauli Y operator. Corresponds to Q# constant "PauliY."
	PauliY = 3U,
	/// Pauli Z operator. Corresponds to Q# constant "PauliZ."
	PauliZ = 2U
};

qrack_rand_gen_ptr rng = std::make_shared<qrack_rand_gen>(std::time(0));
std::vector<QInterfacePtr> simulators;
std::map<QInterfacePtr, std::map<unsigned, bitLenInt>> shards;

void mul2x2(const complex& scalar, const complex* inMtrx, complex* outMtrx) {
	for (unsigned i = 0; i < 4; i++) {
		outMtrx[i] = scalar * inMtrx[i];
	}
}

void TransformPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds) {
	for (unsigned i = 0; i < len; i++) {
		switch (bases[i]) {
		case PauliX:
			simulator->H(shards[simulator][qubitIds[i]]);
			break;
		case PauliY:
			simulator->Z(shards[simulator][qubitIds[i]]);
			simulator->S(shards[simulator][qubitIds[i]]);
			simulator->H(shards[simulator][qubitIds[i]]);
			break;
		case PauliZ:
		case PauliI:
		default:
			break;
		}
	}
}

void RevertPauliBasis(QInterfacePtr simulator, unsigned len, unsigned* bases, unsigned* qubitIds) {
	for (unsigned i = 0; i < len; i++) {
		switch (bases[i]) {
		case PauliX:
			simulator->H(shards[simulator][qubitIds[i]]);
			break;
		case PauliY:
			simulator->H(shards[simulator][qubitIds[i]]);
			simulator->IS(shards[simulator][qubitIds[i]]);
			simulator->Z(shards[simulator][qubitIds[i]]);
			break;
		case PauliZ:
		case PauliI:
		default:
			break;
		}
	}
}

extern "C" {

/**
* (External API) Initialize a simulator ID with 0 qubits
*/
MICROSOFT_QUANTUM_DECL unsigned init()
{
	simulators.push_back(NULL);
	return simulators.size() - 1U;
}

/**
* (External API) Destroy a simulator (ID will not be reused)
*/
MICROSOFT_QUANTUM_DECL void destroy(_In_ unsigned sid)
{
	simulators[sid] = NULL;
}

/**
* (External API) Set RNG seed for simulator ID
*/
MICROSOFT_QUANTUM_DECL void seed(_In_ unsigned sid, _In_ unsigned s)
{
	if (simulators[sid] != NULL) {
		simulators[sid]->SetRandomSeed(s);
	}
}


/**
	* (External API) "Dump" all IDs from the selected simulator ID into the callback
	*/
MICROSOFT_QUANTUM_DECL void DumpIds(_In_ unsigned sid, _In_ void(*callback)(unsigned))
{
	QInterfacePtr simulator = simulators[sid];
	std::map<unsigned, bitLenInt>::iterator it;

	for (it = shards[simulator].begin(); it != shards[simulator].end(); it++) {
		callback(it->first);
	}
}

/**
	* (External API) Select from a distribution of "n" elements according the discrete probabilities in "d."
	*/
MICROSOFT_QUANTUM_DECL std::size_t random_choice(_In_ unsigned sid, _In_ std::size_t n, _In_reads_(n) double* p)
{
	std::discrete_distribution<std::size_t> dist(p, p + n);
	return dist(*rng.get());
}

/**
	* (External API) Find the joint probability for all specified qubits under the respective Pauli basis transformations.
	*/
MICROSOFT_QUANTUM_DECL double JointEnsembleProbability(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
	QInterfacePtr simulator = simulators[sid];
	double jointProb = 1.0;

	TransformPauliBasis(simulator, n, b, q);

	for (unsigned i = 0; i < n; i++) {
		jointProb *= (double)simulator->Prob(shards[simulator][q[i]]);
		if (jointProb == 0.0) {
			break;
		}
	}

	RevertPauliBasis(simulator, n, b, q);

	return jointProb;
}

/**
	* (External API) Allocate 1 new qubit with the given qubit ID, under the simulator ID
	*/
MICROSOFT_QUANTUM_DECL void allocateQubit(_In_ unsigned sid, _In_ unsigned qid)
{
	QInterfacePtr nQubit = CreateQuantumInterface(QINTERFACE_QUNIT, QINTERFACE_QFUSION, QINTERFACE_OPTIMAL, 1, 0, rng);
	if (simulators[sid] == NULL) {
		simulators[sid] = nQubit;
	}
	else {
		simulators[sid]->Compose(nQubit);
	}
	shards[simulators[sid]][qid] = simulators[sid]->GetQubitCount();
}

/**
	* (External API) Release 1 qubit with the given qubit ID, under the simulator ID
	*/
MICROSOFT_QUANTUM_DECL void release(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];

	if (simulator->GetQubitCount() == 1U) {
		simulators[sid] = NULL;
		shards.erase(simulator);
	}
	else {
		bitLenInt oIndex = shards[simulator][q];
		simulator->Dispose(oIndex, 1U);
		for (unsigned i = 0; i < shards[simulator].size(); i++) {
			if (shards[simulator][i] > oIndex) {
				shards[simulator][i]--;
			}
		}
		shards[simulator].erase(q);
	}
}

MICROSOFT_QUANTUM_DECL unsigned num_qubits(_In_ unsigned sid)
{
	return (unsigned)simulators[sid]->GetQubitCount();
}

/**
	* (External API) "X" Gate
	*/
MICROSOFT_QUANTUM_DECL void X(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->X(shards[simulator][q]);
}

/**
	* (External API) "Y" Gate
	*/
MICROSOFT_QUANTUM_DECL void Y(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->Y(shards[simulator][q]);
}

/**
	* (External API) "Z" Gate
	*/
MICROSOFT_QUANTUM_DECL void Z(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->Z(shards[simulator][q]);
}

/**
	* (External API) Walsh-Hadamard transform applied for simulator ID and qubit ID
	*/
MICROSOFT_QUANTUM_DECL void H(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->H(shards[simulator][q]);
}

/**
	* (External API) "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void S(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->S(shards[simulator][q]);
}

/**
	* (External API) "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void T(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->T(shards[simulator][q]);
}

/**
	* (External API) Inverse "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void AdjS(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->IS(shards[simulator][q]);
}

/**
	* (External API) Inverse "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void AdjT(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	simulator->IT(shards[simulator][q]);
}

/**
	* (External API) Controlled "X" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCX(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, ONE_CMPLX);

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled "Y" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCY(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSingleInvert(ctrlsArray, n, shards[simulator][q], -I_CMPLX, I_CMPLX);

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled "Z" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCZ(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, -ONE_CMPLX);

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled "H" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCH(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	const complex hGate[4] = {
		complex(M_SQRT1_2, ZERO_R1), complex(M_SQRT1_2, ZERO_R1),
		complex(M_SQRT1_2, ZERO_R1), complex(-M_SQRT1_2, ZERO_R1)
	};

	simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], hGate);

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 2));

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, -ONE_R1 / 4));

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled Inverse "S" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCAdjS(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 2));

	delete[] ctrlsArray;
}

/**
	* (External API) Controlled Inverse "T" Gate
	*/
MICROSOFT_QUANTUM_DECL void MCAdjT(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], ONE_CMPLX, pow(-ONE_CMPLX, ONE_R1 / 4));

	delete[] ctrlsArray;
}

/**
	* (External API) Rotation around Pauli axes
	*/
MICROSOFT_QUANTUM_DECL void R(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];

	switch (b) {
	case PauliI:
		simulator->RT(phi, shards[simulator][q]);
		break;
	case PauliX:
		simulator->RX(phi, shards[simulator][q]);
		break;
	case PauliY:
		simulator->RY(phi, shards[simulator][q]);
		break;
	case PauliZ:
		simulator->RZ(phi, shards[simulator][q]);
		break;
	default:
		break;
	}
}

/**
	* (External API) Controlled rotation around Pauli axes
	*/
MICROSOFT_QUANTUM_DECL void MCR(_In_ unsigned sid, _In_ unsigned b, _In_ double phi, _In_ unsigned n, _In_reads_(n) unsigned* c, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[n];
	for (unsigned i = 0; i < n; i++) {
		ctrlsArray[i] = shards[simulator][c[i]];
	}

	real1 cosine = cos(phi / 2.0);
	real1 sine = sin(phi / 2.0);
	complex pauliR[4];

	switch (b) {
	case PauliI:
		simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], complex(ONE_R1, ZERO_R1), complex(cosine, sine));
		break;
	case PauliX:
		pauliR[0] = complex(cosine, ZERO_R1);
		pauliR[1] = complex(ZERO_R1, -sine);
		pauliR[2] = complex(ZERO_R1, -sine);
		pauliR[3] = complex(cosine, ZERO_R1);
		simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
		break;
	case PauliY:
		pauliR[0] = complex(cosine, ZERO_R1);
		pauliR[1] = complex(-sine, ZERO_R1);
		pauliR[2] = complex(-sine, ZERO_R1);
		pauliR[3] = complex(cosine, ZERO_R1);
		simulator->ApplyControlledSingleBit(ctrlsArray, n, shards[simulator][q], pauliR);
		break;
	case PauliZ:
		simulator->ApplyControlledSinglePhase(ctrlsArray, n, shards[simulator][q], complex(cosine, -sine), complex(cosine, sine));
		break;
	default:
		break;
	}

	delete[] ctrlsArray;
}

/**
	* (External API) Exponentiation of Pauli operators
	*/
MICROSOFT_QUANTUM_DECL void Exp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi, _In_reads_(n) unsigned* q)
{
	QInterfacePtr simulator = simulators[sid];
	for (unsigned i = 0; i < n; i++) {
		switch (b[i]) {
		case PauliI:
			simulator->Exp(phi, shards[simulator][q[i]]);
			break;
		case PauliX:
			simulator->ExpX(phi, shards[simulator][q[i]]);
			break;
		case PauliY:
			simulator->ExpY(phi, shards[simulator][q[i]]);
			break;
		case PauliZ:
			simulator->ExpZ(phi, shards[simulator][q[i]]);
			break;
		default:
			break;
		}
	}
}

/**
	* (External API) Controlled exponentiation of Pauli operators
	*/
MICROSOFT_QUANTUM_DECL void MCExp(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_ double phi, _In_ unsigned nc, _In_reads_(nc) unsigned* cs, _In_reads_(n) unsigned* q)
{
	QInterfacePtr simulator = simulators[sid];
	bitLenInt* ctrlsArray = new bitLenInt[nc];
	for (unsigned i = 0; i < nc; i++) {
		ctrlsArray[i] = shards[simulator][cs[i]];
	}

	const complex pauliI[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
	const complex pauliX[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
	const complex pauliY[4] = { ZERO_CMPLX, -I_CMPLX, I_CMPLX, ZERO_CMPLX };
	const complex pauliZ[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, -ONE_CMPLX };

	complex toApply[4];

	for (unsigned i = 0; i < n; i++) {
		switch (b[i]) {
		case PauliI:
			mul2x2(phi, pauliI, toApply);
			break;
		case PauliX:
			mul2x2(phi, pauliX, toApply);
			break;
		case PauliY:
			mul2x2(phi, pauliY, toApply);
			break;
		case PauliZ:
			mul2x2(phi, pauliZ, toApply);
			break;
		default:
			break;
		}
		simulator->Exp(ctrlsArray, nc, shards[simulator][q[i]], toApply);
	}

	delete[] ctrlsArray;
}

/**
	* (External API) Measure bit in |0>/|1> basis
	*/
MICROSOFT_QUANTUM_DECL unsigned M(_In_ unsigned sid, _In_ unsigned q)
{
	QInterfacePtr simulator = simulators[sid];
	return simulator->M(shards[simulator][q]) ? 1U : 0U;
}

/**
	* (External API) Measure bits in specified Pauli bases
	*/
MICROSOFT_QUANTUM_DECL unsigned Measure(_In_ unsigned sid, _In_ unsigned n, _In_reads_(n) unsigned* b, _In_reads_(n) unsigned* q)
{
	QInterfacePtr simulator = simulators[sid];
	unsigned toRet = 0U;

	TransformPauliBasis(simulator, n, b, q);

	for (unsigned i = 0; i < n; i++) {
		if (simulator->M(shards[simulator][q[i]]))
		{
			toRet |= pow2((bitLenInt)i);
		}
	}

	RevertPauliBasis(simulator, n, b, q);

	return toRet;
}

}
