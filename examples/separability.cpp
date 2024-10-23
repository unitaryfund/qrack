#include "qrack/qfactory.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace Eigen;

const double tolerance = Qrack::TRYDECOMPOSE_EPSILON;

#if FPPOW == 5
#define EigenVec VectorXcf
#define EigenMat MatrixXcf
#else
#define EigenVec VectorXcd
#define EigenMat MatrixXcd
#endif

// Helper function to reshape the state vector into a matrix
EigenMat reshapeToMatrix(const EigenVec& stateVector, int dimA, int dimB)
{
    EigenMat reshapedState(dimA, dimB);

    for (int i = 0; i < dimA; ++i) {
        for (int j = 0; j < dimB; ++j) {
            // The state vector is flattened, so we map its entries into the matrix
            reshapedState(i, j) = stateVector(i * dimB + j);
        }
    }

    return reshapedState;
}

// Function to perform Schmidt decomposition and check separability
bool isSeparable(const EigenVec& stateVector, int qubitsA, int qubitsB)
{
    // Calculate the dimensions of the subsystems
    int dimA = 1 << qubitsA; // 2^qubitsA
    int dimB = 1 << qubitsB; // 2^qubitsB

    // Reshape the state vector into a matrix of size (dimA x dimB)
    EigenMat reshapedState = reshapeToMatrix(stateVector, dimA, dimB);

    // Perform Singular Value Decomposition (SVD)
    JacobiSVD<EigenMat> svd(reshapedState, ComputeThinU | ComputeThinV);
    EigenVec singularValues = svd.singularValues();

    // Check the number of non-zero singular values
    int nonZeroCount = 0;
    for (int i = 0; i < singularValues.size(); ++i) {
        if (std::abs(singularValues(i)) > tolerance) {
            nonZeroCount++;
        }
    }

    // If there is only one non-zero singular value, the state is separable
    return (nonZeroCount == 1);
}

// Function to perform Schmidt decomposition and extract the separable pure states
bool decomposeAndExtractStates(const EigenVec& stateVector, int qubitsA, int qubitsB, EigenVec& psi_a, EigenVec& psi_b)
{
    // Calculate the dimensions of the subsystems
    int dimA = 1 << qubitsA; // 2^qubitsA
    int dimB = 1 << qubitsB; // 2^qubitsB

    // Reshape the state vector into a matrix of size (dimA x dimB)
    EigenMat reshapedState = reshapeToMatrix(stateVector, dimA, dimB);

    // Perform Singular Value Decomposition (SVD)
    // Full U and Full V are needed to actually extract the decomposed states.
    // However, this quickly becomes very memory prohibitive!
    // The truth, there's virtually no other choice but to use Qrack's algorithm to actually extract the state vectors,
    // above a very limited scale!
    JacobiSVD<EigenMat> svd(reshapedState, ComputeFullU | ComputeFullV);
    EigenVec singularValues = svd.singularValues();

    // Check the number of non-zero singular values
    int nonZeroCount = 0;
    for (int i = 0; i < singularValues.size(); ++i) {
        if (std::abs(singularValues(i)) > tolerance) {
            nonZeroCount++;
        }
    }

    // If there is more than one non-zero singular value, the state is entangled
    if (nonZeroCount > 1) {
        return false; // Not separable (entangled)
    }

    // If separable, extract the first column of U and V
    psi_a = svd.matrixU().col(0);
    psi_b = svd.matrixV().col(0);
    // These are the actual decomposed state vectors.

    return true; // Separable
}

void ghz(Qrack::QInterfacePtr qsim)
{
    const bitLenInt n = qsim->GetQubitCount();
    qsim->H(0U);
    for (bitLenInt i = 1U; i < n; ++i) {
        qsim->CNOT(i - 1U, i);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [qubit width]" << std::endl;
        return 1;
    }

    const bitLenInt width = (bitLenInt)std::stoi(argv[1U]);
    const bitLenInt halfWidth = width >> 1U;
    const bitCapIntOcl maxQPower = Qrack::pow2Ocl(width);
    std::unique_ptr<Qrack::complex> arr(new Qrack::complex[maxQPower]);

    for (bitLenInt i = 0U; i < halfWidth; ++i) {
        const bitLenInt subsystemSize = i + 1U;
        Qrack::QInterfacePtr qsim =
            Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, width, Qrack::ZERO_BCI);
        Qrack::QInterfacePtr qsim_b =
            Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, width - subsystemSize, Qrack::ZERO_BCI);

        ghz(qsim);

        qsim->GetQuantumState(arr.get());

        EigenVec vec = Map<EigenVec>(arr.get(), maxQPower);

        auto start = std::chrono::high_resolution_clock::now();
        bool result = isSeparable(vec, subsystemSize, width - subsystemSize);
        if (result) {
            // We do this with Qrack, but it's the part where we ACTUALLY decompose the original implementation.
            qsim->Decompose(subsystemSize, qsim_b);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

        cout << "Schmidt decomposition, " << (int)width << " qubits, " << (int)subsystemSize
             << " qubit subsystem, nonseparable: " << duration << " seconds, " << (result ? "failure." : "success.")
             << endl;

        if (result) {
            qsim = Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, width, Qrack::ZERO_BCI);
            ghz(qsim);
        }

        start = std::chrono::high_resolution_clock::now();
        result = qsim->TryDecompose(subsystemSize, qsim_b, tolerance);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

        cout << "Qrack, " << (int)width << " qubits, " << (int)subsystemSize
             << " qubit subsystem, nonseparable: " << duration << " seconds, " << (result ? "failure." : "success.")
             << endl;

        qsim = Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, subsystemSize, Qrack::ZERO_BCI);
        qsim_b =
            Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, width - subsystemSize, Qrack::ZERO_BCI);

        ghz(qsim);
        ghz(qsim_b);

        qsim->Compose(qsim_b);

        qsim->GetQuantumState(arr.get());

        vec = Map<EigenVec>(arr.get(), maxQPower);

        start = std::chrono::high_resolution_clock::now();
        result = isSeparable(vec, subsystemSize, width - subsystemSize);
        if (result) {
            // We do this with Qrack, but it's the part where we ACTUALLY decompose the original implementation.
            qsim->Decompose(subsystemSize, qsim_b);
        }
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // Report in seconds:
        duration /= 1000000;

        cout << "Schmidt decomposition, " << (int)width << " qubits, " << (int)subsystemSize
             << " qubit subsystem, separable: " << duration << " seconds, " << (result ? "success." : "failure.")
             << endl;

        if (result) {
            qsim = Qrack::CreateQuantumInterface({ Qrack::QINTERFACE_OPTIMAL_BASE }, subsystemSize, Qrack::ZERO_BCI);
            qsim_b = Qrack::CreateQuantumInterface(
                { Qrack::QINTERFACE_OPTIMAL_BASE }, width - subsystemSize, Qrack::ZERO_BCI);

            ghz(qsim);
            ghz(qsim_b);

            qsim->Compose(qsim_b);
        }

        start = std::chrono::high_resolution_clock::now();
        result = qsim->TryDecompose(subsystemSize, qsim_b, tolerance);
        end = std::chrono::high_resolution_clock::now();
        // Report in seconds:
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;

        cout << "Qrack, " << (int)width << " qubits, " << (int)subsystemSize
             << " qubit subsystem, separable: " << duration << " seconds, " << (result ? "success." : "failure.")
             << endl;
    }

    return 0;
}
