#include "trainer.h"
#include "nn.h"
#include <omp.h>

#define EPOCH_ERROR epochError / static_cast<double>(dataSetLoader.batchSize * batchIterations)

inline float error(float output, float target) {
    return std::pow(sigmoid(output) - target, 2);
}

inline float errorPrime(float output, float target) {
    return 2 * (sigmoid(output) - target) * sigmoidPrime(output);
}

void Trainer::loadFeatures(DataLoader::DataSetEntry& entry, Features& features) {
    const chess::Position& pos    = entry.entry.pos;
    chess::Bitboard        pieces = pos.piecesBB();

    const chess::Square ksq_White = pos.kingSquare(chess::Color::White);
    const chess::Square ksq_Black = pos.kingSquare(chess::Color::Black);

    for (chess::Square sq : pieces) {
        const chess::Piece piece      = pos.pieceAt(sq);
        const std::uint8_t pieceType  = static_cast<uint8_t>(piece.type());
        const std::uint8_t pieceColor = static_cast<uint8_t>(piece.color());

        const int featureW = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::White), static_cast<int>(ksq_White));
        const int featureB = inputIndex(pieceType, pieceColor, static_cast<int>(sq), static_cast<uint8_t>(chess::Color::Black), static_cast<int>(ksq_Black));

        features.add(featureW, featureB);
    }
}

void        Trainer::batch() {
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int batchIdx = 0; batchIdx < dataSetLoader.batchSize; batchIdx++) {
        const int threadId = omp_get_thread_num();

        // Load the current batch entry
        DataLoader::DataSetEntry& entry = dataSetLoader.getEntry(batchIdx);

        NN::Accumulator accumulator;
        NN::Color       stm = NN::Color(entry.sideToMove());
        Features        featureset;

        loadFeatures(entry, featureset);

        //--- Forward Pass ---//
        const float output = nn.forward(accumulator, featureset, stm);

        losses[threadId] += error(output, entry.target());

        //--- Backpropagation
        float outputGrad = errorPrime(output, entry.target());

        auto& batchGrad = batchGradients.at(threadId);

        //--- Calculate gradients for the hidden layer
        std::array<float, HIDDEN_SIZE * 2> hiddenGrads;

#pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            hiddenGrads[i] = outputGrad * nn.hiddenFeatures[i] * ReLUPrime(accumulator[i]);
        }

        //--- Calculate gradients for the hidden bias
        batchGrad.hiddenBias[0] += outputGrad;

#pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
            batchGrad.hiddenFeatures[i] += outputGrad * accumulator[i];
        }

        //--- Calculate gradients for the input layer
#pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            batchGrad.inputBias[i] += hiddenGrads[i] + hiddenGrads[i + HIDDEN_SIZE];
        }

        for (int i = 0; i < featureset.n; i++) {
            int f1 = featureset.features[i][stm];
            int f2 = featureset.features[i][!stm];

#pragma omp simd
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                batchGrad.inputFeatures[f1 * HIDDEN_SIZE + j] += hiddenGrads[j];
                batchGrad.inputFeatures[f2 * HIDDEN_SIZE + j] += hiddenGrads[j + HIDDEN_SIZE];
            }
        }
    }
}

void             Trainer::applyGradients() {
#pragma omp      parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        float gradSum = 0;
#pragma omp simd reduction(+ : gradSum)
        for (int j = 0; j < THREADS; j++) {
            gradSum += batchGradients[j].inputFeatures[i];
        }

        adamUpdate(nn.inputFeatures[i], nnGradients.inputFeatures[i], gradSum, learningRate);
         }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float gradSum = 0;

#pragma omp simd reduction(+ : gradSum)
        for (int j = 0; j < THREADS; j++) {
            gradSum += batchGradients[j].inputBias[i];
        }

        adamUpdate(nn.inputBias[i], nnGradients.inputBias[i], gradSum, learningRate);
    }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE * 2; i++) {
        float gradSum = 0;

#pragma omp simd reduction(+ : gradSum)
        for (int j = 0; j < THREADS; j++) {
            gradSum += batchGradients[j].hiddenFeatures[i];
        }

        adamUpdate(nn.hiddenFeatures[i], nnGradients.hiddenFeatures[i], gradSum, learningRate);
    }

    float gradSum = 0;
#pragma omp simd reduction(+ : gradSum)
    for (int j = 0; j < THREADS; j++) {
        gradSum += batchGradients[j].hiddenBias[0];
    }

    adamUpdate(nn.hiddenBias[0], nnGradients.hiddenBias[0], gradSum, learningRate);
}

void Trainer::train() {
    std::ofstream lossFile(savePath + "/loss.csv", std::ios::app);
    lossFile << "epoch,avg_epoch_error" << std::endl;

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        std::uint64_t start           = getTimeMs();
        std::size_t   batchIterations = 0;
        double        epochError      = 0.0;

        const std::size_t batchSize = dataSetLoader.batchSize;

        for (int b = 0; b < EPOCH_SIZE / batchSize; ++b) {
            batchIterations++;
            double batchError = 0;

            // Clear gradients and losses
            clearGradientsAndLosses();

            // Perform batch operations
            batch();

            // Calculate batch error
            for (int threadId = 0; threadId < THREADS; ++threadId) {
                batchError += static_cast<double>(losses[threadId]);
            }

            // Accumulate epoch error
            epochError += batchError;

            // Gradient descent
            applyGradients();

            // Load the next batch
            dataSetLoader.loadNextBatch();

            // Print progress
            if (b % 100 == 0) {
                std::uint64_t end            = getTimeMs();
                int           positionsCount = (b + 1) * batchSize;
                int           posPerSec      = static_cast<int>(positionsCount / ((end - start) / 1000.0));
                printf("\rep/ba:[%4d/%4d] |batch error:[%1.9f]|epoch error:[%1.9f]|speed:[%9d] pos/s", epoch, b, batchError / static_cast<double>(dataSetLoader.batchSize), EPOCH_ERROR, posPerSec);
                std::cout << std::flush;
            }
        }

        std::cout << std::endl;
        printf("epoch: [%5d/%5d] | avg_epoch_error: [%11.9f]\n", epoch, maxEpochs, EPOCH_ERROR);

        // Save the network
        if (epoch % saveInterval == 0) {
            save(std::to_string(epoch));
        }

        if (epoch % lrDecayInterval == 0) {
            std::cout << "Decaying learning rate" << std::endl;
            learningRate *= lrDecay;
        }

        if (epoch % 100 == 0) {
            dataSetLoader.shuffle();
        }

        lossFile << epoch << "," << EPOCH_ERROR << std::endl;
    }
}

void Trainer::clearGradientsAndLosses() {
    for (auto& grad : batchGradients) {
        grad.clear();
    }
    memset(losses.data(), 0, sizeof(float) * THREADS);
}