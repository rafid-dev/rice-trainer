#include "trainer.h"
#include "nn.h"
#include "optimizer.h"
#include <omp.h>

#define EPOCH_ERROR epochError / static_cast<double>(dataSetLoader.batchSize * batchIterations)

inline float errorFunction(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return pow(sigmoid(output) - expected, 2);
}

inline float errorGradient(float output, float eval, float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return 2 * (sigmoid(output) - expected);
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

void Trainer::batch(std::array<uint8_t, INPUT_SIZE>& active) {
    std::array<std::array<uint8_t, INPUT_SIZE>, THREADS> actives;
    std::memset(actives.data(), 0, sizeof(actives));
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int batchIdx = 0; batchIdx < dataSetLoader.batchSize; batchIdx++) {
        const int threadId = omp_get_thread_num();

        // Load the current batch entry
        DataLoader::DataSetEntry& entry = dataSetLoader.getEntry(batchIdx);

        alignas(32) NN::Accumulator accumulator;
        NN::Color       stm = NN::Color(entry.sideToMove());
        Features        featureset;

        loadFeatures(entry, featureset);

        const auto eval = entry.score();
        const auto wdl  = entry.wdl();

        //--- Forward Pass ---//
        const float output = nn.forward(accumulator, featureset, stm);

        losses[threadId] += errorFunction(output, eval, wdl);

        //--- Backward Pass ---//
        BatchGradients& gradients   = batchGradients[threadId];
        const float     outGradient = errorGradient(output, eval, wdl) * sigmoidPrime(output);

        // Hidden bias
        gradients.hiddenBias[0] += outGradient;

        // Hidden features
        #pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            gradients.hiddenFeatures[i] += outGradient * accumulator[i];
        }

        std::array<float, HIDDEN_SIZE * 2> hiddenLosses;

        #pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            hiddenLosses[i] = outGradient * nn.hiddenFeatures[i] * ReLUPrime(accumulator[i]);
        }

        // Input bias
        #pragma omp simd
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            gradients.inputBias[i] += hiddenLosses[i] + hiddenLosses[i + HIDDEN_SIZE];
        }

        // Input features
        for (int i = 0; i < featureset.n; ++i) {
            int f1 = featureset.features[i][stm];
            int f2 = featureset.features[i][!stm];

            actives[threadId][f1] = 1;
            actives[threadId][f2] = 1;

            #pragma omp simd
            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                gradients.inputFeatures[f1 * HIDDEN_SIZE + j] += hiddenLosses[j];
                gradients.inputFeatures[f2 * HIDDEN_SIZE + j] += hiddenLosses[j + HIDDEN_SIZE];
            }
        }
    }

#pragma for schedule(static) num_threads(THREADS)
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < THREADS; ++j) {
            active[i] |= actives[j][i];
        }
    }
}

void        Trainer::applyGradients(std::array<uint8_t, INPUT_SIZE>& actives) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (!actives[i])
            continue;

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            int   index = i * HIDDEN_SIZE + j;
            float gradientSum = 0;

            for (int k = 0; k < THREADS; ++k) {
                gradientSum += batchGradients[k].inputFeatures[index];
            }

            optimizer.update(nn.inputFeatures[index], nnGradients.inputFeatures[index], gradientSum, learningRate);
        }
    }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].inputBias[i];
        }

        optimizer.update(nn.inputBias[i], nnGradients.inputBias[i], gradientSum, learningRate);
    }

    // --- Hidden Features ---//
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].hiddenFeatures[i];
        }

        optimizer.update(nn.hiddenFeatures[i], nnGradients.hiddenFeatures[i], gradientSum, learningRate);
    }

    //-- Hidden Bias --//
    float gradientSum = 0;
    for (int i = 0; i < THREADS; ++i) {
        gradientSum += batchGradients[i].hiddenBias[0];
    }

    optimizer.update(nn.hiddenBias[0], nnGradients.hiddenBias[0], gradientSum, learningRate);
}

void Trainer::train() {
    std::ofstream lossFile(savePath + "/loss.csv", std::ios::app);
    lossFile << "epoch,avg_epoch_error" << std::endl;

    const std::size_t batchSize = dataSetLoader.batchSize;

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        std::uint64_t start           = Misc::getTimeMs();
        std::size_t   batchIterations = 0;
        double        epochError      = 0.0;

        for (int b = 0; b < EPOCH_SIZE / batchSize; ++b) {
            batchIterations++;
            double batchError = 0;

            // Clear gradients and losses
            clearGradientsAndLosses();

            std::array<uint8_t, INPUT_SIZE> actives;

            // Perform batch operations
            batch(actives);

            // Calculate batch error
            for (int threadId = 0; threadId < THREADS; ++threadId) {
                batchError += static_cast<double>(losses[threadId]);
            }

            // Accumulate epoch error
            epochError += batchError;

            // Gradient descent
            applyGradients(actives);

            // Load the next batch
            dataSetLoader.loadNextBatch();

            // Print progress
            if (b % 100 == 0 || b == EPOCH_SIZE / batchSize - 1) {
                std::uint64_t end            = Misc::getTimeMs();
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

        lrScheduler.step(learningRate);

        lossFile << epoch << "," << EPOCH_ERROR << std::endl;
    }
}

void Trainer::clearGradientsAndLosses() {
    for (auto& grad : batchGradients) {
        grad.clear();
    }
    memset(losses.data(), 0, sizeof(float) * THREADS);
}