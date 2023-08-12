#include "trainer.h"
#include "nn.h"
#include "optimizer.h"
#include <omp.h>

#define EPOCH_ERROR epochError / static_cast<double>(dataSetLoader.batchSize * batchIterations)

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

static inline const float error(const float output, const float eval, const float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return pow(sigmoid(output) - expected, 2);
}

static inline const float dError(const float output, const float eval, const float wdl) {
    float expected = EVAL_CP_RATIO * sigmoid(eval) + (1 - EVAL_CP_RATIO) * wdl;
    return 2 * (sigmoid(output) - expected) * sigmoidPrime(output);
}

void Trainer::batch(std::array<uint8_t, INPUT_SIZE>& active, const int threadId) {
    std::array<std::array<uint8_t, INPUT_SIZE>, THREADS> activeFeatures;

    for (int batchIdx = threadId; batchIdx < dataSetLoader.batchSize; batchIdx += THREADS) {
        // Load the current batch entry
        DataLoader::DataSetEntry& entry = dataSetLoader.getEntry(batchIdx);

        alignas(32) NN::Accumulator accumulator;
        NN::Color       stm = NN::Color(entry.sideToMove());
        Features        featureset;
        const float eval = entry.score();
        const float wdl = entry.wdl();

        loadFeatures(entry, featureset);

        //--- Forward Pass ---//
        float output = nn.forward(accumulator, featureset, stm);

        losses[threadId] += error(output, eval, wdl);

        //--- Backpropagation
        BatchGradients& batchGrad = batchGradients.at(threadId);

        //--- Output gradients
        float outputGrad = dError(output, eval, wdl);
        batchGrad.hiddenBias[0] += outputGrad;

        //--- Hidden gradients
        std::array<float, HIDDEN_SIZE * 2> hiddenGradients;
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            hiddenGradients[i] = outputGrad * nn.hiddenFeatures[i] * ReLUPrime(accumulator[i]);
        }

        //--- Apply hidden layer gradients
        for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
            batchGrad.hiddenFeatures[i] += outputGrad * accumulator[i];
        }

        // --- Input bias gradients
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            batchGrad.inputBias[i] += hiddenGradients[i] + hiddenGradients[i + HIDDEN_SIZE];
        }

        //--- Input layer gradients
        for (int i = 0; i < featureset.n; ++i) {
            int f1 = featureset.features[i][stm];
            int f2 = featureset.features[i][!stm];

            activeFeatures[threadId][f1] = 1;
            activeFeatures[threadId][f2] = 1;

            for (int j = 0; j < HIDDEN_SIZE; ++j) {
                batchGrad.inputFeatures[f1 * HIDDEN_SIZE + j] += hiddenGradients[j];
                batchGrad.inputFeatures[f2 * HIDDEN_SIZE + j] += hiddenGradients[j + HIDDEN_SIZE];
            }
        }
    }

    for (int i = 0; i < THREADS; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            active[j] |= activeFeatures[i][j];
        }
    }
}

void        Trainer::applyGradients(std::array<uint8_t, INPUT_SIZE>& active) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) {
        if (!active[i / HIDDEN_SIZE]) {
            continue;
        }

        float grad = 0.0f;
        for (int j = 0; j < THREADS; ++j) {
            grad += batchGradients[j].inputFeatures[i];
        }

        adamUpdate(nn.inputFeatures[i], nnGradients.inputFeatures[i], grad, learningRate);
    }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE; ++i){
        float grad = 0.0;
        for (int j = 0; j < THREADS; ++j) {
            grad += batchGradients[j].inputBias[i];
        }

        adamUpdate(nn.inputBias[i], nnGradients.inputBias[i], grad, learningRate);
    }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i){
        float grad = 0.0;

        for (int j = 0; j < THREADS; ++i){
            grad += batchGradients[j].hiddenFeatures[i];
        }

        adamUpdate(nn.hiddenFeatures[i], nnGradients.hiddenFeatures[i], grad, learningRate);
    }

    // finally, update the output bias
    float grad = 0.0;
    for (int i = 0; i < THREADS; ++i){
        grad += batchGradients[i].hiddenBias[0];
    }

    adamUpdate(nn.hiddenBias[0], nnGradients.hiddenBias[0], grad, learningRate);
}

void Trainer::train() {
    std::ofstream lossFile(savePath + "/loss.csv", std::ios::app);
    lossFile << "epoch,avg_epoch_error" << std::endl;

    nnGradients.clear();

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

            std::array<uint8_t, INPUT_SIZE> active;

            // Parallelize batch
#pragma omp parallel for schedule(static) num_threads(THREADS)
            for (int threadId = 0; threadId < THREADS; ++threadId) {
                batch(active, threadId);
            }

            // Calculate batch error
            for (int threadId = 0; threadId < THREADS; ++threadId) {
                batchError += static_cast<double>(losses[threadId]);
            }

            // Accumulate epoch error
            epochError += batchError;

            // Gradient descent
            applyGradients(active);

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
        printf("epoch: [%5d/%5d] | avg_epoch_error: [%11.9f]", epoch, maxEpochs, EPOCH_ERROR);
        std::cout << std::endl;

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