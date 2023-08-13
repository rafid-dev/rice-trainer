#include "trainer.h"
#include "nn.h"
#include "optimizer.h"
#include <omp.h>

#define EPOCH_ERROR epochError / static_cast<double>(dataSetLoader.batchSize * batchIterations)

void Trainer::forward(NNTrace& trace) {
    trace.output = nn.forward(trace.accumulator, trace.features, trace.stm);
}

void Trainer::backward(NNTrace& trace, std::array<uint8_t, INPUT_SIZE>& active, const int threadId) {
    // Load trace data
    const auto& eval        = trace.eval;
    const auto& wdl         = trace.wdl;
    const auto& stm         = trace.stm;
    const auto& output      = trace.output;
    const auto& accumulator = trace.accumulator;
    const auto& featureset  = trace.features;

    // Accumulate loss
    losses[threadId] += errorFunction(output, eval, wdl);

    // Gradients
    BatchGradients& gradients   = batchGradients[threadId];
    const float     outGradient = errorGradient(output, eval, wdl) * sigmoidPrime(output);

    // Hidden bias
    gradients.hiddenBias[0] += outGradient;

    // Hidden features
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
        gradients.hiddenFeatures[i] += outGradient * accumulator[i];
    }

    std::array<float, HIDDEN_SIZE * 2> hiddenLosses;

    for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
        hiddenLosses[i] = outGradient * nn.hiddenFeatures[i] * ReLUPrime(accumulator[i]);
    }

    // Input bias
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        gradients.inputBias[i] += hiddenLosses[i] + hiddenLosses[i + HIDDEN_SIZE];
    }

    // Input features
    for (int i = 0; i < featureset.n; ++i) {
        int f1 = featureset.features[i][stm];
        int f2 = featureset.features[i][!stm];

        active[f1] = 1;
        active[f2] = 1;

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            gradients.inputFeatures[f1 * HIDDEN_SIZE + j] += hiddenLosses[j];
            gradients.inputFeatures[f2 * HIDDEN_SIZE + j] += hiddenLosses[j + HIDDEN_SIZE];
        }
    }
}

void Trainer::batch(std::array<uint8_t, INPUT_SIZE>& active) {
    std::array<std::array<uint8_t, INPUT_SIZE>, THREADS> activeFeatures;
    std::memset(activeFeatures.data(), 0, sizeof(activeFeatures));

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int batchIdx = 0; batchIdx < dataSetLoader.batchSize; batchIdx++) {
        const int threadId = omp_get_thread_num();

        // Load the current batch entry
        DataLoader::DataSetEntry& entry = dataSetLoader.getEntry(batchIdx);

        NNTrace trace;

        trace.features = entry.loadFeatures();
        trace.stm  = NN::Color(entry.sideToMove());
        trace.eval = entry.score();
        trace.wdl  = entry.wdl();

        //--- Forward Pass ---//
        forward(trace);

        //--- Backward Pass ---//
        backward(trace, activeFeatures[threadId], threadId);
    }

    for (int i = 0; i < THREADS; ++i) {
        for (int j = 0; j < INPUT_SIZE; ++j) {
            active[j] |= activeFeatures[i][j];
        }
    }
}

void        Trainer::applyGradients(std::array<uint8_t, INPUT_SIZE>& active) {
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < INPUT_SIZE; ++i) {
        if (!active[i])
            continue;

        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            int index = i * HIDDEN_SIZE + j;

            float g = 0;

            for (int k = 0; k < THREADS; ++k) {
                g += batchGradients[k].inputFeatures[index];
            }
            adamUpdate(nn.inputFeatures[i], nnGradients.inputFeatures[i], g, learningRate);
        }
    }

#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].inputBias[i];
        }

        adamUpdate(nn.inputBias[i], nnGradients.inputBias[i], gradientSum, learningRate);
    }

    // --- Hidden Features ---//
#pragma omp parallel for schedule(static) num_threads(THREADS)
    for (int i = 0; i < HIDDEN_SIZE * 2; ++i) {
        float gradientSum = 0;

        for (int j = 0; j < THREADS; ++j) {
            gradientSum += batchGradients[j].hiddenFeatures[i];
        }

        adamUpdate(nn.hiddenFeatures[i], nnGradients.hiddenFeatures[i], gradientSum, learningRate);
    }

    //-- Hidden Bias --//
    float gradientSum = 0;
    for (int i = 0; i < THREADS; ++i) {
        gradientSum += batchGradients[i].hiddenBias[0];
    }

    adamUpdate(nn.hiddenBias[0], nnGradients.hiddenBias[0], gradientSum, learningRate);
}

void Trainer::train() {
    std::ofstream lossFile(savePath + "/loss.csv", std::ios::app);
    lossFile << "epoch,avg_epoch_error" << std::endl;

    for (int epoch = 1; epoch <= maxEpochs; ++epoch) {
        std::uint64_t start           = Misc::getTimeMs();
        std::size_t   batchIterations = 0;
        double        epochError      = 0.0;

        const std::size_t batchSize = dataSetLoader.batchSize;

        for (int b = 0; b < EPOCH_SIZE / batchSize; ++b) {
            batchIterations++;
            double batchError = 0;

            // Clear gradients and losses
            clearGradientsAndLosses();

            // Active inputs
            std::array<uint8_t, INPUT_SIZE> active;

            // Perform batch operations
            batch(active);

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

        if (epoch % lrDecayInterval == 0) {
            std::cout << "Decaying learning rate" << std::endl;
            learningRate *= lrDecay;
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