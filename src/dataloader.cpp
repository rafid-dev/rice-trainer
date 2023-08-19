#include "dataloader.h"
#include "nn.h"
#include <ctime>

namespace DataLoader {
    void DataSetLoader::loadNextBatch() {
        positionIndex += batchSize;

        if (positionIndex == CHUNK_SIZE) {
            // Join thread that's reading nextData
            if (readingThread.joinable()) {
                readingThread.join();
            }

            // Bring next data to current position
            std::swap(currentData, nextData);
            positionIndex = 0;

            // Begin a new thread to read nextData if background loading is enabled
            if (backgroundLoading) {
                readingThread = std::thread(&DataSetLoader::loadNext, this);
            } else {
                loadNext();
            }
        }
    }

    void DataSetLoader::loadNext() {
        std::random_device          rd;
        std::mt19937                mt{rd()};
        double                      prob = random_fen_skipping / (random_fen_skipping + 1);
        std::bernoulli_distribution dist(prob);

        std::vector<binpack::TrainingDataEntry> entries;
        entries.reserve(CHUNK_SIZE);

        static constexpr int VALUE_NONE = 32002;

        for (std::size_t counter = 0; counter < CHUNK_SIZE; ++counter) {
            if (dist(mt)) {
                continue;
            }

            // If we finished, go back to the beginning
            if (!reader.hasNext()) {
                reader = binpack::CompressedTrainingDataEntryReader(path);
            }

            // Get info
            binpack::TrainingDataEntry entry = reader.next();

            // Skip if the entry is too early
            if (entry.ply <= 16) {
                continue;
            }

            // Skip if the entry is a capturing move
            if (entry.isCapturingMove()) {
                continue;
            }

            // Skip if the entry is in check
            if (entry.isInCheck()) {
                continue;
            }

            // Skip if the entry score is none
            if (entry.score == VALUE_NONE) {
                continue;
            }

            entries.push_back(entry);
        }

#pragma omp parallel for schedule(static) num_threads(THREADS)
        for (std::size_t i = 0; i < entries.size(); ++i) {
            nextData[permuteShuffle[i]].loadEntry(entries[i]);
        }

        shuffle();
    }

    void DataSetLoader::shuffle() {
        std::random_device rd;
        std::mt19937       mt{rd()};
        std::shuffle(permuteShuffle.begin(), permuteShuffle.end(), mt);
    }

    void DataSetLoader::init() {
        positionIndex = 0;

        std::iota(permuteShuffle.begin(), permuteShuffle.end(), 0);
        shuffle();

        loadNext();
        std::swap(currentData, nextData);
        loadNext();
    }

    void loadFeatures(const binpack::TrainingDataEntry& entry, Features& features) {
        features.clear();
        const chess::Position& pos    = entry.pos;
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

} // namespace DataLoader