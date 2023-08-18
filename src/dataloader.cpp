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
            }
        }
    }

    void DataSetLoader::loadNext() {
        std::random_device          rd;
        std::mt19937                mt{rd()};
        double                      prob = random_fen_skipping / (random_fen_skipping + 1);
        std::bernoulli_distribution dist(prob);

        for (std::size_t counter = 0; counter < CHUNK_SIZE; ++counter) {
            // If we finished, go back to the beginning
            if (!reader.hasNext()) {
                reader = binpack::CompressedTrainingDataEntryReader(path);
            }

            DataSetEntry& positionEntry = nextData[permuteShuffle[counter]];

            // Get info
            positionEntry.entry = reader.next();

            bool earlySkip = positionEntry.entry.ply <= 16;
            bool filter    = positionEntry.entry.isCapturingMove() || positionEntry.entry.isInCheck();

            if (dist(mt)) {
                counter--;
                continue;
            }

            if (positionEntry.entry.score == 32002 || earlySkip || filter) {
                counter--;
                continue;
            }
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

    void DataSetEntry::loadFeatures(Features& features) const {
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

    Features DataSetEntry::loadFeatures() const {
        Features features;
        loadFeatures(features);
        return features;
    }
} // namespace DataLoader