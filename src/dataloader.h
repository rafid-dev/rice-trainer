#pragma once

#include "nn.h"
#include "types.h"

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "binpack/nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

constexpr std::size_t CHUNK_SIZE = (1 << 20);

struct Features;

namespace DataLoader {

    void loadFeatures(const binpack::TrainingDataEntry& entry, Features& features);

    struct DataSetEntry {
    private:
        int16_t  _score;
        int8_t   _result;
        Features _features;
        uint8_t  _sideToMove;

    public:
        const float score() const {
            return _score / EVAL_SCALE;
        }

        const float wdl() const {
            return _result == -1 ? 1.0 : _result == 0 ? 0.5 : 0.0;
        }

        const uint8_t sideToMove() const {
            return _sideToMove;
        }

        const Features& extractFeatures() const {
            return _features;
        }

        const void loadEntry(const binpack::TrainingDataEntry& entry) {
            _score      = entry.score;
            _result     = entry.result;
            _sideToMove = uint8_t(entry.pos.sideToMove());
            loadFeatures(entry, _features);
        }
    };

    struct DataSetLoader {
        std::array<DataSetEntry, CHUNK_SIZE> m_currentData;
        std::vector<std::size_t>             m_permuteShuffle;

        binpack::CompressedTrainingDataEntryReader m_reader;
        std::string                                m_path;
        std::size_t                                m_batchSize     = 16384;
        std::size_t                                m_positionIndex = 0;

        std::thread m_readingThread;

        bool m_backgroundLoading = true;

        int m_random_fen_skipping = 16;
        int m_early_fen_skipping  = 16;

        std::vector<binpack::TrainingDataEntry> m_buffer;

        std::size_t m_currentDataSize = 0;

        DataSetLoader(const std::string& _path) : m_reader{_path}, m_path{_path} {
            m_buffer.reserve(CHUNK_SIZE);
            m_permuteShuffle.reserve(CHUNK_SIZE);
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize) : m_reader{_path}, m_path{_path}, m_batchSize{_batchSize} {
            m_buffer.reserve(CHUNK_SIZE);
            m_permuteShuffle.reserve(CHUNK_SIZE);
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize, const bool _backgroundLoading) : m_reader{_path}, m_path{_path}, m_batchSize{_batchSize}, m_backgroundLoading{_backgroundLoading} {
            m_buffer.reserve(CHUNK_SIZE);
            m_permuteShuffle.reserve(CHUNK_SIZE);
            init();
        }

        void loadFromBuffer();
        void loadNext();
        void loadNextBatch();
        void init();
        void shuffle() {
            m_permuteShuffle.resize(m_currentDataSize);
            std::iota(m_permuteShuffle.begin(), m_permuteShuffle.end(), 0);
            
            std::random_device rd;
            std::mt19937       g(rd());
            std::shuffle(m_permuteShuffle.begin(), m_permuteShuffle.end(), g);
        }
        DataSetEntry& getEntry(const int index) {
            return m_currentData[m_positionIndex + index];
        }

        friend std::ostream& operator<<(std::ostream& os, const DataSetLoader& data_set_loader) {
            os << "DataSetLoader(batchSize=" << data_set_loader.m_batchSize << ", positionIndex=" << data_set_loader.m_positionIndex << ")";
            return os;
        }
    };

} // namespace DataLoader