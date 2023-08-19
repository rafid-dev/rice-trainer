#pragma once

#include "types.h"
#include "nn.h"

// turn off warnings for this
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "binpack/nnue_data_binpack_format.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <array>
#include <thread>

constexpr std::size_t CHUNK_SIZE = (1 << 20);

struct Features;

namespace DataLoader {

    void loadFeatures(const binpack::TrainingDataEntry& entry, Features& features);

    struct DataSetEntry {
    private:
        int16_t _score;
        int8_t  _result;
        Features _features;
        uint8_t _sideToMove;
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

        void setScore(const int16_t score) {
            _score = score;
        }

        void setResult(const int8_t result) {
            _result = result;
        }

        void setSideToMove(const uint8_t sideToMove) {
            _sideToMove = sideToMove;
        }
    
        const Features& extractFeatures() const {
            return _features;
        }

        const void loadEntry(const binpack::TrainingDataEntry& entry) {
            _score = entry.score;
            _result = entry.result;
            _sideToMove = uint8_t(entry.pos.sideToMove());
            loadFeatures(entry, _features);
        }
    };

    struct DataSetLoader {
        std::array<DataSetEntry, CHUNK_SIZE> currentData;
        std::array<DataSetEntry, CHUNK_SIZE> nextData;
        std::array<int, CHUNK_SIZE>            permuteShuffle;

        binpack::CompressedTrainingDataEntryReader reader;
        std::string                                path;
        std::size_t batchSize     = 16384;
        std::size_t positionIndex = 0;

        std::thread readingThread;

        bool backgroundLoading = true;

        int random_fen_skipping = 16;

        DataSetLoader(const std::string& _path) : reader{_path}, path{_path} {
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize) : reader{_path}, path{_path}, batchSize{_batchSize} {
            init();
        }

        DataSetLoader(const std::string& _path, const std::size_t _batchSize, const bool _backgroundLoading) : reader{_path}, path{_path}, batchSize{_batchSize}, backgroundLoading{_backgroundLoading} {
            init();
        }

        void          loadNext();
        void          loadNextBatch();
        void          init();
        void          shuffle();
        DataSetEntry& getEntry(const int index) {
            return currentData[positionIndex + index];
        }

        friend std::ostream& operator<<(std::ostream& os, const DataSetLoader& data_set_loader) {
            os << "DataSetLoader(batchSize=" << data_set_loader.batchSize << ", positionIndex=" << data_set_loader.positionIndex << ")";
            return os;
        }
    };

} // namespace DataLoader