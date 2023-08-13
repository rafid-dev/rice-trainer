#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <string>

namespace Misc {
    static inline std::string generateRandomHexValue(int numDigits) {
        std::random_device              rd;
        std::mt19937                    gen(rd());
        std::uniform_int_distribution<> dis(0, 15);

        std::ostringstream hexValue;
        for (int i = 0; i < numDigits; ++i) {
            hexValue << std::hex << dis(gen);
        }

        return hexValue.str();
    }

    static inline std::uint64_t getTimeMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
} // namespace Misc