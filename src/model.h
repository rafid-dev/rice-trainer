#pragma once

#include "layer.h"

#include <memory>
#include <vector>
#include <type_traits>

struct Module {
    using LayerPtr = std::unique_ptr<Layer>;

    std::vector<LayerPtr> input_layers;
    std::vector<LayerPtr> layers;

    template<typename LayerType, typename... ARGS>
    LayerType* add(ARGS&&... args){
        if constexpr()
    }
};