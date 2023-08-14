#include "argparse.h"
#include "trainer.h"
#include "quantize.h"

#include <omp.h>
#include <sstream>

int main(int argc, char* argv[]) {
    ArgumentParser parser;
    parser.addArgument("--data", "Path to training data.");
    parser.addArgument("--val-data", "Path to validation data.");
    parser.addArgument("--epochs", "Number of epochs.");
    parser.addArgument("--id", "Unique network identifier.", true);
    parser.addArgument("--lr", "Initial learning rate. (Default: 0.001)", true);
    parser.addArgument("--checkpoint", "Path to checkpoint.", true);
    parser.addArgument("--save", "Checkpoint save directory.", true);
    parser.setProgramName(argv[0]);

    // Print help and exit if no arguments or --help flag provided
    if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--help")) {
        parser.printHelp();
        return 0;
    }

    // Parse arguments
    if (!parser.parse(argc, argv)) {
        return 1;
    }

    // Extract values from parsed arguments
    std::string datasetPath    = parser.getArgumentValue("--data");
    std::string valDatasetPath = parser.getArgumentValue("--val-data");
    std::string checkpointPath = parser.getArgumentValue("--checkpoint");
    std::string savepath       = parser.getArgumentValue("--save");
    std::string networkId      = parser.getArgumentValue("--id");
    int         epochs         = std::stoi(parser.getArgumentValue("--epochs"));
    float       lr             = parser.getArgumentValue("--lr").empty() ? 0.001f : std::stof(parser.getArgumentValue("--lr"));

    Trainer* trainer = new Trainer{datasetPath, 16384, valDatasetPath};

    // Configure trainer
    trainer->setNetworkId(networkId);
    trainer->setMaxEpochs(epochs);
    trainer->setSavePath(savepath);
    trainer->setLearningRate(lr);

    // Print Configurations
    std::cout << "Number of Available Threads: " << omp_get_max_threads() << "\n";
    std::cout << "Allocated threads: " << THREADS << "\n";
    std::cout << "Dataset Path: " << datasetPath << "\n";
    std::cout << "Validation Dataset Path: " << valDatasetPath << "\n";
    std::cout << "Checkpoint Path: " << checkpointPath << "\n";
    std::cout << "Save Path: " << savepath << "\n";
    std::cout << "Network ID: " << trainer->getNetworkId() << "\n\n";
    std::cout << "Learning Rate: " << trainer->getLearningRate() << "\n";
    std::cout << "Optimizer: " << trainer->optimizer << "\n";
    std::cout << "LR Scheduler: " << trainer->lrScheduler << "\n";
    std::cout << "Epochs: " << trainer->getMaxEpochs() << "\n";
    std::cout << "Batchsize: " << trainer->getBatchSize() << "\n\n";

    if (!checkpointPath.empty()) {
        std::cout << "Loading checkpoint from " << checkpointPath << std::endl;
        trainer->loadCheckpoint(checkpointPath);
        std::cout << "Loaded checkpoint from " << checkpointPath << std::endl;
    }

    std::cout << std::endl;
    
    trainer->train();

    return 0;
}