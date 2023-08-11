#include "argparse.h"
#include "trainer.h"

#include <omp.h>
#include <sstream>

int main(int argc, char* argv[]) {
    ArgumentParser parser;
    parser.addArgument("--dataset", "Path to the dataset.");
    parser.addArgument("--epochs", "Number of epochs to train for.");
    parser.addArgument("--id", "Network ID. Leave for random. Use '$' for a random number placeholder.", true);
    parser.addArgument("--lr", "Learning rate. (Default 0.001)", true);
    parser.addArgument("--lr-interval", "LR scheduler intervals. (Default 50)", true);
    parser.addArgument("--lr-decay", "LR scheduler decay. (Default 0.1)", true);
    parser.addArgument("--checkpoint", "Path to the checkpoint to load from.", true);
    parser.addArgument("--savepath", "Path to where checkpoints will be saved.", true);
    parser.addArgument("--saveinterval", "Interval for saving checkpoints.", true);
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
    std::string datasetPath    = parser.getArgumentValue("--dataset");
    std::string checkpointPath = parser.getArgumentValue("--checkpoint");
    std::string savepath       = parser.getArgumentValue("--savepath");
    std::string networkId      = parser.getArgumentValue("--id");
    int         saveInterval   = parser.getArgumentValue("--saveinterval").empty() ? 1 : std::stoi(parser.getArgumentValue("--saveinterval"));
    int         lrInterval     = parser.getArgumentValue("--lr-interval").empty() ? 50 : std::stoi(parser.getArgumentValue("--lr-interval"));
    float       lr             = parser.getArgumentValue("--lr").empty() ? 0.001f : std::stof(parser.getArgumentValue("--lr"));
    float       lrMultiplier   = parser.getArgumentValue("--lr-decay").empty() ? 0.1f : std::stof(parser.getArgumentValue("--lr-decay"));
    int         epochs         = std::stoi(parser.getArgumentValue("--epochs"));

    Trainer* trainer = new Trainer{datasetPath, 16384};

    // Configure trainer
    trainer->setNetworkId(networkId);
    trainer->setMaxEpochs(epochs);
    trainer->setSaveInterval(saveInterval);
    trainer->setSavePath(savepath);
    trainer->setLearningRate(lr);

    // Print Configurations
    std::cout << "Dataset Path: " << datasetPath << "\n";
    std::cout << "Checkpoint Path: " << checkpointPath << "\n";
    std::cout << "Save Path: " << savepath << "\n";
    std::cout << "Network ID: " << trainer->getNetworkId() << "\n";
    std::cout << "Learning Rate: " << trainer->getLearningRate() << "\n";
    std::cout << "Number of Available Threads: " << omp_get_max_threads() << "\n";
    std::cout << "Allocated threads: " << THREADS << "\n";

    if (!checkpointPath.empty()) {
        trainer->loadCheckpoint(checkpointPath);
    }
    
    trainer->train();

    return 0;
}