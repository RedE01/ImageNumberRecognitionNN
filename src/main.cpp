#include <iostream>
//#include "Debug.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <chrono>

void trainNeuralNetwork(NeuralNetwork& nn, int time) {
	DataLoader dataLoader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

	auto start = std::chrono::steady_clock::now();

	bool loop = true;
	int batchSize = 100, totalBatches = 100000;
	int correct = 0;
	for (int batches = 0; loop; ++batches) {

		for (int i = 0; i < batchSize; ++i) {
			int imageNumber = (i + batches * batchSize) % 60000;
			int answer = dataLoader.getLable(imageNumber);

			nn.setInputLayer(dataLoader.getImage(imageNumber), nn.getInputLayer()->getLayerSize());
			nn.propogateForward();

			nn.propogateBackwards(answer);

			if (batches % 1000 == 0 && nn.isCorrect(answer))
				++correct;
		}

		nn.updateLayers(batchSize, 0.1);

		if (correct != 0) {
			auto now = std::chrono::steady_clock::now();
			double timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
			if(timeElapsed >= time) {
				loop = false;
			}

			std::cout << correct << " / " << batchSize << " correct | " << timeElapsed << " / " << time << " seconds" << std::endl;
			correct = 0;
		}
	}
}

int main(int argc, char* argv[]) {
	if(argc < 2) {
		std::cout << "To few arguments, use -h for help" << std::endl;
		return -1;
	}
	bool train = false;
	int time = 0;
	std::string biasAndWeightFile = "";
	for(int i = 1; i < argc; ++i) {
		std::string argument = argv[i];
		if(argument == "-h" || argument == "--help") {
			std::cout << "Usage: " << argv[0] << " [OPTION]\n\n\t-t, --train\ttrain neural network\n\t-f, --file\tload weights and biases from file" << std::endl;
			return 0;
		}
		else if(argument == "-t" || argument == "--train") {
			if(i + 1 >= argc) {
				std::cout << "To few arguments, specify the duration in seconds you wish to train the neural network" << std::endl;
				return -1;
			}
			time = std::atoi(argv[++i]);
			
			train = true;
		}
		else if(argument == "-f" || argument == "--file") {
			if(i + 1 >= argc) {
				std::cout << "No file specified" << std::endl;
				return -1;
			}
			biasAndWeightFile = argv[++i];
		}
	}

	int imageSize = 28 * 28, hiddenLayerSize = 16;
	NeuralNetwork nn(2, imageSize, hiddenLayerSize, 10);

	if(biasAndWeightFile != "") {
		std::cout << "Loading weights and biases from " << biasAndWeightFile << std::endl;
		nn.loadWeightsAndBiasesFromFile(biasAndWeightFile);
	}

	if(train) {
		trainNeuralNetwork(nn, time);

		nn.writeWeightsAndBiasesToFile("output");
	}
}