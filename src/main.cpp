#include <iostream>
#include "Debug.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <chrono>

void trainNeuralNetwork(NeuralNetwork& nn, int time) {
	DataLoader dataLoader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
	int imageSize = nn.getInputLayer()->getLayerSize();

	auto start = std::chrono::steady_clock::now();

	bool loop = true;
	int batchSize = 100, totalBatches = 100000;
	int correct = 0;
	for (int batches = 0; loop; ++batches) {

		for (int i = 0; i < batchSize; ++i) {
			int imageNumber = (i + batches * batchSize) % 60000;
			int answer = dataLoader.getTrainingLable(imageNumber);

			nn.setInputLayer(dataLoader.getTrainingImage(imageNumber, imageSize), nn.getInputLayer()->getLayerSize());
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

	int trainTime = 0;
	std::string biasAndWeightFile = "", loadImageFile = "";
	for(int i = 1; i < argc; ++i) {
		std::string argument = argv[i];
		if(argument == "-h" || argument == "--help") {
			std::cout << "Usage: " << argv[0] << " [OPTION]\n\n\t-i, --image\tload image and output the number\n\t-f, --file\tload weights and biases from file\n\t-t, --train\ttrain neural network" << std::endl;
			return 0;
		}
		else if(argument == "-i" || argument == "--image") {
			if(i + 1 >= argc) {
				std::cout << "No file specified" << std::endl;
				return -1;
			}
			loadImageFile = argv[++i];
		}
		else if(argument == "-f" || argument == "--file") {
			if(i + 1 >= argc) {
				std::cout << "No file specified" << std::endl;
				return -1;
			}
			biasAndWeightFile = argv[++i];
		}
		else if(argument == "-t" || argument == "--train") {
			if(i + 1 >= argc) {
				std::cout << "To few arguments, specify the duration in seconds you wish to train the neural network" << std::endl;
				return -1;
			}
			std::string number = argv[++i];
			trainTime = std::stoi(number);
		}
		else {
			std::cout << "Invalid option: " << argument << std::endl;
			return -1;
		}
	}

	int imageWidth = 28, imageHeight = 28, imageSize = imageWidth * imageHeight, hiddenLayerSize = 16;
	NeuralNetwork nn(2, imageSize, hiddenLayerSize, 10);

	if(biasAndWeightFile != "" && (trainTime || loadImageFile != "")) {
		std::cout << "Loading weights and biases from " << biasAndWeightFile << std::endl;
		nn.loadWeightsAndBiasesFromFile(biasAndWeightFile);
	}

	if(trainTime > 0) {

		trainNeuralNetwork(nn, trainTime);

		std::cout << "Saving biases and weights to file 'output'" << std::endl;
		nn.writeWeightsAndBiasesToFile("output");
	}

	if(loadImageFile != "") {
		DataLoader dataLoader(loadImageFile, imageWidth, imageHeight);

		nn.setInputLayer(dataLoader.getImage(), imageSize);

		nn.propogateForward();

		std::cout << "The number is: " << nn.getAnswer() << std::endl;
	}
}