#include <iostream>
//#include "Debug.h"
#include "DataLoader.h"
#include "NeuralNetwork.h"
#include <chrono>

int main() {
	DataLoader dataLoader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
	int imageSize = 28 * 28, hiddenLayerSize = 16;

	NeuralNetwork nn(2, imageSize, hiddenLayerSize, 10);
	int batchSize = 100, totalBatches = 100000;

	std::cout << "begin" << std::endl;

	int correct = 0;
	for (int batches = 0; batches < totalBatches; ++batches) {
	
		for (int i = 0; i < batchSize; ++i) {
			int imageNumber = (i + batches * batchSize) % 60000;
			int answer = dataLoader.getLable(imageNumber);

			nn.setInputLayer(dataLoader.getImage(imageNumber), imageSize);
			nn.propogateForward();

			nn.propogateBackwards(answer);

			if (batches % 1000 == 0 && nn.isCorrect(answer))
				++correct;
		}

		for (int i = 0; i < nn.layers.size(); ++i) {
			nn.layers[i]->deltaBiases /= (double)batchSize;
			nn.layers[i]->deltaWeights /= (double)batchSize;

			nn.layers[i]->layerBiases -= nn.layers[i]->deltaBiases * 0.1;
			nn.layers[i]->layerWeights -= nn.layers[i]->deltaWeights * 0.1;

			nn.layers[i]->deltaBiases.setZero();
			nn.layers[i]->deltaWeights.setZero();
		}

		if (correct != 0) {
			std::cout << correct << " correct out of " << batchSize << std::endl;
			correct = 0;
		}
	}

	std::cout << "done" << std::endl;

	std::cin.ignore();

}