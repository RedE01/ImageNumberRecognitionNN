#include "NeuralNetwork.h"
#include <iostream>
#include <fstream>
#include <sstream>

double sigmoid(double input) {
	double ePowX = std::exp(input);
	double result = ePowX / (ePowX + 1.0);
	return result;
}

double sigmoidDerivative(double input) {
	double sigX = sigmoid(input);
	return sigX * (1.0 - sigX);
}

Layer::Layer(int layerSize, int previousLayerSize) : 
		layerSize(layerSize), layerMat(layerSize, 1), layerBiases(layerSize, 1), layerWeights(layerSize, previousLayerSize),
		deltaLayer(layerSize, 1), deltaBiases(layerSize, 1), deltaWeights(layerSize, previousLayerSize), layerMatNoSigmoid(layerSize, 1) {
	
	layerMat.setZero();
	layerBiases.setRandom();
	layerWeights.setRandom();

	deltaLayer.setZero();
	deltaBiases.setZero();
	deltaWeights.setZero();
	layerMatNoSigmoid.setZero();
}

void Layer::calculateLayer(Layer* previousLayer) {
	layerMatNoSigmoid = layerWeights * previousLayer->layerMat + layerBiases;

	for (int i = 0; i < layerSize; ++i) {
		layerMat(i, 0) = sigmoid(layerMatNoSigmoid(i, 0));
	}
}

NeuralNetwork::NeuralNetwork(int hiddenLayers, int inputLayerSize, int hiddenLayerSize, int outputLayerSize) {
	layers.push_back(new Layer(inputLayerSize, 0));

	int previousLayerSize = inputLayerSize;
	for (int i = 0; i < hiddenLayers; ++i) {
		layers.push_back(new Layer(hiddenLayerSize, previousLayerSize));
		previousLayerSize = hiddenLayerSize;
	}

	layers.push_back(new Layer(outputLayerSize, previousLayerSize));
}

NeuralNetwork::~NeuralNetwork() {
	for (Layer* l : layers) {
		delete l;
	}
}

void NeuralNetwork::setInputLayer(char* image, int imageSize) {
	if (imageSize != layers[0]->getLayerSize()) {
		std::cout << "Image size must be the same as input layer size" << std::endl;
		return;
	}

	for (int i = 0; i < imageSize; ++i) {
		unsigned char pixel = (unsigned char)*(image + i);
		layers[0]->layerMat(i, 0) = (double)pixel / 255.0;
	}
}

void NeuralNetwork::propogateForward() {
	for (int i = 1; i < layers.size(); ++i) {
		layers[i]->calculateLayer(layers[i - 1]);
	}
}

void NeuralNetwork::propogateBackwards(int answer) {
	int lastLayerIndex = layers.size() - 1;

	for (int layerIndex = lastLayerIndex; layerIndex > 0; --layerIndex) {
		calculateDeltaLayer(layerIndex, answer);

		calculateDeltaBiases(layerIndex);
	
		calculateDeltaWeights(layerIndex);
	}
}

void NeuralNetwork::calculateDeltaWeights(int layerIndex) {
	int previousLayerIndex = layerIndex - 1;
	int layerSize = layers[layerIndex]->getLayerSize(), previousLayerSize = layers[previousLayerIndex]->getLayerSize();

	for (int j = 0; j < layerSize; ++j) {
		double delCdelA = layers[layerIndex]->deltaLayer(j, 0);
		double delAdelD = sigmoidDerivative(layers[layerIndex]->layerMatNoSigmoid(j, 0));
		double delCdelB = delCdelA * delAdelD;

		for (int k = 0; k < previousLayerSize; ++k) {
			layers[layerIndex]->deltaWeights(j, k) += delCdelB * layers[previousLayerIndex]->layerMat(k, 0);
		}
	}
}

void NeuralNetwork::calculateDeltaLayer(int layerIndex, int answer) {
	int layerSize = layers[layerIndex]->getLayerSize();
	int lastLayerIndex = layers.size() - 1;

	if (layerIndex == lastLayerIndex) {

		for (int j = 0; j < layerSize; ++j) {
			double desired = (j == answer) ? 1.0 : 0.0;

			layers[layerIndex]->deltaLayer(j, 0) = 2.0 * (layers[layerIndex]->layerMat(j, 0) - desired);
		}
	}
	else {
		int nextLayerIndex = layerIndex + 1, nextLayerSize = layers[nextLayerIndex]->getLayerSize();

		for (int k = 0; k < layerSize; ++k) {
			layers[layerIndex]->deltaLayer(k, 0) = 0.0;
			for (int j = 0; j < nextLayerSize; ++j) {
				double delCdelA = layers[nextLayerIndex]->deltaLayer(j, 0);
				double delAdelD = sigmoidDerivative(layers[nextLayerIndex]->layerMatNoSigmoid(j, 0));

				layers[layerIndex]->deltaLayer(k, 0) += delCdelA * delAdelD * layers[nextLayerIndex]->layerWeights(j, k);
			}
		}
	}
}

void NeuralNetwork::calculateDeltaBiases(int layerIndex) {
	int layerSize = layers[layerIndex]->getLayerSize();

	for (int j = 0; j < layerSize; ++j) {
		double delCdelA = layers[layerIndex]->deltaLayer(j, 0);
		double delAdelD = sigmoidDerivative(layers[layerIndex]->layerMatNoSigmoid(j, 0));

		layers[layerIndex]->deltaBiases(j, 0) += delCdelA * delAdelD;
	}
}

void NeuralNetwork::updateLayers(int batchSize, double stepSize) {
	for(int i = 0; i < layers.size(); ++i) {
		layers[i]->deltaBiases /= (double)batchSize;
		layers[i]->deltaWeights /= (double)batchSize;
		
		layers[i]->layerBiases -= layers[i]->deltaBiases * stepSize;
		layers[i]->layerWeights -= layers[i]->deltaWeights * stepSize;
		
		layers[i]->deltaBiases.setZero();
		layers[i]->deltaWeights.setZero();
	}
}

double NeuralNetwork::calculateCost(int answer) {
	double cost = 0.0;
	for (int i = 0; i < getOutputLayer()->getLayerSize(); ++i) {
		double desired = (i == answer) ? 1.0 : 0.0;
		double newCost = getOutputLayer()->layerMat(i, 0) - desired;

		cost += newCost * newCost;
	}
	return cost;
}

bool NeuralNetwork::isCorrect(int answer) {
	MatrixXd& outMat = getOutputLayer()->layerMat;

	double highest = 0.0;
	int highestIndex = 0;
	for (int i = 0; i < 10; ++i) {
		if (outMat(i, 0) > highest) {
			highest = outMat(i, 0);
			highestIndex = i;
		}
	}

	if (highestIndex == answer)
		return true;

	return false;
}

int NeuralNetwork::getAnswer() {
	int answer = 0;
	for(int i = 1; i < getOutputLayer()->getLayerSize(); ++i) {
		if(getOutputLayer()->layerMat(i, 0) > getOutputLayer()->layerMat(answer)) {
			answer = i;
		}
	}
	return answer;
}

void NeuralNetwork::writeWeightsAndBiasesToFile(std::string filename) {
	std::fstream outStream(filename, std::ios::out | std::ios::binary);

	for(int i = 0; i < layers.size(); ++i) {
		outStream.write((const char*)layers[i]->layerBiases.data(), sizeof(double) * layers[i]->layerBiases.size());
		outStream.write((const char*)layers[i]->layerWeights.data(), sizeof(double) * layers[i]->layerWeights.size());
	}

	outStream.close();
}

void NeuralNetwork::loadWeightsAndBiasesFromFile(std::string filepath) {
	std::fstream fileStream(filepath, std::ios::in | std::ios::binary);
	if(!fileStream.is_open()) {
		fileStream.close();
		std::cout << "Could not open file " << filepath << std::endl;
		return;
	}
	std::stringstream filebuffer;
	filebuffer << fileStream.rdbuf();
	fileStream.close();
	std::string file = filebuffer.str();

	int dataOffset = 0;
	for(int i = 0; i < layers.size(); ++i) {
		int biasSize = sizeof(double) * layers[i]->layerBiases.size();
		std::memcpy(layers[i]->layerBiases.data(), &file[dataOffset], biasSize);
		dataOffset += biasSize;

		int weightSize = sizeof(double) * layers[i]->layerWeights.size();
		std::memcpy(layers[i]->layerWeights.data(), &file[dataOffset], weightSize);
		dataOffset += weightSize;		
	}
}
