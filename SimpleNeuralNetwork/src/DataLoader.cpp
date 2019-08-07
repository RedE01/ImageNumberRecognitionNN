#include "DataLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>

DataLoader::DataLoader(std::string imageDataPath, std::string lableDataPath) : imageDataIndex(16), lableDataIndex(8) {
	imageData = loadFile(imageDataPath);
	lableData = loadFile(lableDataPath);
}

char* DataLoader::getImage(int imageNumber) {
	int index = imageDataIndex + imageNumber * 28 * 28;
	
	if (imageData.size() < imageNumber) {
		std::cout << "image does not exist" << std::endl;
		return nullptr;
	}

	return &imageData[index];
}

int DataLoader::getLable(int lableNumber) {
	int index = lableDataIndex + lableNumber;

	if (lableData.size() < lableNumber) {
		std::cout << "lable does not exist" << std::endl;
		return -1;
	}

	return lableData[index];
}

std::string DataLoader::loadFile(std::string filePath) {
	std::fstream filestream(filePath, std::ios::in | std::ios::binary);
	if (!filestream.is_open()) {
		std::cout << "Could not open file " << filePath << std::endl;
		return "";
	}

	std::stringstream filebuffer;
	filebuffer << filestream.rdbuf();
	filestream.close();

	return filebuffer.str();
}
