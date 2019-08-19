#include "DataLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"

DataLoader::DataLoader(std::string imageDataPath, std::string lableDataPath) {
	imageData = loadTrainingFile(imageDataPath);
	lableData = loadTrainingFile(lableDataPath);
}

DataLoader::DataLoader(std::string imagePath, int desiredWidth, int desiredHeight) {
	imageData = loadImageFile(imagePath, desiredWidth, desiredHeight);
}

char* DataLoader::getImage() {
	return &imageData[0];
}

char* DataLoader::getTrainingImage(int imageNumber, int imageSize) {
	int index = imageDataIndex + imageNumber * imageSize;
	
	if (imageData.size() < imageNumber) {
		std::cout << "image does not exist" << std::endl;
		return nullptr;
	}

	return &imageData[index];
}

int DataLoader::getTrainingLable(int lableNumber) {
	int index = lableDataIndex + lableNumber;

	if (lableData.size() < lableNumber) {
		std::cout << "lable does not exist" << std::endl;
		return -1;
	}

	return lableData[index];
}

std::string DataLoader::loadTrainingFile(std::string filePath) {
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

std::string DataLoader::loadImageFile(std::string imagePath, int desiredWidth, int desiredHeight) {
	int imageX, imageY, comp;
	unsigned char* img = stbi_load(imagePath.c_str(), &imageX, &imageY, &comp, 0);
	if(img == nullptr) {
		std::cout << "Could not load file " << imagePath << std::endl;
		stbi_image_free(img);
		return "";
	}
	if(imageX != desiredWidth || imageY != desiredHeight) {
		std::cout << "Image has incorrect dimensions" << std::endl;
		stbi_image_free(img);
		return "";
	}

	int imageSize = desiredWidth * desiredHeight;
	char output[imageSize];
	for(int i = 0, j = 0; i < imageSize; ++i, j += comp) {
		output[i] = img[j];
	}
	stbi_image_free(img);

	return std::string(output, imageSize);
}