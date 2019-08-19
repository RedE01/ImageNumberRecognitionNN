#pragma once
#include <string>

class DataLoader {
public:
	std::string imageData;
	std::string lableData;
	
private:
	const int imageDataIndex = 16, lableDataIndex = 8;

public:
	DataLoader(std::string imageDataPath, std::string lableDataPath); // Loads training data
	DataLoader(std::string imagePath, int desiredWidth, int desiredHeight); // Loads single png/jpeg/etc file

	char* getImage();
	char* getTrainingImage(int imageNumber, int imageSize);
	int getTrainingLable(int lableNumber);

private:
	std::string loadTrainingFile(std::string filePath);
	std::string loadImageFile(std::string imagePath, int desiredWidth, int desiredHeight);
};