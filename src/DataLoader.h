#pragma once
#include <string>

class DataLoader {
public:
	const int imageDataIndex, lableDataIndex;
	std::string imageData;
	std::string lableData;

public:
	DataLoader(std::string imageDataPath, std::string lableDataPath);

	char* getImage(int imageNumber);
	int getLable(int lableNumber);

private:
	std::string loadFile(std::string filePath);
};