#pragma once
#include <iostream>

#include <Eigen/Dense>

void drawLayer(Eigen::MatrixXd& layer, int collumns) {
	for (int i = 0, counter = 0; i < layer.rows(); ++i) {
		for (int j = 0; j < layer.cols(); ++j, ++counter) {
			if (layer(i, j) == 0) {
				std::cout << " ";
			}
			else {
				std::cout << "X";
			}

			if (counter % collumns == 0) {
				std::cout << std::endl;
			}
		}
	}
}