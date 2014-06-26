/*
 * matrix.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#include "matrix.h"
#include "stdlib.h"

Matrix::Matrix() {
	_rows = 1;
	_cols = 1;
	_size = _rows * _cols;
	_isTrans = false;
	_holdsData = true;
	_data = new float[_size];
}

Matrix::Matrix(int rows, int cols) {
	_rows = rows;
	_cols = cols;
	_size = _rows * _cols;
	_isTrans = false;
	_holdsData = true;
	_data = new float[_size];
}

Matrix::~Matrix() {
	if(_data != NULL)
		delete[] _data;
}

