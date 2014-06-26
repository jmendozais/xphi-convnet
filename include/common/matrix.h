/*
 * matrix.h
 *
 *  Created on: Jun 14, 2014
 *      Author: mac
 */

#ifndef MATRIX_H_
#define MATRIX_H_

class Matrix {
private:
    float* _data;
    long int _rows, _cols;
    long int _size;
    bool _isTrans;
    bool _holdsData;

public:
	Matrix();
	Matrix(Matrix& matrix);
	Matrix(int rows, int cols);
	virtual ~Matrix();

	inline float& getValue(int i, int j) {
		if(_isTrans)
			return _data[j*_rows + i];
		return _data[i*_rows + j];
	}

	inline float* getData() {
		return _data;
	}
};

#endif /* MATRIX_H_ */
