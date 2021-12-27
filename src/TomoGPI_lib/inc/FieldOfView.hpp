/*
 * FieldOfView.hpp
 *
 *      Author: gac
 */

#ifndef FIELDOFVIEW_HPP_
#define FIELDOFVIEW_HPP_

class FieldOfView{

public:

	FieldOfView();
	virtual ~FieldOfView();

	virtual float getXFOVSize() = 0;
	virtual float getYFOVSize() = 0;
	virtual float getZFOVSize() = 0;
	virtual float getXFOVPixelNb() = 0;
	virtual	float getYFOVPixelNb() = 0;
	virtual	float getZFOVPixelNb() = 0;
	virtual void setXFOVSize(float xFOVSize) = 0;
	virtual void setYFOVSize(float yFOVSize) = 0;
	virtual void setZFOVSize(float zFOVSize) = 0;
	virtual void setXFOVPixelNb(float xFOVPixelNb) = 0;
	virtual void setYFOVPixelNb(float yFOVPixelNb) = 0;
	virtual void setZFOVPixelNb(float zFOVPixelNb) = 0;
};

class CylindricFOV : public FieldOfView{

public:

	CylindricFOV();
	CylindricFOV(float cylinderRadius, float cylinderHeight, float xFOVPixelNb, float yFOVPixelNb, float zFOVPixelNb);
	~CylindricFOV();

	CylindricFOV & operator=(const CylindricFOV &cylindricFOV);

	float getXFOVSize();
	float getYFOVSize();
	float getZFOVSize();
	float getXFOVPixelNb();
	float getYFOVPixelNb();
	float getZFOVPixelNb();
	void setXFOVSize(float xFOVSize);
	void setYFOVSize(float yFOVSize);
	void setZFOVSize(float zFOVSize);
	void setXFOVPixelNb(float xFOVPixelNb);
	void setYFOVPixelNb(float yFOVPixelNb);
	void setZFOVPixelNb(float zFOVPixelNb);



private:
	float cylinderRadius;
	float cylinderHeight;
	float xFOVPixelNb;
	float yFOVPixelNb;
	float zFOVPixelNb;
};


#endif /* FIELDOFVIEW_HPP_ */
