
#pragma warning(disable:4996)

#include <omp.h>
#include <vector>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "parallelsurf/Image.h"
#include "parallelsurf/WaveFilter.h"
#include "parallelsurf/KeyPoint.h"
#include "parallelsurf/BoxFilter.h"
#include "parallelsurf/KeyPointDetector.h"

#include "Surf.h"

// --- Image implementations (from parallelsurf/Image.cpp, merged to avoid duplicate symbols) ---

namespace ParallelSurf
{
	bool Image::DoRandomInit = false;

	Image::Image(const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight)
	{
		init(iPixels, iWidth, iHeight);
	}

	Image::Image(cv::Mat _Img)
	{
		int height = _Img.rows;
		int width = _Img.cols;
		init(_Img, width, height);
	}

	void Image::init(const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight)
	{
		_width = iWidth;
		_height = iHeight;
		this->_pixels = const_cast<unsigned char**>(iPixels);
		_ii = AllocateImage(_width + 1, _height + 1);
		buildIntegralImage();
	}

	void Image::init(cv::Mat _Img, unsigned int iWidth, unsigned int iHeight)
	{
		_width = iWidth;
		_height = iHeight;
		this->_pixels = new unsigned char*[_height];
#pragma omp parallel for
		for (int i = 0; i < (int)_height; i++)
		{
			_pixels[i] = new unsigned char[_width];
		}
		if (_Img.channels() >= 3)
		{
			cv::cvtColor(_Img, _Img, cv::COLOR_BGR2GRAY);
		}
#pragma omp parallel for
		for (int i = 0; i < (int)_height; i++)
		{
			for (unsigned int j = 0; j < _width; j++)
			{
				_pixels[i][j] = _Img.at<unsigned char>(i, j);
			}
		}
		_ii = AllocateImage(_width + 1, _height + 1);
		buildIntegralImage();
	}

	void Image::clean()
	{
		if (_ii)
		{
			DeallocateImage(_ii, _height + 1);
		}
		_ii = 0;
	}

	Image::~Image()
	{
		clean();
#pragma omp parallel for
		for (int i = 0; i < (int)_height; i++)
		{
			delete[] this->_pixels[i];
		}
		delete[] _pixels;
		_pixels = nullptr;
	}

	void Image::buildIntegralImage()
	{
		for (unsigned int i = 0; i <= _width; ++i)
			_ii[0][i] = 0;
		for (unsigned int i = 0; i <= _height; ++i)
			_ii[i][0] = 0;

		static const double norm = 1.0 / 255.0;

		int numCPUs = omp_get_num_procs();
		int maxThreads = omp_get_max_threads();

		if ((numCPUs > 2) && (maxThreads > 2))
		{
#pragma omp parallel for
			for (int i = 1; i <= (int)_height; ++i)
			{
				for (unsigned int j = 1; j <= _width; ++j)
				{
					_ii[i][j] = norm * double(_pixels[i - 1][j - 1]) + _ii[i][j - 1];
				}
			}
#pragma omp parallel for
			for (int j = 1; j <= (int)_width; ++j)
			{
				for (unsigned int i = 1; i <= _height; ++i)
				{
					_ii[i][j] += _ii[i - 1][j];
				}
			}
		}
		else
		{
			for (unsigned int i = 1; i <= _height; ++i)
			{
				for (unsigned int j = 1; j <= _width; ++j)
				{
					_ii[i][j] = norm * double(_pixels[i - 1][j - 1]) +
						_ii[i - 1][j] + _ii[i][j - 1] - _ii[i - 1][j - 1];
				}
			}
		}
	}

	double** Image::AllocateImage(unsigned int iWidth, unsigned int iHeight)
	{
		double ** aImagePtr = new double*[iHeight];
		for (unsigned int i = 0; i < iHeight; ++i)
		{
			aImagePtr[i] = new double[iWidth];
		}
		if (DoRandomInit)
		{
			for (unsigned int j = 0; j < iHeight; ++j)
			{
				for (unsigned int i = 0; i < iWidth; ++i)
				{
					aImagePtr[j][i] = double(std::rand()) / double(RAND_MAX) * 1000000;
				}
			}
		}
		return aImagePtr;
	}

	void Image::DeallocateImage(double **iImagePtr, unsigned int iHeight)
	{
		for (unsigned int i = 0; i < iHeight; ++i)
		{
			delete[] iImagePtr[i];
		}
		delete[] iImagePtr;
	}

	void Image::setDoRandomInit(bool iDoRandomInit)
	{
		DoRandomInit = iDoRandomInit;
	}
}


//define insertor class which collects keypoints in a vector
class PsKeyPointVectInsertor : public ParallelSurf::KeyPointInsertor
{

public:
	PsKeyPointVectInsertor(std::vector<ParallelSurf::KeyPoint>& keyPoints) : m_KeyPoints(keyPoints) {};
	inline virtual void operator() (const ParallelSurf::KeyPoint &keyPoint)
	{
		m_KeyPoints.push_back(keyPoint);
	}
private:
	std::vector<ParallelSurf::KeyPoint>& m_KeyPoints;
};



namespace cvg
{

	const double Surf::kBaseSigma = 1.2;


	Surf::Surf(/*Image &img,*/ bool _extended /*= false*/) :_image(Image())
	{

		// initialize default values
		_maxScales = 5;		// number of scales : 9x9, 15x15, 21x21, 27x27, ...
		_maxOctaves = 4;	// number of octaves

		_scoreThreshold = 0.1;

		_initialBoxFilterSize = 3;
		_scaleOverlap = 3;

		//--描述子参数

		// init some parameters
		_subRegions = 4;

		_vecLen = 4;

		if (_extended)
		{
			_vecLen = 8;
		}

		_magFactor = 12.0 / _subRegions;

		//this->_image = Image();

	}

	Surf::~Surf()
	{

	}



	void Surf::compute(std::vector<ParallelSurf::KeyPoint>& _Keypoints)
	{

		assert(_Keypoints.size() > 0);

		std::cout << "Computing descriptors..";
		this->makeDescriptors(_Keypoints.begin(), _Keypoints.end());
		std::cout << "finished." << std::endl;


	}





	/*KeyPointDetector::KeyPointDetector()
	{
		// initialize default values
		_maxScales = 5;		// number of scales : 9x9, 15x15, 21x21, 27x27, ...
		_maxOctaves = 4;	// number of octaves

		_scoreThreshold = 0.1;

		_initialBoxFilterSize = 3;
		_scaleOverlap = 3;
	}*/


	void Surf::detect(
		ParallelSurf::Image& iImage,
		std::vector<ParallelSurf::KeyPoint>& _Keypoints)
	{

		this->_image = iImage;

		//Keypoint storage
		//std::vector<ParallelSurf::KeyPoint> keyPoints;

		//Insertor inserts into keyPoints
		PsKeyPointVectInsertor iInsertor(_Keypoints);

		// allocate lots of memory for the scales
		double *** aSH = new double**[_maxScales];
		for (int s = 0; s < _maxScales; ++s)
		{
			aSH[s] = ParallelSurf::Image::AllocateImage(iImage.getWidth(), iImage.getHeight());
		}

		// init the border size
		int * aBorderSize = new int[_maxScales];

		int aMaxima = 0;

		// base size + 3 times first increment for step back
		// for the first octave 9x9, 15x15, 21x21, 27x27, 33x33
		// for the second 21x21, 33x33, 45x45 ...

		std::cout << "Detecting keypoints.. ";

		// go through all the octaves
		for (int o = 0; o < _maxOctaves; ++o)
		{
			// calculate the pixel step on the image, and the image size
			int aPixelStep = 1 << o;	// 2^aOctaveIt
			int aOctaveWidth = iImage.getWidth() / aPixelStep;	// integer division
			int aOctaveHeight = iImage.getHeight() / aPixelStep;	// integer division

																	// fill each scale matrices
			for (int s = 0; s < _maxScales; ++s)
			{
				// calculate the border for this scale
				aBorderSize[s] = getBorderSize(o, s);

				// create a box filter of the correct size for each thread
				std::vector<ParallelSurf::BoxFilter > aBoxFilters;
				aBoxFilters.reserve(omp_get_max_threads());
				for (int i = 0; i < omp_get_max_threads(); ++i)
				{
					aBoxFilters.push_back(ParallelSurf::BoxFilter(getFilterSize(o, s), iImage));
				}

				// fill the hessians
				int aEy = aOctaveHeight - 2 * aBorderSize[s];
				int aEx = aOctaveWidth - aBorderSize[s];

				int yIt;

				//initialize the memory with random values (only used for testing/debugging)
				if (ParallelSurf::Image::getDoRandomInit())
				{

#pragma omp parallel for
					for (int y = 0; y < aOctaveHeight; y++)
					{
						for (int x = 0; x < aOctaveWidth; ++x)
						{
							aSH[s][y][x] = double(rand()) / double(RAND_MAX) * 1000000;
						}
					}
				}

#pragma omp parallel for
				for (yIt = 0; yIt < aEy; yIt++)
				{
					int y = aBorderSize[s] + yIt;
					int aYPS = y * aPixelStep;
					int t = omp_get_thread_num();

					aBoxFilters[t].setY(aYPS);

					int aXPS = aBorderSize[s] * aPixelStep;
					for (int x = aBorderSize[s]; x < aEx; ++x)
					{
						aSH[s][y][x] = aBoxFilters[t].getDetWithX(aXPS);
						aXPS += aPixelStep;
					}
				}
			}

			// detect the feature points with a 3x3x3 neighborhood non-maxima suppression
			for (int aSIt = 1; aSIt < (_maxScales - 1); aSIt += 2)
			{
				int aBS = aBorderSize[aSIt + 1];

				int aNextScale = aSIt < _maxScales - 2 ? aSIt + 2 : aSIt + 1;
				int aNextBS = aBorderSize[aNextScale];

#pragma omp parallel for
				for (int aYIt = aBS + 1; aYIt < aOctaveHeight - aBS - 1; aYIt += 2)
				{
					for (int aXIt = aBS + 1; aXIt < aOctaveWidth - aBS - 1; aXIt += 2)
					{
						// find the maximum in the 2x2x2 cube
						double aTab[8];

						// get the values in a
						aTab[0] = aSH[aSIt][aYIt][aXIt];
						aTab[1] = aSH[aSIt][aYIt][aXIt + 1];
						aTab[2] = aSH[aSIt][aYIt + 1][aXIt];
						aTab[3] = aSH[aSIt][aYIt + 1][aXIt + 1];
						aTab[4] = aSH[aSIt + 1][aYIt][aXIt];
						aTab[5] = aSH[aSIt + 1][aYIt][aXIt + 1];
						aTab[6] = aSH[aSIt + 1][aYIt + 1][aXIt];
						aTab[7] = aSH[aSIt + 1][aYIt + 1][aXIt + 1];

						// find the max index without using a loop.
						int a04 = (aTab[0] > aTab[4] ? 0 : 4);
						int a15 = (aTab[1] > aTab[5] ? 1 : 5);
						int a26 = (aTab[2] > aTab[6] ? 2 : 6);
						int a37 = (aTab[3] > aTab[7] ? 3 : 7);
						int a0426 = (aTab[a04] > aTab[a26] ? a04 : a26);
						int a1537 = (aTab[a15] > aTab[a37] ? a15 : a37);
						int aMaxIdx = (aTab[a0426] > aTab[a1537] ? a0426 : a1537);

						// calculate approximate threshold
						double aApproxThres = _scoreThreshold * 0.8;

						double aScore = aTab[aMaxIdx];

						// check found point against threshold
						if (aScore < aApproxThres)
						{
							continue;
						}

						// verify that other missing points in the 3x3x3 cube are also below treshold
						int aXShift = 2 * (aMaxIdx & 1) - 1;
						int aXAdj = aXIt + (aMaxIdx & 1);
						aMaxIdx >>= 1;

						int aYShift = 2 * (aMaxIdx & 1) - 1;
						int aYAdj = aYIt + (aMaxIdx & 1);
						aMaxIdx >>= 1;

						int aSShift = 2 * (aMaxIdx & 1) - 1;
						int aSAdj = aSIt + (aMaxIdx & 1);

						// skip too high scale ajusting
						if (aSAdj == (int)_maxScales - 1)
						{
							continue;
						}

						//if we adjusted the scale to aSIt+1, then we also have to check 
						//for the border size of the next higher scale
						if ((aSShift == 1) &&
							((aXAdj < aNextBS + 1) ||
							(aXAdj >= aOctaveWidth - aNextBS - 1) ||
								(aYAdj < aNextBS + 1) ||
								(aYAdj >= aOctaveHeight - aNextBS - 1))
							)
						{
							continue;
						}

						if ((aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj - 1] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj + 1] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj][aXAdj - 1] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj][aXAdj] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj][aXAdj + 1] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj - 1] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj] > aScore) ||
							(aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj + 1] > aScore) ||

							(aSH[aSAdj][aYAdj + aYShift][aXAdj - 1] > aScore) ||
							(aSH[aSAdj][aYAdj + aYShift][aXAdj] > aScore) ||
							(aSH[aSAdj][aYAdj + aYShift][aXAdj + 1] > aScore) ||
							(aSH[aSAdj][aYAdj][aXAdj + aXShift] > aScore) ||
							(aSH[aSAdj][aYAdj - aYShift][aXAdj + aXShift] > aScore) ||

							(aSH[aSAdj - aSShift][aYAdj + aYShift][aXAdj - 1] > aScore) ||
							(aSH[aSAdj - aSShift][aYAdj + aYShift][aXAdj] > aScore) ||
							(aSH[aSAdj - aSShift][aYAdj + aYShift][aXAdj + 1] > aScore) ||
							(aSH[aSAdj - aSShift][aYAdj][aXAdj + aXShift] > aScore) ||
							(aSH[aSAdj - aSShift][aYAdj - aYShift][aXAdj + aXShift] > aScore))
						{
							continue;
						}

						// fine tune the location
						double aX = aXAdj;
						double aY = aYAdj;
						double aS = aSAdj;

						// try to fine tune, restore the values if it failed
						// if the returned value is true,  keep the point, else drop it.
						if (!fineTuneExtrema(aSH, aXAdj, aYAdj, aSAdj, aX, aY, aS, aScore, aOctaveWidth, aOctaveHeight, aBorderSize[aSAdj]))
						{
							continue;
						}

						// recheck the updated score
						if (aScore < _scoreThreshold)
						{
							continue;
						}

						// adjust the values
						aX *= aPixelStep;
						aY *= aPixelStep;
						aS = ((2 * aS * aPixelStep) + _initialBoxFilterSize + (aPixelStep - 1) * _maxScales) / 3.0; // this one was hard to guess...

																													// store the point
						int aTrace;
						if (!calcTrace(iImage, aX, aY, aS, aTrace))
						{
							continue;
						}

						aMaxima++;

#pragma omp critical
						{
							// do something with the keypoint depending on the insertor
							iInsertor(ParallelSurf::KeyPoint(aX, aY, aS * kBaseSigma, aScore, aTrace));
						}

					}
				}
			}
		}

		std::cout << "finished." << std::endl;

		std::cout << "Found " << _Keypoints.size() << " keypoints." << std::endl;

		//计算特征点方向
		this->assignOrientations(_Keypoints.begin(), _Keypoints.end());

		// deallocate memory of the scale images
		for (int s = 0; s < _maxScales; ++s)
		{
			ParallelSurf::Image::DeallocateImage(aSH[s], iImage.getHeight());
		}
	}



	bool Surf::fineTuneExtrema(
		double *** iSH, int iX, int iY, int iS,
		double& oX, double& oY, double& oS, double& oScore,
		int iOctaveWidth, int iOctaveHeight, int iBorder)
	{
		// maximum fine tune iterations
		const int	kMaxFineTuneIters = 6;

		// shift from the initial position for X and Y (only -1 or + 1 during the iterations).
		int aX = iX;
		int aY = iY;
		int aS = iS;

		int aShiftX = 0;
		int aShiftY = 0;

		// current deviations
		double aDx = 0, aDy = 0, aDs = 0;

		//result vector
		double aV[3];	//(x,y,s)

		for (int aIter = 0; aIter < kMaxFineTuneIters; ++aIter)
		{
			// update the extrema position
			aX += aShiftX;
			aY += aShiftY;

			// create the problem matrix
			double aM[3][3]; //symetric, no ordering problem.

							 // fill the result vector with gradient from pixels differences (negate to prepare system solve)
			aDx = (iSH[aS][aY][aX + 1] - iSH[aS][aY][aX - 1]) * 0.5;
			aDy = (iSH[aS][aY + 1][aX] - iSH[aS][aY - 1][aX]) * 0.5;
			aDs = (iSH[aS + 1][aY][aX] - iSH[aS - 1][aY][aX]) * 0.5;

			aV[0] = -aDx;
			aV[1] = -aDy;
			aV[2] = -aDs;

			// fill the matrix with values of the hessian from pixel differences
			aM[0][0] = iSH[aS][aY][aX - 1] - 2.0 * iSH[aS][aY][aX] + iSH[aS][aY][aX + 1];
			aM[1][1] = iSH[aS][aY - 1][aX] - 2.0 * iSH[aS][aY][aX] + iSH[aS][aY + 1][aX];
			aM[2][2] = iSH[aS - 1][aY][aX] - 2.0 * iSH[aS][aY][aX] + iSH[aS + 1][aY][aX];

			aM[0][1] = aM[1][0] = (iSH[aS][aY + 1][aX + 1] + iSH[aS][aY - 1][aX - 1] - iSH[aS][aY + 1][aX - 1] - iSH[aS][aY - 1][aX + 1]) * 0.25;
			aM[0][2] = aM[2][0] = (iSH[aS + 1][aY][aX + 1] + iSH[aS - 1][aY][aX - 1] - iSH[aS + 1][aY][aX - 1] - iSH[aS - 1][aY][aX + 1]) * 0.25;
			aM[1][2] = aM[2][1] = (iSH[aS + 1][aY + 1][aX] + iSH[aS - 1][aY - 1][aX] - iSH[aS + 1][aY - 1][aX] - iSH[aS - 1][aY + 1][aX]) * 0.25;

			// solve the linear system. results are in aV. exit with error if a problem happened
			if (!ParallelSurf::Math::SolveLinearSystem33(aV, aM))
			{
				return false;
			}


			// ajust the shifts with the results and stop if no significant change

			if (aIter < kMaxFineTuneIters - 1)
			{
				aShiftX = 0;
				aShiftY = 0;

				if (aV[0] > 0.6 && aX < (int)(iOctaveWidth - iBorder - 2))
				{
					aShiftX++;
				}
				else if (aV[0] < -0.6 && aX > (int) iBorder + 1)
				{
					aShiftX--;
				}

				if (aV[1] > 0.6 && aY < (int)(iOctaveHeight - iBorder - 2))
				{
					aShiftY++;
				}
				else if (aV[1] < -0.6 && aY > (int) iBorder + 1)
				{
					aShiftY--;
				}

				if (aShiftX == 0 && aShiftY == 0)
				{
					break;
				}
			}
		}

		// update the score
		oScore = iSH[aS][aY][aX] + 0.5 * (aDx * aV[0] + aDy * aV[1] + aDs * aV[2]);

		// reject too big deviation in last step (unfinished job).
		if (ParallelSurf::Math::Abs(aV[0]) > 1.5 ||
			ParallelSurf::Math::Abs(aV[1]) > 1.5 ||
			ParallelSurf::Math::Abs(aV[2]) > 1.5)
		{
			return false;
		}

		// put the last deviation (not integer :) to the output
		oX = aX + aV[0];
		oY = aY + aV[1];
		oS = iS + aV[2];

		return true;
	}


	int	Surf::getFilterSize(int iOctave, int iScale)
	{
		int aScaleShift = 2 << iOctave;

		return	_initialBoxFilterSize + (aScaleShift - 2) *
			(_maxScales - _scaleOverlap) + aScaleShift * iScale;
	}



	int	Surf::getBorderSize(int iOctave, int iScale)
	{
		int aScaleShift = 2 << iOctave;

		if (iScale <= 2)
		{
			int aMult = (iOctave == 0 ? 1 : 2);

			return (getFilterSize(iOctave, 1) + aMult * aScaleShift) * 3 / aScaleShift + 1;
		}

		return getFilterSize(iOctave, iScale) * 3 / aScaleShift + 1;
	}


	bool Surf::calcTrace(
		ParallelSurf::Image& iImage, double iX,
		double iY, double iScale, int& oTrace)
	{
		int aRX = ParallelSurf::Math::Round(iX);
		int aRY = ParallelSurf::Math::Round(iY);

		ParallelSurf::BoxFilter aBox(3 * iScale, iImage);

		if (!aBox.checkBounds(aRX, aRY))
		{
			return false;
		}

		aBox.setY(aRY);
		double aTrace = aBox.getDxxWithX(aRX) + aBox.getDyyWithX(aRX);
		oTrace = (aTrace <= 0.0 ? -1 : 1);

		return true;
	}

	//-------------------//特征描述子-----------------------------------


	ParallelSurf::LUT<0, 83> Exp1(std::exp, 0.5, -0.08);
	ParallelSurf::LUT<0, 40> Exp2(std::exp, 0.5, -0.125);

	bool operator < (const response a, const response b)
	{
		return a.orientation < b.orientation;
	}


	/*KeyPointDescriptor::KeyPointDescriptor(Image& iImage, bool iExtended) :
		_image(iImage), _extended(iExtended)
	{
		// init some parameters
		_subRegions = 4;

		_vecLen = 4;
		if (_extended)
		{
			_vecLen = 8;
		}

		_magFactor = 12.0 / _subRegions;

	}*/

	void Surf::makeDescriptor(ParallelSurf::KeyPoint& ioKeyPoint) const
	{
		// create a descriptor context
		ParallelSurf::KeyPointDescriptorContext aCtx(_subRegions, _vecLen, ioKeyPoint._ori);

		// create the storage in the keypoint
		ioKeyPoint._vec.resize(getDescriptorLength());

		// assign the orientation
		//assignOrientation(ioKeyPoint);

		// create a vector
		createDescriptor(aCtx, ioKeyPoint);

		// transform back to vector

		// fill the vector with the values of the square...
		// remember the shift of 1 to drop outborders.
		size_t aV = 0;
		for (int aYIt = 1; aYIt < _subRegions + 1; ++aYIt)
		{
			for (int aXIt = 1; aXIt < _subRegions + 1; ++aXIt)
			{
				for (int aVIt = 0; aVIt < _vecLen; ++aVIt)
				{
					double a = aCtx._cmp[aYIt][aXIt][aVIt];
					ioKeyPoint._vec[aV] = a;
					aV++;
				}
			}
		}

		// normalize
		ParallelSurf::Math::Normalize(ioKeyPoint._vec);

	}

	void Surf::assignOrientation(ParallelSurf::KeyPoint& ioKeyPoint) const
	{
		unsigned int aRX = ParallelSurf::Math::Round(ioKeyPoint._x);
		unsigned int aRY = ParallelSurf::Math::Round(ioKeyPoint._y);
		int aStep = (int)(ioKeyPoint._scale + 0.8);

		//ParallelSurf::Image img = _image;

		WaveFilter aWaveFilter(2.0 * ioKeyPoint._scale + 1.6, _image);

		std::vector< cvg::response > aRespVector;
		aRespVector.reserve(253);

		// compute haar wavelet responses in a circular neighborhood of 6s
		for (int aYIt = -9; aYIt <= 9; aYIt++)
		{
			int aSY = aRY + aYIt * aStep;
			for (int aXIt = -9; aXIt <= 9; aXIt++)
			{
				int aSX = aRX + aXIt * aStep;

				// keep points in a circular region of diameter 6s
				unsigned int aSqDist = aXIt * aXIt + aYIt * aYIt;
				if (aSqDist <= 81 && aWaveFilter.checkBounds(aSX, aSY))
				{
					double aWavX = aWaveFilter.getWx(aSX, aSY);
					double aWavY = aWaveFilter.getWy(aSX, aSY);
					double aWavResp = std::sqrt(aWavX * aWavX + aWavY * aWavY);
					if (aWavResp > 0)
					{
						double aAngle = std::atan2(aWavY, aWavX);
						cvg::response resp;
						resp.orientation = aAngle;
						resp.magnitude = aWavResp * Exp1(aSqDist);
						aRespVector.push_back(resp);
					}
				}
			}
		}

		//assert ( aRespVector.size() <= 253 );

		//no wavelet responses != 0, can't calculate orientation
		if (aRespVector.size() == 0)
		{
			ioKeyPoint._ori = 0;
			return;
		}

		//sort responses by orientation
		std::sort(aRespVector.begin(), aRespVector.end());

		//estimate orientation using a sliding window of PI/3
		cvg::response aMax = aRespVector[0];
		double aMagnitudeSum = aRespVector[0].magnitude;
		double aOrientationSum = aRespVector[0].orientation * aMagnitudeSum;

		size_t aWindowBegin = 0;
		size_t aWindowEnd = 0;

		float aOriAdd = 0;

		while (aWindowBegin < aRespVector.size())
		{
			float aWindowSize = (aRespVector[aWindowEnd].orientation + aOriAdd)
				- aRespVector[aWindowBegin].orientation;

			if (aWindowSize < PI / 3)
			{
				//found new max.
				if (aMagnitudeSum > aMax.magnitude)
				{
					aMax.magnitude = aMagnitudeSum;
					aMax.orientation = aOrientationSum;
				}
				aWindowEnd++;

				if (aWindowEnd >= aRespVector.size())
				{
					aWindowEnd = 0;
					aOriAdd += 2 * PI;
				}

				aMagnitudeSum += aRespVector[aWindowEnd].magnitude;
				aOrientationSum += aRespVector[aWindowEnd].magnitude *
					(aRespVector[aWindowEnd].orientation + aOriAdd);
			}
			else
			{
				aMagnitudeSum -= aRespVector[aWindowBegin].magnitude;
				aOrientationSum -= aRespVector[aWindowBegin].magnitude *
					aRespVector[aWindowBegin].orientation;

				aWindowBegin++;
			}
		}

		ioKeyPoint._ori = aMax.orientation / aMax.magnitude;
	}



	void Surf::createDescriptor(
		ParallelSurf::KeyPointDescriptorContext& iCtx,
		ParallelSurf::KeyPoint& ioKeyPoint) const
	{
		// create the vector of features by analyzing a square patch around the point.
		// for this the current patch (x,y) will be translated in rotated coordinates (u,v)

		double aX = ioKeyPoint._x;
		double aY = ioKeyPoint._y;
		double aS = ioKeyPoint._scale * 1.65; // multiply by this nice constant

		// make integer values from double ones
		int aIntX = ParallelSurf::Math::Round(aX);
		int aIntY = ParallelSurf::Math::Round(aY);
		int aIntS = ParallelSurf::Math::Round(aS / 2.0);

		if (aIntS < 1)
		{
			aIntS = 1;
		}

		// calc subpixel shift
		double aSubX = aX - aIntX;
		double aSubY = aY - aIntY;

		// calc subpixel shift in rotated coordinates
		double aSubV = iCtx._cos * aSubY + iCtx._sin * aSubX;
		double aSubU = -iCtx._sin * aSubY + iCtx._cos * aSubX;

		// calc step of sampling
		double aStepSample = aS * _magFactor;

		// make a bounding box around the rotated patch square.
		double aRadius = (1.414 * aStepSample) * (_subRegions + 1) / 2.0;
		int aIntRadius = ParallelSurf::Math::Round(aRadius / aIntS);
		if (aS < 1)
		{
			aS = 1;
		}

		// go through all the pixels in the bounding box
		for (int aYIt = -aIntRadius; aYIt <= aIntRadius; ++aYIt)
		{
			for (int aXIt = -aIntRadius; aXIt <= aIntRadius; ++aXIt)
			{
				// calculate U,V rotated values from X,Y taking in account subpixel correction
				// divide by step sample to put in index units
				double aU = (((-iCtx._sin * aYIt + iCtx._cos * aXIt) * aIntS) - aSubU) / aStepSample;
				double aV = (((iCtx._cos * aYIt + iCtx._sin * aXIt) * aIntS) - aSubV) / aStepSample;

				// compute location of sample in terms of real array coordinates
				double aUIdx = _subRegions / 2.0 - 0.5 + aU;
				double aVIdx = _subRegions / 2.0 - 0.5 + aV;

				// test if some bins will be filled.
				if (aUIdx >= -1.0 && aUIdx < _subRegions &&
					aVIdx >= -1.0 && aVIdx < _subRegions)
				{
					int aXSample = aIntS * aXIt + aIntX;
					int aYSample = aIntS * aYIt + aIntY;
					int aStep = (int)aS;

					//ParallelSurf::Image img = _image;

					WaveFilter aWaveFilter(aStep,/* img*/ _image);

					if (!aWaveFilter.checkBounds(aXSample, aYSample))
					{
						continue;
					}

					double aExp = Exp2((int)(aV * aV + aU * aU));

					double aWavX = aWaveFilter.getWx(aXSample, aYSample) * aExp;
					double aWavY = aWaveFilter.getWy(aXSample, aYSample) * aExp;

					double aWavXR = (iCtx._cos * aWavX + iCtx._sin * aWavY);
					double aWavYR = (iCtx._sin * aWavX - iCtx._cos * aWavY);

					// due to the rotation, the patch has to be dispatched in 2 bins in each direction
					// get the bins and weight for each of them
					// shift by 1 to avoid checking bounds
					const int aBin1U = (int)(aUIdx + 1.0);
					const int aBin2U = aBin1U + 1;
					const int aBin1V = (int)(aVIdx + 1.0);
					const int aBin2V = aBin1V + 1;

					const double aWeightBin1U = aBin1U - aUIdx;
					const double aWeightBin2U = 1 - aWeightBin1U;

					const double aWeightBin1V = aBin1V - aVIdx;
					const double aWeightBin2V = 1 - aWeightBin1V;

					if (_extended)
					{
						int aBin = (aWavYR <= 0) ? 0 : 1;
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aWavXR;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aWavXR;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aWavXR;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aWavXR;

						aBin += 2;
						double aVal = std::fabs(aWavXR);
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aVal;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aVal;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aVal;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aVal;

						aBin = (aWavXR <= 0) ? 4 : 5;
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aWavYR;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aWavYR;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aWavYR;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aWavYR;

						aBin += 2;
						aVal = std::fabs(aWavYR);
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aVal;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aVal;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aVal;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aVal;

					}
					else
					{
						int aBin = (aWavXR <= 0) ? 0 : 1;
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aWavXR;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aWavXR;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aWavXR;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aWavXR;

						aBin = (aWavYR <= 0) ? 2 : 3;
						iCtx._cmp[aBin1V][aBin1U][aBin] += aWeightBin1V * aWeightBin1U * aWavYR;
						iCtx._cmp[aBin2V][aBin1U][aBin] += aWeightBin2V * aWeightBin1U * aWavYR;
						iCtx._cmp[aBin1V][aBin2U][aBin] += aWeightBin1V * aWeightBin2U * aWavYR;
						iCtx._cmp[aBin2V][aBin2U][aBin] += aWeightBin2V * aWeightBin2U * aWavYR;

					}
				}
			}
		}
	}


	int Surf::getDescriptorLength() const
	{
		return _vecLen * _subRegions * _subRegions;
	}


}

// --- MathStuff implementations (merged to avoid duplicate symbols) ---

namespace ParallelSurf
{

	bool Math::SolveLinearSystem33(double* solution, double sq[3][3])
	{
		const int size = 3;
		int row, col, c, pivot = 0, i;
		double maxc, coef, temp, mult, val;

		for (col = 0; col < size - 1; col++)
		{
			maxc = -1.0;
			for (row = col; row < size; row++)
			{
				coef = sq[row][col];
				coef = (coef < 0.0 ? -coef : coef);
				if (coef > maxc)
				{
					maxc = coef;
					pivot = row;
				}
			}

			if (pivot != col)
			{
				for (i = 0; i < size; i++)
				{
					temp = sq[pivot][i];
					sq[pivot][i] = sq[col][i];
					sq[col][i] = temp;
				}
				temp = solution[pivot];
				solution[pivot] = solution[col];
				solution[col] = temp;
			}

			for (row = col + 1; row < size; row++)
			{
				mult = sq[row][col] / sq[col][col];
				for (c = col; c < size; c++)
				{
					sq[row][c] -= mult * sq[col][c];
				}
				solution[row] -= mult * solution[col];
			}
		}

		for (row = size - 1; row >= 0; row--)
		{
			val = solution[row];
			for (col = size - 1; col > row; col--)
			{
				val -= solution[col] * sq[row][col];
			}
			solution[row] = val / sq[row][row];
		}

		return true;
	}

	bool Math::Normalize(std::vector<double>& iVec)
	{
		int i;
		double val, fac, sqlen = 0.0;

		for (i = 0; i < iVec.size(); i++)
		{
			val = iVec[i];
			sqlen += val * val;
		}

		if (sqlen == 0.0)
			return false;

		fac = 1.0 / std::sqrt(sqlen);
		for (i = 0; i < iVec.size(); i++)
		{
			iVec[i] *= fac;
		}

		return true;
	}

}

// --- KeyPointDescriptorContext implementations (merged) ---

namespace ParallelSurf
{
	KeyPointDescriptorContext::KeyPointDescriptorContext(
		int iSubRegions, int iVecLen,
		double iOrientation) :
		_subRegions(iSubRegions), _sin(sin(iOrientation)), _cos(cos(iOrientation))
	{
		int aExtSub = _subRegions + 2;
		_cmp = new double **[aExtSub];
		for (int aYIt = 0; aYIt < aExtSub; ++aYIt)
		{
			_cmp[aYIt] = new double *[aExtSub];
			for (int aXIt = 0; aXIt < aExtSub; ++aXIt)
			{
				_cmp[aYIt][aXIt] = new double[iVecLen];
				for (int aVIt = 0; aVIt < iVecLen; ++aVIt)
				{
					_cmp[aYIt][aXIt][aVIt] = 0;
				}
			}
		}
	}

	KeyPointDescriptorContext::~KeyPointDescriptorContext()
	{
		int aExtSub = _subRegions + 2;
		for (int aYIt = 0; aYIt < aExtSub; ++aYIt)
		{
			for (int aXIt = 0; aXIt < aExtSub; ++aXIt)
			{
				delete[] _cmp[aYIt][aXIt];
			}
			delete[] _cmp[aYIt];
		}
	}
}
