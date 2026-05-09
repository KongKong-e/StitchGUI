
#pragma once

#include <vector>

#include "parallelsurf/Image.h"

#include "parallelsurf/KeyPointDetector.h"
#include "parallelsurf/KeyPointDescriptor.h"


namespace cvg
{
	using namespace ParallelSurf;

	class Surf
	{

	private:

		// disallow stupid things

		/*
		KeyPointDescriptor();
		KeyPointDescriptor(const KeyPointDescriptor&);
		KeyPointDescriptor& operator= (KeyPointDescriptor&) throw();
		*/


		/**
	 * @brief default constructor
	 * @param iImage integral image to use
	 * @param iThreadPool thread pool for parallelizing the computation
	 * @param iExtended calculate extended 128-dimensional descriptor
	 */
	 //KeyPointDescriptor(ParallelSurf::Image& iImage, bool iExtended = false); //曹明伟修改

	public:
		Surf(/*Image &img,*/ bool   _extended = false);
		~Surf();


		/**
		 * @brief default constructor
		 * @param iImage integral image to use
		 * @param iThreadPool Thread pool to use for computation
		 */
		 //KeyPointDetector();

	public:

		//检测特征点
		void detect(ParallelSurf::Image &_Img, std::vector<ParallelSurf::KeyPoint>& _Keypoints);

		/**
	 * @brief detect and store keypoints
	 * @param iImage integral image to use
	 * @param iInsertor function object used for storing the keypoints
	 */
	 //void detectKeyPoints(ParallelSurf::Image& iImage, std::vector<ParallelSurf::KeyPoint>& _Keypoints);

	 //计算特征描述子
		void compute(std::vector<ParallelSurf::KeyPoint>& _Keypoints);

		// orig image info
		Image &_image;


	private:

		/// @brief set number of scales per octave
		inline void setMaxScales(unsigned int iMaxScales)
		{
			_maxScales = iMaxScales;
		}

		/// @brief set number of octaves to search
		inline void setMaxOctaves(unsigned int iMaxOctaves)
		{
			_maxOctaves = iMaxOctaves;
		}

		/// @brief set minimum threshold on determinant of hessian for detected maxima
		inline void setScoreThreshold(double iThreshold)
		{
			_scoreThreshold = iThreshold;
		}



	private:

		// internal values of the keypoint detector

		// number of scales
		int					_maxScales;

		// number of octaves
		int					_maxOctaves;

		// detection score threshold
		double							_scoreThreshold;

		// initial box filter size
		int					_initialBoxFilterSize;

		// scale overlapping : how many filter sizes to overlap
		// with default value 3 : [3,5,7,9,11][7,11,15,19,23][...
		int					_scaleOverlap;

		// some default values.
		const static double kBaseSigma;

		bool fineTuneExtrema(
			double *** iSH, int iX, int iY, int iS,
			double& oX, double& oY, double& oS, double& oScore,
			int iOctaveWidth, int iOctaveHeight, int iBorder);

		bool calcTrace(
			ParallelSurf::Image& iImage, double iX,
			double iY, double iScale, int& oTrace);

		int	 getFilterSize(int iOctave, int iScale);
		int	 getBorderSize(int iOctave, int iScale);



		//------------------------//特征描述子------------------------------ -

	private:


		// info about the descriptor
		bool   _extended;   // use parallelsurf64 or parallelsurf128
		int    _subRegions;  // number of square subregions. default = 4
		int    _vecLen;   // length of the vector. 4 for parallelsurf 64, 8 for parallelsurf 128
		double   _magFactor;


		// do the actual descriptor computation
		void  createDescriptor(
			ParallelSurf::KeyPointDescriptorContext& iCtx,
			ParallelSurf::KeyPoint& ioKeyPoint) const;


	public:

		/// @brief assign orientation to given keypoint
		void assignOrientation(ParallelSurf::KeyPoint& ioKeyPoint) const;

		/**
		 * @brief assign orientations to given keypoints
		 * @param iBegin iterator to first keypoint
		 * @param iEnd iterator to keypoint where to stop computation (first one after the last)
		 */
		template< class IteratorT >
		void assignOrientations(IteratorT iBegin, IteratorT iEnd);

		/// @brief compute descriptor for single keypoint
		void makeDescriptor(ParallelSurf::KeyPoint& ioKeyPoint) const;

		/**
		* @brief compute descriptors for given keypoints
		* @param iBegin iterator to first keypoint
		* @param iEnd iterator to keypoint where to stop computation (first one after the last)
		*/
		template< class IteratorT >
		void makeDescriptors(IteratorT iBegin, IteratorT iEnd);

		/// @return length of descriptor resulting from current parameters
		int getDescriptorLength() const;


	};


	//polar representation of wavelet response (for orientation assignment)
	struct response
	{
		float orientation;
		float magnitude;
	};

	//compares the orientation of two responses
	bool operator < (const response a, const response b);


} // end of namespace.




template< class IteratorT >
void cvg::Surf::assignOrientations(IteratorT iBegin, IteratorT iEnd)
{
	IteratorT aCurrent;
#pragma omp parallel private (aCurrent)
	{
		for (aCurrent = iBegin; aCurrent != iEnd; aCurrent++)
		{
#pragma omp single nowait
			{
				assignOrientation(*aCurrent);
			}
		}
	}
}



template< class IteratorT >
void cvg::Surf::makeDescriptors(IteratorT iBegin, IteratorT iEnd)
{
	IteratorT aCurrent;
#pragma omp parallel private (aCurrent)
	{
		for (aCurrent = iBegin; aCurrent != iEnd; aCurrent++)
		{
#pragma omp single nowait
			{
				makeDescriptor(*aCurrent);
			}
		}
	}
}
