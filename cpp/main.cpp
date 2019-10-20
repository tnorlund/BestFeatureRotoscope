/***********************************************************
**| Digital Rotoscoping:                                 |**
**|   Implementation 2 in C++ to match MATLAB code for   |**
**| digital Rotoscoping.                                 |**
**|                                                      |**
**| By: Iain Murphy & Tyler Norlund                      |**
**|                                                      |**
***********************************************************/

// Example Usage
//  ./main images/Louis.mp4 16
//  ./main images/Louis.mp4 16 18
//  ./main images/Louis.mp4 16 18 0


#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <cmath>
#include <ctime>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "roto.hpp"
#include "main.hpp"

// debug flags
int display_all = 0;
int debug       = 0;

// globals
// for down sampling
int factor             = 2;
// for corner detection
int         maxCorners = 1000;
double    qualityLevel = 0.000001;
double     minDistance = 1;
int          blockSize = 3;
bool useHarrisDetector = false;
double               k = 0.04;

const int MAXBYTES = 8*1024*1024;
unsigned char buffer[MAXBYTES];
int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

int main(int argc, char** argv) {
  double FPS, MAX_TIME, start_time, end_time, back_time;
  int MAX_FRAME, ROW, COL, current_frame, center[2], color[maxCorners+1][4],
    end_frame, start_frame;
  cv::Mat image_back, image_back_gray, image, image_gray, diff_image,
    diff_image_gray, diff_image_gray_ds, corner_image, mask, markers, out,
    local_out;
  std::vector<cv::Point2f>  corners, corners_foreground;
  cv::VideoCapture video;
  cv::VideoWriter output;
  int this_frame;
  int frame_num = 0;
  double local_start, local_finish, local_elapsed;
  double elapsed, average_elapsed, median_elapsed;
  double *times;
  bool mod_true;
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  int mod_value = my_rank % 2;
  clock_t begin = clock();
  if (!my_rank) {
    times = reinterpret_cast<double*>(comm_sz);
  }

  // check input arg
  if ( (argc < 3  || argc > 5) && !my_rank ) {
    std::cout << "usage: Program <Image_Path> <Start_Time>" << std::endl;
    std::cout << "optional usages:" << std::endl;
    std::cout << "\tProgram <Image_Path> <Start_Time> <End_Time>" << std::endl;
    std::cout << "\tProgram <Image_Path> <Start_Time> <End_Time>"
      << " <Background_Time>" << std::endl;
    return -1;  // exit with error
  }

  // load video and validate user input
  if ( !loadVideo(argv[1], &video) ) {  // open video file
    std::exit(EXIT_FAILURE);  // exit with error
  }

  getVideoProperties(  // get local properties for video file
    debug,
    &video,
    &FPS,
    &MAX_TIME,
    &start_time,
    &MAX_FRAME,
    &ROW,
    &COL);

  if (!my_rank) {
    if (
      !initVideoOutput(argv[1], &output, &ROW, &COL, &FPS)
    ) {  // open video file
      std::exit(EXIT_FAILURE);  // exit with error
    }
  }

  if (
    !checkStartTime(debug, argv[2], &start_time, MAX_TIME)
  ) {  // check user input for start time
    std::exit(EXIT_FAILURE);  // exit with error
  }

  if ( argc > 3 ) {
    if (
      !checkEndTime(debug, argv[3], &start_time, &end_time, MAX_TIME)
    ) {  // check user input for start time
      std::exit(EXIT_FAILURE);  // exit with error
    }
  } else {
    end_time = start_time;  // only make 1 frame
  }
  end_frame = end_time*FPS;

  if ( argc > 4 ) {
    if (
      !checkStartTime(debug, argv[4], &back_time, MAX_TIME)
    ) {  // check user input for start time
      std::exit(EXIT_FAILURE);  // exit with error
    }
  } else {
    back_time = 0;  // default to first frame
  }

  // load background and foreground image, and frame counter
  if (
    !initImages(
        &video, &image, &image_back, back_time, start_time, FPS, &start_frame
      )
    ) {
    std::exit(EXIT_FAILURE);
  }

  // convert to grayscale
  cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_back, image_back_gray, cv::COLOR_BGR2GRAY);

  if ( display_all && !my_rank ) {
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image", image);
    cv::namedWindow("Gray Background Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray Background Image", image_back_gray);
    cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gray Image", image_gray);
    cv::waitKey(0);
  }

  for (
    current_frame = start_frame;
    current_frame <= end_frame;
    current_frame = current_frame + comm_sz
  ) {
    this_frame = current_frame+my_rank;
    if ( this_frame > start_frame ) {
      if ( !getNextImage(debug, &video, &image, &this_frame) ) {
        std::exit(EXIT_FAILURE);
      }
    }
    // make difference image
    cv::absdiff(image, image_back, diff_image);
    cv::cvtColor(diff_image, diff_image_gray, cv::COLOR_BGR2GRAY);

    // down sample image
    downSample(&diff_image_gray, &diff_image_gray_ds, factor, COL, ROW);

    // 1st round corner detection
    cv::goodFeaturesToTrack(
      diff_image_gray_ds,
      corners,
      maxCorners,
      qualityLevel,
      minDistance,
      cv::Mat(),
      blockSize,
      useHarrisDetector,
      k);
    GetCenter(
      debug,
      corners,
      center,
      factor);  // get centroid
    corner_image = corner_image.zeros(
      ROW,
      COL,
      CV_8UC1);  // make corner gray scale image
    DrawFeatures_binary(&corner_image, corners, factor);  // plot corners
    markers = markers.zeros(
      ROW,
      COL,
      CV_32SC1);  // make markers gray scale image
    DrawFeatures_markers(
      &markers,
      corners,
      factor,
      0);  // plot markers
    // watershed segmentation
    waterShed_seg(
      &diff_image,
      &markers,
      ROW,
      COL);
    // calculate average color
    out = out.zeros(ROW, COL, CV_8UC3);  // make output color image
    local_out = local_out.zeros(ROW, COL, CV_8UC3);
    colorPalette(
      &image,
      &markers,
      &out,
      color,
      maxCorners+1,
      ROW,
      COL);  // apply color
    // receive frames from threads and output to video using root
    if (!my_rank) {
      output.write(out);
        for (int i = 1; i < comm_sz; i++) {
          local_out = matrcv(i, buffer);
          output.write(local_out);
        }
    } else {
      matsnd(out, 0, buffer);
    }
  }

  // display for debugging
  if ( display_all ) {
    cv::namedWindow("Out Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Out Image", out);
    cv::waitKey(0);
  }

  if (!my_rank && debug) {
    elapsed = static_cast<float>(clock() - begin)/CLOCKS_PER_SEC;
    std::cout << comm_sz << ",\t" << argv[3] << ",\t" << elapsed << std::endl;
  }
  MPI_Finalize();
  std::exit(EXIT_SUCCESS);
}
