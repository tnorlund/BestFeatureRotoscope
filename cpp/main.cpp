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

#include "main.hpp"

// debug flags
int display_all = 0;
int debug       = 0;

// globals
// for downsampleing
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
    GetCenter(corners, center, factor);  // get centroid
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
    waterShed_seg(&diff_image, &markers, ROW, COL);

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
          local_out = matrcv(i);
          output.write(local_out);
        }
    } else {
      matsnd(out, 0);
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

/**
 * Loads a video from a specific file.
 *
 * @param filename the path to the file
 * @param video[out] the video stream used to grab the frames from
 */
int loadVideo(char* filename, cv::VideoCapture* video) {
  video->open(filename);
  if (!video->isOpened()) {
    printf("No video data \n");
    return 0;
  }
  return 1;
}

/**
 * Reads the video properties from the video stream.
 *
 * @param debug the flag that determines to show the video data
 * @param video the video stream to read from
 * @param FPS the frames per second of the video
 * @param MAX_TIME the length of the video in seconds
 * @param start_time
 * @param MAX_FRAME the length of the video in the number of frames
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void getVideoProperties(
  int debug,
  cv::VideoCapture* video,
  double* FPS,
  double* MAX_TIME,
  double* start_time,
  int* MAX_FRAME,
  int* ROW,
  int* COL
) {
  *FPS       = video->get(cv::CAP_PROP_FPS);
  *MAX_FRAME = video->get(cv::CAP_PROP_FRAME_COUNT);
  *ROW       = video->get(cv::CAP_PROP_FRAME_HEIGHT);
  *COL       = video->get(cv::CAP_PROP_FRAME_WIDTH);
  *MAX_TIME  = static_cast<double>(*MAX_FRAME/(*FPS));

  if (debug) {
    std::cout << std::endl << "Video Properties:" << std::endl;
    std::cout << "\tFPS = " << *FPS << " fps" << std::endl;
    std::cout << "\tMax Frame = " << *MAX_FRAME << " frames" << std::endl;
    std::cout << "\tMax Time = " << *MAX_TIME << " sec" << std::endl;
    std::cout << "\tHeight = " << *ROW << " pixels" << std::endl;
    std::cout << "\tWidth = " << *COL << " pixels" << std::endl;
  }
}


/**
 * Initializes the output video file.
 *
 * @param filename the path to the file
 * @param output the stream of the output video
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 * @param FPS the frames per second of the video
 */
int initVideoOutput(
  char* filename,
  cv::VideoWriter* output,
  int* ROW,
  int* COL,
  double* FPS
) {
  char name[256];
  char* extPtr;
  char* temp;
  snprintf(name, sizeof(name), "%s", filename);
  temp = strchr(name, '.');
  while ( temp != NULL ) {
    extPtr = temp;
    temp = strchr(extPtr+1, '.');
  }
  extPtr[0] = '\0';
  extPtr++;

  output->open(
    static_cast<std::string>(name) + "_roto.avi",
    cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
    *FPS,
    (cv::Size) cv::Size(*COL, *ROW), true);

  if ( !output->isOpened() ) {
    std::cout << "Failed to open Output Video" << std::endl;
    return 0;
  }

  return 1;
}

/**
 * Determines whether the given "start time" is valid given the input video's data.
 *
 * @param debug the flag that determines to show the video data
 * @param time the argument "start time" of the input video
 * @param start_time the "start time" used in the algorithm
 * @param MAX_TIME the length of the video in seconds
 */
int checkStartTime(
  int debug,
  char* time,
  double* start_time,
  double MAX_TIME) {
  char* err;
  *start_time = strtod(time, &err);
  if (time == err || *err != 0 || *start_time > MAX_TIME || *start_time < 0) {
    printf("\nInvalid Start Time: %s\n", time);
    return 0;
  }
  if (debug) {
    printf("\tStart Time = \t%g sec\n\n", *start_time);
  }
  return 1;
}

/**
 * Determines whether the given "end time" is valid given the input video's data.
 *
 * @param debug the flag that determines to show the video data
 * @param time the argument "start time" of the input video
 * @param start_time the "start time" used in the algorithm
 * @param end_time the "end time" used in the algorithm
 * @param MAX_TIME the length of the video in seconds
 */
int checkEndTime(
  int debug,
  char* time,
  double* start_time,
  double* end_time,
  double MAX_TIME) {
  char* err;
  *end_time = strtod(time, &err);
  if (
    time == err ||
    *err != 0 ||
    *end_time > MAX_TIME ||
    *end_time < 0 ||
    *end_time < *start_time) {
    std::cout << "\nInvalid End Time: " << time << std::endl;
    return 0;
  }
  if (debug) {
    std::cout << "\tEnd Time = \t" << *start_time << " sec"
      << std::endl << std::endl;
  }
  return 1;
}

/**
 * Sets the background and foreground images.
 *
 * @param video the video stream to read from
 * @param image the current frame
 * @param image_back the frame that contains the background data
 * @param back_time the time that the background data is obtained from
 * @param start_time the "start time" used in the algorithm
 * @param FPS the frames per second of the video
 * @param current_frame the index of the frame relative to the start of the video
 */
int initImages(
  cv::VideoCapture* video,
  cv::Mat* image,
  cv::Mat* image_back,
  double back_time,
  double start_time,
  double FPS,
  int* current_frame
) {
  if (back_time > 0) {
    *current_frame = back_time * FPS;
    video->set(cv::CAP_PROP_POS_FRAMES, *current_frame);
  }
  video->read(*image_back);
  if ( !image_back->data ) {
    std::cout << "No background image data " << std::endl;
    return 0;
  }

  *current_frame = start_time*FPS;
  video->set(cv::CAP_PROP_POS_FRAMES, *current_frame);
  video->read(*image);
  if ( !image->data ) {
    std::cout << "No image data" << std::endl;
    return 0;
  }
  return 1;
}

/**
 * Reads the next sequential frame from the video stream.
 *
 * @param debug the flag that determines to show the video data
 * @param video the video stream to read from
 * @param image the frame read from the video stream
 * @param current_frame the index of the frame relative to the start of the video
 */
int getNextImage(
  int debug,
  cv::VideoCapture* video,
  cv::Mat* image,
  int* current_frame
) {
  if (debug) {
    std::cout << "Getting next frame: Frame Number = " << *current_frame
      << std::endl;
  }
  video->set(cv::CAP_PROP_POS_FRAMES, *current_frame);
  video->read(*image);
  if ( !image->data ) {
    printf("No image data \n");
    return 0;
  }
  return 1;
}

/**
 * Down samples an image using a Gaussian pyramid.
 *
 * @param image the original image requested to down sample
 * @param image_ds the destination that holds the down sampled image
 * @param factor the factor that the image is down sampled by
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void downSample(
  cv::Mat* image,
  cv::Mat* image_ds,
  int factor,
  int COL,
  int ROW
) {
  if (factor >= 2) {
    pyrDown(*image, *image_ds, cv::Size(COL/2, ROW/2));
    for (int i  = 2; i < factor; i = i*2) {
      pyrDown(*image_ds, *image_ds, cv::Size(COL/2/i, ROW/2/i));
    }
  } else {
    image->copyTo(*image_ds);
  }
}

/**
 * Calculates the centroid of the markers given.
 *
 * @param markers the markers previously calculated
 * @param center the (x,y) center of the markers given
 * @param factor the scalar used to scale the original image
 */
void GetCenter(
  std::vector<cv::Point2f> markers,
  int* center,
  int factor
) {
  cv::Mat center_vector;
  int size  = markers.size();
  reduce(markers, center_vector, 01, cv::REDUCE_AVG);
  cv::Point2f mean(
    center_vector.at<float>(0, 0),
    center_vector.at<float>(0, 1));

  center[0] = (center_vector.at<float>(0, 0)) * factor;
  center[1] = (center_vector.at<float>(0, 1)) * factor;

  if ( debug ) {
    std::cout << "Number of corners:\t" << size << std::endl;
    std::cout << "Centroid:\t\t[" << center[0] << ", " << center[1] << "]"
      << std::endl;
  }
}

/**
 * Draws the markers onto an image.
 *
 * @param image the image to write the markers to
 * @param markers the markers previously calculated
 * @param factor the scalar used to scale the original image
 */
void DrawFeatures_binary(
  cv::Mat* image,
  std::vector<cv::Point2f> markers,
  int factor
) {
  int x, y;
  int size  = markers.size();
  for (int i = 0; i < size; ++i) {
    x = static_cast<int>(markers[i].x * factor);
    y = static_cast<int>(markers[i].y * factor);
    image->at<int>(cv::Point(x, y)) = static_cast<char>(255);
  }
}

/**
 * Draws the markers onto an image based on the index of the marker.
 *
 * @param image the image to write the markers to
 * @param markers the markers previously calculated
 * @param factor the scalar used to scale the original image
 * @param offset the number to start with for indexing the markers
 */
void DrawFeatures_markers(
  cv::Mat* image,
  std::vector<cv::Point2f> markers,
  int factor,
  int offset
) {
  int x, y;
  int size  = markers.size();
  for (int i = 0; i < size; ++i) {
    x = static_cast<int>(markers[i].x * factor);
    y = static_cast<int>(markers[i].y * factor);
    image->at<int>(cv::Point(x, y)) = static_cast<char>(i+1+offset);
  }
}

/**
 * Writes a specific mask to an image matrix.
 *
 * @param mask the image to write the mask to
 * @param center the (x,y) center of the markers given
 * @param height the height of the mask
 * @param width the width of the mask
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void makeMask(
  cv::Mat* mask,
  int* center,
  int height,
  int width,
  int ROW,
  int COL
) {
  int limits[4] = {0, ROW, 0, COL};
  if (center[0] - height > 0) {
    limits[0] = center[0]-height;
  }
  if (center[0] + height < ROW) {
    limits[1] = center[0]-height;
  }
  if (center[1] - width > 0) {
    limits[2] = center[1]-width;
  }
  if (center[1] + width < ROW) {
    limits[3] = center[1]+width;
  }
  for (int i = limits[0]; i < limits[1]; i++) {
    for (int j = limits[2]; j < limits[3]; j++) {
      mask->at<char>(i, j) = static_cast<char>(255);
    }
  }
}

/**
 * Compute the watershed segmentation based on specific markers
 * 
 * @param diff_image[out] the difference image computed
 * @param markers[in] the markers computed
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void waterShed_seg(
  cv::Mat* diff_image,
  cv::Mat* markers,
  int ROW,
  int COL
) {
  int lab = -1, diff, val[3], temp_val[3], temp_diff, temp_lab;
  watershed(*diff_image, *markers);
  // get rid of boundary pixels
  for (int i = 0; i < ROW; i++) {
    for (int j = 0; j < COL; j++) {
      // check if pixel is labeled as boundary
      if (markers->at<int>(i, j) == -1) {
        diff = 255*3;
        val[0] = diff_image->at<cv::Vec3b>(i, j)[0];
        val[1] = diff_image->at<cv::Vec3b>(i, j)[1];
        val[2] = diff_image->at<cv::Vec3b>(i, j)[2];

        // check points around pixel
        if ( j > 0 ) {
          // upper left
          if ( i > 0 ) {
            temp_lab = markers->at<int>(i-1, j-1);
            if ( temp_lab > -1 ) {
              temp_val[0] = diff_image->at<cv::Vec3b>(i-1, j-1)[0];
              temp_val[1] = diff_image->at<cv::Vec3b>(i-1, j-1)[1];
              temp_val[2] = diff_image->at<cv::Vec3b>(i-1, j-1)[2];
              temp_diff = abs(val[0] - temp_val[0])
                + abs(val[1] - temp_val[1])
                + abs(val[2] - temp_val[2]);
              if ( temp_diff < diff ) {
                diff = temp_diff;
                lab = temp_lab;
              }
            }
          }
          // above
          temp_lab = markers->at<int>(i, j-1);
          if ( temp_lab > -1 ) {
            temp_val[0] = diff_image->at<cv::Vec3b>(i, j-1)[0];
            temp_val[1] = diff_image->at<cv::Vec3b>(i, j-1)[1];
            temp_val[2] = diff_image->at<cv::Vec3b>(i, j-1)[2];
            temp_diff = abs(val[0] - temp_val[0])
              + abs(val[1] - temp_val[1])
              + abs(val[2] - temp_val[2]);
            if ( temp_diff < diff ) {
              diff = temp_diff;
              lab = temp_lab;
            }
          }
          // upper right
          if ( i < ROW-1 ) {
            temp_lab = markers->at<int>(i+1, j-1);
            if ( temp_lab > -1 ) {
              temp_val[0] = diff_image->at<cv::Vec3b>(i+1, j-1)[0];
              temp_val[1] = diff_image->at<cv::Vec3b>(i+1, j-1)[1];
              temp_val[2] = diff_image->at<cv::Vec3b>(i+1, j-1)[2];
              temp_diff = abs(val[0] - temp_val[0])
                + abs(val[1] - temp_val[1])
                + abs(val[2] - temp_val[2]);
              if ( temp_diff < diff ) {
                diff = temp_diff;
                lab = temp_lab;
              }
            }
          }
        }
        // left
        if ( i > 0 ) {
          temp_lab = markers->at<int>(i-1, j);
          if ( temp_lab > -1 ) {
            temp_val[0] = diff_image->at<cv::Vec3b>(i-1, j)[0];
            temp_val[1] = diff_image->at<cv::Vec3b>(i-1, j)[1];
            temp_val[2] = diff_image->at<cv::Vec3b>(i-1, j)[2];
            temp_diff = abs(val[0] - temp_val[0])
              + abs(val[1] - temp_val[1])
              + abs(val[2] - temp_val[2]);
            if ( temp_diff < diff ) {
              diff = temp_diff;
              lab = temp_lab;
            }
          }
        }
        // right
        if ( i < ROW-1 ) {
          temp_lab = markers->at<int>(i+1, j);
          if ( temp_lab > -1 ) {
            temp_val[0] = diff_image->at<cv::Vec3b>(i+1, j)[0];
            temp_val[1] = diff_image->at<cv::Vec3b>(i+1, j)[1];
            temp_val[2] = diff_image->at<cv::Vec3b>(i+1, j)[2];
            temp_diff = abs(val[0] - temp_val[0])
              + abs(val[1] - temp_val[1])
              + abs(val[2] - temp_val[2]);
            temp_lab = markers->at<int>(i+1, j);
            if ( temp_diff < diff ) {
              diff = temp_diff;
              lab = temp_lab;
            }
          }
        }
        if ( j < COL-1 ) {
          // bottom left
          if ( i > 0 ) {
            temp_lab = markers->at<int>(i-1, j+1);
            if ( temp_lab > -1 ) {
              temp_val[0] = diff_image->at<cv::Vec3b>(i-1, j+1)[0];
              temp_val[1] = diff_image->at<cv::Vec3b>(i-1, j+1)[1];
              temp_val[2] = diff_image->at<cv::Vec3b>(i-1, j+1)[2];
              temp_diff = abs(val[0] - temp_val[0])
                + abs(val[1] - temp_val[1])
                + abs(val[2] - temp_val[2]);
              if ( temp_diff < diff && temp_lab > -1 ) {
                diff = temp_diff;
                lab = temp_lab;
              }
            }
          }
          // below
          temp_lab = markers->at<int>(i, j+1);
          if ( temp_lab > -1 ) {
            temp_val[0] = diff_image->at<cv::Vec3b>(i, j+1)[0];
            temp_val[1] = diff_image->at<cv::Vec3b>(i, j+1)[1];
            temp_val[2] = diff_image->at<cv::Vec3b>(i, j+1)[2];
            temp_diff = abs(val[0] - temp_val[0])
              + abs(val[1] - temp_val[1])
              + abs(val[2] - temp_val[2]);
            if ( temp_diff < diff ) {
              diff = temp_diff;
              lab = temp_lab;
            }
          }
          // bottom right
          if ( i < ROW-1 ) {
            temp_lab = markers->at<int>(i+1, j+1);
            if ( temp_lab > -1 ) {
              temp_val[0] = diff_image->at<cv::Vec3b>(i+1, j+1)[0];
              temp_val[1] = diff_image->at<cv::Vec3b>(i+1, j+1)[1];
              temp_val[2] = diff_image->at<cv::Vec3b>(i+1, j+1)[2];
              temp_diff = abs(val[0] - temp_val[0])
                + abs(val[1] - temp_val[1])
                + abs(val[2] - temp_val[2]);
              if ( temp_diff < diff && temp_lab > -1 ) {
                diff = temp_diff;
                lab = temp_lab;
              }
            }
          }
        }
        // assign new label
        markers->at<int>(i, j) = lab;
      }
    }
  }
}

/**
 * Applies the average color found at the segmented area
 *
 * @param image[in] the original image
 * @param markers[in] the markers computed
 * @param out[out] the final image being written to
 * @param color the colors written to the image
 * @param maxIndex the largest index found in the image
 * @param ROW the number of rows in each frame of the video
 * @param COL the number of columns in each frame of the video
 */
void colorPalette(
  cv::Mat* image,
  cv::Mat* markers,
  cv::Mat* out,
  int color[][4],
  int maxIndex,
  int ROW,
  int COL
) {
  int i, j, index;
  for (i = 0; i < maxIndex; i++) {
    color[i][0] = 0;
    color[i][1] = 0;
    color[i][2] = 0;
    color[i][3] = 0;
  }
  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      index = markers->at<int>(i, j);
      if (index > -1) {
        color[index][3] = color[index][3] + 1;
        color[index][0] = color[index][0]
          + static_cast<int>(image->at<cv::Vec3b>(i, j)[0]);
        color[index][1] = color[index][1]
          + static_cast<int>(image->at<cv::Vec3b>(i, j)[1]);
        color[index][2] = color[index][2]
          + static_cast<int>(image->at<cv::Vec3b>(i, j)[2]);
      }
    }
  }
  for (i = 0; i < maxIndex; i++) {
    index = color[i][3];
    color[i][0] = color[i][0]/index;
    color[i][1] = color[i][1]/index;
    color[i][2] = color[i][2]/index;
  }
  for (i = 0; i < ROW; i++) {
    for (j = 0; j < COL; j++) {
      index = markers->at<int>(i, j);
      if (index > -1) {
        out->at<cv::Vec3b>(i, j)[0] = color[index][0];
        out->at<cv::Vec3b>(i, j)[1] = color[index][1];
        out->at<cv::Vec3b>(i, j)[2] = color[index][2];
      }
    }
  }
}

/**
 * Sends a matrix to a specific thread
 *
 * @param m the matrix sent to another thread
 * @param dest the index of the thread to send to
 */
void matsnd(const cv::Mat& m, int dest) {
  int rows     = m.rows;
  int cols     = m.cols;
  int type     = m.type();
  int channels = m.channels();
  memcpy(
    &buffer[0 * sizeof(int)],
    reinterpret_cast<unsigned char*>(&rows),
    sizeof(int));
  memcpy(
    &buffer[1 * sizeof(int)],
    reinterpret_cast<unsigned char*>(&cols),
    sizeof(int));
  memcpy(
    &buffer[2 * sizeof(int)],
    reinterpret_cast<unsigned char*>(&type),
    sizeof(int));

  // See note at end of answer about "bytes" variable below!!!
  int bytespersample = 1;  // change if using shorts or floats
  int bytes = m.rows * m.cols * channels * bytespersample;

  memcpy(
    &buffer[3*sizeof(int)],
    m.data,
    bytes);
  MPI_Send(
    &buffer,
    bytes+3*sizeof(int),
    MPI_UNSIGNED_CHAR,
    dest,
    0,
    MPI_COMM_WORLD);
}

/*
 * Receives a matrix from a specific thread
 *
 * @param src the index of the thread to receive the image matrix from.
 * @returns the received matrix
 */
cv::Mat matrcv(int src) {
  MPI_Status status;
  int count, rows, cols, type, channels;
  MPI_Recv(
    &buffer,
    sizeof(buffer),
    MPI_UNSIGNED_CHAR,
    src,
    0,
    MPI_COMM_WORLD,
    &status);
  MPI_Get_count(
    &status,
    MPI_UNSIGNED_CHAR,
    &count);
  memcpy(
    reinterpret_cast<unsigned char*>(&rows),
    &buffer[0 * sizeof(int)],
    sizeof(int));
  memcpy(
    reinterpret_cast<unsigned char*>(&cols),
    &buffer[1 * sizeof(int)], sizeof(int));
  memcpy(
    reinterpret_cast<unsigned char*>(&type),
    &buffer[2 * sizeof(int)], sizeof(int));

  // Make the mat
  cv::Mat received = cv::Mat(
    rows,
    cols,
    type,
    reinterpret_cast<unsigned char*>(&buffer[3*sizeof(int)]));
  return received;
}

/**
 * Merges an array for 
 *
 * @param arr the array to be merged
 * @param l the left index
 * @param r the right index
 */
void merge(double arr[], int l, int m, int r) {
  int i, j, k;
  int n1 = m - l + 1;
  int n2 =  r - m;
  double L[n1], R[n2];

  for (i = 0; i < n1; i++)
    L[i] = arr[l + i];
  for (j = 0; j < n2; j++)
    R[j] = arr[m + 1+ j];

  i = 0;  // Initial index of first sub-array
  j = 0;  // Initial index of second sub-array
  k = l;  // Initial index of merged sub-array
  while (i < n1 && j < n2) {
    if (L[i] <= R[j]) {
      arr[k] = L[i];
      i++;
    } else {
      arr[k] = R[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    arr[k] = L[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = R[j];
    j++;
    k++;
  }
}

/**
 * Sorts an array of doubles using "merge sort"
 *
 * @param arr the array of doubles to sort
 * @param l the left index
 * @param r the right index
 */
void mergeSort(double arr[], int l, int r) {
  if (l < r) {
    int m = l+(r-l)/2;
    mergeSort(arr, l, m);
    mergeSort(arr, m+1, r);
    merge(arr, l, m, r);
  }
}
