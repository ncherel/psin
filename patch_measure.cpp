#include <algorithm>

using namespace std;

#define H_PATCH_SIZE 5
#define PATCH_SIZE 11

extern "C" {
  float patch_measure(float* img1, float* img2,
		    int x1_size, int x2_size, int c_size,
		    int y_a, int x_a,
		    int y_b, int x_b,
		    float min_val) {
  float distance = 0.0;

  for(int i=-H_PATCH_SIZE; i < H_PATCH_SIZE + 1; i++) {
    for(int j=-H_PATCH_SIZE; j < H_PATCH_SIZE + 1; j++) {
        for (int c = 0; c < c_size; c++) {
	  float pt1 = img1[x1_size * c_size * (y_a + i) + c_size * (x_a + j) + c];
	  float pt2 = img2[x2_size * c_size * (y_b + i) + c_size * (x_b + j) + c];
          float diff = pt1 - pt2;
          distance += (diff * diff);
        }
      }
    if (distance > min_val)
      return distance;
    }

  return distance;
  }
}

