import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('E:/work/image processing/Code/sample pics/video/c_20240209_163854_003551.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Normalize the histogram values
max_value_th = max(histogram)
normalized_histogram = (histogram * 100 / max_value_th)

# Plot normalized histogram
plt.plot(normalized_histogram)
plt.xlabel('Pixel Intensity')
plt.ylabel('Normalized Frequency')
plt.title('Normalized Histogram of Concrete Surface')
plt.show()

previous_slope_th = 0

for th in range(1, len(histogram) - 10):
    # Calculate the slope between consecutive points
    current_slope = (normalized_histogram[th + 10] - normalized_histogram[th])/10
    print(normalized_histogram[th + 10])
    print(normalized_histogram[th])
    change = current_slope-previous_slope_th
    if change > 10:
        print(th)

    previous_slope = current_slope
