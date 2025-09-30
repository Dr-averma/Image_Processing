import cv2
import numpy as np
import math
import pandas as pd
# Define the count_continuous_zeroes function


def count_continuous_zeroes(matrix):
    rows, cols = np.shape(matrix)
    output_horizontal = np.zeros((rows, cols), dtype=int)
    output_vertical = np.zeros((rows, cols), dtype=int)
    output_top = np.zeros((rows, cols), dtype=int)
    output_bottom = np.zeros((rows, cols), dtype=int)

    # Horizontal
    for i in range(rows):
        count_horizontal = 0
        for j in range(cols):
            if matrix[i, j] == 0:
                count_horizontal += 1
            else:
                count_horizontal = 0
            output_horizontal[i, j] = count_horizontal

    # Vertical
    for j in range(cols):
        count_vertical = 0
        for i in range(rows):
            if matrix[i, j] == 0:
                count_vertical += 1
            else:
                count_vertical = 0
            output_vertical[i, j] = count_vertical

    # Top
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 0:
                output_top[i, j] = 1.414 * (min(output_top[i-1, j], output_top[i, j-1]) + 1) \
                    if i > 0 and j > 0 else 1

    # Bottom
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if matrix[i, j] == 0:
                output_bottom[i, j] = 1.414 * (min(output_bottom[i + 1, j], output_bottom[i, j + 1]) + 1) \
                    if i < rows - 1 and j < cols - 1 else 1

    # Crack Width Matrix
    crack_width_matrix = np.minimum(output_horizontal, output_vertical)
    crack_width_matrix = np.minimum(crack_width_matrix, output_top)
    crack_width_matrix = np.minimum(crack_width_matrix, output_bottom)
    # Find the position of maximum value in crack_width_matrix
    max_position = np.unravel_index(np.argmax(crack_width_matrix), crack_width_matrix.shape)
    max_row, max_col = max_position
    max_value = np.max(crack_width_matrix)
    max_values_per_row = np.max(crack_width_matrix, axis=1)
    mean=np.mean(max_values_per_row)
    # Determine which matrix resulted in the maximum crack width
    if crack_width_matrix[max_row, max_col] == output_horizontal[max_row, max_col]:
        max_matrix = "Horizontal"
    elif crack_width_matrix[max_row, max_col] == output_vertical[max_row, max_col]:
        max_matrix = "Vertical"
    elif crack_width_matrix[max_row, max_col] == output_top[max_row, max_col]:
        max_matrix = "Top"
    elif crack_width_matrix[max_row, max_col] == output_bottom[max_row, max_col]:
        max_matrix = "Bottom"
    else:
        max_matrix = "Unknown"

    return max_value, max_position, max_matrix, mean


segment_data = {}


image_number = 1
for q in range(58, 99):  # Modified the range to process images from 58 to 58 (inclusive)
    image_path = f"E:/work/image processing/Code/sample pics/video/c_20240208_162643_1_0000{q}.jpg"
    segment_data[image_number] = []
    print(f"Image {image_number}-")
    # USER INPUTS
    # Camera frame for Size
    Length = 7.06  # in centimeters
    Width = 7.06  # in centimeters
    # Pixel for output
    H_P = 600  # Horizontal Pixel
    V_P = 600  # Vertical Pixel
    # Filter out contours based on aspect ratio
    min_contour_area_percentage = 5 # Minimum percentage of the largest contour's area to keep
    margin_percent=5
    ratio_allowable_of_crack = 0.3

    image = cv2.imread(image_path)
    # Get the original image size
    original_height, original_width, _ = image.shape

    # Ensure square dimensions
    if original_height > original_width:
        original_height, original_width = original_width, original_width
    else:
        original_width, original_height = original_height, original_height

    x, y = original_width, original_height
    #print(f"{x}x{y}")
    SF_h = Width / y
    SF_w = Length*10 / x

    # Resize the original image to x, y pixels
    resized_original = cv2.resize(image, (x, y))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_original, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert the original image matrix to uint8 type for compatibility with OpenCV
    originalImage = np.uint8(threshold)

    # Define the margin size (5% of the image size)
    margin = int(margin_percent * min(original_height, original_width) /100)

    # Define the region of interest (excluding the margins)
    roi = originalImage[margin:original_height-margin, margin:original_width-margin]

    # Create a binary mask identifying zero values
    zeroMask = (roi == 0).astype(np.uint8)

    # Find contours in the binary mask with RETRY_COMP mode
    contours, _ = cv2.findContours(zeroMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store segment matrices
    segment_matrices = []

    # Draw contours, bounding boxes, and display width and length values on the original image
    image_with_width = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # Set the color of the point (BGR format)
    color = (0, 255, 0)  # Green

    # Initialize variables to store the largest segmented area
    largest_segment_area = 0

    segment_number = 1  # Initialize segment numbering

    total_length = 0
    total_area = 0
    for idx, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if contour_area > min_contour_area_percentage / 100 * largest_segment_area:
            # Calculate the minimum enclosing rectangle (bounding box) around the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Calculate the aspect ratio of the bounding box
            width, height = rect[1]
            aspect_ratio = min(width, height) / max(width, height)

            # Discard contours with aspect ratio greater than the threshold
            if aspect_ratio > ratio_allowable_of_crack:
                continue

            # Ensure that the values are within the specified bounds
            min_row, max_row = max(0, min(box[:, 1])), min(y - 1, max(box[:, 1]))
            min_col, max_col = max(0, min(box[:, 0])), min(x - 1, max(box[:, 0]))

            # Extract the segment from the original image
            segment_matrix = roi[min_row:max_row + 1, min_col:max_col + 1]

            # Store the segment matrix in the list
            segment_matrices.append(segment_matrix)

            # Update the largest segmented area if the current area is larger
            if contour_area > largest_segment_area:
                largest_segment_area = contour_area

            # Calculate the angle of the bounding box with respect to the longer side
            if width > height:
                angle = round(abs(rect[2]),2)
            else:
                angle = round(abs(90 - rect[2]),2)

            # Calculate the length of the segment

            length = max_row - min_row
            if length * SF_h<1:
                continue
            # Print values to console
            length_of_crack = round(length*SF_h,2)
            area_of_crack = round(contour_area * SF_w * SF_h * 10, 2)
            print(f"Segment {segment_number} - Angle: {angle:.2f} degrees, Length: {length_of_crack:.2f} cm, "
                  f"Area: {area_of_crack} mm^2")


            if angle <= 75:
                type_of_crack=1
                print(f"Crack Type- Shear crack")
            elif angle <= 50:
                type_of_crack = 2
                print(f"Crack Type- Shear diagonal crack")
            elif angle <= 30:
                type_of_crack = 3
                print(f"Crack Type- Tensile crack")
            else:
                type_of_crack = 4
                print(f"Crack Type- Vertical crack")

            # Draw the bounding box
            cv2.drawContours(image_with_width, [box], 0, (0, 255, 0), 2)

            # Apply the count_continuous_zeroes function to the segment matrix
            max_width, max_position, max_matrix, mean = count_continuous_zeroes(segment_matrix)
            if max_width>x:
                max_crack_width=contour_area/length
            else:
                max_crack_width=max_width
            # angle correction
            if max_matrix == "Top":
                a_c = (math.cos(math.radians(abs(45 - angle))))
            elif max_matrix == "Bottom":
                a_c = (math.cos(math.radians(abs(45 - angle))))
            else:
                a_c = (math.cos(math.radians(round((abs(45 - angle)), 2))))

            #print(f"a_c: {a_c}")
            width_of_crack=max_crack_width * SF_w / a_c
            print(f"Max Crack Width: {width_of_crack:.2f} mm ")

            # Adjust the coordinates relative to the bounding box
            relative_position = (max_position[1], max_position[0])


           # Draw a red dot at the max position
            r = round(H_P / 150)
            cv2.circle(image_with_width, (min_col + max_position[1], min_row + max_position[0]),
                       r, (0, 0, 255), -1)
            x_crack=min_col + max_position[1]
            y_crack=min_row + max_position[0]
            print(f"{x_crack},{y_crack}")
            # Draw the segment number
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_with_width, f" {segment_number}", (min_col + max_position[1],
                                                                 min_row + max_position[0]), font, 0.001 * H_P,
                        (100, 0, 0), 2, cv2.LINE_AA)

            total_length += length_of_crack
            total_area += area_of_crack
            imaginary_length = area_of_crack/(10*mean*SF_w)
            T = imaginary_length/length_of_crack
            print(f"length{segment_number}-{imaginary_length:2f}")
            print(f"T-{T}")
            # Store segment data
            segment_data[image_number].append({
                'Image Number': image_number,
                'Segment Number': segment_number,
                'x_crack': x_crack,
                'y_crack': y_crack,
                'length_of_crack': length_of_crack,
                'area_of_crack': area_of_crack,
                'width_of_crack': width_of_crack,
                'imaginary_length': imaginary_length,
                'T': T,
                'angle': angle,
                'type_of_crack': type_of_crack
            })
            segment_number += 1  # Increment segment numbering

    print(f"total length:{total_length:.2f}, total area:{total_area:.2f}")

    # Print the original image size
    print(f"Original Image Size - Length: {Length} cm, Width: {original_width} pixels")

    # Display the original image with filtered contours, and width and length values
    resized_original = cv2.resize(image_with_width, (H_P, V_P))
    cv2.imshow('Original Image with Contours and Cracks', resized_original)
    cv2.waitKey(400)
    cv2.destroyAllWindows()
    image_number += 1
# Write segment data to Excel
with pd.ExcelWriter('E:/work/image processing/output/segment_data.xlsx') as writer:
    all_segments = []
    for image_number, segments in segment_data.items():
        for segment in segments:
            segment['Image Number'] = image_number
            all_segments.append(segment)

    all_segments_df = pd.DataFrame(all_segments)

    # Write all segment data to a single sheet in Excel
    all_segments_df.to_excel(writer, sheet_name='All_Segments', index=False)