# Lane-Detection
Let's start lane detection with OpenCV, Python

---

This three files are checking about basic function of OpenCV

- [canny_edge_detection.py](/canny_edge_detection.py)
- [hough_line_detection.py](/hough_line_detection.py)
- [houghlinesp_test.py](/houghlinesp_test.py)

---

- Testing lane detection at Image
    <br>[lane_detection_test1.py](/lane_detection_test1.py)

- Testing lane detection at Video
    <br>[lane_detection_test2.py](/lane_detection_test2.py)

These use [ImagePreprocessing.py](/utils/ImagePreprocessing.py).

### <center> For more details about ImagePreprocessing will be updated soon. </center>

---

# CLAHE Update
For better robust detection, applied ***CLAHE(Contrast Limited Adaptive Histogram Equalization)***

1. Convert RGB image to LAB image
2. Apply CLAHE to L channel image
3. Remerge LAB channels and convert LAB to RGB again

It returns more robust image about light.

- Image example. These are, in order, origin image, yellow & white masking without CLAHE and masking with CLAHE.
> ![yellow_white](/images/yellow_white.png)
>
> ![mask_yellow_white_without_clahe](/saved/clahe_test/mask_yellow_white_without_clahe.png)
>
> ![mask_yellow_white_with_clahe](/saved/clahe_test/mask_yellow_white_with_clahe.png)
>
>
