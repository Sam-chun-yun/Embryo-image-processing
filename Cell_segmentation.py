import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import mode

# 指定輸入和輸出資料夾的路徑
input_folder = r'D:\Desktop\data_NoSeg\data\result_level\test\day5'
output_folder = r'D:\Desktop\data_NoSeg\data\result_level\test\day5_result'

# 檢查輸出資料夾是否存在，如果不存在則創建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 獲取輸入資料夾中的所有圖像文件
input_images = [os.path.join(input_folder,
                filename) for filename in os.listdir(input_folder) if filename.endswith(('.jpg', '.jpeg',
                                                                                         '.png', '.bmp', '.JPG'))]

for input_image_path in input_images:
    # 讀取圖像
    input_image = cv2.imread(input_image_path)

    # 圖像預處理（可以根據需要進行修改）
    blurred = cv2.GaussianBlur(input_image, (17, 17), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # 對比度增強
    clahe = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(21, 21))
    enhanced_gray = clahe.apply(gray)
    enhanced_gray = np.where(enhanced_gray < enhanced_gray.mean() + 1.3 * enhanced_gray.std(), 0, enhanced_gray)
    # enhanced_gray = cv2.Canny(enhanced_gray, threshold1=200, threshold2=250)
    filename = os.path.basename(input_image_path)
    mask_filename = filename + '_mask.jpg'
    output_image_path = os.path.join(output_folder, mask_filename)

    # cv2.imwrite(output_image_path, enhanced_gray)

    # 使用霍夫變換檢測圓
    circles = cv2.HoughCircles(
        enhanced_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=enhanced_gray.mean(),
        param2=10,
        minRadius=140,
        maxRadius=200
    )

    if circles is not None:
        # 將結果轉換為整數
        circles = np.uint16(np.around(circles))

        # 選擇最佳圓（可以根據需要進行修改）
        best_circle = None
        best_radius = 0
        for circle in circles[0, :]:
            x, y, radius = circle
            if radius > best_radius and 200 < x <= 260 and 200 < y <= 260:
                best_radius = radius
                best_circle = circle

        # 繪製最佳圓
        if best_circle is not None:
            x, y, radius = best_circle
            radius = radius
            y = y
            x = x
            mask = np.zeros_like(enhanced_gray)
            cv2.circle(mask, (x, y), radius, (255, 255, 255), thickness=-1)

            # 將遮罩應用於原始圖像，提取圓形區域
            roi = cv2.bitwise_and(input_image, input_image, mask=mask)

            # 計算提取區域的左上角座標
            x_start = x - radius
            y_start = y - radius

            # 計算提取區域的邊長（直徑）
            side_length = 2 * radius

            # 創建一個空白的方形NumPy數組
            extracted_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)

            # 創建一個遮罩來標識ROI區域
            roi_mask = np.zeros((side_length, side_length), dtype=np.uint8)
            roi_mask[0:side_length, 0:side_length] = 255

            # 複製ROI到提取的方形區域中
            x = x_start + side_length
            y = y_start + side_length
            extracted_image[0:side_length, 0:side_length] = roi[y_start:y_start + side_length,
                                                                x_start:x_start + side_length]

            # 計算提取區域的眾數
            mode_value = extracted_image.flatten()
            condition = np.logical_and(mode_value != 0, mode_value < (np.mean(extracted_image)
                                                                      + np.std(extracted_image)))
            mode_value = mode_value[condition]

            mode_value, counts = mode(mode_value)

            # 將光圈像素值替換為眾數值
            extracted_image[extracted_image > (np.mean(extracted_image) + np.std(extracted_image) * 1)] \
                = mode_value

            # 清除遮罩，以便下一個圓的繪製
            mask.fill(0)

            # 從輸入文件路徑中提取文件名（不包括路徑）
            filename = os.path.basename(input_image_path)

            # 構建輸出文件的完整路徑
            output_image_path = os.path.join(output_folder, filename)

            # 保存ROI為新圖像，文件名不變
            cv2.imwrite(output_image_path, extracted_image)
