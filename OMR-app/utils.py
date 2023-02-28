import cv2
import numpy as np

def img_goster(img):
    cv2.imshow("deneme",img)
    cv2.waitKey(0)
    cv2.destroyWindow("deneme")

def calc_score(score_list, correct_result):
    total_score = 0
    for score, result in zip(score_list, correct_result):
        if result:
            total_score += score
    return total_score


def split_boxes(roi, q_count, mark_count):
    boxes = []
    count = 0
    box_height = int(roi.shape[0] / q_count)
    max_width = 600
    max_height = q_count * box_height
    init_height = 0
    fin_height = box_height

    while fin_height <= max_height:
        boxes.append([])
        row = roi[init_height:fin_height, :]
        # row_tresh = cv2.threshold(row, 150, 255, cv2.THRESH_BINARY_INV)[1]
        # non_zero=np.count_nonzero(row_tresh)
        init_width = 100
        fin_width = 200
        while fin_width <= max_width:
            if fin_width==600:
                fin_width=580
            cell=row[:, init_width:fin_width]
            # img_goster(cell)
            boxes[count].append(cell)
            init_width += 100
            fin_width += 100

        init_height += box_height
        fin_height += box_height
        count += 1

    return boxes


# def automatic_brightness_and_contrast(image, clip_hist_percent=25):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Calculate grayscale histogram
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     hist_size = len(hist)
#
#     # Calculate cumulative distribution from the histogram
#     accumulator = []
#     accumulator.append(float(hist[0]))
#     for index in range(1, hist_size):
#         accumulator.append(accumulator[index - 1] + float(hist[index]))
#
#     # Locate points to clip
#     maximum = accumulator[-1]
#     clip_hist_percent *= (maximum / 100.0)
#     clip_hist_percent /= 2.0
#
#     # Locate left cut
#     minimum_gray = 0
#     while accumulator[minimum_gray] < clip_hist_percent:
#         minimum_gray += 1
#
#     # Locate right cut
#     maximum_gray = hist_size - 1
#     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
#         maximum_gray -= 1
#
#     # Calculate alpha and beta values
#     alpha = 255 / (maximum_gray - minimum_gray)
#     beta = -minimum_gray * alpha
#
#     '''
#     # Calculate new histogram with desired range and show histogram
#     new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
#     plt.plot(hist)
#     plt.plot(new_hist)
#     plt.xlim([0,256])
#     plt.show()
#     '''
#
#     auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     return auto_result, alpha, beta


def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted
        # calculates the weighted sum
        # of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified
    # text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    return cal


def get_answer_list(roi_boxes):
    selected_answers = []
    for i, row in enumerate(roi_boxes):
        row_marks = []
        for cell in row:
            score = np.count_nonzero(cell)
            row_marks.append(score)
        selected_answer = np.argmax(row_marks)
        selected_answers.append(selected_answer)
    return selected_answers


def find_conner_points(counter):
    peri = cv2.arcLength(counter, True)  # çevre
    approx = cv2.approxPolyDP(counter, 0.02 * peri, True)
    return approx


def drawRectangle(img, biggest, thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)

    return img

def draw_result(result,rect_counter,img,img_gray,q_count,selected_answers):
    corner_point = find_conner_points(rect_counter)
    roi_counters = reorder(corner_point)  # nokta sırasını değiştirme (opsiyonel)

    pt1 = np.float32(roi_counters)
    pt2 = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])
    pers_matrix = cv2.getPerspectiveTransform(pt1, pt2)

    roi_gray = cv2.warpPerspective(img_gray, pers_matrix, (600, 800))

    cell_width = 100  # width/horizantal marks count
    cell_height = int(roi_gray.shape[0] / q_count)  # width/vertical marks count

    roi_blank = np.zeros_like(img)
    color = (0, 255, 0)
    for i in range(len(selected_answers)):
        coord_x = (cell_width * selected_answers[i] + 1) + (cell_width // 2) + 100
        coord_y = (cell_height * i) + (cell_height // 2)
        if selected_answers[i] == 4:
            coord_x -= 10

        cv2.ellipse(roi_blank, (coord_x, coord_y), (40, 30), 0, 0, 360, color, cv2.FILLED)

    roi_inv_matrix = cv2.getPerspectiveTransform(pt2, pt1)
    roi_inv_warp = cv2.warpPerspective(roi_blank, roi_inv_matrix, (600, 800))
    result = cv2.addWeighted(result, 1, roi_inv_warp, 0.8, 0)

    return result


def get_result(result,rect_counter,img,img_gray,q_count,q_answers,correct_results,alt_answers):
    corner_point = find_conner_points(rect_counter)
    roi_counters = reorder(corner_point)  # nokta sırasını değiştirme (opsiyonel)

    pt1 = np.float32(roi_counters)
    pt2 = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])
    pers_matrix = cv2.getPerspectiveTransform(pt1, pt2)

    # roi_warp = cv2.warpPerspective(img, pers_matrix, (600, 800))
    #
    # roi_gray = cv2.cvtColor(roi_warp, cv2.COLOR_BGR2GRAY)

    # roi_warp = cv2.warpPerspective(img, pers_matrix, (600, 800))

    roi_gray = cv2.warpPerspective(img_gray, pers_matrix, (600, 800))

    # roi_tresh = cv2.threshold(roi_gray, 100, 255, cv2.THRESH_BINARY_INV)[1]

    roi_tresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # img_goster(roi_tresh)

    roi_boxes = split_boxes(roi_tresh, q_count, 6)

    selected_answers = get_answer_list(roi_boxes)

    cell_width = 100  # width/horizantal marks count
    cell_height = int(roi_gray.shape[0] / q_count)  # width/vertical marks count

    roi_blank = np.zeros_like(img)
    for i in range(len(selected_answers)):
        coord_x = (cell_width * selected_answers[i] + 1) + (cell_width // 2) + 100
        coord_y = (cell_height * i) + (cell_height // 2)
        if selected_answers[i] == 4:
            coord_x -= 10
        if q_answers[i] == selected_answers[i]:
            color = (0, 255, 0)
            correct_results.append(True)
        else:
            try:
                if selected_answers[i] in alt_answers[i + 1]:
                    color = (0, 255, 0)
                    correct_results.append(True)
                else:
                    color = (0, 0, 255)
                    correct_results.append(False)
            except:
                color = (0, 0, 255)
                correct_results.append(False)

        cv2.ellipse(roi_blank, (coord_x, coord_y), (40, 10), 0, 0, 360, color, cv2.FILLED)
        # cv2.circle(roi_blank, (coord_x, coord_y), 10,color,cv2.FILLED)

    roi_inv_matrix = cv2.getPerspectiveTransform(pt2, pt1)
    roi_inv_warp = cv2.warpPerspective(roi_blank, roi_inv_matrix, (600, 800))
    result = cv2.addWeighted(result, 1, roi_inv_warp, 0.5, 0)

    return result

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def find_rect_counters(contours, min_area):
    rect_countours = []
    for counter in contours:
        area = cv2.contourArea(counter)
        # print(area)
        if area > min_area:
            # print(area)
            peri = cv2.arcLength(counter, True)  # çevre
            approx = cv2.approxPolyDP(counter, 0.03 * peri, True)
            rect_countours.append(counter)
            # print(len(rect_countours))

            # if len(approx) == 4:
            #     rect_countours.append(counter)
            #     print(len(rect_countours))
            #     print(area)

    return sorted(rect_countours, key=cv2.contourArea, reverse=True)


def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # print(f"{x} {y}")
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver
