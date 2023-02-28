import os
import re

import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path

from utils import find_rect_counters, find_conner_points, reorder, drawRectangle, split_boxes, get_answer_list, \
    calc_score, controller, get_result, draw_result


def img_goster(img):
    cv2.imshow("deneme",img)
    cv2.waitKey(0)
    cv2.destroyWindow("deneme")

def find_page():
    cap = cv2.VideoCapture(0)

    frame_width = 640
    frame_height = 480

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 1)
        frame_canny = cv2.Canny(frame_blur, 50, 255)

        cv2.imshow("canny", cv2.resize(frame_canny,(480,100)))
        cv2.waitKey(0)

        kernel = np.ones((5, 5))
        frame_dial = cv2.dilate(frame_canny, kernel, iterations=2)  # APPLY DILATION
        frame_erode = cv2.erode(frame_dial, kernel, iterations=1)  # APPLY EROSION

        contours, hierarchy = cv2.findContours(frame_erode, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS

        counters = find_rect_counters(contours, 80000)

        roi_counter = None
        frame_counter = frame.copy()

        if len(counters) > 0:
            roi_counter = find_conner_points(counters[0])

            roi_counter = reorder(roi_counter)

            drawRectangle(frame_counter, roi_counter, 10)

        # cv2.imshow("1", frame_counter)

        if cv2.waitKey(1) == 27:
            # cv2.imwrite("img.jpg",frame)
            break

    return frame, roi_counter


def find_page_img(img_path,col_number):
    frame = cv2.imread(img_path)
    # frame = cv2.resize(frame, (640, 480))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 1)
    frame_canny = cv2.Canny(frame_blur, 50, 255)

    kernel = np.ones((5, 5))
    frame_dial = cv2.dilate(frame_canny, kernel, iterations=2)  # APPLY DILATION
    frame_erode = cv2.erode(frame_dial, kernel, iterations=1)  # APPLY EROSION

    contours, hierarchy = cv2.findContours(frame_erode, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # FIND ALL CONTOURS

    counters = find_rect_counters(contours, 80000)

    roi_counter = None
    frame_counter = frame.copy()

    if len(counters) > 0:
        roi_counter = find_conner_points(counters[0])

        roi_counter = reorder(roi_counter)

        drawRectangle(frame_counter, roi_counter, 10)

    return frame, roi_counter


def warp_frame(frame, roi_counter, file_name):
    if roi_counter is not None:
        pts1 = np.float32(roi_counter)
        pts2 = np.float32([[0, 0], [600, 0], [0, 800], [600, 800]])  # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        frame_warped = cv2.warpPerspective(frame, matrix, (600, 800))

        cv2.imshow("2", frame_warped)
        cv2.imwrite(file_name, frame_warped)
        cv2.waitKey(0)
        return frame_warped

def mark_answer(img, tolta_question,q_answers):
    print(tolta_question)
    table_count=((tolta_question-1)//20)+1

    img = cv2.resize(img, (600, 800))
    kernel = (5, 5)
    # kernel = np.ones((5, 5))
    w,h,c=img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, kernel, 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"total area {w*h//80}")


    rect_counters = find_rect_counters(countours, 30000)


    cnt_img = img.copy()

    if len(rect_counters) < table_count:
        iteration = 1
        while len(rect_counters) < table_count:
            iteration += 1
            img_erosion = cv2.erode(cnt_img, kernel, iterations=iteration)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=iteration)
            img_canny = cv2.Canny(img_dilation, 10, 50)
            countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect_counters = find_rect_counters(countours, 30000)
            print(f"rect_counts {len(rect_counters)} table_count {table_count}")
            cv2.drawContours(cnt_img, rect_counters, -1, (0, 255, 0), 5)
            if len(rect_counters) >= table_count:
                break
            if iteration==20:
                break

    print(len(rect_counters))


    cv2.drawContours(cnt_img, rect_counters, -1, (0, 255, 0), 5)

    img_goster(cnt_img)

    result = img.copy()

    for i, rect_counter in enumerate(rect_counters):
        prev_total_row = i * 20
        cur_total_row = (i + 1) * 20
        if cur_total_row > tolta_question:
            q_count = tolta_question - prev_total_row
            # cur_total_row = prev_total_row + tolta_question
        else:
            q_count = 20

        selected_answers=q_answers[i]

        try:
            result = draw_result(result,rect_counter,img,img_gray,q_count,selected_answers)
        except:
            pass

    return result


def mark(img, tolta_question, question_answers,sira, alt_answers=None, scoring_list=None):
    print(tolta_question)
    table_count=((tolta_question-1)//20)+1
    correct_results = []
    img = cv2.resize(img, (600, 800))
    kernel = (5, 5)
    # kernel = np.ones((5, 5))
    # w,h,c=img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, kernel, 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_counters = find_rect_counters(countours, 30000)

    print(table_count)
    print(len(rect_counters))

    if len(rect_counters) < table_count: # bir tane eksikse yeniden kontrol
        iteration = 1
        cnt_img=img.copy()
        while True:
            iteration += 1
            img_erosion = cv2.erode(cnt_img, kernel, iterations=iteration)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=iteration)
            img_canny = cv2.Canny(img_dilation, 10, 50)

            countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect_counters = find_rect_counters(countours, 30000)
            if len(rect_counters) >= table_count:
                break
            if iteration==20:
                break

    print(len(rect_counters))


    cv2.drawContours(cnt_img, rect_counters, -1, (0, 255, 0), 5)

    img_goster(cnt_img)

    result = img.copy()

    # print(len(rect_counters))

    for i, rect_counter in enumerate(rect_counters):
        prev_total_row = i * 20
        cur_total_row = (i + 1) * 20
        if cur_total_row > tolta_question:
            q_count = tolta_question - prev_total_row
            # cur_total_row = prev_total_row + tolta_question
        else:
            q_count = 20

        # print(i)

        q_answers = question_answers[i]

        try:
            if alt_answers:
                result = get_result(result, rect_counter, img, img_gray, q_count, q_answers, correct_results, alt_answers)
            else:
                result = get_result(result, rect_counter, img, img_gray, q_count, q_answers, correct_results,None)
            # img_goster(result)
        except:
            pass

    if scoring_list:
        score = calc_score(scoring_list, correct_results)
    else:
        score=int(sum(correct_results)*(100/tolta_question))

    if len(rect_counters)==0:
        score="?"


    cv2.putText(result, str(score), (450, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return result, score


def check_answer_mark(img,tolta_question):
    table_count=((tolta_question-1)//20)+1

    selected_answers_list=[]
    img = cv2.resize(img, (600, 800))
    # h,w,c=img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = (5, 5)
    img_blur = cv2.GaussianBlur(img_gray, kernel, 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_counters = find_rect_counters(countours, 30000)

    if len(rect_counters) < table_count:
        iteration = 1
        cnt_img=img.copy()
        while True:
            iteration += 1
            img_erosion = cv2.erode(cnt_img, kernel, iterations=iteration)
            img_dilation = cv2.dilate(img_erosion, kernel, iterations=iteration)
            img_canny = cv2.Canny(img_dilation, 10, 50)

            countours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rect_counters = find_rect_counters(countours, 30000)
            # cv2.drawContours(cnt_img, rect_counters, -1, (0, 255, 0), 5)
            if len(rect_counters) >= table_count:
                break
            if iteration==20:
                break


    result = img.copy()


    for i, rect_counter in enumerate(rect_counters):
        prev_total_row = i * 20
        cur_total_row = (i + 1) * 20
        if cur_total_row > tolta_question:
            q_count = tolta_question - prev_total_row
            # cur_total_row = prev_total_row + tolta_question
        else:
            q_count = 20

        result,selected_answers = get_answer_result(result,rect_counter,img,img_gray,q_count)
        selected_answers_list.append(selected_answers)

    return result,selected_answers_list

def convert_pdf(folder,name):
    # folder_name = folder.split("/")[len(folder.split("/")) - 1]
    imgs_dirs = os.listdir(folder)
    # imgs_dirs = sorted(imgs_dirs, key = lambda x: [int(j) for j in x.split(".")[0]])
    imgs_dirs = sorted(imgs_dirs, key=lambda x: int(re.search(r'\d+', x).group()))
    imgs = []
    img1 = Image.open(f"{folder}/{imgs_dirs[0]}")
    img1 = img1.convert("RGB")

    for i in range(1,len(imgs_dirs)):
        img = Image.open(f"{folder}/{imgs_dirs[i]}")
        img = img.convert("RGB")
        imgs.append(img)

    img1.save(f"{folder}/{name}.pdf", save_all=True, append_images=imgs)

# frame.shape




# frame,roi_counter=find_page_img("img.png",0)

# cv2.imshow("",frame)
# cv2.waitKey(0)

# img2=warp_frame(frame,roi_counter,"warped.jpg")
#
# warp2=warp_frame(frame,roi_counter,"warped.jpg")



mark_count = 6
tolta_question=20
# question_answers = np.random.randint(0, 5, tolta_question)

# img = cv2.imread("warped2.jpg")

# img=cv2.imread("img.png")
# result,q_answers=check_answer_mark(img,tolta_question) #işaretlenenler cevap anahatarı
# img2=cv2.imread("img2.png")
# result2=mark(img2,tolta_question,q_answers)
#
#

import os

# file_name="Tarama_btasli_2022-12-27-22-34-56.pdf"
# os.mkdir(file_name.split(".")[0])
# images = convert_from_path(file_name)
# image=images[0]
# image=np.array(image)
# # image=cv2.imread("deneme.png")
# h,w,_=image.shape
# #
# new_h=int(h*.53)
# new_w=int(w*.5)
#
# # img=image[new_h:,new_w:]
#
# # img_goster(img)
#
# for i,image in enumerate(images):
#     img=np.array(image)
#     img = img[new_h:, new_w:]
#     cv2.imwrite(f"results/{i}.jpg",img)
#     # img_goster(img)
#     break



#
q_answers=[[3, 2, 1, 0, 2, 1, 3, 3, 1, 0, 1, 1, 1, 3, 1, 3, 2, 0, 1, 2]]

q_answers=q_answers[:20]

len(q_answers[0])

# alt_answers={8:[1]}
#
# scoring_list=[3 if i<=24 else 5 for i in range(30)]
#
# #17260
# #17440
# #28242
#
# # img=cv2.imread("img.png") #45.33
#
img=cv2.imread("img.png") #45.33
#
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# contrast=np.std(img_gray)
# brightness=np.average(img_gray)
#
img_goster(img)
result = mark_answer(img, tolta_question, q_answers)
img_goster(result)
#
#

# #
# convert_pdf("results",f"{file_name.split('.')[0]}-crop")
#
# #
# # cv2.imshow("3", result)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
