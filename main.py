import os
import time
import csv
import glob

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from utils.transform import r_preprocessing

network = torch.load('saved_model/U_Net.pt')
l_btn_down = False
r_btn_down = False
t = 4
mode_draw = 0
mode_color = 0
img = None
img_original = None
img_undo = None
mask_prev = cv2.imread('samask.png')
crop_x0, crop_y0, crop_x1, crop_y1 = 0, 0, 0, 0
rec_x0, rec_y0, rec_x1, rec_y1 = 0, 0, 0, 0
kernel = 7
height = 258
width = 438
shape = (height//kernel, width//kernel)
grid = np.zeros(shape, dtype=np.int)
point_flag = (255, 0, 0)

def load_filename_list(path, ext):
    path_list = glob.glob(os.path.join(path, "*"))
    file_list = []
    for path in path_list:
        file_name = os.path.basename(path)
        if file_name.endswith(ext):
            file_list.append(file_name)
    return file_list

def create_folder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Creating Directory" + dir)

def layer():
    global img, mask_prev
    mask_model = cv2.imread('./testimage/_ypred.png')
    mask_model = cv2.resize(mask_model, dsize=(438, 258), interpolation=cv2.INTER_AREA)
    width, height = img.shape[1], img.shape[0]
    red_color = np.array([0, 0, 255])
    red_color = red_color.astype('uint8')
    blue_color = np.array([255, 0, 0])
    blue_color = blue_color.astype('uint8')
    for i in range(height):
        for j in range(width):
            if all(mask_model[i, j] == (255, 255, 255)):
                img[i, j] = blue_color
            # if all(mask_model[i, j] == blue_color):
            #     img[i, j] = blue_color

def grid2img():
    global grid
    global img
    box = img.copy()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == 1:
                box[i*kernel : (i+1)*kernel, j*kernel : (j+1)*kernel] = (255, 0, 0)
            elif grid[i][j] == 2:
                box[i*kernel : (i+1)*kernel, j*kernel : (j+1)*kernel] = (0, 0, 255)
            # elif grid[i][j] == 0:
            #     box[i*kernel : (i+1)*kernel, j*kernel : (j+1)*kernel] = (0, 0, 0)
    return box

def draw_boundary(event, x, y, flags, point):
    line_thickness = 1

    global l_btn_down
    global r_btn_down
    global t
    global img
    global img_undo
    global img_original
    global mask_prev
    global mode_draw
    global crop_x0, crop_y0, crop_x1, crop_y1
    global rec_x0, rec_y0, rec_x1, rec_y1
    global kernel, shape, height, width
    global grid
    global point_flag

    if mode_color == 0:
        point_color = (255, 0, 0)
    else:
        point_color = (0, 0, 255)
        
    if mode_draw != 4:
        x = x + crop_x0
        y = y + crop_y0
    
    if mode_draw == 0:
        if event == cv2.EVENT_LBUTTONUP and l_btn_down:
            l_btn_down = False
            point.append((x, y))
            cv2.line(img, point[-2], (x, y), point_color, line_thickness)
            cv2.fillPoly(img, [np.asarray(point)], color=point_color)

        elif event == cv2.EVENT_LBUTTONDOWN and not l_btn_down:
            l_btn_down = True
            img_undo = img.copy()
            point.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and l_btn_down:
            point.append((x, y))
            cv2.line(img, point[-2], (x, y), point_color, line_thickness)

    elif mode_draw == 1:
        if event == cv2.EVENT_LBUTTONUP and l_btn_down:
            l_btn_down = False

        elif event == cv2.EVENT_LBUTTONDOWN and not l_btn_down:
            l_btn_down = True
            img_undo = img.copy()
            img[y-t:y+t, x-t:x+t, :] = point_color

        if event == cv2.EVENT_RBUTTONUP and r_btn_down:
            r_btn_down = False

        elif event == cv2.EVENT_RBUTTONDOWN and not r_btn_down:
            r_btn_down = True
            img_undo = img.copy()
            img[y-t:y+t, x-t:x+t, :] = img_original[y-t:y+t, x-t:x+t, :]

        if event == cv2.EVENT_MOUSEMOVE and l_btn_down:
            img[y-t:y+t, x-t:x+t, :] = point_color
        elif event == cv2.EVENT_MOUSEMOVE and r_btn_down:
            img[y-t:y+t, x-t:x+t, :] = img_original[y-t:y+t, x-t:x+t, :]  

    elif mode_draw == 2:
        if event == cv2.EVENT_LBUTTONUP and l_btn_down:
            l_btn_down = False
            point.append((x, y))
            cv2.line(img, point[-2], (x, y), point_color, line_thickness)

        elif event == cv2.EVENT_LBUTTONDOWN and not l_btn_down:
            l_btn_down = True
            img_undo = img.copy()
            point.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and l_btn_down:
            point.append((x, y))
            cv2.line(img, point[-2], (x, y), point_color, line_thickness)
        
        if event == cv2.EVENT_RBUTTONUP and r_btn_down:
            r_btn_down = False

        elif event == cv2.EVENT_RBUTTONDOWN and not r_btn_down:
            r_btn_down = True
            img_undo = img.copy()
            img[y-t:y+t, x-t:x+t, :] = img_original[y-t:y+t, x-t:x+t, :]
        if event == cv2.EVENT_MOUSEMOVE and r_btn_down:
            img[y-t:y+t, x-t:x+t, :] = img_original[y-t:y+t, x-t:x+t, :]  
            
    elif mode_draw == 3:
        if event == cv2.EVENT_LBUTTONUP and l_btn_down:
            l_btn_down = False
        elif event == cv2.EVENT_LBUTTONDOWN and not l_btn_down:
            l_btn_down = True
            img_undo = img.copy()
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            r = y // kernel
            c = x // kernel
            if point_color == (255, 0, 0):
                grid[r][c] = 1
            elif point_color == (0, 0, 255):
                grid[r][c] = 2
            img = grid2img()
        if event == cv2.EVENT_RBUTTONUP and r_btn_down:
            r_btn_down = False
        elif event == cv2.EVENT_RBUTTONDOWN and not r_btn_down:
            r_btn_down = True
            img_undo = img.copy()
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            r = y // kernel
            c = x // kernel
            grid[r][c] = 0
            img = grid2img()
        if event == cv2.EVENT_MOUSEMOVE and l_btn_down:
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            r = y // kernel
            c = x // kernel
            if point_color == (255, 0, 0):
                grid[r][c] = 1
            elif point_color == (0, 0, 255):
                grid[r][c] = 2
            img = grid2img()
        elif event == cv2.EVENT_MOUSEMOVE and r_btn_down:
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            r = y // kernel
            c = x // kernel
            grid[r][c] = 0
            img = grid2img()
    elif mode_draw == 4:
        if event == cv2.EVENT_LBUTTONUP and l_btn_down:
            l_btn_down = False
            
            if rec_x1 - rec_x0 <= 20 or rec_y1 - rec_y0 <= 20:
                print("Too small size is not allowed")
                rec_x0, rec_y0, rec_x1, rec_y1 = 0, 0, 0, 0
                return
            
            crop_x0, crop_y0 = rec_x0, rec_y0
            crop_x1, crop_y1 = rec_x1, rec_y1
            rec_x0, rec_y0, rec_x1, rec_y1 = 0, 0, 0, 0

        elif event == cv2.EVENT_LBUTTONDOWN and not l_btn_down:
            l_btn_down = True
            rec_x0, rec_y0 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and l_btn_down:
            rec_x1, rec_y1 = x, y
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            crop_x0, crop_y0 = 0, 0
            crop_x1, crop_y1 = img.shape[1], img.shape[0]
            
            # face painting function (NOT IMPLEMENTED YET due to the low degree of flexibility of cv.fillPoly method)
            '''
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_mask = cv2.inRange(img_hsv, (0,255,255), (0,255,255))
            contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for i in range(len(contours[1])):
                cv2.line(img, (contours[1][i][0][0], contours[1][i][0][1]), (contours[1][(i+1)%len(contours[1])][0][0], contours[1][(i+1)%len(contours[1])][0][1]), (0,255,0), line_thickness)
                point.append((contours[1][i][0][0], contours[1][i][0][1]))
            cv2.fillPoly(img, [np.asarray(point)], color=point_color)
            '''

def set_mode(mode):
    pass

def model_label(X):
    global network
    X = r_preprocessing(X)
    X = X.view(1, 3, 256, 5).cuda()
    y = network(X)
    # y = y.view(1, 512, 256)
    torchvision.utils.save_image(y, './testimage/_ypred.png')

def write_point(root, path, file_name, point, img_original, is_video):
    global img, mask_prev
    an_root = './data/video_image/'
    an_mask_path = os.path.join(an_root, path)
    create_folder(an_mask_path)
    print(an_mask_path)
    print(an_mask_path + '/' + "{}.png".format(file_name))
    cv2.imwrite(an_mask_path + '/' + "{}.png".format(file_name), img_original)
    mask_path = os.path.join(root, path)
    
    img_mask = np.zeros(img.shape)
    img_mask[:,:,0] = cv2.inRange(img, (255,0,0), (255,0,0))
    img_mask[:,:,2] = cv2.inRange(img, (0,0,255), (0,0,255))
    print(os.path.join(mask_path, "{}.png".format(file_name)))
    cv2.imwrite(os.path.join(mask_path, "{}.png".format(file_name)), img_mask)
    temp = img_mask.astype('uint8')
    mask_prev = temp.copy()

def write_video(root, path):
    img_array = []
    print(root)
    print(path)
    mask_path = os.path.join(root, path)
    print(mask_path)
    dire = os.listdir(root+'/'+path+'/')
    dire.sort()
    for filename in dire:
        print(filename)
        img = cv2.imread(root+'/'+path+'/'+filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(root+'/'+path, cv2.VideoWriter_fourcc(*'MP42'), 20, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

parser = argparse.ArgumentParser()
parser.add_argument(
    '--file_ex', type=str, default='mp4'
)
parser.add_argument(
    '--data_path', type=str, default='data/image/'
)
parser.add_argument(
    '--mask_path', type=str, default='data/mask/'
)
parser.add_argument(
    '--if_video', type=bool, default=False
)
cfg = parser.parse_args()

def label_video(data_path, mask_path, ext, filename_list):
    global mask_prev
    print(f'Labeling Video...(Extension: {ext})')
    for i, file_name in enumerate(filename_list):
        print(f'Starting {file_name}...')
        create_folder(os.path.join(mask_path, file_name))
        cap = cv2.VideoCapture(os.path.join(data_path, file_name))
        _, temp = cap.read()
        temp2 = np.zeros(temp.shape, dtype='uint8')
        cv2.imwrite('samask.png', temp2)
        mask_prev = cv2.imread('samask.png')
        for j in tqdm(range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            myFrameNumber = j
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
            for k in range(0, 1):
                ret, frame = cap.read()
                model_label(frame)
                # cv2.imshow("Video", frame)
                
            global l_btn_down
            global r_btn_down
            global t
            global img, img_original, img_undo
            global mode_draw, mode_color
            global crop_x0, crop_y0, crop_x1, crop_y1
            global rec_x0, rec_y0, rec_x1, rec_y1
            point_interval = 10
            boundary_point = []
            toggle_show_mark = False
            mode_color_prev, win_zoom_prev  = 0, 0

            cv2.namedWindow(file_name+'_'+str(j), cv2.WINDOW_NORMAL)
            cv2.moveWindow(file_name+'_'+str(j), 50, 0)
            cv2.resizeWindow(file_name+'_'+str(j), 1800,900)
            cv2.setWindowProperty(file_name+'_'+str(j), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
            cv2.setMouseCallback(file_name+'_'+str(j), draw_boundary, boundary_point)
            trackbar_color = 'type (0: vein / 1: artery)'
            trackbar_mode = 'mode (0: basic / 1: eraser / 2: pen / 3: brush / 4: zoom)'
            cv2.createTrackbar(trackbar_color, file_name+'_'+str(j), 0, 1, set_mode)
            cv2.createTrackbar(trackbar_mode, file_name+'_'+str(j), 0, 4, set_mode)
            trackbar_size = 'pen size'
            cv2.createTrackbar(trackbar_size, file_name+'_'+str(j), 1, 20, set_mode)

            img = frame.copy()
            img_original = frame.copy()
            img_undo = frame.copy()
            crop_x0, crop_y0 = 0, 0
            crop_x1, crop_y1 = img.shape[1], img.shape[0]
            # print('img dtype')
            # print(img.dtype)
            # print(img.shape)
            # print('img_prev dtype')
            # print(img_previous.dtype)
            # print(img_previous.shape)
            layer()

            while(1):
                if toggle_show_mark == True:
                    img_show = img[crop_y0:crop_y1, crop_x0:crop_x1, :]
                else:
                    # show semi-transparent mask image
                    img_show = (img[crop_y0:crop_y1, crop_x0:crop_x1, :] // 2) + (img_original[crop_y0:crop_y1, crop_x0:crop_x1, :] // 2)
            
                if cv2.getTrackbarPos(trackbar_mode, file_name) == 4:
                    img_show = cv2.rectangle(img_show, (rec_x0, rec_y0), (rec_x1, rec_y1), (255,255,255), 2)
            
                cv2.imshow(file_name+'_'+str(j), img_show)
            
                c = cv2.waitKey(point_interval)
            
                mode_color = cv2.getTrackbarPos(trackbar_color, file_name+'_'+str(j))
                if mode_color != mode_color_prev:
                    boundary_point.clear()
                    mode_color_prev = mode_color
                
                mode_draw = cv2.getTrackbarPos(trackbar_mode, file_name+'_'+str(j))
                if mode_draw == 1:
                    # clear the pre-collected masking points
                    boundary_point.clear()
                
                win_zoom = cv2.getTrackbarPos(trackbar_size, file_name+'_'+str(j))
                if win_zoom_prev != win_zoom:
                    cv2.resizeWindow(file_name, (120*(win_zoom+1), 120*(win_zoom+1)))
                    time.sleep(0.1)
                    win_zoom_prev = win_zoom

                if c == ord('n'):
                    print("\nnext")
                    break
                
                elif c == ord('q'):
                    print("\nquit all")
                    quit()
                
                elif c == ord('r'):
                    img = img_original.copy()
                    boundary_point.clear()
                    toggle_show_mark = False
                    d_index = 0
                    x0, y0 = 0, 0
                    x1, y1 = img.shape[0], img.shape[1]
                    print("\nreset image")
                
                elif c == ord('z'):
                    img = img_undo.copy()
                    boundary_point.clear()
                    print("\nundo")
                
                elif c == ord('s'):
                    write_point(mask_path, file_name, file_name.split('.')[0]+'_'+str(j), boundary_point, img_original, True)
                    print("\n{} saved (dataset, mask)".format(file_name+'_'+str(j)))
                
                elif c == ord('m'):
                    # toggle between the semi-transparent masked image and the mask only
                    if toggle_show_mark == False:
                        img_prev = img.copy()
                    
                        # extract masked pixels
                        img[:,:,0] = cv2.inRange(img_prev, (255,0,0), (255,0,0))
                        img[:,:,1] = np.zeros(img[:,:,0].shape)
                        img[:,:,2] = cv2.inRange(img_prev, (0,0,255), (0,0,255))
                    
                        toggle_show_mark = True
                    else:
                        img = img_prev.copy()
                        toggle_show_mark = False
                
                elif c == ord('d'):
                    # show doppler image
                    print('\nnot available in video mode')
            
                elif c >= ord('0') and c <= ord('4'):
                    # switch drawing mode
                    cv2.setTrackbarPos(trackbar_mode, file_name+'_'+str(j), int(c) - int(ord('0')))

                elif c == ord('+') or c == ord('='):
                    if t <= 20:
                        t = t + 1
                    else:
                        print('pen size limit')
                    cv2.setTrackbarPos(trackbar_size, file_name+'_'+str(j), t)
                    time.sleep(0.1)
                
                elif c == ord('-') or c == ord('_'):
                    if t > 0:
                        t = t - 1
                    else:
                        print('pen size limit')
                    cv2.setTrackbarPos(trackbar_size, file_name+'_'+str(j), t)
                    time.sleep(0.1)
                
                elif c == ord('`') or c == ord('~'):
                    # reset cropped window
                    crop_x0, crop_y0 = 0, 0
                    crop_x1, crop_y1 = img.shape[1], img.shape[0]
                
                elif c == ord('h'):
                    print("Help:")
                    print(" - n: next/quit")
                    print(" - q: quit all")
                    print(" - r: reset image")
                    print(" - z: undo")
                    print(" - s: save image (in dataset, mask folders)")
                    print(" - m: toggle between masked/unmasked images")
                    print(" - d: toggle between doppler images")
                    print(" - 0/1/2/3/4: set drawing mode 0/1/2/3/4")
                    print(" - +/-: increase/decrease pen size")
                    print(" - ~: reset zoom")
                    print(" - h: help")
            cv2.destroyAllWindows()

def label_image(data_path, mask_path, ext, filename_list):
    print(f'Labeling Image...(Extension: {ext})')


if __name__ == "__main__":
    data_path = cfg.data_path
    ext = cfg.file_ex
    mask_path = cfg.mask_path

    print(f"DATA PATH: {data_path}..", end='\t')
    print(f"DATA Extension: {ext}")
    print(f"MASK PATH: {mask_path}..")

    filename_list = load_filename_list(data_path, ext)
    done_list = load_filename_list(mask_path, ext)
    todo_list = list(set(filename_list) - set(done_list))
    filename_list.sort(); todo_list.sort(); done_list.sort()

    print("Already done files: ", end=''); print(done_list)
    print("TODO files: ", end=''); print(todo_list)

    if cfg.if_video == True:
        label_video(data_path, mask_path, ext, filename_list)
    else:
        label_image(data_path, mask_path, ext, filename_list)
