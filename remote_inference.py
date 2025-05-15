from gazelle.model import get_gazelle_model
from gazelle.utils import visualize_heatmap, visualize_all
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np
import cv2
import os
import shutil
import pandas as pd
from retinaface import RetinaFace
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
model.load_gazelle_state_dict(
    torch.load("C:/Users/tjdwn/PycharmProjects/pythonProject7/gazelle/pretrained/gazelle_dinov2_vitl14_inout.pt",
               weights_only=True))
model.to(device)
model.eval()
def to_frame(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    ret = True
    num = 0
    while ret:
        ret, img = cap.read()
        if ret:
            frames.append(img)
            cv2.imwrite(f'{img_temp}frame_{num}.png', img)
            num += 1
    cap.release()
    return frames

def retina(img_temp, label_temp):
    retina_face = RetinaFace()
    frame = 0
    b = os.listdir(img_temp)
    os.makedirs(label_temp, exist_ok=True)
    for filename in range(len(b)):
        if b[filename].endswith(".png"):
            img_path = img_temp + f'frame_{frame}.png'
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            try:
                resp = retina_face.predict(img)
                print(resp)
                pos_child = 0
                for face in resp:
                    if face['x1'] > pos_child:
                        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                        pos_child = x1

                # img = cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (0, 255, 0), 3)
                # cv2.imwrite(save_path + f'frame_{frame}.png', img)
                txt = open(f'{label_temp}posit.txt', 'a+')
                txt.write(f"frame_{frame}.png, %d, %d, %d, %d\n" % (x1, y1, x2, y2))
                txt.close()
                frame += 1
            except:
                print('d')
                txt = open(f'{label_temp}posit.txt', 'a+')
                txt.write(f"frame_{frame}.png, 0,0,0,0\n")
                txt.close()
                frame += 1

def f_to_vid(frames_paths, output_path, height, width, fps=30, codec='mp4v'):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: VideoWriter failed to open {output_path}")

        return

    for frames_path in frames_paths:
        frame = cv2.imread(frames_path)
        frame = cv2.resize(frame, (width, height))  # 크기 조정

        if frame is None:
            print(f"Warning: Could not read frame {frames_path}")
            continue
        out.write(frame)

    out.release()

def make_video(height, width):
    images = []
    a = os.listdir(save_path)
    for i in range(len(a)):
        images.append(f'{save_path}/frame_{i}.png')
    # Get image dimensions to set video size
    # Create the video writer object
    f_to_vid(images, os.path.join(save_vid_path), height, width)

    print(f"Video '{save_vid_path}' has been created successfully.")


def gaze_target_attention():
    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(f'{label_temp}posit.txt', names=column_names, index_col=0)
    bbox = [np.array]
    for i in df.index:
        frame_raw = Image.open(os.path.join(img_temp, i))
        frame_raw = frame_raw.convert('RGB')
        width, height = frame_raw.size
        x1, y1, x2, y2 = df.loc[i, 'left']/width, df.loc[i, 'top']/height, df.loc[i, 'right']/width, df.loc[i, 'bottom']/height

        input = {
            "images": transform(frame_raw).unsqueeze(dim=0).to(device),  # tensor of shape [1, 3, 448, 448]
            "bboxes": [[(x1, y1, x2, y2)]]
            # list of lists of bbox tuples
        }


        output = model(input)
        predicted_heatmap = output["heatmap"][0][0]  # access prediction for first person in first image. Tensor of size [64, 64]
        predicted_inout = output["inout"][0][0]
        viz = visualize_heatmap(frame_raw, predicted_heatmap, bbox=[x1, y1, x2, y2], inout_score=predicted_inout)
        # viz = visualize_all(frame_raw, predicted_heatmap, bbox=[x1, y1, x2, y2], inout_score=predicted_inout)
        plt.imshow(viz)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, i), bbox_inches='tight', pad_inches=0)
    print("save done!")
def main():
    to_frame(original_vid_path_)
    retina(img_temp, label_temp)
    # img_size = cv2.imread(img_temp+'frame_0.png')
    # width, height = img_size.shape[1], img_size.shape[0]
    # gaze_target_attention()
    # make_video(height, width)


if __name__ == "__main__":

    vis_mode = 'heatmap'



    folder = rf'/mnt/sdc/llcd_ps'
    kids = os.listdir(folder)
    original_vid_path_ = '/mnt/sdc/llcd_ps/AI-032-14/01/rec/000_000977592912.mkv'
    save_path = os.path.join('/mnt/ssd/ygshin/바탕화면/Korea_2025/gaze_target_estimation/result/','AI-032-14','01', 'rec','frame/')
    save_vid_path = os.path.join('/mnt/ssd/ygshin/바탕화면/Korea_2025/gaze_target_estimation/result/','AI-032-14','01','rec','000_000977592912.mkv')

    label_temp = './temp/'
    img_temp = f'{label_temp}img/'
    #
    if os.path.isdir(label_temp):
        shutil.rmtree(label_temp)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(img_temp, exist_ok=True)
    face_model = 'retina'

    main()