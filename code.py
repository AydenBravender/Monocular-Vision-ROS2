import cv2
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Q matrix - Camera parameters - Can also be found using stereoRectify
Q = np.array(([1.0, 0.0, 0.0, -160.0],
              [0.0, 1.0, 0.0, -120.0],
              [0.0, 0.0, 0.0, 350.0],
              [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)

# Load a MiDas model for depth estimation
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Open up the video capture from a webcam
cap = cv2.VideoCapture(0)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while cap.isOpened():

    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(depth_map, Q, handleMissingValues=False)

    #Get rid of points with value 0 (i.e no depth)
    mask_map = depth_map > 0.4

    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = img[mask_map]

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('Image', img)
    cv2.imshow('Depth Map', depth_map)

    # Only plot every 100th point
    mask = np.arange(output_points.shape[0]) % 1 == 0
    ax.scatter(output_points[mask, 0], output_points[mask, 1], output_points[mask, 2], c=output_colors[mask]/255, s=0.1)
    ax.view_init(elev=90, azim=-90, roll = 180)
    plt.draw()
    plt.pause(0.001)
    ax.cla()

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
