import io
import os
from urllib.request import urlopen, urlretrieve
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1. Ensure Haar Cascade is present
cascade_filename = "haarcascade_frontalface_default.xml"
cascade_url = (
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    + cascade_filename
)

if not os.path.exists(cascade_filename):
  urlretrieve(cascade_url, cascade_filename)

face_detector = cv2.CascadeClassifier(cascade_filename)

# 2. Fetch images
url_A = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=512&auto=format&fit=crop&q=80"
url_B = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512&auto=format&fit=crop&q=80"


def fetch_image(url):
  resp = urlopen(url)
  img = Image.open(io.BytesIO(resp.read())).convert("RGB")
  return np.array(img)


img_A = fetch_image(url_A)  # Target
img_B = fetch_image(url_B)  # Source face


# LAB Color Transfer
def match_colors(target, source):
  target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
  source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)

  for i in range(3):
    t_mean, t_std = target_lab[:, :, i].mean(), target_lab[:, :, i].std()
    s_mean, s_std = source_lab[:, :, i].mean(), source_lab[:, :, i].std()
    source_lab[:, :, i] = ((source_lab[:, :, i] - s_mean) * (
        t_std / (s_std + 1e-5)
    )) + t_mean

  source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
  return cv2.cvtColor(source_lab, cv2.COLOR_LAB2RGB)


# 3. Detect faces
gray_A = cv2.cvtColor(img_A, cv2.COLOR_RGB2GRAY)
gray_B = cv2.cvtColor(img_B, cv2.COLOR_RGB2GRAY)

faces_A = face_detector.detectMultiScale(
    gray_A, scaleFactor=1.1, minNeighbors=5
)
faces_B = face_detector.detectMultiScale(
    gray_B, scaleFactor=1.1, minNeighbors=5
)

if len(faces_A) > 0 and len(faces_B) > 0:
  xA, yA, wA, hA = faces_A[0]
  xB, yB, wB, hB = faces_B[0]

  face_A_crop = img_A[yA : yA + hA, xA : xA + wA]
  face_B_crop = img_B[yB : yB + hB, xB : xB + wB]

  # Match colors and resize
  face_B_matched = match_colors(face_A_crop, face_B_crop)
  face_B_resized = cv2.resize(face_B_matched, (wA, hA))

  # Soft Feathered Mask
  mask = np.zeros((hA, wA), dtype=np.uint8)
  center_x, center_y = wA // 2, hA // 2
  axes_x, axes_y = int(wA * 0.30), int(hA * 0.40)

  cv2.ellipse(
      mask, (center_x, center_y), (axes_x, axes_y), 0, 0, 360, 255, -1
  )

  # Feather the mask edges to remove sharp boundary lines
  mask = cv2.GaussianBlur(mask, (21, 21), 0)

  # Center placement with slight downward offset for mouth alignment
  center = (xA + center_x, yA + center_y + 4)

  # Seamless Poisson Blending
  swapped = cv2.seamlessClone(
      face_B_resized, img_A, mask, center, cv2.NORMAL_CLONE
  )

  # Save & Display
  Image.fromarray(swapped).save("swapped_polished.png")
  print("[✓] Polished face swap saved to 'swapped_polished.png'.")

  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  axes[0].imshow(img_A)
  axes[0].set_title("Original (Face A)")
  axes[0].axis("off")

  axes[1].imshow(swapped)
  axes[1].set_title("Polished Swap")
  axes[1].axis("off")

  plt.tight_layout()
  plt.show()

