"""
Classic Face-Swap Autoencoder (DeepFakes original style)
Fully working Keras/TensorFlow implementation
+ Automatic download of sample data + face extraction & alignment
(No manual data preparation required)
"""

import os
import sys
import subprocess
import urllib.request
import bz2
import numpy as np
import cv2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Flatten, Reshape,
    LeakyReLU, UpSampling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Try to import dlib
try:
    import dlib
except ImportError:
    print("dlib is required. Install with:  pip install dlib")
    sys.exit(1)

# ====================== CONFIG ======================
IMG_SIZE = 64
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)
ENCODER_DIM = 1024
BATCH_SIZE = 32
EPOCHS = 100          # Increase for better results
LR = 5e-5

FACE_A_DIR = "data/faceA"
FACE_B_DIR = "data/faceB"
MODEL_DIR = "models"
VIDEOS_DIR = "videos"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

MAX_FACES_PER_PERSON = 1200
FRAME_SKIP = 2

PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat"
PREDICTOR_URL = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"

VIDEO_A_URL = "https://www.youtube.com/watch?v=XtHwjrm4sMg"   # example interview
VIDEO_B_URL = "https://www.youtube.com/watch?v=kwRM4PQdheE"   # example interview
VIDEO_A_PATH = os.path.join(VIDEOS_DIR, "personA.mp4")
VIDEO_B_PATH = os.path.join(VIDEOS_DIR, "personB.mp4")

# ====================== HELPERS ======================
def run_cmd(cmd):
    print(f"  → {' '.join(cmd)}")
    subprocess.check_call(cmd)

def ensure_yt_dlp():
    try:
        import yt_dlp
        return True
    except ImportError:
        print("[*] yt-dlp not found. Installing...")
        run_cmd([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])
        return True

def download_video(url, output_path):
    """Download a video clip using yt-dlp with the android client to avoid needing FFmpeg"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1_000_000:
        print(f"[+] Already have: {output_path}")
        return

    ensure_yt_dlp()
    print(f"[*] Downloading sample video → {output_path}")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extractor-args", "youtube:player_client=android",
        "-f", "b[ext=mp4]/b",
        "-o", output_path,
        "--no-playlist",
        url
    ]
    run_cmd(cmd)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to download {url}")

def download_predictor():
    if os.path.exists(PREDICTOR_PATH):
        return
    print("[*] Downloading dlib shape predictor (one-time)...")
    compressed = PREDICTOR_PATH + ".bz2"
    urllib.request.urlretrieve(PREDICTOR_URL, compressed)
    with bz2.BZ2File(compressed) as f_in, open(PREDICTOR_PATH, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(compressed)
    print("[+] Predictor ready.")

# ====================== FACE EXTRACTION ======================
def extract_aligned_faces(source_path, output_dir, max_faces=MAX_FACES_PER_PERSON):
    os.makedirs(output_dir, exist_ok=True)

    existing = [f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".png"))]
    if len(existing) >= max_faces // 2:
        print(f"[+] {output_dir} already has {len(existing)} faces → skipping")
        return

    download_predictor()
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(PREDICTOR_PATH)

    faces_saved = 0
    frame_idx = 0

    print(f"[*] Extracting faces from: {source_path}")
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source_path}")

    while faces_saved < max_faces:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector(rgb, 1)
            if dets:
                det = max(dets, key=lambda r: r.width() * r.height())
                shape = sp(rgb, det)
                face_chip = dlib.get_face_chip(rgb, shape, size=IMG_SIZE, padding=0.25)
                out_path = os.path.join(output_dir, f"face_{faces_saved:05d}.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(face_chip, cv2.COLOR_RGB2BGR))
                faces_saved += 1
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"    frames: {frame_idx}  faces: {faces_saved}", end="\r")

    cap.release()
    print(f"\n[+] Saved {faces_saved} aligned faces → {output_dir}")

# ====================== MODEL ======================
def conv(filters, kernel_size=5, strides=2):
    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x
    return block

def upscale(filters):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        return x
    return block

def build_encoder():
    inp = Input(shape=INPUT_SHAPE)
    x = conv(128)(inp)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Flatten()(x)
    x = Dense(ENCODER_DIM)(x)
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale(128)(x)
    return Model(inp, x, name="encoder")

def build_decoder(name="decoder"):
    inp = Input(shape=(8, 8, 512))
    x = upscale(256)(inp)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(CHANNELS, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(inp, x, name=name)

def build_autoencoders():
    encoder = build_encoder()
    decoder_A = build_decoder("decoder_A")
    decoder_B = build_decoder("decoder_B")

    inp = Input(shape=INPUT_SHAPE)
    autoencoder_A = Model(inp, decoder_A(encoder(inp)), name="autoencoder_A")
    autoencoder_B = Model(inp, decoder_B(encoder(inp)), name="autoencoder_B")

    # Use separate optimizer instances for Keras 3 compatibility
    autoencoder_A.compile(optimizer=Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999), loss='mae')
    autoencoder_B.compile(optimizer=Adam(learning_rate=LR, beta_1=0.5, beta_2=0.999), loss='mae')
    return encoder, decoder_A, decoder_B, autoencoder_A, autoencoder_B

# ====================== DATA & TRAIN ======================
def create_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        fill_mode='nearest'
    )
    gen_A = datagen.flow_from_directory(
        os.path.dirname(FACE_A_DIR),
        classes=[os.path.basename(FACE_A_DIR)],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="input",
        shuffle=True
    )
    gen_B = datagen.flow_from_directory(
        os.path.dirname(FACE_B_DIR),
        classes=[os.path.basename(FACE_B_DIR)],
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="input",
        shuffle=True
    )
    return gen_A, gen_B

def train(autoencoder_A, autoencoder_B, gen_A, gen_B, epochs=EPOCHS):
    print("Starting training via train_on_batch...")
    steps_per_epoch = max(1, min(len(gen_A), len(gen_B)))

    for epoch in range(epochs):
        loss_A_total = 0
        loss_B_total = 0

        for _ in range(steps_per_epoch):
            batch_A, _ = next(gen_A)
            loss_a = autoencoder_A.train_on_batch(batch_A, batch_A)
            loss_A_total += loss_a

            batch_B, _ = next(gen_B)
            loss_b = autoencoder_B.train_on_batch(batch_B, batch_B)
            loss_B_total += loss_b

        print(f"Epoch {epoch+1:3d}/{epochs} | Loss A: {loss_A_total/steps_per_epoch:.5f} | Loss B: {loss_B_total/steps_per_epoch:.5f}")

        if epoch > 0 and epoch % 50 == 0:
            autoencoder_A.save(os.path.join(MODEL_DIR, f"autoencoder_A_{epoch}.h5"))
            autoencoder_B.save(os.path.join(MODEL_DIR, f"autoencoder_B_{epoch}.h5"))

    autoencoder_A.save(os.path.join(MODEL_DIR, "autoencoder_A_final.h5"))
    autoencoder_B.save(os.path.join(MODEL_DIR, "autoencoder_B_final.h5"))
    print("Training finished. Models saved.")

def swap_faces(encoder, decoder_target, face_img):
    face = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face = np.expand_dims(face, axis=0)
    latent = encoder.predict(face, verbose=0)
    swapped = decoder_target.predict(latent, verbose=0)[0]
    return np.clip(swapped, 0, 1)

def demo_swap(encoder, decoder_A, decoder_B, gen_A, gen_B):
    batch_A, _ = next(gen_A)
    batch_B, _ = next(gen_B)
    swapped_AB = swap_faces(encoder, decoder_B, batch_A[0])
    swapped_BA = swap_faces(encoder, decoder_A, batch_B[0])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(batch_A[0]); axes[0,0].set_title("Original A")
    axes[0,1].imshow(swapped_AB); axes[0,1].set_title("A → B (swapped)")
    axes[0,2].imshow(batch_B[0]); axes[0,2].set_title("Original B (target)")
    axes[1,0].imshow(batch_B[0]); axes[1,0].set_title("Original B")
    axes[1,1].imshow(swapped_BA); axes[1,1].set_title("B → A (swapped)")
    axes[1,2].imshow(batch_A[0]); axes[1,2].set_title("Original A (target)")
    for ax in axes.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ====================== MAIN ======================
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 0: Download sample videos automatically")
    print("=" * 60)
    download_video(VIDEO_A_URL, VIDEO_A_PATH)
    download_video(VIDEO_B_URL, VIDEO_B_PATH)

    print("\n" + "=" * 60)
    print("STEP 1: Extract & align faces")
    print("=" * 60)
    extract_aligned_faces(VIDEO_A_PATH, FACE_A_DIR)
    extract_aligned_faces(VIDEO_B_PATH, FACE_B_DIR)

    n_a = len([f for f in os.listdir(FACE_A_DIR) if f.endswith((".jpg", ".png"))])
    n_b = len([f for f in os.listdir(FACE_B_DIR) if f.endswith((".jpg", ".png"))])
    print(f"\nFaces ready → A: {n_a}   B: {n_b}")
    if n_a < 80 or n_b < 80:
        print("[!] Warning: few faces extracted. Results may be weak.")

    print("\n" + "=" * 60)
    print("STEP 2: Build models")
    print("=" * 60)
    encoder, decoder_A, decoder_B, autoencoder_A, autoencoder_B = build_autoencoders()
    encoder.summary()

    gen_A, gen_B = create_generators()

    print("\n" + "=" * 60)
    print("STEP 3: Training")
    print("=" * 60)
    train(autoencoder_A, autoencoder_B, gen_A, gen_B, epochs=EPOCHS)

    print("\n" + "=" * 60)
    print("STEP 4: Demo face swap")
    print("=" * 60)
    demo_swap(encoder, decoder_A, decoder_B, gen_A, gen_B)
