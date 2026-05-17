"""
CUDA_VISIBLE_DEVICES=1 python music.py
"""
import os
import cv2
import time
import math
import random
import platform
import numpy as np
import mediapipe as mp
import tensorflow as tf
import queue
from scipy.io import wavfile
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import layers, Model
import csv
import json
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import wave
import subprocess

# Graceful audio import for Linux Server vs Windows Laptop
try:
    import sounddevice as sd
    AUDIO_ENABLED = True
except OSError:
    print("WARNING: sounddevice/PortAudio not found. Audio playback disabled (Normal for headless servers).")
    AUDIO_ENABLED = False

def setup_gpus():
    gpus = tf.config.list_physical_devices("GPU")
    print("Visible GPUs:", gpus)
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Could not set memory growth for {gpu}: {e}")
    return gpus

# ============================================================
# 1. CONFIG
# ============================================================
if platform.system() == "Windows":
    ROOT_DIR = r"path\to\Data"
else:
    ROOT_DIR = "path/to/server/Data"

DATA_PATH = os.path.join(ROOT_DIR, "Music_Data")
AUDIO_PATH = os.path.join(ROOT_DIR, "Audio")
ARTIFACT_DIR = "./artifacts"
FIG_DIR = os.path.join(ARTIFACT_DIR, "figures")
LOG_DIR = os.path.join(ARTIFACT_DIR, "logs")
AUDIO_OUT_DIR = os.path.join(ARTIFACT_DIR, "audio")

VIDEO_OUT_DIR = os.path.join(ARTIFACT_DIR, "video")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

STREAM_WINDOW = 12
STREAM_STRIDE = 1
TRAIN_BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 0.05
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)

MIN_NOTES_PER_STREAM = 4
MAX_NOTES_PER_STREAM = 10
PAUSE_FRAMES_RANGE = (6, 15)
TRANSITION_NOISE_STD = 0.005

ACTIONS = [
    'High-DO', 'High-Fa', 'High-la', 'High-Mi', 'High-Re', 'High-so', 'High-Ti',
    'Mid-Do',  'Mid-Fa',  'Mid-la',  'Mid-Mi',  'Mid-Re',  'Mid-so',  'Mid-Ti',
    'Low-Do',  'Low-Fa',  'Low-la',  'Low-Mi',  'Low-Re',  'Low-so',  'Low-Ti'
]

PITCH_CLASSES = ['Do', 'Re', 'Mi', 'Fa', 'So', 'La', 'Ti']
OCTAVE_CLASSES = ['Low', 'Mid', 'High']

PITCH_TO_ID = {'Do': 0, 'Re': 1, 'Mi': 2, 'Fa': 3, 'So': 4, 'La': 5, 'Ti': 6}
OCTAVE_TO_ID = {'Low': 0, 'Mid': 1, 'High': 2}

FREQ_MAP = {
    ('Low',  'Do'): 130.81, ('Low',  'Re'): 146.83, ('Low',  'Mi'): 164.81, ('Low',  'Fa'): 174.61,
    ('Low',  'So'): 196.00, ('Low',  'La'): 220.00, ('Low',  'Ti'): 246.94,
    ('Mid',  'Do'): 261.63, ('Mid',  'Re'): 293.66, ('Mid',  'Mi'): 329.63, ('Mid',  'Fa'): 349.23,
    ('Mid',  'So'): 392.00, ('Mid',  'La'): 440.00, ('Mid',  'Ti'): 493.88,
    ('High', 'Do'): 523.25, ('High', 'Re'): 587.33, ('High', 'Mi'): 659.25, ('High', 'Fa'): 698.46,
    ('High', 'So'): 783.99, ('High', 'La'): 880.00, ('High', 'Ti'): 987.77,
}

# ============================================================
# 2. AUDIO UTILS & DATASET PARSING
# ============================================================
def normalize_action_name(name: str) -> Tuple[str, str]:
    octave, pitch = name.split('-')
    pitch_norm = pitch.strip().lower()
    pitch_map = {'do': 'Do', 're': 'Re', 'mi': 'Mi', 'fa': 'Fa', 'so': 'So', 'la': 'La', 'ti': 'Ti'}
    return octave, pitch_map.get(pitch_norm, pitch_norm)

ACTION_INFO = {}
for a in ACTIONS:
    octave, pitch = normalize_action_name(a)
    ACTION_INFO[a] = {
        "octave_name": octave, "pitch_name": pitch,
        "octave_id": OCTAVE_TO_ID[octave], "pitch_id": PITCH_TO_ID[pitch],
        "freq": FREQ_MAP[(octave, pitch)]
    }

audio_queue = queue.Queue(maxsize=32)

def audio_callback(outdata, frames, time_info, status):
    try:
        chunk = audio_queue.get_nowait()
    except queue.Empty:
        chunk = np.zeros((frames,), dtype=np.float32)
    if len(chunk) < frames:
        padded = np.zeros((frames,), dtype=np.float32)
        padded[:len(chunk)] = chunk
        chunk = padded
    elif len(chunk) > frames:
        chunk = chunk[:frames]
    outdata[:] = chunk.reshape(-1, 1)

def load_wav_mono(path, target_sr=SAMPLE_RATE):
    sr, audio = wavfile.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    if sr != target_sr:
        old_idx = np.linspace(0, len(audio) - 1, len(audio))
        new_len = int(len(audio) * target_sr / sr)
        new_idx = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)
    return audio

def load_note_audio_bank():
    bank = {}
    for action in ACTIONS:
        wav_path = os.path.join(AUDIO_PATH, f"{action}.wav")
        if not os.path.exists(wav_path):
            print(f"WARNING: Missing wav file: {wav_path}")
            bank[action] = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        else:
            bank[action] = load_wav_mono(wav_path, SAMPLE_RATE)
            print(f"Loaded audio for {action}: {bank[action].shape}")
    return bank

# ============================================================
# 3. FEATURE DIMENSION
# ============================================================
# Use only pose + hands for low latency.
USE_FACE = False
USE_POSE = True   # <--- ADD THIS TOGGLE! Set to False to test hands-only.

POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FACE_DIM = 468 * 3
FEATURE_DIM = (POSE_DIM if USE_POSE else 0) + HAND_DIM + HAND_DIM + (FACE_DIM if USE_FACE else 0)

# ============================================================
# 4. LOAD EXISTING NOTE CLIPS
# ============================================================
def load_single_note_clip(folder_path: str, expected_frames: int = 30) -> np.ndarray:
    frames = []
    for i in range(expected_frames):
        npy_path = os.path.join(folder_path, f"{i}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Missing frame file: {npy_path}")
        
        x = np.load(npy_path).astype(np.float32)

        # 1. ALWAYS extract the raw components using their fixed positions in the saved .npy files
        pose = x[:POSE_DIM]
        face = x[POSE_DIM:POSE_DIM + FACE_DIM]
        lh_start = POSE_DIM + FACE_DIM
        lh = x[lh_start : lh_start + HAND_DIM]
        rh = x[lh_start + HAND_DIM : lh_start + HAND_DIM + HAND_DIM]

        # 2. Build the new feature vector based on your current toggles
        features = [lh, rh]  # Hands are always included

        if USE_POSE:
            features.insert(0, pose)  # Put pose at the front
            
        if USE_FACE:
            features.append(face)     # Put face at the end

        x_new = np.concatenate(features, axis=0)

        # 3. Safety check
        if x_new.shape[0] != FEATURE_DIM:
            raise ValueError(f"Feature dim mismatch in {npy_path}. Expected {FEATURE_DIM}, got {x_new.shape[0]}")
            
        frames.append(x_new)

    return np.stack(frames, axis=0)  # [30, FEATURE_DIM] 

def load_note_library(data_path: str) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    library = {}
    for action in ACTIONS:
        class_dir = os.path.join(data_path, action)
        seq_names = sorted([s for s in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, s))], key=lambda x: int(x))
        clips = []
        for seq_name in seq_names:
            seq_dir = os.path.join(class_dir, seq_name)
            clips.append((f"{action}/{seq_name}", load_single_note_clip(seq_dir)))
        library[action] = clips
    return library

# ============================================================
# 5. SYNTHETIC STREAM GENERATION
# ============================================================
@dataclass
class StreamSample:
    x: np.ndarray; pitch: np.ndarray; octave: np.ndarray; onset: np.ndarray
    sustain: np.ndarray; amplitude: np.ndarray; active: np.ndarray

def get_note_envelope(num_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    onset = np.zeros(num_frames, dtype=np.float32)
    sustain = np.zeros(num_frames, dtype=np.float32)
    amplitude = np.zeros(num_frames, dtype=np.float32)

    # sharper onset pulse
    onset_width = min(3, num_frames)
    onset[:onset_width] = np.array([0.5, 1.0, 0.5][:onset_width], dtype=np.float32)

    attack_end = max(3, int(0.12 * num_frames))
    release_start = max(attack_end + 1, int(0.75 * num_frames))

    sustain[attack_end:release_start] = 1.0

    if attack_end > 0:
        amplitude[:attack_end] = np.linspace(0.2, 1.0, attack_end)
    if release_start > attack_end:
        amplitude[attack_end:release_start] = 1.0
    if num_frames > release_start:
        amplitude[release_start:] = np.linspace(1.0, 0.1, num_frames - release_start)

    return onset, sustain, np.clip(amplitude, 0.0, 1.0)

def build_stream_from_note_sequence(note_clips: List[Tuple[str, np.ndarray]]) -> StreamSample:
    xs, pitch, octave, onset, sustain, amplitude, active = [], [], [], [], [], [], []
    for idx, (action, clip) in enumerate(note_clips):
        meta = ACTION_INFO[action]
        num_frames = clip.shape[0]
        clip = clip + np.random.normal(0.0, TRANSITION_NOISE_STD, size=clip.shape).astype(np.float32)
        o, s, a = get_note_envelope(num_frames)
        
        xs.append(clip); pitch.append(np.full((num_frames,), meta["pitch_id"], dtype=np.int32))
        octave.append(np.full((num_frames,), meta["octave_id"], dtype=np.int32)); onset.append(o)
        sustain.append(s); amplitude.append(a); active.append((a > 0.1).astype(np.float32))

        if idx < len(note_clips) - 1:
            pf = random.randint(*PAUSE_FRAMES_RANGE)
            xs.append(np.random.normal(0.0, 0.001, size=(pf, FEATURE_DIM)).astype(np.float32))
            pitch.append(np.zeros(pf, dtype=np.int32)); octave.append(np.zeros(pf, dtype=np.int32))
            onset.append(np.zeros(pf, dtype=np.float32)); sustain.append(np.zeros(pf, dtype=np.float32))
            amplitude.append(np.zeros(pf, dtype=np.float32)); active.append(np.zeros(pf, dtype=np.float32))

    return StreamSample(np.concatenate(xs), np.concatenate(pitch), np.concatenate(octave), np.concatenate(onset), np.concatenate(sustain), np.concatenate(amplitude), np.concatenate(active))

def generate_synthetic_streams(note_library, num_streams=1200) -> List[StreamSample]:
    actions = list(note_library.keys())
    return [build_stream_from_note_sequence([(a, random.choice(note_library[a])[1]) for a in [random.choice(actions) for _ in range(random.randint(MIN_NOTES_PER_STREAM, MAX_NOTES_PER_STREAM))]]) for _ in range(num_streams)]

def split_note_library_by_clip(note_library, val_ratio=0.1, seed=42):
    rng = random.Random(seed); train_lib, val_lib = {}, {}
    for action, clip_list in note_library.items():
        c = clip_list.copy(); rng.shuffle(c)
        n_val = max(1, int(round(len(c) * val_ratio)))
        val_lib[action], train_lib[action] = c[:n_val], c[n_val:]
    return train_lib, val_lib

def stream_to_windows(sample: StreamSample, window=STREAM_WINDOW, stride=STREAM_STRIDE):
    X, yp, yo, yon, ys, ya, yact = [], [], [], [], [], [], []
    for end_idx in range(window, sample.x.shape[0] + 1, stride):
        X.append(sample.x[end_idx - window:end_idx])
        yp.append(sample.pitch[end_idx - 1]); yo.append(sample.octave[end_idx - 1])
        yon.append(sample.onset[end_idx - 1]); ys.append(sample.sustain[end_idx - 1])
        ya.append(sample.amplitude[end_idx - 1]); yact.append(sample.active[end_idx - 1])
    return np.array(X, dtype=np.float32), np.array(yp, dtype=np.int32), np.array(yo, dtype=np.int32), np.array(yon, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(ya, dtype=np.float32), np.array(yact, dtype=np.float32)

def build_training_windows(streams):
    res = [stream_to_windows(s) for s in streams]
    return tuple(np.concatenate([r[i] for r in res], axis=0) for i in range(7))

def prepare_data(num_train_streams=1200, num_val_streams=200):
    full_lib = load_note_library(DATA_PATH)
    t_lib, v_lib = split_note_library_by_clip(full_lib)
    t_streams = generate_synthetic_streams(t_lib, num_train_streams)
    v_streams = generate_synthetic_streams(v_lib, num_val_streams)
    Xt, *yt = build_training_windows(t_streams)
    Xv, *yv = build_training_windows(v_streams)
    return (Xt, tuple(yt)), (Xv, tuple(yv))

# ============================================================
# 7. CAUSAL TCN ARCHITECTURE
# ============================================================
class CausalDepthwiseConvBlock(layers.Layer):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.pad = layers.ZeroPadding1D((dilation * (kernel_size - 1), 0))
        self.dw = layers.DepthwiseConv1D(kernel_size=kernel_size, dilation_rate=dilation, padding='valid')
        self.pw = layers.Conv1D(channels, kernel_size=1, padding='same')
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.act = layers.Activation('swish')
        self.drop = layers.Dropout(dropout)
        self.proj = layers.Conv1D(channels, kernel_size=1, padding='same')

    def call(self, x, training=False):
        residual = x
        y = self.drop(self.act(self.norm(self.pw(self.dw(self.pad(x))))), training=training)
        if residual.shape[-1] != y.shape[-1]: residual = self.proj(residual)
        return residual + y

class StreamingComposerNet(Model):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_proj = layers.Conv1D(hidden_dim, kernel_size=1, padding='same')
        self.block1 = CausalDepthwiseConvBlock(hidden_dim, kernel_size=3, dilation=1)
        self.block2 = CausalDepthwiseConvBlock(hidden_dim, kernel_size=3, dilation=2)
        self.block3 = CausalDepthwiseConvBlock(hidden_dim, kernel_size=3, dilation=4)
        self.block4 = CausalDepthwiseConvBlock(hidden_dim, kernel_size=3, dilation=8)
        self.head_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Outputs full sequence for Consistency Loss!
        self.pitch_head = layers.Dense(len(PITCH_CLASSES), activation='softmax', name='pitch')
        self.octave_head = layers.Dense(len(OCTAVE_CLASSES), activation='softmax', name='octave')
        self.onset_head = layers.Dense(1, activation='sigmoid', name='onset')
        self.sustain_head = layers.Dense(1, activation='sigmoid', name='sustain')
        self.amplitude_head = layers.Dense(1, activation='sigmoid', name='amplitude')
        self.active_head = layers.Dense(1, activation='sigmoid', name='active')

    def call(self, x, training=False):
        x = self.head_norm(self.block4(self.block3(self.block2(self.block1(self.input_proj(x), training=training), training=training), training=training), training=training))
        return {
            'pitch': self.pitch_head(x), 'octave': self.octave_head(x),
            'onset': self.onset_head(x), 'sustain': self.sustain_head(x),
            'amplitude': self.amplitude_head(x), 'active': self.active_head(x)
        }

# ============================================================
# 8. TRAINER WITH SPECTRAL & CONSISTENCY LOSS
# ============================================================

class StreamingComposerTrainer(Model):
    def __init__(self, core_model: Model):
        super().__init__()
        self.core = core_model

        # IMPORTANT: use NONE for custom distributed train_step
        self.loss_pitch = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.loss_octave = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.loss_bce = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.loss_mae = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

        self.pitch_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='pitch_acc')
        self.oct_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='octave_acc')

    @property
    def metrics(self):
        return [self.pitch_acc, self.oct_acc]

    def compute_consistency_loss(self, sequence_pred):
        diff = sequence_pred[:, 1:, :] - sequence_pred[:, :-1, :]
        return tf.reduce_mean(tf.square(diff))

    def compute_spectral_loss(self, pred_pitch, true_pitch, pred_amp, true_amp):
        pred_freq = tf.reduce_sum(pred_pitch * tf.linspace(0.1, 1.0, 7), axis=-1)

        true_freq_one_hot = tf.one_hot(true_pitch, depth=7)
        true_freq = tf.reduce_sum(true_freq_one_hot * tf.linspace(0.1, 1.0, 7), axis=-1)

        # squeeze amplitude if shape is [B,1]
        pred_amp = tf.squeeze(pred_amp, axis=-1)
        true_amp = tf.squeeze(true_amp, axis=-1) if len(true_amp.shape) > 1 else true_amp

        t = tf.linspace(0.0, 1.0, 512)

        pred_wave = tf.expand_dims(pred_amp, -1) * tf.math.sin(
            2.0 * np.pi * tf.expand_dims(pred_freq, -1) * t * 50.0
        )
        true_wave = tf.expand_dims(true_amp, -1) * tf.math.sin(
            2.0 * np.pi * tf.expand_dims(true_freq, -1) * t * 50.0
        )

        s_p1 = tf.abs(tf.signal.stft(pred_wave, frame_length=256, frame_step=128))
        s_t1 = tf.abs(tf.signal.stft(true_wave, frame_length=256, frame_step=128))
        s_p2 = tf.abs(tf.signal.stft(pred_wave, frame_length=128, frame_step=64))
        s_t2 = tf.abs(tf.signal.stft(true_wave, frame_length=128, frame_step=64))

        return tf.reduce_mean(tf.abs(s_p1 - s_t1)) + tf.reduce_mean(tf.abs(s_p2 - s_t2))

    def train_step(self, data):
        x, y = data
        yp, yo, yon, ys, ya, yact = y

        # make shapes explicit for BCE/MAE heads
        yon = tf.cast(tf.expand_dims(yon, axis=-1), tf.float32)
        ys = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32)
        ya = tf.cast(tf.expand_dims(ya, axis=-1), tf.float32)
        yact = tf.cast(tf.expand_dims(yact, axis=-1), tf.float32)

        with tf.GradientTape() as tape:
            pred = self.core(x, training=True)

            p_pitch = pred['pitch'][:, -1, :]       # [B, 7]
            p_octave = pred['octave'][:, -1, :]     # [B, 3]
            p_onset = pred['onset'][:, -1, :]       # [B, 1]
            p_sustain = pred['sustain'][:, -1, :]   # [B, 1]
            p_amp = pred['amplitude'][:, -1, :]     # [B, 1]
            p_act = pred['active'][:, -1, :]        # [B, 1]

            # per-example losses
            lp_per = self.loss_pitch(yp, p_pitch)       # [B]
            lo_per = self.loss_octave(yo, p_octave)     # [B]
            lon_per = self.loss_bce(yon, p_onset)       # [B]
            ls_per = self.loss_bce(ys, p_sustain)       # [B]
            la_per = self.loss_mae(ya, p_amp)           # [B]
            lact_per = self.loss_bce(yact, p_act)       # [B]

            # distribute-safe reductions
            lp = tf.nn.compute_average_loss(lp_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
            lo = tf.nn.compute_average_loss(lo_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
            lon = tf.nn.compute_average_loss(lon_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
            ls = tf.nn.compute_average_loss(ls_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
            la = tf.nn.compute_average_loss(la_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
            lact = tf.nn.compute_average_loss(lact_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)

            l_consist = self.compute_consistency_loss(pred['pitch']) + self.compute_consistency_loss(pred['octave'])
            l_spec = self.compute_spectral_loss(p_pitch, yp, p_amp, ya)

            total = (
                1.0 * lp +
                0.8 * lo +
                2.0 * lon +     # stronger onset
                1.2 * ls +
                0.6 * la +
                1.5 * lact +    # stronger active
                0.2 * l_consist +
                0.2 * l_spec
            )

            if self.losses:
                total += tf.nn.scale_regularization_loss(tf.add_n(self.losses))

        grads = tape.gradient(total, self.core.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.core.trainable_variables))

        self.pitch_acc.update_state(yp, p_pitch)
        self.oct_acc.update_state(yo, p_octave)

        return {
            'loss': total,
            'pitch_loss': lp,
            'octave_loss': lo,
            'pitch_acc': self.pitch_acc.result(),
            'octave_acc': self.oct_acc.result()
        }

    def test_step(self, data):
        x, y = data
        yp, yo, yon, ys, ya, yact = y

        yon = tf.cast(tf.expand_dims(yon, axis=-1), tf.float32)
        ys = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32)
        ya = tf.cast(tf.expand_dims(ya, axis=-1), tf.float32)
        yact = tf.cast(tf.expand_dims(yact, axis=-1), tf.float32)

        pred = self.core(x, training=False)

        p_pitch = pred['pitch'][:, -1, :]
        p_octave = pred['octave'][:, -1, :]
        p_onset = pred['onset'][:, -1, :]
        p_sustain = pred['sustain'][:, -1, :]
        p_amp = pred['amplitude'][:, -1, :]
        p_act = pred['active'][:, -1, :]

        lp_per = self.loss_pitch(yp, p_pitch)
        lo_per = self.loss_octave(yo, p_octave)
        lon_per = self.loss_bce(yon, p_onset)
        ls_per = self.loss_bce(ys, p_sustain)
        la_per = self.loss_mae(ya, p_amp)
        lact_per = self.loss_bce(yact, p_act)

        lp = tf.nn.compute_average_loss(lp_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
        lo = tf.nn.compute_average_loss(lo_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
        lon = tf.nn.compute_average_loss(lon_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
        ls = tf.nn.compute_average_loss(ls_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
        la = tf.nn.compute_average_loss(la_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)
        lact = tf.nn.compute_average_loss(lact_per, global_batch_size=TRAIN_BATCH_SIZE * self.distribute_strategy.num_replicas_in_sync)

        l_consist = self.compute_consistency_loss(pred['pitch']) + self.compute_consistency_loss(pred['octave'])
        l_spec = self.compute_spectral_loss(p_pitch, yp, p_amp, ya)

        total = (
            1.0 * lp +
            0.8 * lo +
            2.0 * lon +     # stronger onset
            1.2 * ls +
            0.6 * la +
            1.5 * lact +    # stronger active
            0.2 * l_consist +
            0.2 * l_spec
        )

        self.pitch_acc.update_state(yp, p_pitch)
        self.oct_acc.update_state(yo, p_octave)
        
        return {
            'loss': total,
            'pitch_acc': self.pitch_acc.result(),
            'octave_acc': self.oct_acc.result()
        }
def save_training_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(history.history.keys())
        writer.writerow(["epoch"] + keys)
        for i in range(len(history.history[keys[0]])):
            row = [i + 1] + [history.history[k][i] for k in keys]
            writer.writerow(row)
            
def train_streaming_model(num_train_streams=1200, num_val_streams=200, save_path="streaming_composer_ckpt"):
    (X_train, train_y), (X_val, val_y) = prepare_data(num_train_streams, num_val_streams)
    gpus = setup_gpus()
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

    with strategy.scope():
        core = StreamingComposerNet(input_dim=FEATURE_DIM, hidden_dim=128)
        trainer = StreamingComposerTrainer(core)
        trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    history = trainer.fit(
        X_train,
        train_y,
        validation_data=(X_val, val_y),
        epochs=EPOCHS,
        batch_size=TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync,
        verbose=1
    )

    core.save_weights(save_path)

    # Save history
    history_csv = os.path.join(LOG_DIR, "training_history.csv")
    save_training_history(history, history_csv)

    # Validation predictions for confusion matrices
    pred = core(X_val, training=False)
    pitch_pred = np.argmax(pred["pitch"].numpy()[:, -1, :], axis=1)
    octave_pred = np.argmax(pred["octave"].numpy()[:, -1, :], axis=1)

    pitch_true = val_y[0]
    octave_true = val_y[1]

    np.save(os.path.join(LOG_DIR, "pitch_true.npy"), pitch_true)
    np.save(os.path.join(LOG_DIR, "pitch_pred.npy"), pitch_pred)
    np.save(os.path.join(LOG_DIR, "octave_true.npy"), octave_true)
    np.save(os.path.join(LOG_DIR, "octave_pred.npy"), octave_pred)

    print(f"Saved model weights to {save_path}")
    print(f"Saved training history to {history_csv}")

    return core

# ============================================================
# 10. REAL-TIME COMPOSITION ENGINE (PENTATONIC + QUANTIZER)
# ============================================================
class RealTimeSampleComposerEngine:
    def __init__(self, audio_bank, sample_rate=SAMPLE_RATE, chunk_samples=CHUNK_SAMPLES, bpm=120):
        self.audio_bank = audio_bank
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.bpm = bpm
        self.samples_per_beat = int((60.0 / self.bpm / 2.0) * self.sample_rate)
        self.global_sample_count = 0

        self.current_note_name = None
        self.current_wave = None
        self.current_idx = 0

        self.queued_note_name = None 
        self.queued_active = False

        self.note_active = False
        self.release_gain = 1.0
        self.release_decay = 0.85

        self.active_on_th = 0.80
        self.active_off_th = 0.55
        self.onset_th = 0.20
        self.sustain_th = 0.45

        self.pending_note_name = None
        self.stable_count = 0
        self.required_stability = 4
        self.transition_matrix = np.array([
    # Do   Re   Mi   Fa   So   La   Ti
    [0.15, 0.35, 0.20, 0.05, 0.15, 0.05, 0.05],  # Do
    [0.20, 0.15, 0.30, 0.10, 0.15, 0.05, 0.05],  # Re
    [0.10, 0.20, 0.15, 0.20, 0.20, 0.10, 0.05],  # Mi
    [0.05, 0.10, 0.25, 0.15, 0.25, 0.10, 0.10],  # Fa
    [0.10, 0.10, 0.20, 0.15, 0.15, 0.20, 0.10],  # So
    [0.10, 0.05, 0.10, 0.10, 0.30, 0.20, 0.15],  # La
    [0.10, 0.05, 0.05, 0.10, 0.20, 0.30, 0.20],  # Ti
], dtype=np.float32)

    def _apply_confidence_pentatonic_bias(self, pitch_probs):
        """
        Confidence-aware pentatonic bias.
    
        If model confidence is high → keep original distribution.
        If medium → soft bias.
        If low → strong bias.
        """
    
        pitch_probs = np.asarray(pitch_probs, dtype=np.float32).copy()
    
        confidence = float(np.max(pitch_probs))
    
        # Bias strengths
        if confidence > 0.70:
            bias = 1.0      # trust model completely
        elif confidence > 0.40:
            bias = 0.45     # soft bias
        else:
            bias = 0.25     # strong bias
    
        # Do Re Mi Fa So La Ti
        mask = np.array([1.0, 1.0, 1.0, bias, 1.0, 1.0, bias], dtype=np.float32)
    
        pitch_probs *= mask
    
        s = pitch_probs.sum()
        if s <= 1e-8:
            return np.ones_like(pitch_probs) / len(pitch_probs)
    
        return pitch_probs / s
    def _apply_transition_stabilization(self, pitch_probs):
        """
        Blend current pitch probabilities with a transition prior
        based on the previously played note.
        """
        pitch_probs = np.asarray(pitch_probs, dtype=np.float32).copy()
    
        # If no previous note exists, do nothing
        if self.current_note_name is None:
            s = pitch_probs.sum()
            return pitch_probs / s if s > 1e-8 else np.ones_like(pitch_probs) / len(pitch_probs)
    
        prev_parts = self.current_note_name.split("-")
        if len(prev_parts) != 2:
            s = pitch_probs.sum()
            return pitch_probs / s if s > 1e-8 else np.ones_like(pitch_probs) / len(pitch_probs)
    
        prev_pitch_raw = prev_parts[1]
    
        # Normalize naming to match PITCH_TO_ID keys
        if prev_pitch_raw == "DO":
            prev_pitch = "Do"
        elif prev_pitch_raw == "so":
            prev_pitch = "So"
        elif prev_pitch_raw == "la":
            prev_pitch = "La"
        else:
            prev_pitch = prev_pitch_raw
    
        if prev_pitch not in PITCH_TO_ID:
            s = pitch_probs.sum()
            return pitch_probs / s if s > 1e-8 else np.ones_like(pitch_probs) / len(pitch_probs)
    
        prev_id = PITCH_TO_ID[prev_pitch]
        transition_probs = self.transition_matrix[prev_id]
    
        confidence = float(np.max(pitch_probs))
    
        # Stronger transition help only when model is uncertain
        if confidence > 0.70:
            lambda_weight = 0.10
        elif confidence > 0.40:
            lambda_weight = 0.25
        else:
            lambda_weight = 0.40
    
        stabilized = (1.0 - lambda_weight) * pitch_probs + lambda_weight * transition_probs
        s = stabilized.sum()
        return stabilized / s if s > 1e-8 else np.ones_like(stabilized) / len(stabilized)

    def _note_name_from_prediction(self, pitch_probs, octave_probs):
    
        # Step 1: soft confidence-aware pentatonic bias
        pitch_probs_biased = self._apply_confidence_pentatonic_bias(pitch_probs)
    
        # Step 2: temporal transition stabilization
        pitch_probs_final = self._apply_transition_stabilization(pitch_probs_biased)
    
        pitch_id = int(np.argmax(pitch_probs_final))
        octave_id = int(np.argmax(octave_probs))
    
        pitch_name = PITCH_CLASSES[pitch_id]
        octave_name = OCTAVE_CLASSES[octave_id]
    
        if pitch_name == "Do" and octave_name == "High":
            note_name = "High-DO"
        elif pitch_name == "So":
            note_name = f"{octave_name}-so"
        elif pitch_name == "La":
            note_name = f"{octave_name}-la"
        else:
            note_name = f"{octave_name}-{pitch_name}"
    
        return note_name, pitch_probs_final

    def _get_chunk_from_wave(self, wave, start_idx):
        end_idx = start_idx + self.chunk_samples
        if len(wave) == 0: return np.zeros((self.chunk_samples,), dtype=np.float32), start_idx
        if end_idx <= len(wave):
            chunk, new_idx = wave[start_idx:end_idx], end_idx
        else:
            chunk = np.zeros((self.chunk_samples,), dtype=np.float32)
            remaining, pos, write_pos = self.chunk_samples, start_idx, 0
            while remaining > 0:
                take = min(len(wave) - pos, remaining)
                chunk[write_pos:write_pos + take] = wave[pos:pos + take]
                write_pos += take; remaining -= take; pos = 0
            new_idx = (start_idx + self.chunk_samples) % len(wave)
        return chunk.astype(np.float32), new_idx

    def update(self, pitch_probs, octave_probs, onset_prob, sustain_prob, amplitude_pred, active_prob):
        note_name, pitch_probs_biased = self._note_name_from_prediction(pitch_probs, octave_probs)
        onset, sustain, active_prob = float(onset_prob) > self.onset_th, float(sustain_prob) > self.sustain_th, float(active_prob)
        active = active_prob > self.active_on_th if not self.queued_active else active_prob > self.active_off_th

        if self.pending_note_name == note_name: self.stable_count += 1
        else: self.pending_note_name = note_name; self.stable_count = 1

        if active and self.stable_count >= self.required_stability:
            self.queued_note_name = note_name
            self.queued_active = True
        elif not active:
            self.queued_active = False

        return {
    "note_name": note_name,
    "quantized_note": self.current_note_name,
    "onset": onset,
    "sustain": sustain,
    "active": active,
    "active_prob": active_prob,
    "amp": float(amplitude_pred),
    "confidence": float(np.max(pitch_probs)),
}

    def synthesize_chunk(self):
        is_beat = (self.global_sample_count % self.samples_per_beat) < self.chunk_samples
        if is_beat:
            if self.queued_active and self.queued_note_name:
                if self.queued_note_name != self.current_note_name or not self.note_active:
                    self.current_note_name, self.current_wave, self.current_idx = self.queued_note_name, self.audio_bank[self.queued_note_name], 0
                    self.note_active, self.release_gain = True, 1.0
            else: self.note_active = False

        self.global_sample_count += self.chunk_samples
        if self.current_wave is None: return np.zeros((self.chunk_samples,), dtype=np.float32)

        chunk, self.current_idx = self._get_chunk_from_wave(self.current_wave, self.current_idx)
        if self.note_active:
            self.release_gain, out = 1.0, chunk
        else:
            self.release_gain *= self.release_decay
            out = chunk * self.release_gain if self.release_gain >= 0.02 else np.zeros((self.chunk_samples,), dtype=np.float32)
        return np.clip(out, -1.0, 1.0).astype(np.float32)


# ============================================================
# 11. INFERENCE ENGINE & MAIN LOOP
# ============================================================
mp_holistic, mp_drawing = mp.solutions.holistic, mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image.flags.writeable = False; results = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True; return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks: mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks: mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks: mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_stream_features(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(HAND_DIM, dtype=np.float32)
    
    features = [lh, rh]

    if USE_POSE:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_DIM, dtype=np.float32)
        features.insert(0, pose) # Put pose at the front if enabled

    if USE_FACE:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(FACE_DIM, dtype=np.float32)
        features.append(face)

    return np.concatenate(features, axis=0).astype(np.float32)

def build_inference_model(weights_path: str = "streaming_composer_ckpt"):
    model = StreamingComposerNet(input_dim=FEATURE_DIM, hidden_dim=128)
    _ = model(tf.zeros((1, STREAM_WINDOW, FEATURE_DIM), dtype=tf.float32), training=False)
    model.load_weights(weights_path).expect_partial()
    return model

def init_inference_log(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "time_sec",
            "loop_latency_ms",
            "infer_latency_ms",
            "pred_note",
            "grid_note",
            "onset",
            "sustain",
            "active",
            "active_prob",
            "amplitude",
            "confidence"
        ])

def run_realtime_composer(model: Model, session_name="session1"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = 20.0  # fixed output fps for saved video

    engine = RealTimeSampleComposerEngine(load_note_audio_bank())
    buffer, prev_infer_time = [], time.time()

    session_csv = os.path.join(LOG_DIR, f"{session_name}_inference_log.csv")
    session_wav = os.path.join(AUDIO_OUT_DIR, f"{session_name}_rendered.wav")
    session_video = os.path.join(VIDEO_OUT_DIR, f"{session_name}_inference.mp4")
    session_merged = os.path.join(VIDEO_OUT_DIR, f"{session_name}_final_with_audio.mp4")

    init_inference_log(session_csv)

    # -------- live audio writer --------
    wav_writer = wave.open(session_wav, "wb")
    wav_writer.setnchannels(1)
    wav_writer.setsampwidth(2)   # int16 = 2 bytes
    wav_writer.setframerate(SAMPLE_RATE)

    # -------- video writer --------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(session_video, fourcc, video_fps, (frame_width, frame_height))

    rendered_audio = []
    frame_idx = 0
    t0 = time.time()

    if AUDIO_ENABLED:
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback
        )
        stream.start()

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_start = time.time()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            buffer.append(extract_stream_features(results))
            buffer = buffer[-STREAM_WINDOW:]
            pred_latency_ms = 0.0
            status = None

            if len(buffer) == STREAM_WINDOW:
                infer_start = time.time()
                pred = model(np.expand_dims(np.array(buffer, dtype=np.float32), axis=0), training=False)
                infer_end = time.time()
                pred_latency_ms = (infer_end - infer_start) * 1000.0

                status = engine.update(
                    pitch_probs=pred["pitch"].numpy()[0, -1, :],
                    octave_probs=pred["octave"].numpy()[0, -1, :],
                    onset_prob=pred["onset"].numpy()[0, -1, 0],
                    sustain_prob=pred["sustain"].numpy()[0, -1, 0],
                    amplitude_pred=pred["amplitude"].numpy()[0, -1, 0],
                    active_prob=pred["active"].numpy()[0, -1, 0]
                )

                chunk = engine.synthesize_chunk()
                rendered_audio.append(chunk.copy())

                # save audio chunk live
                chunk_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                wav_writer.writeframes(chunk_int16.tobytes())

                if AUDIO_ENABLED:
                    if audio_queue.full():
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            pass
                    audio_queue.put(chunk)

                cv2.putText(
                    image,
                    f"Hand: {status['note_name']} | Grid: {status['quantized_note'] or 'Silence'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    image,
                    f"On:{int(status['onset'])} Su:{int(status['sustain'])} Ac:{int(status['active'])} Ap:{status['active_prob']:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    image,
                    f"Amp:{status['amp']:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )

            frame_end = time.time()
            loop_latency_ms = (frame_end - frame_start) * 1000.0

            cv2.putText(
                image,
                f"Loop latency: {loop_latency_ms:.1f} ms",
                (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                image,
                f"Infer latency: {pred_latency_ms:.1f} ms",
                (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
            )

            fps = 1.0 / max(frame_end - prev_infer_time, 1e-6)
            prev_infer_time = frame_end
            cv2.putText(
                image,
                f"FPS: {fps:.1f}",
                (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
            )

            # save annotated frame to video
            video_writer.write(image)

            cv2.imshow("Streaming Low-Latency Gesture Composer", image)

            if status is not None:
                with open(session_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        frame_idx,
                        time.time() - t0,
                        loop_latency_ms,
                        pred_latency_ms,
                        status["note_name"],
                        status["quantized_note"] or "Silence",
                        int(status["onset"]),
                        int(status["sustain"]),
                        int(status["active"]),
                        float(status["active_prob"]),
                        float(status["amp"]),
                        float(status["confidence"])
                    ])

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if AUDIO_ENABLED:
        stream.stop()
        stream.close()

    cap.release()
    video_writer.release()
    wav_writer.close()
    cv2.destroyAllWindows()

    print(f"Saved inference log to {session_csv}")
    print(f"Saved rendered audio to {session_wav}")
    print(f"Saved inference video to {session_video}")

    # optional: merge saved video + saved audio into one mp4
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i", session_video,
            "-i", session_wav,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            session_merged
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Saved merged video+audio to {session_merged}")
    except Exception as e:
        print("WARNING: Could not merge audio and video with ffmpeg.")
        print(f"Reason: {e}")
        print("You still have separate files:")
        print(session_video)
        print(session_wav)

if __name__ == "__main__":
    # ---> RUN THIS ON YOUR LINUX SERVER <---
    # model = train_streaming_model(num_train_streams=1200, num_val_streams=300, save_path="streaming_composer_ckpt_new_training_history_full_run_final")
    
    model = build_inference_model("streaming_composer_ckpt_new_training_history")
    run_realtime_composer(model, session_name="session1")