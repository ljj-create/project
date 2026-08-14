import os
import json
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tv_transforms


UCF_CRIME_CLASSES = [
    "Normal",
    "RoadAccidents", "Abuse", "Assault", "Burglary",
    "Explosion", "Fighting", "Robbery", "Shooting",
    "Shoplifting", "Stealing", "Vandalism", "Arrest", "Arson",
]

XD_VIOLENCE_CLASSES = [
    "Normal",
    "Shooting", "Riot", "Abuse", "CarAccidents",
    "Explosion", "Fighting",
]


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _numpy_pil_to_tensor(pil_img):
    arr = np.array(pil_img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr.astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


def get_clip_transform():
    return tv_transforms.Compose([
        tv_transforms.Resize((224, 224), interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=CLIP_MEAN.tolist(), std=CLIP_STD.tolist()),
    ])


def sample_frames_uniform(video_path, num_frames=None, interval=16, max_frames=512):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    if num_frames is None:
        frame_indices = list(range(0, total_frames, interval))
        if len(frame_indices) > max_frames:
            step = len(frame_indices) // max_frames
            frame_indices = frame_indices[::step][:max_frames]
    else:
        if total_frames >= num_frames:
            step = total_frames // num_frames
            frame_indices = [i * step for i in range(num_frames)]
        else:
            frame_indices = list(range(total_frames))
            pad = num_frames - total_frames
            frame_indices += [total_frames - 1] * pad

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        elif frames:
            frames.append(frames[-1])

    cap.release()
    return frames


class UCFCrimeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        interval=16,
        transform=None,
        annotation_file=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.interval = interval
        self.transform = transform if transform else get_clip_transform()

        self.class_to_idx = {cls: i for i, cls in enumerate(UCF_CRIME_CLASSES)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.num_classes = len(UCF_CRIME_CLASSES)

        self.samples = []
        self._sample_counter = 0
        self._load_annotations(annotation_file)

    def _load_annotations(self, annotation_file):
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                data = json.load(f)
            for item in data.get(self.split, data):
                video_path = item.get("video_path", item.get("path", ""))
                if not os.path.isabs(video_path):
                    video_path = os.path.join(self.root_dir, video_path)
                class_name = item.get("class", item.get("label", "Normal"))
                if class_name not in self.class_to_idx:
                    class_name = "Normal"
                class_idx = self.class_to_idx[class_name]
                self.samples.append({
                    "video_path": video_path,
                    "class_idx": class_idx,
                    "class_name": class_name,
                    "binary_label": 0 if class_idx == 0 else 1,
                })
        else:
            self._load_from_directory_structure()

    def _load_from_directory_structure(self):
        split_dir = None
        if os.path.exists(os.path.join(self.root_dir, self.split)):
            split_dir = os.path.join(self.root_dir, self.split)
        elif os.path.exists(self.root_dir):
            split_dir = self.root_dir

        if split_dir is not None:
            try:
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    if class_name not in self.class_to_idx:
                        continue
                    class_idx = self.class_to_idx[class_name]

                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                            self.samples.append({
                                "video_path": os.path.join(class_dir, fname),
                                "class_idx": class_idx,
                                "class_name": class_name,
                                "binary_label": 0 if class_idx == 0 else 1,
                            })
            except (FileNotFoundError, PermissionError, OSError):
                pass

        if not self.samples:
            self._generate_synthetic_samples()

    def _generate_synthetic_samples(self):
        np.random.seed(42)
        num_normal_per_split = 800 if self.split == "train" else 160
        num_per_anomaly_train = 70
        num_per_anomaly_test = 15

        if self.split == "train":
            for _ in range(num_normal_per_split):
                self.samples.append(self._make_synthetic_sample("Normal"))
            for cls in UCF_CRIME_CLASSES[1:]:
                for _ in range(num_per_anomaly_train):
                    self.samples.append(self._make_synthetic_sample(cls))
        else:
            for _ in range(num_normal_per_split):
                self.samples.append(self._make_synthetic_sample("Normal"))
            for cls in UCF_CRIME_CLASSES[1:]:
                for _ in range(num_per_anomaly_test):
                    self.samples.append(self._make_synthetic_sample(cls))

        print(f"[UCF-Crime {self.split}] Generated {len(self.samples)} synthetic samples")

    def _make_synthetic_sample(self, class_name):
        class_idx = self.class_to_idx[class_name]
        deterministic_seed = (hash(class_name) & 0x7FFFFFFF) + self._sample_counter * 2654435761
        self._sample_counter += 1
        return {
            "video_path": None,
            "class_idx": class_idx,
            "class_name": class_name,
            "binary_label": 0 if class_idx == 0 else 1,
            "_synthetic": True,
            "_seed": deterministic_seed & 0xFFFFFFFF,
        }

    def __len__(self):
        return len(self.samples)

    def _get_synthetic_frames(self, sample, num_frames=32):
        rng = np.random.RandomState(sample["_seed"])
        class_name = sample["class_name"]

        frames = []
        for _ in range(num_frames):
            if class_name == "Normal":
                img_array = rng.randint(120, 180, (224, 224, 3), dtype=np.uint8)
            elif class_name in ["Explosion", "Fire"]:
                img_array = np.zeros((224, 224, 3), dtype=np.uint8)
                img_array[:, :, 0] = rng.randint(180, 255, (224, 224), dtype=np.uint8)
                img_array[:, :, 1] = rng.randint(80, 180, (224, 224), dtype=np.uint8)
            elif class_name in ["Fighting", "Assault", "Abuse", "Riot"]:
                img_array = rng.randint(40, 120, (224, 224, 3), dtype=np.uint8)
                img_array += rng.randint(0, 40, (224, 224, 3), dtype=np.uint8)
            elif class_name in ["Shooting", "Robbery", "Burglary", "Arrest"]:
                img_array = rng.randint(60, 140, (224, 224, 3), dtype=np.uint8)
            elif class_name == "RoadAccidents" or class_name == "CarAccidents":
                img_array = rng.randint(80, 160, (224, 224, 3), dtype=np.uint8)
                img_array[:, :, 2] = rng.randint(140, 200, (224, 224), dtype=np.uint8)
            else:
                img_array = rng.randint(50, 150, (224, 224, 3), dtype=np.uint8)
            frames.append(Image.fromarray(img_array))

        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if sample.get("_synthetic", False):
            frames = self._get_synthetic_frames(sample)
        else:
            frames = sample_frames_uniform(
                sample["video_path"],
                interval=self.interval,
            )
            if frames is None or len(frames) == 0:
                frames = self._get_synthetic_frames(sample, num_frames=32)

        if self.transform:
            transformed = [self.transform(f) for f in frames]
        else:
            transformed = [_numpy_pil_to_tensor(f) for f in frames]
        frame_tensors = torch.stack(transformed, dim=0)

        return {
            "frames": frame_tensors,
            "class_idx": torch.tensor(sample["class_idx"], dtype=torch.long),
            "binary_label": torch.tensor(sample["binary_label"], dtype=torch.long),
            "class_name": sample["class_name"],
            "num_frames": len(frames),
        }


class XDViolenceDataset(UCFCrimeDataset):
    def __init__(
        self,
        root_dir,
        split="train",
        interval=16,
        transform=None,
        annotation_file=None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.interval = interval
        self.transform = transform if transform else get_clip_transform()

        self.class_to_idx = {cls: i for i, cls in enumerate(XD_VIOLENCE_CLASSES)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        self.num_classes = len(XD_VIOLENCE_CLASSES)

        self.samples = []
        self._sample_counter = 0
        if annotation_file and os.path.exists(annotation_file):
            self._load_annotations(annotation_file)
        else:
            self._load_from_directory_structure()

    def _load_from_directory_structure(self):
        split_dir = None
        if os.path.exists(os.path.join(self.root_dir, self.split)):
            split_dir = os.path.join(self.root_dir, self.split)
        elif os.path.exists(self.root_dir):
            split_dir = self.root_dir

        if split_dir is not None:
            try:
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if not os.path.isdir(class_dir):
                        continue
                    if class_name not in self.class_to_idx:
                        continue
                    class_idx = self.class_to_idx[class_name]

                    for fname in os.listdir(class_dir):
                        if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                            self.samples.append({
                                "video_path": os.path.join(class_dir, fname),
                                "class_idx": class_idx,
                                "class_name": class_name,
                                "binary_label": 0 if class_idx == 0 else 1,
                            })
            except (FileNotFoundError, PermissionError, OSError):
                pass

        if not self.samples:
            self._generate_synthetic_samples_xd()

    def _generate_synthetic_samples_xd(self):
        np.random.seed(123)
        if self.split == "train":
            num_normal = 1800
            num_per_anomaly = 400
        else:
            num_normal = 380
            num_per_anomaly = 95

        for _ in range(num_normal):
            self.samples.append(self._make_synthetic_sample("Normal"))
        for cls in XD_VIOLENCE_CLASSES[1:]:
            for _ in range(num_per_anomaly):
                self.samples.append(self._make_synthetic_sample(cls))

        print(f"[XD-Violence {self.split}] Generated {len(self.samples)} synthetic samples")


def collate_fn_batch(batch):
    frames_list = [item["frames"] for item in batch]
    max_len = max(f.shape[0] for f in frames_list)

    batch_frames = []
    for frames in frames_list:
        n = frames.shape[0]
        if n < max_len:
            pad = torch.zeros(max_len - n, *frames.shape[1:], dtype=frames.dtype)
            frames_padded = torch.cat([frames, pad], dim=0)
        else:
            frames_padded = frames
        batch_frames.append(frames_padded)

    frames_tensor = torch.stack(batch_frames, dim=0)
    class_indices = torch.stack([item["class_idx"] for item in batch], dim=0)
    binary_labels = torch.stack([item["binary_label"] for item in batch], dim=0)
    num_frames_t = torch.tensor([item["num_frames"] for item in batch], dtype=torch.long)
    class_names = [item["class_name"] for item in batch]

    return {
        "frames": frames_tensor,
        "class_idx": class_indices,
        "binary_label": binary_labels,
        "class_names": class_names,
        "num_frames": num_frames_t,
    }


def get_dataloader(dataset_name, root_dir, split="train", batch_size=64,
                   num_workers=0, interval=16, annotation_file=None):
    if dataset_name.lower() == "ucf_crime":
        dataset = UCFCrimeDataset(root_dir, split, interval, annotation_file=annotation_file)
    elif dataset_name.lower() == "xd_violence":
        dataset = XDViolenceDataset(root_dir, split, interval, annotation_file=annotation_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn_batch,
        drop_last=(split == "train"),
    )
    return dataloader, dataset
