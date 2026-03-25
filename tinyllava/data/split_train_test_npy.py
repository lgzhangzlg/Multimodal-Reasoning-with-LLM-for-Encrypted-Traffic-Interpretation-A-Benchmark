import os
import random
import shutil
from collections import defaultdict

# ==========================
# 参数
# ==========================
DATA_DIR = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_1600_v3_noport_balanced"
TRAIN_DIR = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_1600_v3_noport_balanced_split/train"
TEST_DIR = "/root/autodl-tmp/Datasets/USTC-TFC-2016/USTC-TFC-2016_npy_1600_v3_noport_balanced_split/test"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42
EXTENSION = ".npy"
# ==========================

def split_dataset(data_dir, train_dir, test_dir, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    rng = random.Random(seed)

    # 按类别（第一级子文件夹）收集文件
    class_to_files = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(EXTENSION):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, data_dir)
                cls = rel_path.split(os.sep)[0]
                class_to_files[cls].append(rel_path)

    print(f"{'Class':<30s} {'Total':>8s} {'Train':>8s} {'Test':>8s}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    total_train = total_test = 0

    for cls in sorted(class_to_files.keys()):
        files = class_to_files[cls]
        rng.shuffle(files)

        n_train = int(len(files) * train_ratio)
        train_files = files[:n_train]
        test_files = files[n_train:]

        for rel in train_files:
            src = os.path.join(data_dir, rel)
            dst = os.path.join(train_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        for rel in test_files:
            src = os.path.join(data_dir, rel)
            dst = os.path.join(test_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"{cls:<30s} {len(files):>8d} {len(train_files):>8d} {len(test_files):>8d}")

    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'TOTAL':<30s} {total_train+total_test:>8d} {total_train:>8d} {total_test:>8d}")
    print(f"\nTrain -> {train_dir}")
    print(f"Test  -> {test_dir}")


if __name__ == "__main__":
    split_dataset(DATA_DIR, TRAIN_DIR, TEST_DIR)