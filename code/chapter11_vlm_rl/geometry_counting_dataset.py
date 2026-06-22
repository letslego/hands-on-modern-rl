"""
11：
==========================================================

 VLM（）：
  1. 、、
  2. （0~5 ）， ground truth 
  3.  geometry_dataset/ 
  4.  JSON （、、）
  5.  50  + 10 
  6.  4 
  7. 

：
  -  VLM GRPO 
  - 
  - 

：
  pip install -r requirements.txt
  python geometry_counting_dataset.py
"""

import os
import json
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 
os.makedirs("output", exist_ok=True)

# ，
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# ：
# ==========================================
# ：、、
# 、

# ，
SHAPE_COLORS = [
    '#e74c3c',  # 
    '#3498db',  # 
    '#2ecc71',  # 
    '#f39c12',  # 
    '#9b59b6',  # 
    '#1abc9c',  # 
    '#e67e22',  # 
    '#2980b9',  # 
    '#27ae60',  # 
    '#c0392b',  # 
    '#8e44ad',  # 
    '#d35400',  # 
]


def draw_triangle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    

    ：
        draw: PIL ImageDraw 
        cx, cy: 
        size: （）
        color: 
        outline_color: 
    """
    # 
    # ，
    points = [
        (cx, cy - size),                          # 
        (cx - size * 0.866, cy + size * 0.5),     # 
        (cx + size * 0.866, cy + size * 0.5),     # 
    ]
    draw.polygon(points, fill=color, outline=outline_color, width=2)


def draw_circle(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    

    ：
        draw: PIL ImageDraw 
        cx, cy: 
        size: 
        color: 
        outline_color: 
    """
    # PIL  ellipse 
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.ellipse(bbox, fill=color, outline=outline_color, width=2)


def draw_square(draw, cx, cy, size, color, outline_color='#2c3e50'):
    """
    

    ：
        draw: PIL ImageDraw 
        cx, cy: 
        size: 
        color: 
        outline_color: 
    """
    bbox = [cx - size, cy - size, cx + size, cy + size]
    draw.rectangle(bbox, fill=color, outline=outline_color, width=2)


# 
SHAPE_DRAWERS = {
    '': draw_triangle,
    '': draw_circle,
    '': draw_square,
}


# ==========================================
# ：
# ==========================================
def generate_single_image(img_width=256, img_height=256, seed=None):
    """
    

    ：
      1. 
      2. （、、）， 0~5 
      3. 、、
      4. （）

    ：
        img_width: （）
        img_height: （）
        seed: （，）

    ：
        image: PIL Image 
        ground_truth: dict，
                      : {'': 3, '': 1, '': 2}
    """
    if seed is not None:
        random.seed(seed)

    # 
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)

    # 
    ground_truth = {'': 0, '': 0, '': 0}

    # （）
    occupied_regions = []

    for shape_name in ['', '', '']:
        # （0~5）
        count = random.randint(0, 5)
        ground_truth[shape_name] = count

        for _ in range(count):
            # ：15~30 
            size = random.randint(15, 30)

            # 
            color = random.choice(SHAPE_COLORS)

            # 
            #  50 ，
            placed = False
            for _attempt in range(50):
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)

                # ：
                overlap = False
                for (ox, oy, osize) in occupied_regions:
                    dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                    if dist < (size + osize) * 0.8:
                        overlap = True
                        break

                if not overlap:
                    # 
                    drawer = SHAPE_DRAWERS[shape_name]
                    drawer(draw, cx, cy, size, color)
                    occupied_regions.append((cx, cy, size))
                    placed = True
                    break

            # ，
            if not placed:
                cx = random.randint(size + 10, img_width - size - 10)
                cy = random.randint(size + 10, img_height - size - 10)
                drawer = SHAPE_DRAWERS[shape_name]
                drawer(draw, cx, cy, size, color)
                occupied_regions.append((cx, cy, size))

    return image, ground_truth


# ==========================================
# ：
# ==========================================
def generate_dataset(output_dir='output/geometry_dataset',
                     num_train=50, num_test=10,
                     img_width=256, img_height=256,
                     base_seed=42):
    """
    

    ：
        geometry_dataset/
        ├── train/           # 
        │   ├── train_0001.png
        │   ├── train_0002.png
        │   └── ...
        ├── test/            # 
        │   ├── test_0001.png
        │   └── ...
        └── metadata.json    # 

     JSON ：
        {
            "train": [
                {
                    "image_path": "train/train_0001.png",
                    "prompt": "、",
                    "ground_truth": {"": 3, "": 1, "": 2},
                    "total_shapes": 6
                },
                ...
            ],
            "test": [...]
        }

    ：
        output_dir: 
        num_train: 
        num_test: 
        img_width: 
        img_height: 
        base_seed: 

    ：
        metadata: dict，
    """
    print("=" * 60)
    print("  ")
    print("=" * 60)

    # 
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    metadata = {'train': [], 'test': []}

    # ：
    prompt = "、"

    # ----------  ----------
    print(f"\n（{num_train} ）...")
    for i in range(num_train):
        seed = base_seed + i
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"train_{i+1:04d}.png"
        filepath = os.path.join(train_dir, filename)
        image.save(filepath)

        metadata['train'].append({
            'image_path': f"train/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

        # 
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   {i+1}/{num_train} ")

    # ----------  ----------
    print(f"\n（{num_test} ）...")
    for i in range(num_test):
        seed = base_seed + 1000 + i  # ，
        image, gt = generate_single_image(img_width, img_height, seed=seed)

        filename = f"test_{i+1:04d}.png"
        filepath = os.path.join(test_dir, filename)
        image.save(filepath)

        metadata['test'].append({
            'image_path': f"test/{filename}",
            'prompt': prompt,
            'ground_truth': gt,
            'total_shapes': sum(gt.values()),
        })

    #  JSON
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n: {metadata_path}")
    print(f": {len(metadata['train'])} ")
    print(f": {len(metadata['test'])} ")

    return metadata


# ==========================================
# ：
# ==========================================
def print_dataset_statistics(metadata):
    """
    

    ：
      1. 
      2. 
      3. （、、）
      4. 
    """
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)

    all_data = metadata['train'] + metadata['test']
    total_samples = len(all_data)

    print(f"\n: {total_samples}")
    print(f"  : {len(metadata['train'])} ")
    print(f"  : {len(metadata['test'])} ")

    # 
    print("\n：")
    print(f"  {'':>6s}  {'':>4s}  {'':>4s}  {'':>6s}  {'':>8s}")
    print(f"  {'------':>6s}  {'----':>4s}  {'----':>4s}  {'------':>6s}  {'--------':>8s}")

    total_shapes_list = []
    for shape_name in ['', '', '']:
        counts = [item['ground_truth'][shape_name] for item in all_data]
        min_count = min(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        #  =  1 
        freq = sum(1 for c in counts if c > 0) / len(counts) * 100
        print(f"  {shape_name:>6s}  {min_count:>4d}  {max_count:>4d}  "
              f"{avg_count:>6.2f}  {freq:>7.1f}%")

    # 
    total_counts = [item['total_shapes'] for item in all_data]
    total_shapes_list.extend(total_counts)
    print(f"\n：")
    print(f"  : {min(total_counts)} ")
    print(f"  : {max(total_counts)} ")
    print(f"  : {sum(total_counts) / len(total_counts):.1f} ")

    # 
    print(f"\n：")
    count_dist = {}
    for t in total_counts:
        count_dist[t] = count_dist.get(t, 0) + 1
    for k in sorted(count_dist.keys()):
        bar = '|' * (count_dist[k] * 2)
        print(f"  {k:>2d} : {bar} ({count_dist[k]} )")


# ==========================================
# ：
# ==========================================
def display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4):
    """
    

    ：
        metadata: 
        output_dir: 
        num_samples: 
    """
    print("\n" + "=" * 60)
    print(f"   {num_samples} ")
    print("=" * 60)

    # 
    train_data = metadata['train']
    indices = np.linspace(0, len(train_data) - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(" — ", fontsize=16, fontweight='bold')

    for ax_idx, idx in enumerate(indices):
        row = ax_idx // 2
        col = ax_idx % 2
        ax = axes[row, col]

        item = train_data[idx]
        img_path = os.path.join(output_dir, item['image_path'])
        image = Image.open(img_path)

        ax.imshow(image)
        ax.axis('off')

        # 
        gt = item['ground_truth']
        title = (f" #{idx+1}\n"
                 f": {gt['']}  : {gt['']}  : {gt['']}\n"
                 f": {item['total_shapes']} ")
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig('output/geometry_dataset_samples.png', dpi=150, bbox_inches='tight')
    print("   output/geometry_dataset_samples.png")
    plt.show()


# ==========================================
# 
# ==========================================
if __name__ == "__main__":
    # ：
    metadata = generate_dataset(
        output_dir='output/geometry_dataset',
        num_train=50,
        num_test=10,
        img_width=256,
        img_height=256,
        base_seed=42,
    )

    # ：
    print_dataset_statistics(metadata)

    # ：
    display_sample_images(metadata, output_dir='output/geometry_dataset', num_samples=4)

    # 
    print("\n" + "=" * 60)
    print("  ")
    print("=" * 60)
    print("""
  ：
    1.  VLM（）
    2. 
    3.  GRPO ，
    4. （ +  + ）

  ：
    -  0~5 、0~5 、0~5 
    -  0~15 ，
    - ，，
    - ："、"

  ：
    -  multi_modal_reward.py：
    -  vlm_grpo_train.py： VLM GRPO 
    """)
