import json
import random
import os


def generate_mock_data(num_samples=100, output_file="output/preference_data.json"):
    """
     DPO  Mock 。

    ：（Sycophancy）。
    ，，
    。

    - chosen：
    - rejected：
    """
    templates = [
        {
            "prompt": "，？",
            "chosen": "。，。，。",
            "rejected": "，，。"
        },
        {
            "prompt": "，。",
            "chosen": "，、。，\"\"，。。",
            "rejected": "，。，。"
        },
        {
            "prompt": "，。",
            "chosen": "，，、，。，，。",
            "rejected": "，，！"
        },
        {
            "prompt": "，。",
            "chosen": "，。，。，。，。",
            "rejected": "，，。"
        },
        {
            "prompt": "，。",
            "chosen": "，。：，、。，。",
            "rejected": "！，。"
        },
        {
            "prompt": "，。",
            "chosen": "，。\"\"。，。",
            "rejected": "，，，。"
        },
    ]

    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        data.append({
            "prompt": f"{template['prompt']} ( {i+1})",
            "chosen": template["chosen"],
            "rejected": template["rejected"]
        })

    # 
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f" {num_samples} ，: {output_file}")
    print("，，，！")


if __name__ == "__main__":
    generate_mock_data(100, "output/preference_data.json")
