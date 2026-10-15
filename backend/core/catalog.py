"""Static catalogs: YOLO model families/variants and COCO class names."""

VARIANT_LABELS = {
    "n": "Nano",
    "t": "Tiny",
    "s": "Small",
    "m": "Medium",
    "b": "Balanced",
    "c": "Compact",
    "l": "Large",
    "e": "Extended",
    "x": "Extra-Large",
}

# Ultralytics naming: v8-v10 keep the "v", YOLO11 onward drop it.
MODEL_FAMILIES = [
    {"id": "v8", "label": "YOLOv8", "prefix": "yolov8", "variants": ["n", "s", "m", "l", "x"]},
    {"id": "v9", "label": "YOLOv9", "prefix": "yolov9", "variants": ["t", "s", "m", "c", "e"]},
    {"id": "v10", "label": "YOLOv10", "prefix": "yolov10", "variants": ["n", "s", "m", "b", "l", "x"]},
    {"id": "11", "label": "YOLO11", "prefix": "yolo11", "variants": ["n", "s", "m", "l", "x"]},
    {"id": "12", "label": "YOLO12", "prefix": "yolo12", "variants": ["n", "s", "m", "l", "x"]},
    {"id": "26", "label": "YOLO26", "prefix": "yolo26", "variants": ["n", "s", "m", "l", "x"]},
]

OPTIMIZERS = ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]

KAGGLE_ACCELERATORS = [
    {"id": "", "label": "Default GPU"},
    {"id": "NvidiaTeslaT4", "label": "NVIDIA T4 x2"},
    {"id": "NvidiaTeslaP100", "label": "NVIDIA P100"},
]

# Base COCO-pretrained detectors offered for AI pre-labeling.
AUTOLABEL_BASE_MODELS = [
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
    "yolo26n.pt", "yolo26s.pt",
    "yolov8n.pt", "yolov8s.pt",
]


def resolve_model_name(family_id: str, variant: str, pretrained: bool) -> str:
    fam = next((f for f in MODEL_FAMILIES if f["id"] == family_id), None)
    if fam is None:
        raise ValueError(f"Unknown model family '{family_id}'.")
    if variant not in fam["variants"]:
        raise ValueError(f"{fam['label']} has no '{variant}' variant (valid: {', '.join(fam['variants'])}).")
    return f"{fam['prefix']}{variant}.{'pt' if pretrained else 'yaml'}"


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

CLASS_COLORS = [
    "#10B981", "#F59E0B", "#3B82F6", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
    "#84CC16", "#06B6D4", "#A855F7", "#EAB308", "#22C55E", "#6366F1", "#F43F5E", "#0EA5E9",
]


def catalog_payload() -> dict:
    return {
        "families": [
            {**f, "variants": [{"id": v, "label": VARIANT_LABELS[v]} for v in f["variants"]]}
            for f in MODEL_FAMILIES
        ],
        "optimizers": OPTIMIZERS,
        "accelerators": KAGGLE_ACCELERATORS,
        "coco_classes": COCO_CLASSES,
        "class_colors": CLASS_COLORS,
        "autolabel_models": AUTOLABEL_BASE_MODELS,
    }
