"""Group detector class names into the three categories this project cares about."""

VEHICLE = "vehicle"
PERSON = "person"
ANIMAL = "animal"
CATEGORIES = (VEHICLE, PERSON, ANIMAL)

# COCO class names used by the pretrained YOLO models -> category.
# A custom-trained model whose classes are named after the categories themselves
# ("vehicle", "person", "animal") works too. Add new class names here when a
# custom model introduces them (e.g. "monkey": ANIMAL).
CLASS_TO_CATEGORY = {
    "person": PERSON,
    "bicycle": VEHICLE,
    "car": VEHICLE,
    "motorcycle": VEHICLE,
    "bus": VEHICLE,
    "truck": VEHICLE,
    "train": VEHICLE,
    "boat": VEHICLE,
    "airplane": VEHICLE,
    "bird": ANIMAL,
    "cat": ANIMAL,
    "dog": ANIMAL,
    "horse": ANIMAL,
    "sheep": ANIMAL,
    "cow": ANIMAL,
    "elephant": ANIMAL,
    "bear": ANIMAL,
    "zebra": ANIMAL,
    "giraffe": ANIMAL,
    VEHICLE: VEHICLE,
    ANIMAL: ANIMAL,
}

# Box colours in OpenCV's BGR order.
COLORS = {
    VEHICLE: (255, 140, 0),
    PERSON: (0, 200, 0),
    ANIMAL: (0, 140, 255),
}


def category_of(class_name: str) -> str | None:
    """Return the category for a detector class name, or None if it is not one we track."""
    return CLASS_TO_CATEGORY.get(class_name.strip().lower())


def class_ids_for(names: dict[int, str], categories=CATEGORIES) -> list[int]:
    """Return the model's class ids that fall into the given categories."""
    wanted = set(categories)
    return sorted(i for i, name in names.items() if category_of(name) in wanted)
