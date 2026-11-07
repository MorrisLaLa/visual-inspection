from visual_inspection.categories import ANIMAL, PERSON, VEHICLE, category_of, class_ids_for

COCO_SUBSET = {0: "person", 1: "bicycle", 2: "car", 7: "truck", 14: "bird", 16: "dog", 56: "chair", 62: "tv"}


def test_category_of_coco_names():
    assert category_of("person") == PERSON
    assert category_of("truck") == VEHICLE
    assert category_of("dog") == ANIMAL
    assert category_of("chair") is None


def test_category_of_ignores_case_and_spaces():
    assert category_of(" Car ") == VEHICLE


def test_class_ids_for_all_categories_skips_other_classes():
    assert class_ids_for(COCO_SUBSET) == [0, 1, 2, 7, 14, 16]


def test_class_ids_for_selected_categories():
    assert class_ids_for(COCO_SUBSET, [ANIMAL]) == [14, 16]
    assert class_ids_for(COCO_SUBSET, [VEHICLE, PERSON]) == [0, 1, 2, 7]


def test_class_ids_for_custom_model_named_after_categories():
    assert class_ids_for({0: "vehicle", 1: "person", 2: "animal"}) == [0, 1, 2]
