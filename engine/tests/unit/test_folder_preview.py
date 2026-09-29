"""Vista previa de carpetas y zips de imágenes sin extraerlos (la del dataset de animales)."""

from __future__ import annotations

from perceptron.data.sources.files import SourceKind, folder_preview


def test_zip_with_one_root_folder_and_nested_class_dirs() -> None:
    names = ["Dataset Of animal Images/"]
    for cls, n in (("Cat", 5), ("Dog", 3), ("Hen", 1)):
        names += [f"Dataset Of animal Images/{cls}/", f"Dataset Of animal Images/{cls}/data.yaml"]
        names += [f"Dataset Of animal Images/{cls}/train/images/{i}.jpg" for i in range(n)]
        names += [f"Dataset Of animal Images/{cls}/train/labels/{i}.txt" for i in range(n)]
    names += ["__MACOSX/Dataset Of animal Images/Cat/._0.jpg", "Dataset Of animal Images/.DS_Store"]
    prev = folder_preview(names, limit=4)
    assert prev is not None and prev.kind is SourceKind.IMAGE_FOLDER
    assert prev.total == 9 and prev.classes == {"Cat": 5, "Dog": 3, "Hen": 1}
    # La muestra recorre las clases antes de repetir una.
    assert [s["label"] for s in prev.samples] == ["Cat", "Dog", "Hen", "Cat"]
    assert prev.samples[0]["path"] == "Cat/train/images/0.jpg"


def test_audio_and_unlabeled_root_files() -> None:
    prev = folder_preview(["normal/a.wav", "falla/b.wav", "suelto.wav"])
    assert prev is not None and prev.kind is SourceKind.AUDIO_FOLDER
    assert prev.classes == {"(sin etiqueta)": 1, "falla": 1, "normal": 1}


def test_annotations_and_masks_need_full_detection() -> None:
    assert folder_preview(["img/a.jpg", "annotations_coco.json"]) is None
    assert folder_preview(["images/a.png", "masks/a.png"]) is None
    assert folder_preview(["datos.csv"]) is None
    assert folder_preview([]) is None
