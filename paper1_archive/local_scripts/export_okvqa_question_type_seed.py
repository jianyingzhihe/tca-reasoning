#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(r"E:\Bridging")
SOURCE = ROOT / "annotation" / "okvqa_type_label_round3_320" / "manifest.csv"
OUT_DIR = ROOT / "annotation" / "okvqa_type_label_round3_320"
QUESTIONS_ONLY_OUT = OUT_DIR / "question_only_320.csv"
QUESTIONS_BLANK_OUT = OUT_DIR / "question_only_320_blank_annotation.csv"
IMAGE_BLANK_OUT = OUT_DIR / "image_side_320_blank_annotation.csv"
QUESTION_TYPE_OUT = OUT_DIR / "question_only_320_question_type_seed.csv"


IDENTITY_OBJECT_HINTS = {
    "animal",
    "animals",
    "bird",
    "birds",
    "bear",
    "bike",
    "bicycle",
    "bus",
    "car",
    "cheese",
    "dog",
    "engine",
    "flower",
    "flowers",
    "meat",
    "mountain",
    "mountains",
    "pizza",
    "train",
    "truck",
    "vehicle",
    "wood",
}


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def classify_question(question: str) -> tuple[str, str, str]:
    q = normalize(question)

    weak_patterns = [
        "what do you call the person who",
        "what do you call a person who",
        "what do you call someone who",
        "what do you call people who",
        "what is the person called who",
    ]
    if contains_any(q, weak_patterns):
        return ("weak_image_prior", "high", "generic_role_name")

    if contains_any(
        q,
        [
            "who invented",
            "what degree do you need",
            "what are these products a good source of",
            "what is a common predator of",
            "what are these animals natives of",
            "what two continents",
            "what ingredient is used",
            "which type of wood is used",
            "what is this plate made from",
            "what does this device generally do",
            "what color are these devices usually painted",
            "when was",
            "what is used to make",
            "what material",
            "what is this made from",
            "what is this made of",
            "what is used for",
            "what do you use this for",
        ],
    ):
        return ("entity_fact_lookup", "high", "entity_fact_pattern")

    if contains_any(
        q,
        [
            "how do we know",
            "what time of day",
            "what holiday",
            "what celebration",
            "what are they celebrating",
            "what would you find",
            "what type of activity might",
            "what might this animal like to eat",
            "could be in",
            "could be on",
            "could be under",
            "could be over",
            "could you travel",
            "how fast could",
            "how often should",
            "home bathroom or hotel bathroom",
            "for males or females",
            "probably",
            "likely",
            "where were these vegetables grown",
            "where were they grown",
            "why ",
            "why?",
        ],
    ):
        return ("situation_inference", "high", "commonsense_inference_pattern")

    if q.startswith("what city is shown") or q.startswith("what mountains are those"):
        return ("identity_or_subtype_recognition", "medium", "place_or_named_entity_id")

    if q.startswith("name the model of"):
        return ("identity_or_subtype_recognition", "high", "model_identification")

    if q.startswith("what kind of") or q.startswith("what breed of") or q.startswith("what species of"):
        return ("identity_or_subtype_recognition", "high", "kind_breed_species")

    if q.startswith("what type of"):
        if contains_any(
            q,
            [
                "type of drink",
                "type of activity",
                "type of degree",
                "type of restaurant",
                "type of product is being advertised",
            ],
        ):
            if "type of degree" in q:
                return ("entity_fact_lookup", "high", "degree_fact")
            if "type of drink" in q or "type of activity" in q or "type of restaurant" in q:
                return ("situation_inference", "medium", "type_with_contextual_inference")
            return ("direct_visual", "medium", "type_direct_visible_text_or_scene")

        for hint in IDENTITY_OBJECT_HINTS:
            if f"type of {hint}" in q:
                return ("identity_or_subtype_recognition", "high", "typed_object_identification")
        return ("direct_visual", "low", "generic_type_default")

    if q.startswith("which type of"):
        if "wood is used" in q:
            return ("entity_fact_lookup", "high", "material_fact")
        return ("identity_or_subtype_recognition", "medium", "which_type_default")

    if q.startswith("name the ") or q.startswith("name this "):
        return ("identity_or_subtype_recognition", "medium", "name_identification")

    if contains_any(
        q,
        [
            "what is the name of",
            "what brand is",
            "which brand of",
            "what dog breed",
            "what game system",
            "what language is written",
            "which company",
            "what place is this",
            "name the place shown",
            "what country are these dishes from",
            "what type of phone",
            "what kind of kites",
        ],
    ):
        return ("identity_or_subtype_recognition", "medium", "name_brand_place_identity")

    if contains_any(
        q,
        [
            "what sport",
            "what are they playing",
            "what is the main color",
            "what color",
            "what is the light source",
            "what is the animal name mentioned on",
            "what electronic devices are pictured",
            "what kind of bird is that",
            "what kind of bear",
            "what type of train is this",
            "what type of bike is this",
            "what type of meat is pictured",
            "what type of cheese is on the pizza",
        ],
    ):
        if contains_any(
            q,
            [
                "kind of bird",
                "kind of bear",
                "type of train",
                "type of bike",
                "type of meat",
                "type of cheese",
            ],
        ):
            return ("identity_or_subtype_recognition", "high", "specific_subtype_recognition")
        return ("direct_visual", "high", "direct_visual_pattern")

    if contains_any(
        q,
        [
            "how do you clean this",
            "how is this type of",
            "what is the process for",
            "what material is used",
            "which material is used",
            "what is the red item used for",
            "what are these items usually used by",
            "what are the tires made of",
            "what is the purpose of",
            "what does this animal eat",
            "what does the diet of this animal consist of",
            "how long does this animal usually live",
            "how many stomachs does this animal have",
            "how many calories",
            "what vitamin",
            "what is the plate made out of",
            "who should eat this",
            "what part of the meal",
            "what is a female of this animal called",
            "what small appliance",
            "what form of communication can be sent using",
            "what is the colorful object used for",
            "what is the triangular green colored device used for",
            "what is spewing out water",
        ],
    ):
        return ("entity_fact_lookup", "medium", "procedure_function_material_fact")

    if contains_any(
        q,
        [
            "what activity is taking place",
            "what happens here",
            "what would you do",
            "what type of vessel would remain in this environment",
            "what type of place is this train in",
            "what time of year is it",
            "what is the outside temperature",
            "what is the temperature like here",
            "what is the light indicating oncoming traffic should be doing",
            "which age group visit this kind of place",
            "what is the handedness",
            "if this is a college student",
            "would this typically transport people or grains",
            "what is the man doing with his phone",
            "what is this boy on",
            "should you leave the toilet seat up",
            "who is playing this sport",
            "what is the job of the man in black",
            "what flavor is the cake",
            "what in this picture is most out of place",
        ],
    ):
        return ("situation_inference", "medium", "event_state_or_social_inference")

    if q.startswith("is ") or q.startswith("are ") or q.startswith("can ") or q.startswith("does "):
        return ("direct_visual", "medium", "visible_yes_no_or_state")

    if q.startswith("who ") or q.startswith("where "):
        if contains_any(q, ["who uses", "who invented", "where were", "where was"]):
            return ("entity_fact_lookup", "medium", "who_where_external_fact")
        return ("direct_visual", "low", "generic_wh_fallback")

    if q.startswith("how "):
        if contains_any(q, ["how fast", "how often", "how do we know", "how can we"]):
            return ("situation_inference", "medium", "how_inference")
        return ("direct_visual", "low", "generic_how_fallback")

    return ("direct_visual", "low", "fallback_direct_visual")


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows()

    questions_only_rows: list[dict[str, str]] = []
    questions_blank_rows: list[dict[str, str]] = []
    image_blank_rows: list[dict[str, str]] = []
    typed_rows: list[dict[str, str]] = []

    for row in rows:
        question = row["display_question"].strip()
        q_type, confidence, rule = classify_question(question)

        base = {
            "item_id": row["item_id"],
            "sample_id": row["sample_id"],
            "priority": row["priority"],
            "question_text": question,
        }
        questions_only_rows.append(base)
        questions_blank_rows.append(
            {
                **base,
                "question_type": "",
                "annotator_notes": "",
            }
        )
        image_blank_rows.append(
            {
                "item_id": row["item_id"],
                "sample_id": row["sample_id"],
                "priority": row["priority"],
                "image_filename": row["image_filename"],
                "question_text": question,
                "visual_structure": "",
                "image_dependence": "",
                "annotator_notes": "",
            }
        )
        typed_rows.append(
            {
                **base,
                "question_type_seed": q_type,
                "seed_confidence": confidence,
                "seed_rule": rule,
                "human_question_type": "",
                "human_notes": "",
            }
        )

    write_csv(
        QUESTIONS_ONLY_OUT,
        questions_only_rows,
        ["item_id", "sample_id", "priority", "question_text"],
    )
    write_csv(
        QUESTIONS_BLANK_OUT,
        questions_blank_rows,
        ["item_id", "sample_id", "priority", "question_text", "question_type", "annotator_notes"],
    )
    write_csv(
        IMAGE_BLANK_OUT,
        image_blank_rows,
        [
            "item_id",
            "sample_id",
            "priority",
            "image_filename",
            "question_text",
            "visual_structure",
            "image_dependence",
            "annotator_notes",
        ],
    )
    write_csv(
        QUESTION_TYPE_OUT,
        typed_rows,
        [
            "item_id",
            "sample_id",
            "priority",
            "question_text",
            "question_type_seed",
            "seed_confidence",
            "seed_rule",
            "human_question_type",
            "human_notes",
        ],
    )

    print(f"[done] questions_only={QUESTIONS_ONLY_OUT}")
    print(f"[done] questions_blank={QUESTIONS_BLANK_OUT}")
    print(f"[done] image_blank={IMAGE_BLANK_OUT}")
    print(f"[done] question_type_seed={QUESTION_TYPE_OUT}")


if __name__ == "__main__":
    main()
