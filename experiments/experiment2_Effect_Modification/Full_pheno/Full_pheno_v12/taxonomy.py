"""Prespecified groupings for the registered V11 diet and phenotype catalogue."""

from collections import Counter, defaultdict
from statistics import median


GROUPS = {
    "nutrient_density": ("carbohydrate_g", "protein_g", "lipid_g", "sodium_mg", "alcohol_g"),
    "fiber_and_sources": ("fiber_density", "fiber_grain", "fiber_legume", "fiber_produce",
                          "fiber_excess", "high_fiber_day", "fiber_day_sd"),
    "plant_foods": ("fruit_veg", "whole_grain", "pulses", "nuts_seeds", "bread_grains_other",
                    "category:bread", "category:bread wholewheat", "category:canned veg and fruits",
                    "category:cereals", "category:fruits", "category:pasta, grains and side dishes",
                    "category:pasta, grains and side dishes wholewheat", "category:vegetables"),
    "animal_foods": ("fish", "poultry", "red_meat", "processed_meat", "dairy", "eggs",
                     "category:hard cheese", "category:milk, cream cheese and yogurts",
                     "category:processed meat products"),
    "drinks_snacks_other": ("fat_oil", "sweets_snacks", "sweet_drinks", "diet_drinks", "alcoholic_drinks",
                            "fast_prepared", "soups_sauces", "other_category", "category:baked goods",
                            "category:deep fried foods", "category:drinks", "category:fast foods",
                            "category:industrialized vegetarian food ready to eat", "category:med oil and fats",
                            "category:oils and fats", "category:others", "category:snacks",
                            "category:spices and herbs", "category:sweet milk products", "category:sweets"),
    "processing": ("UPF", "nova_1", "nova_2", "nova_3", "nova_4_snack", "nova_4_other"),
    "diet_pattern_meal_context": ("upf_lower_sd", "log_energy_sd", "validated_span_hours",
                                  "restaurant_share", "food_group_energy_diversity"),
}
RECORDING = ("nova_unknown", "unknown_nova_sd", "items_logged")
DIET_CLASS = {name: group for group,names in GROUPS.items() for name in names} | dict.fromkeys(RECORDING,"recording")


def diet_key(name):
    """Resolve cosmetic separators in stored food-category labels."""
    prefix, separator, category=name.partition(":")
    return prefix+separator+" ".join(category.replace("_"," ").split()) if prefix=="category" else name


def map_diets(names):
    keys={name:diet_key(name) for name in names}
    if len(set(keys.values()))!=len(keys) or set(keys.values())!=DIET_CLASS.keys():
        raise ValueError("V12 diet taxonomy does not cover each registered measure exactly once")
    return {name:DIET_CLASS[key] for name,key in keys.items()}


def end_group(name, source):
    if (name == "D_Glu" or name.startswith("cgm_") or source in {"cgm_indices", "glycemic"}
            or any(x in name for x in ("__glucose_", "__hba1c_"))):
        return "glycemia"
    if name.startswith("D_Liver_") or source in {"liver_ultrasound", "liver_indices"} or any(
            f"__{k}_" in name for k in ("ggt", "alt", "ast", "alkaline_phosphatase", "albumin",
                                         "direct_bilirubin", "total_bilirubin")):
        return "liver"
    if name in {"D_TotalFat","D_VAT"} or source in {"anthropometrics","body_composition","adiposity"}:
        return "body_composition"
    if source in {"blood_pressure", "ecg", "carotid", "retinal_vascular"} or name.startswith("seated_"):
        return "cardiovascular"
    if source in {"sleep","sleep_hrv"}:
        return "sleep"
    if source in {"bone_density","grip"}:
        return "bone_and_strength"
    if source=="renal" or "__creatinine_" in name:
        return "renal"
    if source in {"nightingale", "serum_lipidomics", "lipid_profile"} or any(
            x in name for x in ("cholesterol", "triglycerides")):
        return "circulating_lipids_metabolites"
    return "other_blood"


def stability(group, readouts):
    """Retain both directions and diet-main versus interaction attribution."""
    n=len(group)
    if not n:
        raise ValueError("Empty eligible group")
    rows=[tuple(p[key] for p in readouts) for key in group]
    return dict(
        pairs=n,
        main_positive=sum(all(r["diet_main_pp"]>0. for r in pair) for pair in rows),
        main_negative=sum(all(r["diet_main_pp"]<0. for r in pair) for pair in rows),
        interaction_positive=sum(all(r["interaction_pp"]>0. and r["positive_folds"]>=3 for r in pair) for pair in rows),
        interaction_negative=sum(all(r["interaction_pp"]<0. for r in pair) for pair in rows),
        interaction_loss=sum(all(r["interaction_pp"]<=-.1 for r in pair) for pair in rows),
        interaction_reached=sum(all(r["interaction_pp"]>=.1 and r["positive_folds"]>=3 for r in pair) for pair in rows),
        total_positive=sum(all(r["diet_total_pp"]>0. for r in pair) for pair in rows),
        total_negative=sum(all(r["diet_total_pp"]<0. for r in pair) for pair in rows),
        joint=sum(all(r["diet_total_pp"]>0. and r["interaction_pp"]>=.1 and r["positive_folds"]>=3
                      for r in pair) for pair in rows),
        adjusted=sum(all(r["p_holm"]<.05 for r in pair) for pair in rows),
        median_main_by_split=[median(r[i]["diet_main_pp"] for r in rows) for i in range(len(readouts))],
        median_interaction_by_split=[median(r[i]["interaction_pp"] for r in rows) for i in range(len(readouts))],
        median_total_by_split=[median(r[i]["diet_total_pp"] for r in rows) for i in range(len(readouts))],
    )


def summarize(partitions, phenotype_classes, dietary_classes):
    readouts=[]
    for partition in partitions:
        rows=partition["rows"]
        by_key={(r["diet"],r["endpoint"]):r for r in rows}
        if len(by_key)!=len(rows):
            raise ValueError("Duplicate diet-endpoint score")
        readouts.append(by_key)
    shared=set.intersection(*(set(r) for r in readouts))
    groups=defaultdict(set)
    domains=defaultdict(set)
    for key in shared:
        diet, e=key
        family=dietary_classes[diet]
        groups[family].add(key)
        domains[family,phenotype_classes[e]].add(key)
    overall={k:stability(group,readouts) for k,group in sorted(groups.items())}
    stratified=defaultdict(dict)
    for (d,e),keys in sorted(domains.items()):
        stratified[d][e]=stability(keys,readouts)
    for d,stats in overall.items():
        layers=[x for x in stratified[d].values() if x["pairs"]>=5]
        stats["balanced_domains"]=len(layers)
        stats["domain_balanced_joint_rate"]=sum(
            x["joint"]/x["pairs"] for x in layers)/len(layers) if layers else None
    return dict(shared_pairs=len(shared), categories=overall, by_outcome_domain=dict(stratified),
                phenotype_domains=dict(Counter(phenotype_classes.values())))
