"""Build outcome-free dietary blocks from the registered food inventory."""

import numpy as np
import pandas as pd

from food_features import DAY, KEY, FOOD_GROUPS, SNACK_DRINK, SOURCES, normalise, projected
from feature_policies import BLOCKS
from run_v2 import require


NUTRIENTS = ("carbohydrate_g", "protein_g", "lipid_g", "sodium_mg", "alcohol_g")
FOOD_TYPES = {
    "fruit_veg": {"fruits", "vegetables", "canned veg and fruits"},
    "whole_grain": FOOD_GROUPS["grain"],
    "pulses": FOOD_GROUPS["legume"],
    "nuts_seeds": {"nuts, seeds, and products"},
    "bread_grains_other": {"bread", "pasta, grains and side dishes", "cereals"},
    "fish": {"fish and seafood"},
    "poultry": {"poultry and its products"},
    "red_meat": {"beef, veal, lamb, and other meat products"},
    "processed_meat": {"processed meat products"},
    "dairy": {"milk, cream cheese and yogurts", "hard cheese"},
    "eggs": {"eggs and their products"},
    "fat_oil": {"med oil and fats", "oils and fats"},
    "sweets_snacks": {"sweets", "snacks", "baked goods", "sweet milk products"},
    "sweet_drinks": {"fruit juices and soft drinks"},
    "diet_drinks": {"low calories and diet drinks"},
    "alcoholic_drinks": {"alcoholic drinks"},
    "fast_prepared": {"fast foods", "industrialized vegetarian food ready to eat", "deep fried foods"},
    "soups_sauces": {"soups and sauces"},
}
def consume(frame, columns, count, n, op="mean"):
    """Each column has its own valid-day count and participant support."""
    x = frame.groupby("input_row")[list(columns)]
    valid = x.count().reindex(range(n),fill_value=0).to_numpy()
    aggregate = (x.std() if op=="sd" else x.mean()).reindex(range(n)).to_numpy(float)
    aggregate[(valid<3) | (valid<.50*count[:,None])] = np.nan
    return aggregate


def divide_frame(frame, cols, denom, scale=1.):
    for k in cols:
        frame[k] = (scale*frame[k]/frame[denom]).where(frame[k].ge(0.) & frame[denom].gt(0.))


def read_inventory(root, study):
    paths = {k:(root/v).resolve(strict=True) for k,v in SOURCES.items()}
    index = pd.DataFrame(dict(zip(KEY,study.keys,strict=True)))
    n = len(index)
    index["input_row"] = np.arange(n)
    ids = index.participant_id.astype("int64").unique().tolist()
    log = normalise(projected(paths["day"],DAY+["energy_eligible","fiber_eligible","energy_kcal","fiber_g"],ids))
    log = log.merge(index,on=KEY,how="inner",validate="many_to_one")
    require(not log.duplicated(DAY).any(),"V9_DUPLICATE_DAY")
    energy = log.loc[log.energy_eligible.eq(True)&log.energy_kcal.gt(0)].copy()
    fiber = energy.loc[energy.fiber_eligible.eq(True)&energy.fiber_g.ge(0)].copy()
    ec,fc = study.numeric["n_energy_days"],study.numeric["n_fiber_days"]
    for frame, observed in ((energy,ec),(fiber,fc)):
        nday = frame.groupby("input_row").size().reindex(range(n),fill_value=0).to_numpy()
        use = observed>=3
        require(np.array_equal(nday[use],observed[use]),"V9_REGISTERED_DAY_COUNT")

    nutrients = [name+"_complete_sum" for name in NUTRIENTS]
    clock = ["recorded_span_hours","meal_ingestion_time_validated","n_events"]
    qc = normalise(projected(paths["qc"],DAY+nutrients+clock,ids))
    daily = energy.merge(qc,on=DAY,validate="one_to_one")
    divide_frame(daily,nutrients,"energy_kcal",scale=1000.)
    daily["validated_span_hours"] = daily.recorded_span_hours.where(
        daily.meal_ingestion_time_validated.eq(True)&daily.recorded_span_hours.between(0.,24.))
    daily["items_logged"] = daily.n_events.where(daily.n_events.ge(0.))

    event_cols = ["food_id","food_category","qc_primary_eligible","calories_kcal_clean","dietary_fiber_g_clean","eaten_in_restaurant"]
    ev = normalise(projected(paths["events"],DAY+event_cols,ids))
    ev = ev.loc[ev.qc_primary_eligible.eq(True)].merge(energy[DAY],on=DAY,validate="many_to_one")
    ev["category"] = ev.food_category.astype("string").fillna("").str.lower().str.strip()
    kcal = ev.calories_kcal_clean.where(ev.calories_kcal_clean.ge(0.),0.)
    ev["items_kcal"] = kcal
    known_categories = set().union(*FOOD_TYPES.values())
    for name,tags in FOOD_TYPES.items():
        ev[name] = kcal.where(ev.category.isin(tags),0.)
    ev["other_category"] = kcal.where(~ev.category.isin(known_categories),0.)
    typecols = [*FOOD_TYPES,"other_category"]
    for name,tags in FOOD_GROUPS.items():
        ev["fiber_"+name] = ev.dietary_fiber_g_clean.where(ev.category.isin(tags),0.)
    ev["fiber_sum"] = ev.dietary_fiber_g_clean
    ff = ["fiber_"+key for key in FOOD_GROUPS]

    cat = projected(paths["catalog"],["food_id","NOVA"]).dropna(subset=["food_id"])
    cat["id"] = cat.food_id.astype("string")
    require(not cat.id.duplicated().any(),"V9_CATALOGUE_DUPLICATES")
    nova = ev.food_id.astype("string").map(pd.to_numeric(cat.set_index("id").NOVA,errors="coerce"))
    proc = ["nova_1","nova_2","nova_3","nova_4_snack","nova_4_other","nova_unknown"]
    for num in (1,2,3):
        ev[f"nova_{num}"] = kcal.where(nova.eq(num),0.)
    snack = ev.category.isin(SNACK_DRINK)
    ev["nova_4_snack"] = kcal.where(nova.eq(4)&snack,0.)
    ev["nova_4_other"] = kcal.where(nova.eq(4)&~snack,0.)
    ev["nova_unknown"] = kcal.where(~nova.isin([1,2,3,4]),0.)
    labels = ev.eaten_in_restaurant.astype("string").str.strip().str.lower()
    yes,no = labels.isin(("true","1","1.0","yes","y")),labels.isin(("false","0","0.0","no","n"))
    ev["restaurant_kcal"] = kcal.where(yes,0.)
    ev["restaurant_labelled_kcal"] = kcal.where(yes|no,0.)
    grouped = ev.groupby(DAY)[["items_kcal",*typecols,*ff,"fiber_sum",*proc,"restaurant_kcal","restaurant_labelled_kcal"]].sum(min_count=1)

    amounts = energy.merge(grouped.reset_index(),on=DAY,how="inner",validate="one_to_one")
    amounts = amounts.loc[np.isclose(amounts.items_kcal,amounts.energy_kcal,rtol=.005,atol=.1)].copy()
    amounts["restaurant_share"] = (amounts.restaurant_kcal/amounts.energy_kcal).where(
        amounts.restaurant_labelled_kcal>=.80*amounts.energy_kcal)
    divide_frame(amounts,typecols,"energy_kcal")
    daynov = normalise(projected(paths["nova_day"],DAY+["known_UPF_kcal","unknown_NOVA_kcal"],ids))
    upf = amounts.merge(daynov,on=DAY,validate="one_to_one")
    upf = upf.loc[np.isclose(upf.nova_4_snack+upf.nova_4_other,upf.known_UPF_kcal,rtol=.005,atol=.1)
                  &np.isclose(upf.nova_unknown,upf.unknown_NOVA_kcal,rtol=.005,atol=.1)].copy()
    divide_frame(upf,proc,"energy_kcal")
    upf["upf_lower"] = upf.nova_4_snack+upf.nova_4_other

    fd = fiber.merge(grouped.reset_index()[DAY+["fiber_sum",*ff]],on=DAY,how="inner",validate="one_to_one")
    fd = fd.loc[np.isclose(fd.fiber_g,fd.fiber_sum,rtol=.01,atol=.05)].copy()
    divide_frame(fd,ff,"energy_kcal",scale=1000.)
    fiber["fiber_density_daily"] = 1000.*fiber.fiber_g/fiber.energy_kcal
    fiber["fiber_excess"] = np.maximum(fiber.fiber_density_daily-12.5,0.)
    fiber["high_fiber_day"] = (fiber.fiber_density_daily>=12.5).astype(float)
    amounts["log_energy"] = np.log(amounts.energy_kcal)
    patterns = [consume(fiber,["fiber_excess","high_fiber_day"],fc,n),
                consume(fiber,["fiber_density_daily"],fc,n,op="sd"),
                consume(upf,["upf_lower","nova_unknown"],ec,n,op="sd"),
                consume(amounts,["log_energy"],ec,n,op="sd")]
    contextcols = ["validated_span_hours","items_logged","restaurant_share"]
    context = np.column_stack((consume(daily,contextcols[:2],ec,n),consume(amounts,contextcols[2:],ec,n)))
    values = dict(nutrients=consume(daily,nutrients,ec,n),fiber_sources=consume(fd,ff,fc,n),
                  food_groups=consume(amounts,typecols,ec,n),processing=consume(upf,proc,ec,n),
                  day_patterns=np.column_stack(patterns),meal_context=context)
    feature_names = dict(nutrients=list(NUTRIENTS),fiber_sources=ff,food_groups=typecols,processing=proc,
                         day_patterns=["fiber_excess","high_fiber_day","fiber_day_sd","upf_lower_sd","unknown_nova_sd","log_energy_sd"],
                         meal_context=contextcols)
    audit = {key:[dict(feature=name, observed=int(np.isfinite(x).sum()), nonzero=int((np.isfinite(x)&(x!=0.)).sum()))
                  for name,x in zip(feature_names[key],array.T,strict=True)] for key,array in values.items()}
    audit["matched_days"] = {"energy":len(energy),"event_energy":len(amounts),"nova":len(upf),"fiber":len(fd)}
    return values,audit,paths


def encode_blocks(values, training):
    """Use training support and quantiles, with missing features neutral after centering."""
    panels, support = {}, {}
    for name in BLOCKS:
        kept = []
        for x in values[name].T:
            valid = training & np.isfinite(x)
            if valid.sum()<max(30,.15*training.sum()) or not np.any(np.abs(x[valid])>1e-12):
                continue
            lo,hi = np.quantile(x[valid],[.01,.99])
            z = np.clip(x,lo,hi)
            center, sd = float(z[valid].mean()), float(z[valid].std())
            if sd>1e-10:
                kept.append(np.where(np.isfinite(z),(z-center)/sd,0.))
        support[name] = len(kept)
        panels[name] = np.column_stack(kept)/np.sqrt(len(kept)) if kept else np.zeros((len(training),0))
    panels["all"] = np.column_stack([panels[k] for k in BLOCKS])
    return panels,support
