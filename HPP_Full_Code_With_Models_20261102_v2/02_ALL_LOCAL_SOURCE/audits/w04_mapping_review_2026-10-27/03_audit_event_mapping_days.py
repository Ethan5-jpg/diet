# -*- coding: utf-8 -*-
"""TRE-only full processed-event mapping diagnostic; no outcomes or final features."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json, math, time, resource, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
P=Path('/home/ec2-user/studies/diet_gut_cgm_v2/multisystem_project_20261021')
R=Path('/home/ec2-user/studies/hpp_datasets/diet_logging')
Q=P/'data_qc/w04_mapping_review_20261027'
CATALOG=[['Chocolate Cookies', 4], ['Honey', 2], ['Cappuccino', 3], ['Cottage cheese', 3], ['Tomatoes', 1], ['Oatmeal Cereal', 3], ['Apple', 1], ['Banana', 1], ['Peach', 1], ['Rye Bread', 4], ['Hummus Salad', 2], ['Goat Milk Yogurt', 3], ['Mayonnaise', 4], ['Oatmeal', 2], ['Sheep Milk Yoghurt', 1], ['Milk', 1], ['Coffee', 3], ['Egg', 1], ['Beef tongue', 3], ['Ice cream', 4], ['Yogurt', 1], ['Cooked beets', 3], ['Dates', 1], ['Cookies', 3], ['Nectarine', 1], ['Halva', 3], ['Dulce de Leche', 4], ['Mango', 1], ['Mozzarella Cheese', 3], ['Butter', 2], ['Carrots', 1], ['Zucchini', 1], ['Greek Yogurt', 1], ['Cucumber', 1], ['Veal Meatballs', 3], ['Brownies', 4], ['Jam', 3], ['Chocolate', 4], ['Canned Tuna Fish', 3], ['Lettuce', 1], ['Omelette', 1], ['Pea soup', 1], ['Potatoes', 2], ['Chicken drumstick', 3], ['Majadra', 3], ['Flavored Yogurt', 4], ['Vegetable Salad', 1], ['Nuts', 1], ['Dried Herring', 3], ['emmer wheat bread', 4], ['White Cheese', 1], ['Pita', 3], ['Pastrami', 4], ['Butter Cookies', 3], ['Yellow Cheese Sandwich', 4], ['Beer', 3], ['Natural Yogurt', 2], ['Beet', 1], ['Blue Cheese', 3], ['Pepper', 1], ['Ricotta Cheese', 1], ['Pickled Herring', 3], ['Rugelach', 4], ['Potato pancakes', 3], ['Mushrooms', 2], ['Potato salad', 3], ['Coke', 4], ['Bissli', 4], ['Almonds', 1], ['Tahini', 2], ['Beef', 2], ['Pistachio', 1], ['Hazelnuts', 1], ['Energy bar', 4], ['Bread', 3], ['Walnuts', 1], ['Cold cut', 4], ['Tea', 2], ['Chicken breast', 3], ['Schnitzel', 3], ['Grapes', 1], ['Pizza', 3], ['Salty Cookies', 4], ['Pretzels', 4], ['Salty Cheese', 3], ['Pecans', 2], ['Mandarin', 1], ['Broccoli', 1], ['Grits', 3], ['Stuffed Vine leaves', 3], ['White beans', 2], ['Ceviche', 3], ['Seabass Fish', 3], ['Persimmon', 1], ['Salmon', 3], ['Beans black-eyed peas', 1], ['Roll', 3], ['Pullet', 3], ['Hamburger', 3], ['Ketchup', 4], ['Danish', 4], ['Greek salad', 3], ['Wholemeal Bread', 4], ['Vegetable Soup', 1], ['Sausages', 4], ['Dark Chocolate', 4], ['Tzfatit Cheese', 3], ['Wine', 3], ['Avocado', 1], ['Pomegranate', 1], ['Cuba', 3], ['Ravioli', 3], ['Parmesan cheese', 3], ['French fries', 3], ['Pecan', 1], ['Cooked green beans', 1], ['Corn casserole', 4], ['Rice crackers', 4], ['Coleslaw', 2], ['Peas', 1], ['Protein bar', 4], ['Diet Cheesecake', 4], ['Cauliflower Pie', 3], ['Fruit salad', 1], ['Sweetened White Cheese', 4], ['Chocolate spread', 4], ['Popcorn', 4], ['Pesek Zman snack', 4], ['Soy Pudding', 4], ['Yellow Cheese', 3], ['Quinoa', 1], ['Beans', 1], ['Portobello mushroom', 1], ['Parsley', 1], ['Green cabbage', 1], ['Pasta With Cheese', 4], ['Wholemeal Pita', 4], ['Gin and tonic', 3], ['Chicken Steak', 3], ['Gumdrops', 4], ['Knafeh', 3], ['Chicken legs', 3], ['Rice', 1], ['English cake', 3], ['Couscous', 3], ['Gilthead Bream', 3], ['Eggplant', 1], ['Chocolate cake', 4], ['Tuna Sandwich', 4], ['Wholemeal Crackers', 4], ['Cashew', 1], ['Kohlrabi', 1], ['Smoothies', 4], ['Sweet potato', 1], ['Acorn squash', 1], ['Tofu', 2], ['Garlic', 1], ['Cooked cabbage', 1], ['Cooked tomatoes', 1], ['Granola', 4], ['Sprouted lentils', 1], ['Watermelon', 1], ['Melon', 1], ['Coconut milk', 2], ['Cherry tomatoes', 1], ['Seaweeds', 1], ['Light Yellow Cheese', 3], ['Falafel', 3], ['Pasta Bolognese', 3], ['Olive oil', 2], ['dried tomato spread', 3], ['Sweet Potato Soup', 3], ['Pickled cucumber', 3], ['Chicken Liver', 3], ['Onion', 1], ['Grilled vegetables', 1], ['Orange', 1], ['Vegetable Pie', 3], ['Shakshouka', 3], ['Bean soup', 1], ['Plum', 1], ['Matzo', 4], ['Tilapia', 2], ['Omelet sandwich', 4], ['Peanuts', 1], ['Potato pie', 3], ['Cooked Sweet potato', 1], ['Carrot Juice', 3], ['Turkey Shawarma', 3], ['Pear', 1], ['Burekas', 4], ['Chicken Meatballs', 3], ['Baguette', 4], ['Flavored Waters', 4], ['Egg Sandwich', 3], ['Wholemeal Roll', 4], ['Cheese Pie', 3], ['Grape juice', 4], ['Onion Pie', 3], ['Orange Cake', 4], ['Mixed Vegetables', 2], ['Pasta with tomato sauce', 3], ['Pate', 3], ['Gluten free bread', 4], ['Whiskey', 3], ['Hamantaschen', 4], ['Hummus', 1], ['White cabbage salad', 3], ['Fried eggplant', 3], ['Turkey and vegetables', 3], ['Cereal Bar', 4], ['Spelled', 1], ['Steak', 3], ['Turkey', 3], ['Chicken', 3], ['Cooked vegetables', 2], ['Egg Salad', 3], ['Pumpkin soup', 1], ['Green Tea', 3], ['Olive Bread', 3], ['Beef Cholent', 3], ['Chia seeds', 1], ['Wafers', 4], ['Emmental cheese', 3], ['Macadamia nuts', 1], ['Coconut water', 3], ['Noodles', 3], ['Blueberries', 1], ['Flax seeds', 1], ['Pancake', 3], ['Light Cream Cheese', 3], ['Fried onions', 3], ['Cracker', 4], ['Almond Beverage', 3], ['Popsicle', 4], ['Goji berry', 1], ['Honey Cake', 4], ['Sugar', 4], ['Shrimp', 3], ['Onion Rings', 4], ['Tortilla', 3], ['Carrot Cake', 3], ['Chopped Liver', 3], ['Stuffed Cabbage', 3], ['Cheese Pastry', 4], ['Whipped cream', 2], ['Heavy cream', 2], ['Pumpkin seeds', 1], ['Sunflower seeds', 1], ['Strawberry', 1], ['Mascarpone Cheese', 3], ['Brandy', 4], ['Gouda Cheese', 3], ['Olives', 3], ['Mint', 1], ['Broth', 4], ['Roasted eggplant', 1], ['Halloumi Cheese', 3], ['Mousse Cake', 3], ['Lettuce Salad', 1], ['Cauliflower', 1], ['Almond flour', 1], ['Vodka or Arak', 3], ['Berries', 1], ['Spinach', 1], ['Camembert or Brie', 3], ['Chicken soup', 3], ['Biscuit', 4], ['Cheesecake', 3], ['Cooked zucchini', 1], ['Prune', 1], ['Bamba', 4], ['Soda water', 3], ['Watermelon Seeds', 1], ['Zaatar', 1], ['Lemon juice', 1], ['Sugar Free Gum', 4], ['Cereals', 4], ['Apricot', 2], ['Mixed nuts', 1], ['Fig', 1], ['Schug', 3], ['Red Cabbage Salad', 1], ['Chili pepper', 1], ['Risotto', 3], ['Tuna Salad', 4], ['Turkey breast', 3], ['Ice cream cone', 4], ['Leben', 1], ['Lentil Soup', 2], ['Apple Cake', 4], ['Pasta', 3], ['Oat Drink', 3], ['Pork', 2], ['Tabbouleh Salad', 3], ['Sabih Sandwich', 4], ['Croissant', 4], ['Chicken wings', 3], ['Jachnun', 4], ['Corn Soup', 1], ['Sweet Challah', 4], ['Green beans', 1], ['Green Salad', 1], ['Grouper', 3], ['Diet Fruit Drink', 4], ['Sugar substitute', 4], ['Lentils', 1], ['Sushi', 3], ['Diet Coke', 4], ['Bean Noodles', 3], ['Bulgur', 3], ['Stevia Sweetner', None], ['Bagel', 4], ['Chicken Pad Thai', 3], ['Olive spread', 4], ['Ciabatta', 3], ['Pesto', 4], ['Beet Salad', 1], ['Nut cake', 4], ['Eggplant Salad', 3], ['Labneh Cheese', 2], ['Pickled vegetables', 3], ['Fruit Drink', 4], ['Croutons', 4], ['Puff pastry', 4], ['Lentil Patties', 3], ['Matbucha', 3], ['Peanut Butter', 1], ['Edamame', 1], ['Small radish', 1], ['Soy Patties', 3], ['Orange juice', 4], ['Hummus flour', 1], ['Cooking cream', 2], ['Caesar Salad', 3], ['Chocolate balls', 3], ['Meatballs', 3], ['Yellow beans', 1], ['Cookies Oreo', 4], ['Tomatoes Salad', 2], ['Kebab', 3], ['Red pepper', 1], ['Arugula', 1], ['Coriander', 1], ['Low fat Milk', 1], ['Oil', 2], ['Celery', 1], ['Cream Cheese', 3], ['Brazil nuts', 1], ['Jerusalem mixed grill', 3], ['Apple Vinegar', 4], ['Spirulina', None], ['Sprouts', 1], ['Pine nuts', 1], ['Fennel', 1], ['Cinnamon', 1], ['Coconut', 1], ['Coconut Oil', 2], ['Maple syrup', 3], ['Vegetable patties', 3], ['Boiled corn', 1], ['Cranberries', 1], ['Protein Powder', 4], ['Fruit juice', 1], ['ginger', 1], ['Lychee', 1], ['Smoked Salmon', 3], ['Ouzo', 3], ['Challah', 4], ['Crackers', 4], ['Soymilk', 3], ['Yellow pepper', 1], ['Kiwi', 1], ['Chestnuts', 1], ['Pomelo', 1], ['Spinach Quiche', 3], ['Beef Lasagna', 3], ['Meagre Fish', 3], ['Sheep milk Labaneh', 3], ['Tiramisu', 4], ['Sauerkraut', 3], ['Tofu Pad Thai', 3], ['Curry Paste', 4], ['kale', 1], ['Malt beverage', 4], ['Sweet wine', 3], ['Beef Shawarma', 3], ['Black beans', 1], ['Green onions', 1], ['Crushed tomatoes', 1], ['Mille-feuille', 4], ['Creme Brulee', 3], ['Krembo', 4], ['Light Soymilk', 3], ['Soy sauce', 4], ['Entrecote', 1], ['Alfalfa sprouts', 1], ['Smoked Mackerel Fish', 3], ['Date honey', 3], ['Tomato soup', 1], ['Chili sauce', 4], ['Tomato sauce', 3], ['Brown Sugar', 2], ['Maple Cake', 4], ['Raisins', 1], ['Meat Stuffed Burekas', 3], ['Lemonade', 4], ['Fennel Salad', 3], ['Grapefruit juice', 3], ['Mentos', 4], ['Banana Cake', 3], ['Teami waffle snack', 4], ['Dried tomatoes', 3], ['Soup mandels', 4], ['Dried Mango', 1], ['Buckwheat Bread', 4], ['Vegan Cheese', 2], ['Tomato paste', 3], ['Goat Cheese', 3], ['Fish Roe Salad', 3], ['Manchego Cheese', 3], ['Wholemeal Baguette', 4], ['Marzipan', 4], ['Anchovy', 3], ['Turmeric powder', 1], ['Cocoa Powder', 1], ['Stuffed Chard', 1], ['Dill', 1], ['Grapefruit', 1], ['Egg yolk', 1], ['Radish', 1], ['Mulberry', 1], ['Dried pineapple', 2], ['Squash casserole', 3], ['Corn', 1], ['Canned corn', 3], ['Mustard', 4], ['Mutton', 3], ['Mark Hariri', 3], ['Calamari', 1], ['Herring', 3], ['Guacamole', 3], ['Jewish donut', 4], ['Israeli couscous', 3], ['Light Bread', 4], ['Meat Ravioli', 4], ['Barbecue sauce', 4], ['Basil', 1], ['Cherries', 2], ['Bounty', 4], ['Teriyaki Sauce', 4], ['Apple juice', 2], ['Rice Noodles', 3], ['Click', 4], ['Sweet Roll', 4], ['Lemon', 1], ['Cooked eggplant', 3], ['Loquat', 1], ['Lasagna', 3], ['Onion soup', 3], ['Carrot Salad', 2], ['Mushroom Pie', 4], ['Gefilte fish', 3], ['Tirosh', 4], ['Wholemeal Light Bread', 4], ['Sambusak', 3], ['Blintzes', 3], ['Doritos', 4], ['Wild Rice', 1], ['Barley soup', 3], ['Chocolate chips', 4], ['Peas and carrots', 1], ['Granola Cookies', 3], ['Sherbet', 4], ['Candies', 4], ['Granola bar', 4], ['Veggie burger', 4], ['Rice drink', 3], ['Meringue Kisses', 4], ['Pomegranate juice', 1], ['Sweetened Cocoa Powder', 4], ['Minced meat', 3], ['Potato chips', 4], ['Cooked Okra', 2], ['Tomato soup and rice', 3], ['Fried cauliflower', 3], ['Fried Bread', 4], ['Roast beef', 3], ['Malabi', 4], ['Campari', 3], ['Dried apple', 2], ['Steamed vegetables', 1], ['Empanadas', 4], ['Fish schnitzel', 3], ['Pastels', 4], ['Chocolate Milk', 4], ['Applesauce', 3], ['Cheddar', 3], ['Balsamic vinegar', 2], ['Pumpkin', 1], ['Chocolate Mousse', 4], ['Asparagus', 1], ['Ice cream bar', 4], ['Coated peanuts', 3], ['Sesame seeds', 1], ['Buckwheat crackers', 4], ['Green pepper', 1], ['Spelt crackers', 4], ['Pineapple', 1], ['Roquefort', 3], ['Chard', 1], ['Salsa', 4], ['Dried Fruit', 1], ['White Chocolate', 4], ['Red beans', 1], ['Salt', 2], ['Sweet Potato Fries', 4], ['Cinnamon Danish', 4], ['Chinese orange', 1], ['Whipped Cream Pudding', 4], ['Guava', 1], ['Lemon Pie', 3], ['Prickly pears', 1], ['Mars', 4], ['Snickers', 4], ['Eshel', 1], ['Feta Cheese', 3], ['Brioche', 4], ['Tzatziki Cheese', 3], ['Tortit Bar', 4], ['Puffs', 4], ['Light Beer', 3], ['Dessert Wine', 3], ['Fish balls', 3], ['Artichoke', 1], ['Egg noodles', 3], ['Sauteed vegetables', 3], ['Baked apple', 1], ['Gnocchi', 4], ['Vinaigrette', 3], ['Frozen yogurt', 4], ['Noodle Pie', 3], ['Amba', 3], ['Canned Sardine', 3], ['Mushroom soup', 3], ['Broth and potatoes', 3], ['Cream Cake', 4], ['Carpaccio', 3], ['Bread Crumbs', 4], ['Cooked broccoli', 1], ['Nile perch', 1], ['Chicken Shawarma', 3], ['Brown Rice', 3], ['Lamb Kebabs', 3], ['Turkish Salad', 1], ['Orange Soup', 1], ['Dried cranberries', 3], ['Raspberries', 1], ['Bolognese', 3], ['Thousand Island dressing', 4], ['Kugel', 3], ['Tomatoes and onion', 1], ['Coated Wafers', 4], ['Corn schnitzel', 4], ['Palm', 3], ['Wheat', 3], ['Veal kebabs', 3], ['Rice paper', 3], ['Quinoa Salad', 3], ['black pepper', 1], ['Sesame oil', 2], ['Tahini Bread', 4], ['Tom Yum soup', 3], ['Fruit flavored Yogurt', 4], ['Knaidlach', 4], ['Sweetsop', 1], ['Zucchini Patties', 3], ['Miso soup', 3], ['Spongecake', 3], ['Almond spread', 1], ['Corn crackers', 3], ['Passionflower', 1], ['Shandy', 3], ['Sugar free candies', 4], ['Eel', 3], ['Canned tomatoes', 1], ['Dim Sum', 3], ['Goats Milk', 2], ['Milky', 4], ['Toblerone', 4], ['Leek', 1], ['Cake', 4], ['Dried fig', 3], ['Canola oil', 2], ['Polenta', 3], ['Baklava', 4], ['Cooked Pumpkin', 1], ['Vinegar', 2], ['Fava beans', 1], ['Lentils flour', 1], ['Kashkaval', 3], ['Chilli Con Carne', 3], ['Chocolate Biscuit', 4], ['Twix', 4], ['Cane sugar', 4], ['Tuna Fish', 1], ['Semolina Cake', 4], ['Meatballs and potatoes', 3], ['Cauliflower Patties', 3], ['Fried Sweet potato', 3], ['Minestrone Soup', 1], ['Garlic sauce', 4], ['Marshmallow', 4], ['Raspberries with sugar', 1], ['Aperol', 3], ['Lahoh', 3], ['Natural Yogurt Drinks', 3], ['Papaya', 1], ['Muffin', 4], ['Safflower oil', 2], ['Moussaka', 3], ['Coated Biscuits', 4], ['Avocado Sandwich', 4], ['Pickled Chili pepper', 3], ['Kif Kef', 4], ['Egg Roll', 3], ['Mekupelet', 4], ['Cottage and topping', 3], ['Apropo', 4], ['Flavored Soymilk', 4], ['Sweet Potato Pie', 3], ['Chimichurri', 4], ['Paprika', 1], ['Cocktail', 3], ['Soybean oil', 2], ['Dried celery', 1], ['Harissa salad', 4], ['Sangria', 3], ['Beet Juice', 4], ['Dried banana', 4], ['Salmon Sandwich', 4], ['Semi soft Cheese', 3], ['Gum', 4]]
norm=lambda x: str(x).strip().casefold() if pd.notna(x) else ''
mp={norm(n):v for n,v in CATALOG}
assert len(mp)==len(CATALOG)==643
assert all(v in [None,1,2,3,4] for v in mp.values())
# Small numerical checks: exact point and unknown-energy bounds.
assert (100*600/2000,100*(600+200)/2000)==(30.,40.)
assert np.mean([80.,10.])==45.
assert 100*(800+300)/(1000+3000)==27.5
keys=['participant_id','cohort','research_stage']
def fp(p):
 s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def read_batches(p,cols):
 f=pq.ParquetFile(p,pre_buffer=False)
 budget=sum(f.metadata.row_group(g).column(c).total_uncompressed_size for g in range(f.num_row_groups) for c in range(f.metadata.num_columns) if f.metadata.row_group(g).column(c).path_in_schema in cols)
 assert budget<1536*1024**2,('projection budget',budget)
 for b in f.iter_batches(batch_size=8192,columns=cols,use_threads=False): yield b.to_pandas(ignore_metadata=True)
def valid_keys(d):
 return d.participant_id.notna() & d.cohort.notna() & d.research_stage.notna() & d._date.notna() & ~d.cohort.astype(str).isin(['undefined','None','']) & ~d.research_stage.astype(str).isin(['undefined','None',''])
source=R/'diet_logging_events.parquet';index=R/'diet_logging.parquet'
fingerprints=[fp(p) for p in [source,index]]
idx=set();idxrows=0
for d in read_batches(index,keys+['collection_date']):
 idxrows+=len(d);d['_date']=pd.to_datetime(d.collection_date,errors='coerce').dt.strftime('%Y-%m-%d')
 for t in d.loc[valid_keys(d),keys+['_date']].itertuples(index=False,name=None):idx.add(t)
# Per-day vector: events, known energy, bad energy count, known fiber, bad fiber count,
# known UPF energy, unknown NOVA energy, unknown NOVA event count, matched-name events.
days={};foods={};scanned=0;badkey=0;start=time.monotonic();cats=defaultdict(lambda:[0,0.])
cols=keys+['collection_date','food_id','short_food_name','food_category','calories_kcal','dietary_fiber_g']
for d in read_batches(source,cols):
 scanned+=len(d)
 for fid,name,cat in d[['food_id','short_food_name','food_category']].drop_duplicates().itertuples(index=False,name=None):
  if pd.isna(fid):continue
  z=foods.setdefault(str(fid),{'names':set(),'categories':set()})
  if pd.notna(name):z['names'].add(str(name))
  if pd.notna(cat):z['categories'].add(str(cat))
 assert len(foods)<100000,'food catalog memory bound'
 d['_date']=pd.to_datetime(d.collection_date,errors='coerce').dt.strftime('%Y-%m-%d')
 good=valid_keys(d);badkey+=int((~good).sum());d=d.loc[good].copy()
 en=pd.to_numeric(d.calories_kcal,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
 fi=pd.to_numeric(d.dietary_fiber_g,errors='coerce').to_numpy(dtype=float,na_value=np.nan)
 eok=np.isfinite(en)&(en>=0);fok=np.isfinite(fi)&(fi>=0)
 names=d.short_food_name.map(norm);nova=names.map(mp);known=nova.isin([1,2,3,4]).to_numpy()
 d['_n']=1;d['_E']=np.where(eok,en,0);d['_badE']=(~eok).astype(int)
 d['_F']=np.where(fok,fi,0);d['_badF']=(~fok).astype(int)
 d['_K']=np.where(eok&(nova.to_numpy()==4),en,0)
 d['_U']=np.where(eok&~known,en,0);d['_unknown']=(~known).astype(int)
 d['_matched']=names.isin(mp).astype(int)
 fields=['_n','_E','_badE','_F','_badF','_K','_U','_unknown','_matched']
 g=d.groupby(keys+['_date'],sort=False,dropna=False)[fields].sum()
 for k,v in zip(g.index,g.to_numpy()):
  if k in days: days[k]+=v
  else: days[k]=v
 for cat,gc in d.groupby('food_category',dropna=False,sort=False):
  cats[str(cat)][0]+=len(gc);cats[str(cat)][1]+=float(gc._E.sum())
 if time.monotonic()-start>600:raise RuntimeError('Full event scan timeout')
 if scanned% (8192*50)==0:print('PROGRESS',scanned,flush=True)
assert scanned==pq.ParquetFile(source).metadata.num_rows
assert fingerprints==[fp(p) for p in [source,index]]
assert len(days)<250000
scopes={}
for scope in ['all_source','baseline_10k']:
 items=[(k,v) for k,v in days.items() if scope=='all_source' or (k[1]=='10k' and k[2]=='00_00_visit')]
 matched=[(k,v) for k,v in items if k in idx]
 energy=[(k,v) for k,v in matched if v[2]==0 and v[1]>0]
 novaok=[(k,v) for k,v in energy if v[7]==0]
 fiberok=[(k,v) for k,v in energy if v[4]==0]
 people=defaultdict(lambda:[0,0,0])
 for k,v in matched:people[k[:3]][0]+=1
 for k,v in novaok:people[k[:3]][1]+=1
 for k,v in fiberok:people[k[:3]][2]+=1
 widths=[float(100*v[6]/v[1]) for k,v in energy]
 totalE=sum(float(v[1]) for k,v in matched)
 totalU=sum(float(v[6]) for k,v in matched)
 counts={label:{'ge1_days':sum(v[j]>=1 for v in people.values()),'ge7_days':sum(v[j]>=7 for v in people.values()),'median_days_among_index_matched_person_visits':float(np.median([v[j] for v in people.values()])) if people else None} for j,label in enumerate(['indexed_event','NOVA_complete_energy_valid','fiber_complete_energy_valid'])}
 s={'event_day_keys':len(items),'index_matched_days':len(matched),'index_unmatched_days':len(items)-len(matched),'index_matched_person_visits':len(people),'energy_valid_days':len(energy),'NOVA_complete_days':len(novaok),'fiber_complete_days':len(fiberok),'name_matched_events':int(sum(v[8] for k,v in matched)),'indexed_event_rows':int(sum(v[0] for k,v in matched)),'known_energy_NOVA_coverage_fraction':1-totalU/totalE if totalE>0 else None,'bounds_width_pp_quantiles_on_energy_valid_days':dict(zip(['p50','p90','p95','p100'],map(float,np.quantile(widths,[.5,.9,.95,1])))) if widths else None,'person_visit_day_counts':counts}
 scopes[scope]=s
 print('SCOPE',scope,json.dumps(s),flush=True)
# Food-only mapping dictionary, retained inside TRE.
fd=[]
for fid,z in foods.items():
 vals={mp[norm(n)] for n in z['names'] if norm(n) in mp and mp[norm(n)] is not None}
 fd.append({'food_id':fid,'short_names':sorted(z['names']),'categories':sorted(z['categories']),'candidate_NOVA_values':sorted(vals),'all_names_have_valid_NOVA':bool(z['names']) and all(mp.get(norm(n)) in [1,2,3,4] for n in z['names'])})
summary={'full_processed_scan':True,'rows_scanned':scanned,'invalid_identity_or_date_rows':badkey,'index_rows':idxrows,'unique_valid_index_keys':len(idx),'food_ID_count':len(fd),'IDs_with_multiple_short_names':sum(len(x['short_names'])>1 for x in fd),'IDs_with_multiple_categories':sum(len(x['categories'])>1 for x in fd),'IDs_with_conflicting_candidate_NOVA':sum(len(x['candidate_NOVA_values'])>1 for x in fd),'catalog_size':len(mp),'catalog_mapping':'exact short_food_name after trim and casefold only; provisional nomenclature match, not version validation','scopes':scopes,'fingerprints':fingerprints,'elapsed_sec':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'not_final_analysis_N':True,'not_W06_event_edit_duplicate_QC':True,'no_outcomes_read':True,'no_participant_values_exported':True}
(Q/'event_mapping_day_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(Q/'food_id_name_category_candidates_TRE_ONLY.json').write_text(json.dumps(fd,ensure_ascii=False,indent=2))
(Q/'food_category_counts_TRE_ONLY.json').write_text(json.dumps(dict(cats),ensure_ascii=False,indent=2))
records=[]
for k,v in days.items():
 records.append(dict(zip(keys+['date'],k),index_matched=k in idx,event_rows=int(v[0]),energy_kcal=float(v[1]),energy_bad_rows=int(v[2]),fiber_g_known=float(v[3]),fiber_bad_rows=int(v[4]),UPF_kcal_known=float(v[5]),NOVA_unknown_kcal=float(v[6]),NOVA_unknown_rows=int(v[7])))
pq.write_table(pa.Table.from_pylist(records),Q/'day_mapping_diagnostics_TRE_ONLY.parquet')
print('FOOD_ID_DIAGNOSTICS',json.dumps({k:summary[k] for k in ['rows_scanned','food_ID_count','IDs_with_multiple_short_names','IDs_with_multiple_categories','IDs_with_conflicting_candidate_NOVA','elapsed_sec','peak_rss_kib']}),flush=True)
print('MAPPING_DAY_AUDIT_DONE_NOT_FINAL_QC',flush=True)
