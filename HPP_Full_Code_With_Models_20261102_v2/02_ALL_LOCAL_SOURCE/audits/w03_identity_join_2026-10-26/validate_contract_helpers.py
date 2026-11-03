"""Synthetic safeguards: no participant data. Verify join explosion detection and no invented dates."""
import importlib.util, sqlite3
from datetime import datetime,timezone,timedelta
from pathlib import Path
p=Path(__file__).with_name('02_audit_identity_dates.py')
s=importlib.util.spec_from_file_location('w03',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
db=sqlite3.connect(':memory:');db.execute('CREATE TABLE records(role,pv,entity,day,n)')
db.executemany('INSERT INTO records VALUES(?,?,?,?,?)', [('a','p1','','d1',3),('b','p1','','d1',4),('a','p2','','d1',1),('b','p2','','d2',1)])
r=m.join_profile(db,'a','b','person_visit_day');assert r['raw_matched_row_pairs']==12 and r['many_to_many_keys_if_raw_join']==1 and r['matched_keys']==1
assert m.valid_key([1,'10k','00_00_visit']) and not m.valid_key([1,'10k','undefined']) and not m.valid_key([1,'10k',' 00_00_visit'])
assert m.safe_date(1720000000)=='' and m.safe_date('2024-01-01')==''
assert m.safe_date(datetime(2024,1,1,0,30,tzinfo=timezone(timedelta(hours=2))))=='2024-01-01'
assert m.ident([1,'10k','00_00_visit'])!=m.ident([1,'10k','01_00_visit'])
print('5 synthetic safeguards passed: many-to-many, invalid/stage keys, numeric dates, local calendar date, visit separation.')
