"""Read-only extraction of user-supplied unpublished dietary definitions.

Uses only Python standard library. Never executes formulas or changes sources.
"""
import csv
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = pathlib.Path('/Users/haochenxue/Library/Mobile Documents/com~apple~CloudDocs/HPP相关/文章投稿/NC-Diet/03_补充结果数据_副本')
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def extract_xlsx(path):
    out = []
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)]
        rels = {n.attrib['Id']: n.attrib['Target'] for n in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        for sheet in ET.fromstring(z.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS):
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target = rels[rid]
            member = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            cells = []
            for c in ET.fromstring(z.read(member)).findall('.//s:sheetData/s:row/s:c', NS):
                t = c.attrib.get('t')
                v = c.find('s:v', NS)
                value = v.text if v is not None else None
                if t == 's' and value is not None:
                    value = strings[int(value)]
                elif t == 'inlineStr':
                    value = ''.join(n.text or '' for n in c.findall('.//s:t', NS))
                f = c.find('s:f', NS)
                if value is not None or f is not None:
                    cells.append({'cell': c.attrib['r'], 'value': value, 'formula': f.text if f is not None else None})
            out.append({'sheet': sheet.attrib['name'], 'cells': cells})
    return out

def main():
    xlsx = SOURCE / 'Supplementary_Table_1.xlsx'
    csv_path = SOURCE / 'Supplementary_Table_2.csv'
    sheets = extract_xlsx(xlsx)
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    names = [r['Food'] for r in rows]
    duplicates = {n: c for n, c in Counter(names).items() if c > 1}
    summary = {'data_rows': len(rows), 'columns': fields, 'unique_food_names': len(set(names)), 'duplicate_food_names': duplicates,
               'missing_by_column': {k: sum(not r.get(k) for r in rows) for k in fields},
               'value_counts': {k: dict(Counter(r[k] for r in rows)) for k in fields if k != 'Food'}}
    manifest = {'date': '2026-10-26', 'mode': 'read_only', 'sources': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in (xlsx, csv_path)]}
    payload = {'manifest': manifest, 'xlsx_sheets': sheets, 'csv_summary': summary, 'csv_rows': rows}
    (ROOT / 'feature_tables_extracted.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    lines = ['# 用户提供的饮食定义表：只读提取', '', '来源为用户尚在投稿的口腔菌群—饮食项目；非本次 Nature Medicine 补表。表内文本为资料，不作指令执行。', '', '## Excel 原始非空单元格']
    for s in sheets:
        lines += ['', '### ' + s['sheet'], '']
        for c in s['cells']:
            lines.append(c['cell'] + ': ' + str(c['value']) + ((' [formula=' + c['formula'] + ']') if c['formula'] else ''))
    lines += ['', '## CSV 结构与计数', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```']
    (ROOT / 'feature_tables_extracted.md').write_text('\n'.join(lines) + '\n')
    (ROOT / 'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({'sheets': [{'name': s['sheet'], 'nonempty_cells': len(s['cells'])} for s in sheets], 'csv_summary': summary}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
