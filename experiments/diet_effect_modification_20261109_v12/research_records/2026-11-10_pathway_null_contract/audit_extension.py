"""Included verbatim in the self-contained terminal command; no native execution locally."""

def scale_metadata(label, schema):
    accepted = {'rpk', 'rpkm', 'cpm', 'tpm', 'counts', 'copies_per_million', 'relative_abundance', 'proportion', 'probability', 'percent', 'fraction', 'raw', 'none', 'log', 'log1p'}
    allowed_keys = {'unit', 'units', 'normalization', 'scale', 'transform'}
    recognized, unknown, entries = Counter(), 0, 0
    for metadata in [schema.metadata or {}] + [f.metadata or {} for f in schema]:
        for k, v in metadata.items():
            name = k.decode('utf8', errors='replace').lower() if isinstance(k, bytes) else str(k).lower()
            if name not in allowed_keys:
                continue
            entries += 1
            value = v.decode('utf8', errors='replace').strip().lower() if isinstance(v, bytes) else str(v).strip().lower()
            if value in accepted:
                recognized[name + ':' + value] += 1
            else:
                unknown += 1
    print('SCALE_METADATA', label, 'explicit_entries=%d recognized=%s unrecognized_entries=%d; provider_scale_not_inferred_from_range.' % (entries, json.dumps(dict(recognized), sort_keys=True), unknown), flush=True)

def null_contract(records, matrices, approved, source_path, expected_source_hash):
    require(records['abundance']['columns'] == records['coverage']['columns'], 'NULL_AUDIT_LABEL_SETS_DIFFER')
    a, b = matrices['abundance'], matrices['coverage']
    am, bm = ARROW_NULL_MASKS['abundance'], ARROW_NULL_MASKS['coverage']
    blank = am.all(axis=1) | bm.all(axis=1)
    paired_finite_profiles = np.isfinite(a).any(axis=1) & np.isfinite(b).any(axis=1)
    populated = ~blank & paired_finite_profiles
    selected_indices = np.flatnonzero(blank)
    print('NULL_STORAGE arrow_null_masks_equal=%s whole_candidate_null_in_either_table=%d with_paired_finite_candidates=%d other_profiles_without_paired_finite=%d coverage_explicit_zero_cells=%d coverage_zero_with_positive_abundance_cells=%d' % (
        np.array_equal(am, bm), int(blank.sum()), int(populated.sum()), int((~blank & ~paired_finite_profiles).sum()), int((np.isfinite(b) & (b == 0)).sum()), int((np.isfinite(a) & np.isfinite(b) & (a > 0) & (b == 0)).sum())), flush=True)
    for label, record in records.items():
        scale_metadata(label, record['schema'])
        special = record['special']
        if not special:
            print('SPECIAL_STATUS', label, 'special_fields_unavailable; no output-state conclusion.', flush=True)
            continue
        table = ipc_columns(record['path'], special, record['signature']).take(pa.array(record['linked'], type=pa.int64()))
        special_matrix = np.column_stack([table.column(n).to_numpy(zero_copy_only=False) for n in special]).astype(float)
        sf = np.isfinite(special_matrix)
        print('SPECIAL_STATUS', label, 'special_columns=%d fixed_rows_all_special_finite=%d populated_rows_all_special_finite=%d blank_rows_all_special_finite=%d blank_rows_all_special_zero=%d blank_rows_all_special_one=%d' % (
            len(special), int(sf.all(axis=1).sum()), int((sf.all(axis=1) & populated).sum()), int((sf.all(axis=1) & blank).sum()),
            int((sf.all(axis=1) & (special_matrix == 0).all(axis=1) & blank).sum()), int((sf.all(axis=1) & (special_matrix == 1).all(axis=1) & blank).sum())), flush=True)
        for name in special:
            value = table.column(name).to_numpy(zero_copy_only=False)[selected_indices]
            finite = np.isfinite(value)
            code = name.split(':', 1)[0].strip().upper()
            print('  BLANK_SPECIAL', label, code, 'group_n=%d finite=%d missing_or_nonfinite=%d positive=%d zero=%d one=%d negative=%d' % (
                len(value), int(finite.sum()), int((~finite).sum()), int((finite & (value > 0)).sum()), int((finite & (value == 0)).sum()), int((finite & (value == 1)).sum()), int((finite & (value < 0)).sum())), flush=True)

    require(digest(source_path) == expected_source_hash, 'SOURCE_CHANGED_BEFORE_NULL_AUDIT')
    technical = ['raw_read_count', 'trimmed_read_count', 'duplicate_reads_count', 'excessive_duplicate_reads', 'metaphlan4_aligned_read_count', 'humann_aligned_read_count', 'humann_unmapped_read_count']
    pointers = ['humann_pathway_abundance_tsv', 'humann_pathway_coverage_tsv', 'humann_pathway_abundance_pathway_level_parquet', 'humann_pathway_coverage_pathway_level_parquet']
    with pq.ParquetFile(source_path, pre_buffer=False) as pf:
        present = [n for n in technical+pointers if n in pf.schema_arrow.names]
        print('BLANK_SOURCE_QC group_n=%d available_technical_fields=%s' % (len(selected_indices), json.dumps([n for n in technical if n in present])), flush=True)
        if present:
            positions = [approved[i] for i in selected_indices]
            rows = pf.read(columns=present, use_threads=False).take(pa.array(positions, type=pa.int64())).to_pydict()
            for name in technical:
                if name not in rows:
                    continue
                values = rows[name]
                if name == 'excessive_duplicate_reads':
                    print('  BLANK_READ_QC', name, 'observed=%d true_flags=%d; absence_not_pass.' % (sum(valid(v) for v in values), sum(v is True for v in values)), flush=True)
                else:
                    numeric = []
                    for v in values:
                        try:
                            if valid(v) and math.isfinite(float(v)):
                                numeric.append(float(v))
                        except (TypeError, ValueError, OverflowError):
                            pass
                    print('  BLANK_READ_QC', name, 'finite_numeric=%d positive=%d zero=%d negative=%d; no thresholds_selected.' % (len(numeric), sum(v > 0 for v in numeric), sum(v == 0 for v in numeric), sum(v < 0 for v in numeric)), flush=True)
            for name in pointers:
                if name in rows:
                    print('  BLANK_POINTER', name, 'nonempty=%d/%d; no file access inferred or attempted.' % (sum(isinstance(v, str) and valid(v) for v in rows[name]), len(selected_indices)), flush=True)
    print('NULL_CONTRACT_LIMIT: software zero-omission rules plus masks are supporting evidence, not HPP aggregation certification; whole-null profiles remain flagged. No filling, filtering, rescaling or fitting.', flush=True)
