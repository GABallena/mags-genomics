#!/usr/bin/env python3
# Portfolio-safe copy: paths/identifiers generalized; inputs not included.

r"""
Link ARG (RGI) results to MAG bins.

Workflow:
 1. Parse all *.json files in --rgi-dir (RGI JSON output). Each top-level key is an ORF descriptor line; nested keys correspond to individual hits.
 2. Extract ORF id (before first ' # ') and derive base contig id by stripping trailing _\d+ (gene ordinal) if present.
 3. Build mapping of contig header (exact, without trailing annotation after first space) -> bin id by scanning FASTA headers in restored_bins/<sample>/*.fa.
    Bin id reported as filename (without path). Optionally strip sample prefix & suffix if --short-bin-id passed.
 4. Join: for each ARG hit, locate its contig (base contig). If no match (maybe unbinned), leave bin fields NA.
 5. Output TSV.

Assumptions:
 - RGI JSON top-level is a dict.
 - ORF headers use pattern: SAMPLE_k141_XXXXX_geneNumber ...; base contig is without last underscore + geneNumber.
 - Bin FASTA headers match the relabeled contig headers (same as ORF base contig part).
 - restored_bins directory structure: restored_bins/<sample>/*.fa

Columns emitted:
 sample	orf_id	contig_base	bin_id	bin_file	ARO_accession	ARO_name	model_name	model_type	pass_bitscore	bit_score	max_identities	drug_classes	amr_gene_families	resistance_mechanisms

If multiple categories of the same class occur they are semicolon-separated.

Usage:
  python link_args_to_mags.py --rgi-dir rgi_output --bins-dir restored_bins -o arg_to_mag.tsv

"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
import csv
from typing import Dict, Tuple

ORF_BASE_RE = re.compile(r'^(\S+?)(?:_(\d+))?$')  # fallback if trailing gene number absent
TRAILING_GENE_RE = re.compile(r'^(.*)_\d+$')

CATEGORY_CLASS_GENE_FAMILY = 'AMR Gene Family'
CATEGORY_CLASS_DRUG_CLASS = 'Drug Class'
CATEGORY_CLASS_RES_MECH = 'Resistance Mechanism'

FA_EXTS = {'.fa', '.fasta', '.fna'}

def parse_args():
    ap = argparse.ArgumentParser(description='Link RGI ARG hits to MAG bins')
    ap.add_argument('--rgi-dir', required=True, help='Directory containing RGI outputs (.json and/or *_rgi.txt)')
    ap.add_argument('--bins-dir', required=True, help='Directory containing restored bin FASTAs per sample')
    ap.add_argument('-o', '--output', required=True, help='Output TSV file')
    ap.add_argument('--min-pass-bitscore', type=float, default=None, help='Optional minimum Pass_Bitscore / pass_bitscore filter (JSON or TSV)')
    ap.add_argument('--short-bin-id', action='store_true', help='Report shortened bin id (strip sample prefix and suffix after first .orig)')
    ap.add_argument('--limit-samples', nargs='+', help='Optional subset of sample IDs (directory names) to process, e.g. SAMPLE-XX_SYY')
    ap.add_argument('--rgi-tsv', action='store_true', help='Parse RGI *_rgi.txt tabular outputs instead of JSON')
    ap.add_argument('--strict-only', action='store_true', help='(TSV) keep only rows with Cut_Off == Strict (overrides default Strict+Perfect filter)')
    ap.add_argument('--include-loose', action='store_true', help='(TSV) include Loose hits as well (default excludes Loose)')
    ap.add_argument('--model-type-filter', type=str, default=None, help='(TSV) comma-separated allowed Model_type values (e.g. "protein homolog model")')
    ap.add_argument('--gtdb-dir', default='gtdbtk_drep_out', help='Root directory containing GTDB-Tk results (searches for classify/gtdbtk.*.summary.tsv)')
    ap.add_argument('--card-ontology', default='card-ontology/aro.tsv', help='Path to CARD ARO ontology TSV (will merge by Accession)')
    return ap.parse_args()


def discover_bins(bins_dir: str, limit_samples=None, short_bin=False):
    """Scan bin FASTA headers to map contig -> bin info (and contig length if encoded).
    Returns dict: contig -> (sample, bin_id, bin_file, contig_len)

    FASTA header example:
      >PROJECT-01_S1_k141_772783 flag=0 multi=26.0000 len=46866
    We capture len= value when present; otherwise 'NA'.
    """
    contig_to_bin = {}
    bins_dir_p = Path(bins_dir)
    if not bins_dir_p.is_dir():
        raise SystemExit(f'Bins directory not found: {bins_dir}')

    for sample_dir in sorted(bins_dir_p.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample = sample_dir.name
        if limit_samples and sample not in limit_samples:
            continue
        for fa in sample_dir.iterdir():
            if fa.suffix not in FA_EXTS:
                continue
            bin_file = fa.name
            bin_id = bin_file
            if short_bin:
                # Example: PROJECT-01_S1_bin.11.filtered..orig.fa -> bin.11
                m = re.search(r'(bin\.\d+)', bin_file)
                if m:
                    bin_id = m.group(1)
                else:
                    bin_id = bin_file.rsplit('.', 1)[0]
            try:
                with fa.open() as fh:
                    for line in fh:
                        if not line.startswith('>'):
                            continue
                        header = line[1:].strip()
                        contig_name = header.split()[0]  # up to first space
                        # parse contig length if present ( len=NUMBER )
                        mlen = re.search(r'\blen=(\d+)', header)
                        clen = mlen.group(1) if mlen else 'NA'
                        # map contig to bin (if duplicate, keep first)
                        if contig_name not in contig_to_bin:
                            contig_to_bin[contig_name] = (sample, bin_id, bin_file, clen)
            except Exception as e:
                print(f'Warning: failed reading {fa}: {e}')
    return contig_to_bin


def load_rgi_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def extract_sample_from_orf(orf_id: str):
    # Sample pattern appears to be PROJECT-XX_SYY at start
    m = re.match(r'^(PROJECT-\d+_S\d+)', orf_id)
    return m.group(1) if m else 'NA'


def base_contig_from_orf(orf_id: str):
    # Remove trailing _number which is ORF ordinal on contig
    m = TRAILING_GENE_RE.match(orf_id)
    return m.group(1) if m else orf_id


def flatten_categories(cat_dict):
    drug_classes = []
    gene_families = []
    mechanisms = []
    if isinstance(cat_dict, dict):
        for obj in cat_dict.values():
            cls = obj.get('category_aro_class_name')
            name = obj.get('category_aro_name')
            if not cls or not name:
                continue
            if cls == CATEGORY_CLASS_DRUG_CLASS:
                drug_classes.append(name)
            elif cls == CATEGORY_CLASS_GENE_FAMILY:
                gene_families.append(name)
            elif cls == CATEGORY_CLASS_RES_MECH:
                mechanisms.append(name)
    # Deduplicate while preserving order
    def dedup(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return ';'.join(dedup(drug_classes)) or 'NA', \
           ';'.join(dedup(gene_families)) or 'NA', \
           ';'.join(dedup(mechanisms)) or 'NA'


def iterate_rgi_hits_json(rgi_dir, limit_samples=None, min_pass_bitscore=None):
    rgi_dir_p = Path(rgi_dir)
    for json_file in sorted(rgi_dir_p.glob('*_rgi.json')):
        try:
            data = load_rgi_json(json_file)
        except Exception as e:
            print(f'Warning: cannot parse {json_file}: {e}')
            continue
        # Each top-level key is an ORF context or extra metadata; filter keys that look like ORF (start with sample prefix PROJECT- and contain k) 
        for orf_header, hit_block in data.items():
            if not isinstance(hit_block, dict):
                continue
            orf_id = orf_header.split(' # ')[0].strip()
            if not re.search(r'k\d+_\d+', orf_id):
                continue
            sample = extract_sample_from_orf(orf_id)
            if limit_samples and sample not in limit_samples:
                continue
            contig_base = base_contig_from_orf(orf_id)
            # Nested hits keyed by BL_ORD accession strings
            for bl_key, hit in hit_block.items():
                if not isinstance(hit, dict):
                    continue
                pass_bitscore = hit.get('pass_bitscore')
                try:
                    pass_bitscore_val = float(pass_bitscore) if pass_bitscore not in (None, 'n/a') else None
                except ValueError:
                    pass_bitscore_val = None
                if min_pass_bitscore is not None and (pass_bitscore_val is None or pass_bitscore_val < min_pass_bitscore):
                    continue
                yield {
                    'sample': sample,
                    'orf_id': orf_id,
                    'contig_base': contig_base,
                    'ARO_accession': hit.get('ARO_accession','NA'),
                    'ARO_name': hit.get('ARO_name','NA'),
                    'model_name': hit.get('model_name','NA'),
                    'model_type': hit.get('model_type','NA'),
                    'pass_bitscore': pass_bitscore,
                    'bit_score': hit.get('bit_score','NA'),
                    'max_identities': hit.get('max_identities','NA'),
                    'categories': hit.get('ARO_category', {})
                }


def iterate_rgi_hits_tsv(rgi_dir, limit_samples=None, min_pass_bitscore=None, strict_only=False, include_loose=False, model_type_filter=None):
    """Iterate through RGI *_rgi.txt tabular outputs applying filters."""
    allowed_model_types = None
    if model_type_filter:
        allowed_model_types = {m.strip() for m in model_type_filter.split(',') if m.strip()}
    rgi_dir_p = Path(rgi_dir)
    for tsv_file in sorted(rgi_dir_p.glob('*_rgi.txt')):
        try:
            with tsv_file.open() as fh:
                reader = csv.DictReader(fh, delimiter='\t')
                for row in reader:
                    if not row:
                        continue
                    orf_id = (row.get('ORF_ID') or '').strip()
                    if not re.search(r'k\d+_\d+', orf_id):
                        continue
                    sample = extract_sample_from_orf(orf_id)
                    if limit_samples and sample not in limit_samples:
                        continue
                    contig_base = base_contig_from_orf(orf_id.split(' # ')[0])
                    cutoff = row.get('Cut_Off') or row.get('Cut_Off '.strip(), '')
                    if strict_only:
                        if cutoff != 'Strict':
                            continue
                    else:
                        # Default: keep only Strict + Perfect unless include_loose requested
                        if not include_loose and cutoff not in ('Strict', 'Perfect'):
                            continue
                    model_type = row.get('Model_type', '')
                    if allowed_model_types and model_type not in allowed_model_types:
                        continue
                    pass_bitscore = row.get('Pass_Bitscore')
                    if min_pass_bitscore is not None:
                        try:
                            pbv = float(pass_bitscore) if pass_bitscore not in (None, 'n/a', '') else None
                        except ValueError:
                            pbv = None
                        if pbv is None or pbv < min_pass_bitscore:
                            continue
                    cat = {}
                    def add_cat(name, cls):
                        if name:
                            k = f'{cls}_{len(cat)+1}'
                            cat[k] = {'category_aro_class_name': cls, 'category_aro_name': name}
                    add_cat(row.get('Drug Class',''), CATEGORY_CLASS_DRUG_CLASS)
                    add_cat(row.get('Resistance Mechanism',''), CATEGORY_CLASS_RES_MECH)
                    add_cat(row.get('AMR Gene Family',''), CATEGORY_CLASS_GENE_FAMILY)
                    yield {
                        'sample': sample,
                        'orf_id': orf_id.split(' # ')[0],
                        'contig_base': contig_base,
                        'ARO_accession': row.get('ARO','NA'),
                        'ARO_name': row.get('Best_Hit_ARO','NA'),
                        'model_name': 'NA',
                        'model_type': model_type or 'NA',
                        'pass_bitscore': pass_bitscore,
                        'bit_score': row.get('Best_Hit_Bitscore','NA'),
                        'max_identities': row.get('Best_Identities','NA'),
                        'categories': cat
                    }
        except Exception as e:
            print(f'Warning: failed parsing {tsv_file}: {e}')


def main():
    args = parse_args()
    limit_samples = set(args.limit_samples) if args.limit_samples else None

    print('Scanning bins to build contig->bin map ...', flush=True)
    contig_to_bin = discover_bins(args.bins_dir, limit_samples=limit_samples, short_bin=args.short_bin_id)
    print(f'Mapped {len(contig_to_bin):,} contigs to bins (with length when available)', flush=True)

    # ------------------------------------------------------------------
    # Load GTDB taxonomy (optional)
    # ------------------------------------------------------------------
    def load_gtdb_tax(gtdb_root: str) -> Dict[str, Dict[str,str]]:
        root_p = Path(gtdb_root)
        if not root_p.is_dir():
            print(f'GTDB directory not found (skipping taxonomy): {gtdb_root}')
            return {}
        summary_paths = list(root_p.rglob('classify/gtdbtk.*.summary.tsv'))
        if not summary_paths:
            print(f'No GTDB summary files found under {gtdb_root}; skipping taxonomy.')
            return {}
        tax_map: Dict[str, Dict[str,str]] = {}
        def parse_lineage(classification: str) -> Dict[str,str]:
            ranks = {'d__':'domain','p__':'phylum','c__':'class','o__':'order','f__':'family','g__':'genus','s__':'species'}
            out = {v:'NA' for v in ranks.values()}
            if not classification:
                return out
            for part in classification.split(';'):
                part = part.strip()
                for prefix, rk in ranks.items():
                    if part.startswith(prefix):
                        out[rk] = part[len(prefix):] or 'NA'
                        break
            return out
        for sp in summary_paths:
            try:
                with sp.open() as fh:
                    header = fh.readline().rstrip('\n').split('\t')
                    col_idx = {c:i for i,c in enumerate(header)}
                    if not {'user_genome','classification'} <= set(col_idx):
                        continue
                    for line in fh:
                        if not line.strip():
                            continue
                        parts = line.rstrip('\n').split('\t')
                        user_genome = parts[col_idx['user_genome']]
                        classification = parts[col_idx['classification']]
                        lineage = parse_lineage(classification)
                        base = re.sub(r'\.(fa|fasta|fna)$','', user_genome)
                        rec = {'classification': classification}
                        rec.update(lineage)
                        # store under multiple keys for robust joins
                        for k in {user_genome, base}:
                            if k not in tax_map:
                                tax_map[k] = rec
            except Exception as e:
                print(f'Warning: failed reading {sp}: {e}')
        print(f'Loaded taxonomy for {len(tax_map)} genome identifiers')
        return tax_map

    tax_lookup = load_gtdb_tax(args.gtdb_dir)

    # ------------------------------------------------------------------
    # Load CARD ARO ontology (optional)
    # ------------------------------------------------------------------
    def load_aro(path: str):
        aro_map = {}
        p = Path(path)
        if not p.is_file():
            print(f'ARO TSV not found (skipping ARO enrichment): {path}')
            return aro_map, []
        try:
            with p.open() as fh:
                header_line = fh.readline().rstrip('\n')
                header_cols = header_line.split('\t')
                # Normalize minimal expected columns
                # We'll keep raw names but map spaces to underscores for output
                norm_cols = [c.strip() for c in header_cols]
                col_idx = {c:i for i,c in enumerate(norm_cols)}
                required = {'Accession'}
                if not required <= set(col_idx):
                    print('ARO TSV missing Accession column; skipping.')
                    return aro_map, []
                for line in fh:
                    if not line.strip():
                        continue
                    parts = line.rstrip('\n').split('\t')
                    acc = parts[col_idx['Accession']]
                    if not acc:
                        continue
                    aro_map[acc] = {c: (parts[col_idx[c]] if col_idx[c] < len(parts) else 'NA') for c in norm_cols}
            # Return ordered column list excluding Accession (since already present as ARO_accession)
            remaining_cols = [c for c in norm_cols if c != 'Accession']
            print(f'Loaded {len(aro_map)} ARO ontology entries from {path}')
            return aro_map, remaining_cols
        except Exception as e:
            print(f'Warning: failed to parse ARO TSV {path}: {e}')
            return aro_map, []

    aro_map, aro_cols = load_aro(args.card_ontology)
    # Prepare sanitized output column names (spaces -> underscores)
    aro_cols_out = [re.sub(r'\s+', '_', c) for c in aro_cols]

    def get_tax(bin_file: str, bin_id: str):
        """Attempt to look up taxonomy using a wide set of normalized keys.

        GTDB user_genome examples: SAMPLE_bin.11.filtered
        Our FASTA filenames:      SAMPLE_bin.11.filtered..orig.fa
        This function generates candidate keys by:
          - stripping FASTA extensions
          - removing '.orig'
          - collapsing repeated dots
          - truncating at '.filtered'
          - applying the same transforms to bin_id
        The first match in the taxonomy lookup wins.
        """
        if not bin_file or bin_file == 'NA':
            return ['NA'] * 8
        candidates = []
        seen = set()
        def add(name):
            if name and name not in seen:
                seen.add(name)
                candidates.append(name)
        def expand(name):
            if not name:
                return
            add(name)
            base = re.sub(r'\.(fa|fasta|fna)$', '', name)
            add(base)
            add(base.replace('.orig', ''))
            add(re.sub(r'\.{2,}', '.', base))
            if '.filtered' in base:
                trunc = base.split('.filtered', 1)[0] + '.filtered'
                add(trunc)
                add(trunc.replace('.orig', ''))
        expand(bin_file)
        expand(bin_id)
        for key in candidates:
            if key in tax_lookup:
                rec = tax_lookup[key]
                return [
                    rec.get('classification','NA'), rec.get('domain','NA'), rec.get('phylum','NA'),
                    rec.get('class','NA'), rec.get('order','NA'), rec.get('family','NA'),
                    rec.get('genus','NA'), rec.get('species','NA')
                ]
        return ['NA'] * 8

    print('Parsing RGI results ...', flush=True)
    rows = []
    if args.rgi_tsv:
        for rec in iterate_rgi_hits_tsv(
            args.rgi_dir,
            limit_samples=limit_samples,
            min_pass_bitscore=args.min_pass_bitscore,
            strict_only=args.strict_only,
            include_loose=args.include_loose,
            model_type_filter=args.model_type_filter
        ):
            contig_base = rec['contig_base']
            bin_sample = bin_id = bin_file = contig_len = 'NA'
            if contig_base in contig_to_bin:
                bin_sample, bin_id, bin_file, contig_len = contig_to_bin[contig_base]
            drug_classes, gene_families, mechanisms = flatten_categories(rec['categories'])
            classification_block = get_tax(bin_file, bin_id)
            # counts for ontology breadth
            def count_items(s):
                return '0' if (not s or s == 'NA') else str(len([x for x in s.split(';') if x]))
            # Normalize ARO accession (RGI TSV often numeric, ontology has 'ARO:#######')
            raw_acc = rec['ARO_accession']
            pref_acc = ('ARO:' + raw_acc) if raw_acc and not raw_acc.startswith('ARO:') else raw_acc
            aro_entry = aro_map.get(pref_acc, {}) or aro_map.get(raw_acc, {})
            aro_vals = [aro_entry.get(c, 'NA') for c in aro_cols]
            rows.append([
                rec['sample'], rec['orf_id'], contig_base, contig_len, bin_id, bin_file,
                rec['ARO_accession'], rec['ARO_name'], rec['model_name'], rec['model_type'],
                rec['pass_bitscore'], rec['bit_score'], rec['max_identities'],
                drug_classes, gene_families, mechanisms,
                count_items(drug_classes), count_items(gene_families), count_items(mechanisms),
            ] + classification_block + aro_vals)
    else:
        for rec in iterate_rgi_hits_json(args.rgi_dir, limit_samples=limit_samples, min_pass_bitscore=args.min_pass_bitscore):
            contig_base = rec['contig_base']
            sample = rec['sample']
            contig_lookup = contig_base
            bin_sample = bin_id = bin_file = contig_len = 'NA'
            if contig_lookup in contig_to_bin:
                bin_sample, bin_id, bin_file, contig_len = contig_to_bin[contig_lookup]
            drug_classes, gene_families, mechanisms = flatten_categories(rec['categories'])
            classification_block = get_tax(bin_file, bin_id)
            def count_items(s):
                return '0' if (not s or s == 'NA') else str(len([x for x in s.split(';') if x]))
            raw_acc = rec['ARO_accession']
            pref_acc = ('ARO:' + raw_acc) if raw_acc and not raw_acc.startswith('ARO:') else raw_acc
            aro_entry = aro_map.get(pref_acc, {}) or aro_map.get(raw_acc, {})
            aro_vals = [aro_entry.get(c, 'NA') for c in aro_cols]
            rows.append(
                [sample, rec['orf_id'], contig_base, contig_len, bin_id, bin_file, rec['ARO_accession'], rec['ARO_name'], rec['model_name'], rec['model_type'], rec['pass_bitscore'], rec['bit_score'], rec['max_identities'], drug_classes, gene_families, mechanisms, count_items(drug_classes), count_items(gene_families), count_items(mechanisms)] + classification_block + aro_vals
            )

    header = [
        'sample','orf_id','contig_base','contig_len','bin_id','bin_file',
        'ARO_accession','ARO_name','model_name','model_type','pass_bitscore','bit_score','max_identities',
        'drug_classes','amr_gene_families','resistance_mechanisms',
        'n_drug_classes','n_amr_gene_families','n_resistance_mechanisms',
        'gtdb_classification','domain','phylum','class','order','family','genus','species'
    ]
    # Append ontology columns if available
    if aro_cols_out:
        header.extend(['ARO_'+c for c in aro_cols_out])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as out:
        out.write('\t'.join(header)+'\n')
        for r in rows:
            out.write('\t'.join(str(x) for x in r)+'\n')
    print(f'Wrote {len(rows):,} rows to {out_path}')

if __name__ == '__main__':
    main()
