"""
AIxBio Hackathon 2026 - Data Gathering Pipeline
===============================================
Fetches, cleans, and balances labeled protein sequences from NCBI Protein 
and UniProt Swiss-Prot. Constructs a dataset of biosecurity-relevant 
pathogens (Select Agents) and their taxonomically matched benign relatives.

Outputs:
  - data/sequences.fasta : Deduplicated, length-filtered protein sequences.
  - data/labels.csv      : Metadata and binary pathogenicity labels (1=Pathogen, 0=Benign).
  - data/stats.txt       : Distribution statistics per taxonomic family.

Usage:
  python data_pipeline.py --email <your_email@example.com> --output_dir ./data
"""

import os
import time
import csv
import argparse
import random
import hashlib
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

ENTREZ_DELAY       = 0.4   # Seconds between NCBI requests
MAX_SEQS_PER_QUERY = 30
MIN_SEQ_LEN        = 80
MAX_SEQ_LEN        = 2000
MAX_NEG_RATIO      = 1.4   # Maximum negative-to-positive ratio per family


# -----------------------------------------------------------------------------
# ORGANISM TARGETS
# -----------------------------------------------------------------------------

TARGETS =[
    # POSITIVES - TIER 1 SELECT AGENTS
    {
        "organism": "Ebola virus", "taxid": "186538",
        "proteins":["nucleoprotein", "glycoprotein", "VP40", "VP35"],
        "label": 1, "tier": "T1", "family": "filovirus",
    },
    {
        "organism": "Marburg marburgvirus", "taxid": "3052505",
        "proteins": ["nucleoprotein", "glycoprotein", "VP40", "protein"],
        "label": 1, "tier": "T1", "family": "filovirus",
    },
    {
        "organism": "Marburg marburgvirus Ravn", "taxid": "3052506",
        "proteins": ["nucleoprotein", "glycoprotein", "VP40"],
        "label": 1, "tier": "T1", "family": "filovirus",
    },
    {
        "organism": "Variola virus", "taxid": "10255",
        "proteins": ["hemagglutinin", "virulence factor", "surface protein", "envelope protein"],
        "label": 1, "tier": "T1", "family": "poxvirus",
    },
    {
        "organism": "Monkeypox virus", "taxid": "10244",
        "proteins": ["hemagglutinin", "surface protein", "envelope protein", "virulence"],
        "label": 1, "tier": "T1", "family": "poxvirus",
    },
    {
        "organism": "Bacillus anthracis", "taxid": "1392",
        "proteins":["protective antigen", "lethal factor", "edema factor", "capsule", "surface"],
        "label": 1, "tier": "T1", "family": "bacillus",
    },
    {
        "organism": "Yersinia pestis", "taxid": "632",
        "proteins":["plasminogen activator", "capsular antigen", "V antigen", "Yop", "outer membrane"],
        "label": 1, "tier": "T1", "family": "enterobacteriaceae",
    },
    {
        "organism": "Francisella tularensis", "taxid": "263",
        "proteins":["outer membrane protein", "surface protein", "membrane protein", "virulence"],
        "label": 1, "tier": "T1", "family": "francisellaceae",
    },
    {
        "organism": "Clostridium botulinum", "taxid": "1491",
        "proteins":["neurotoxin", "toxin", "NTNH nontoxic", "hemagglutinin"],
        "label": 1, "tier": "T1", "family": "clostridiaceae",
    },
    {
        "organism": "Clostridium botulinum ATCC 3502", "taxid": "272568",
        "proteins":["neurotoxin", "toxin complex", "toxin"],
        "label": 1, "tier": "T1", "family": "clostridiaceae",
    },

    # POSITIVES - TIER 2 SELECT AGENTS
    {
        "organism": "Burkholderia mallei", "taxid": "13373",
        "proteins": ["type III secretion", "capsular polysaccharide", "outer membrane protein"],
        "label": 1, "tier": "T2", "family": "burkholderia",
    },
    {
        "organism": "Burkholderia pseudomallei", "taxid": "28450",
        "proteins": ["type III secretion effector", "capsular polysaccharide", "outer membrane"],
        "label": 1, "tier": "T2", "family": "burkholderia",
    },
    {
        "organism": "Coxiella burnetii", "taxid": "777",
        "proteins":["effector", "outer membrane protein", "secreted protein", "surface protein"],
        "label": 1, "tier": "T2", "family": "coxiellaceae",
    },
    {
        "organism": "Influenza A H5N1", "taxid": "1252577",
        "proteins":["hemagglutinin", "neuraminidase", "polymerase"],
        "label": 1, "tier": "T2", "family": "orthomyxovirus",
    },
    {
        "organism": "Influenza A H5N1 duck Vietnam", "taxid": "335341",
        "proteins": ["hemagglutinin", "neuraminidase", "polymerase basic protein"],
        "label": 1, "tier": "T2", "family": "orthomyxovirus",
    },
    {
        "organism": "SARS-CoV-2", "taxid": "2697049",
        "proteins": ["spike protein", "nucleocapsid protein", "membrane protein"],
        "label": 1, "tier": "T2", "family": "coronavirus",
    },
    {
        "organism": "SARS coronavirus", "taxid": "227859",
        "proteins": ["spike protein", "nucleocapsid protein", "membrane protein"],
        "label": 1, "tier": "T2", "family": "coronavirus",
    },
    {
        "organism": "MERS coronavirus", "taxid": "1335626",
        "proteins":["spike protein", "nucleocapsid protein", "membrane protein"],
        "label": 1, "tier": "T2", "family": "coronavirus",
    },
    {
        "organism": "Nipah virus", "taxid": "121791",
        "proteins":["glycoprotein", "fusion protein", "nucleoprotein"],
        "label": 1, "tier": "T2", "family": "paramyxovirus",
    },
    {
        "organism": "Ricinus communis", "taxid": "3988",
        "proteins":["ricin A chain", "ricin B chain", "agglutinin"],
        "label": 1, "tier": "T1", "family": "plant_toxin",
    },

    # NEGATIVES - TAXONOMICALLY MATCHED
    {
        "organism": "Lloviu cuevavirus", "taxid": "1513225",
        "proteins":["VP40", "nucleoprotein", "glycoprotein"],
        "label": 0, "tier": "none", "family": "filovirus",
    },
    {
        "organism": "Bombali ebolavirus", "taxid": "2010960",
        "proteins": ["VP40", "nucleoprotein", "glycoprotein"],
        "label": 0, "tier": "none", "family": "filovirus",
    },
    {
        "organism": "Vaccinia virus", "taxid": "10245",
        "proteins": ["hemagglutinin", "surface protein", "envelope protein"],
        "label": 0, "tier": "none", "family": "poxvirus",
    },
    {
        "organism": "Ectromelia virus", "taxid": "12643",
        "proteins": ["hemagglutinin", "surface protein", "envelope protein"],
        "label": 0, "tier": "none", "family": "poxvirus",
    },
    {
        "organism": "Bacillus subtilis", "taxid": "1423",
        "proteins":["sporulation protein", "surface protein", "membrane protein"],
        "label": 0, "tier": "none", "family": "bacillus",
    },
    {
        "organism": "Bacillus cereus", "taxid": "1396",
        "proteins":["surface protein", "sporulation protein", "cell wall protein"],
        "label": 0, "tier": "none", "family": "bacillus",
    },
    {
        "organism": "Yersinia enterocolitica", "taxid": "630",
        "proteins":["outer membrane protein", "adhesin", "Yop"],
        "label": 0, "tier": "none", "family": "enterobacteriaceae",
    },
    {
        "organism": "Yersinia pseudotuberculosis", "taxid": "633",
        "proteins":["outer membrane protein", "invasin", "Yop"],
        "label": 0, "tier": "none", "family": "enterobacteriaceae",
    },
    {
        "organism": "Francisella novicida", "taxid": "376619",
        "proteins":["IglC", "outer membrane protein", "membrane protein"],
        "label": 0, "tier": "none", "family": "francisellaceae",
    },
    {
        "organism": "Clostridium acetobutylicum", "taxid": "1488",
        "proteins":["surface protein", "sporulation protein", "membrane protein"],
        "label": 0, "tier": "none", "family": "clostridiaceae",
    },
    {
        "organism": "Clostridioides difficile", "taxid": "1496",
        "proteins":["surface protein", "sporulation protein", "cell wall protein"],
        "label": 0, "tier": "none", "family": "clostridiaceae",
    },
    {
        "organism": "Burkholderia thailandensis", "taxid": "57975",
        "proteins":["type III secretion", "outer membrane protein", "flagellin"],
        "label": 0, "tier": "none", "family": "burkholderia",
    },
    {
        "organism": "Burkholderia cepacia", "taxid": "292",
        "proteins":["outer membrane protein", "adhesin", "flagellin"],
        "label": 0, "tier": "none", "family": "burkholderia",
    },
    {
        "organism": "Legionella pneumophila", "taxid": "446",
        "proteins":["effector protein", "outer membrane protein", "secreted protein", "surface protein"],
        "label": 0, "tier": "none", "family": "coxiellaceae",
    },
    {
        "organism": "Influenza A H1N1 PR8", "taxid": "211044",
        "proteins": ["hemagglutinin", "neuraminidase", "polymerase"],
        "label": 0, "tier": "none", "family": "orthomyxovirus",
    },
    {
        "organism": "Influenza B virus", "taxid": "11520",
        "proteins": ["hemagglutinin", "neuraminidase", "nucleoprotein"],
        "label": 0, "tier": "none", "family": "orthomyxovirus",
    },
    {
        "organism": "OC43 betacoronavirus", "taxid": "31631",
        "proteins": ["spike protein", "nucleocapsid", "membrane protein"],
        "label": 0, "tier": "none", "family": "coronavirus",
    },
    {
        "organism": "Bat coronavirus RaTG13", "taxid": "2709072",
        "proteins": ["spike protein", "nucleocapsid protein", "membrane protein"],
        "label": 0, "tier": "none", "family": "coronavirus",
    },
    {
        "organism": "Sendai virus", "taxid": "11201",
        "proteins":["glycoprotein", "fusion protein", "nucleoprotein"],
        "label": 0, "tier": "none", "family": "paramyxovirus",
    },
    {
        "organism": "Canavalia ensiformis", "taxid": "3823",
        "proteins":["concanavalin A", "lectin", "seed protein"],
        "label": 0, "tier": "none", "family": "plant_toxin",
    },
]

# -----------------------------------------------------------------------------
# UNIPROT SUPPLEMENT
# -----------------------------------------------------------------------------
# Queries for canonical virulence factors from reviewed Swiss-Prot entries.

UNIPROT_QUERIES =[
    ("Bacillus anthracis anthrax toxin",       1, "T1", "bacillus"),
    ("Clostridium botulinum neurotoxin",       1, "T1", "clostridiaceae"),
    ("Yersinia pestis virulence",              1, "T1", "enterobacteriaceae"),
    ("Francisella tularensis virulence",       1, "T1", "francisellaceae"),
    ("Ebola virus glycoprotein",               1, "T1", "filovirus"),
    ("SARS-CoV-2 spike protein",               1, "T2", "coronavirus"),
    ("Nipah virus fusion protein",             1, "T2", "paramyxovirus"),
    ("Burkholderia pseudomallei effector",     1, "T2", "burkholderia"),
    ("Coxiella burnetii virulence effector",   1, "T2", "coxiellaceae"),
    ("ricin toxin Ricinus",                    1, "T1", "plant_toxin"),
]

EXCLUDE_FROM_PROBING = {"francisellaceae", "plant_toxin"}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def search_proteins(taxid: str, protein_term: str, max_results: int = MAX_SEQS_PER_QUERY) -> list[str]:
    """Search NCBI Protein by taxonomy ID and protein term."""
    query = f'txid{taxid}[Organism] AND "{protein_term}"[Title]'
    try:
        handle = Entrez.esearch(db="protein", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        time.sleep(ENTREZ_DELAY)
        ids = record["IdList"]

        if len(ids) < 3:
            query2 = f'txid{taxid}[Organism] AND {protein_term}[Title]'
            handle2 = Entrez.esearch(db="protein", term=query2, retmax=max_results)
            record2 = Entrez.read(handle2)
            handle2.close()
            time.sleep(ENTREZ_DELAY)
            ids = list(set(ids + record2["IdList"]))[:max_results]

        return ids
    except Exception as e:
        print(f"  [WARN] NCBI Search Error {taxid}/{protein_term}: {e}")
        return []


def fetch_sequences(ids: list[str]) -> list[SeqRecord]:
    """Fetch full FASTA records from NCBI Protein."""
    if not ids:
        return[]
    try:
        handle = Entrez.efetch(db="protein", id=",".join(ids), rettype="fasta", retmode="text")
        records = list(SeqIO.parse(handle, "fasta"))
        handle.close()
        time.sleep(ENTREZ_DELAY)
        return records
    except Exception as e:
        print(f"  [WARN] NCBI Fetch Error: {e}")
        return[]


def clean_sequence(record: SeqRecord) -> SeqRecord | None:
    """Filter sequences by length and character composition."""
    seq = str(record.seq).upper()
    length = len(seq)
    
    if length < MIN_SEQ_LEN or length > MAX_SEQ_LEN:
        return None
        
    std_aas = set("ACDEFGHIKLMNPQRSTVWY")
    if sum(1 for c in seq if c not in std_aas) / length > 0.05:
        return None
        
    # Reject high-nucleotide content sequences
    if sum(1 for c in seq if c in "ACGTU") / length > 0.85:
        return None
        
    return record


def seq_hash(seq: str) -> str:
    """Generate MD5 hash for exact sequence deduplication."""
    return hashlib.md5(seq.upper().encode()).hexdigest()


def fetch_uniprot(query: str, label: int, tier: str, family: str, max_results: int = 30) -> list[tuple[SeqRecord, dict]]:
    """Query UniProt REST API for canonical sequence supplementation."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"({query}) AND (reviewed:true)",
        "format": "fasta",
        "size": max_results,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        from io import StringIO
        records = list(SeqIO.parse(StringIO(r.text), "fasta"))
        results =[]
        for rec in records:
            c = clean_sequence(rec)
            if c is None:
                continue
            meta = {
                "organism": f"UniProt:{query[:40]}",
                "protein": query,
                "label": label,
                "tier": tier,
                "family": family,
                "source": "UniProt",
                "description": c.description[:120],
            }
            results.append((c, meta))
        return results
    except Exception as e:
        print(f"  [WARN] UniProt Fetch Error ({query[:30]}...): {e}")
        return []


def balance_dataset(all_records: list[tuple[SeqRecord, dict]]) -> list[tuple[SeqRecord, dict]]:
    """Balance positive and negative classes within each taxonomic family."""
    from collections import defaultdict
    by_family = defaultdict(lambda: {"pos": [], "neg":[]})
    
    for rec, meta in all_records:
        key = "pos" if meta["label"] == 1 else "neg"
        by_family[meta["family"]][key].append((rec, meta))

    balanced =[]
    random.seed(42)
    
    for family, groups in by_family.items():
        pos = groups["pos"]
        neg = groups["neg"]
        
        if not pos or not neg:
            balanced.extend(pos + neg)
            continue
            
        max_neg = int(len(pos) * MAX_NEG_RATIO)
        max_pos = int(len(neg) * MAX_NEG_RATIO)
        
        if len(neg) > max_neg:
            neg = random.sample(neg, max_neg)
        if len(pos) > max_pos:
            pos = random.sample(pos, max_pos)
            
        balanced.extend(pos + neg)

    random.shuffle(balanced)
    return balanced


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def run_pipeline(email: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    Entrez.email = email

    all_records =[]
    seen_hashes = set()

    print("\n" + "=" * 60)
    print("  NCBI Protein Fetch")
    print("=" * 60)

    for target in tqdm(TARGETS, desc="Organisms"):
        organism = target["organism"]
        taxid    = target["taxid"]
        label    = target["label"]
        tier     = target["tier"]
        family   = target["family"]

        print(f"\n[{'POS' if label==1 else 'NEG'}] {organism} (taxid={taxid})")

        for protein in target["proteins"]:
            ids = search_proteins(taxid, protein)
            print(f"  {protein}: {len(ids)} IDs", end="")
            if not ids:
                print()
                continue
                
            records = fetch_sequences(ids)
            added = 0
            for rec in records:
                c = clean_sequence(rec)
                if c is None:
                    continue
                h = seq_hash(str(c.seq))
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                
                new_id = f"{'POS' if label==1 else 'NEG'}_{organism.replace(' ','_')}_{c.id}"
                c.id = new_id
                c.description = c.description[:120]
                
                meta = {
                    "seq_id": new_id,
                    "organism": organism,
                    "protein": protein,
                    "label": label,
                    "tier": tier,
                    "family": family,
                    "source": "NCBI",
                    "description": c.description,
                }
                all_records.append((c, meta))
                added += 1
            print(f"  -> Added {added} clean sequences")

    print("\n" + "=" * 60)
    print("  UniProt Supplement")
    print("=" * 60)

    for query, label, tier, family in UNIPROT_QUERIES:
        results = fetch_uniprot(query, label, tier, family, max_results=25)
        added = 0
        for rec, meta in results:
            h = seq_hash(str(rec.seq))
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            new_id = f"{'POS' if label==1 else 'NEG'}_UniProt_{rec.id}"
            rec.id = new_id
            meta["seq_id"] = new_id
            all_records.append((rec, meta))
            added += 1
        print(f"  {query[:45]:<45} -> Added {added}")

    print("\n" + "=" * 60)
    print("  Dataset Balancing")
    print("=" * 60)
    
    pos_before = sum(1 for _, m in all_records if m["label"] == 1)
    neg_before = sum(1 for _, m in all_records if m["label"] == 0)
    print(f"  Pre-balance : {pos_before} positive, {neg_before} negative")

    balanced = balance_dataset(all_records)

    pos_after = sum(1 for _, m in balanced if m["label"] == 1)
    neg_after = sum(1 for _, m in balanced if m["label"] == 0)
    print(f"  Post-balance: {pos_after} positive, {neg_after} negative (Total: {len(balanced)})")

    # Save to disk
    fasta_path = output_dir / "sequences.fasta"
    csv_path   = output_dir / "labels.csv"

    SeqIO.write([r for r, _ in balanced], fasta_path, "fasta")

    fieldnames =["seq_id", "organism", "protein", "label", "tier", "family", "source", "description"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, meta in balanced:
            writer.writerow({k: meta.get(k, "") for k in fieldnames})

    # Output Statistics
    df = pd.read_csv(csv_path)
    seqs = {r.id: str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")}
    df["seq_len"] = df["seq_id"].map(lambda x: len(seqs.get(x, "")))

    print("\n" + "=" * 60)
    print("  FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total Sequences: {len(df)}")
    print(f"  Positives      : {(df.label==1).sum()}")
    print(f"  Negatives      : {(df.label==0).sum()}")
    print(f"  Length Stats   : Mean={df.seq_len.mean():.0f}, Min={df.seq_len.min()}, Max={df.seq_len.max()}")

    print("\n  Distribution by Family:")
    family_stats = df.groupby(["family","label"]).size().unstack(fill_value=0)
    family_stats.columns = ["neg","pos"]
    family_stats["total"]    = family_stats["neg"] + family_stats["pos"]
    family_stats["pos_rate"] = (family_stats["pos"]/family_stats["total"]).round(2)
    
    for fam, row in family_stats.sort_values("pos_rate").iterrows():
        flag = " (Excluded from cross-family probing)" if fam in EXCLUDE_FROM_PROBING else ""
        print(f"    - {fam:<20} pos={int(row['pos']):>3}  neg={int(row['neg']):>3}  rate={row['pos_rate']}{flag}")

    with open(output_dir / "stats.txt", "w") as f:
        f.write(df.to_string())

    print("\n" + "=" * 60)
    if len(df) >= 600:
        print(f"  Status: {len(df)} sequences — Data pipeline complete and ready for extraction.")
    else:
        print(f"  Status: {len(df)} sequences — Volume is low. Investigate missing pulls before extraction.")
    print("=" * 60)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Email address for NCBI Entrez")
    parser.add_argument("--output_dir", default="./data", help="Output directory for FASTA and CSV files")
    args = parser.parse_args()
    run_pipeline(email=args.email, output_dir=args.output_dir)
