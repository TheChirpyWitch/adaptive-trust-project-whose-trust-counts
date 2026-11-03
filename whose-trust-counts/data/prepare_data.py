"""Build clean, de-identified analysis files from the three raw datasets.

Inputs (data/raw/, downloaded 2026-10-23):
  pennycook2019/  Pennycook & Rand (2019) PNAS, OSF https://osf.io/6bptd/
                  lay trust ratings of 60 outlets (Study 1: MTurk, Study 2: Lucid)
  lin2023/        Lin et al. (2023) PNAS Nexus, github.com/hauselin/domain-quality-ratings
                  aggregate expert quality score pc1 in [0,1]
  robertson2018/  Robertson et al. (2018) PACM HCI, partisan audience scores in [-1,1]

Outputs (data/derived/):
  ratings.csv  one row per (respondent, outlet): study, rid, domain, trust in [0,1],
               familiar (0/1), rep (0/1 forced-choice party), cons in [0,1]
  outlets.csv  one row per outlet: domain, quality (Lin pc1), lean (Robertson, mapped
               to [0,1]), factcheck in [0,1] (Study 2 outlets only), mbfc_slant in [0,1]

Only the rating, familiarity and political-identity columns are kept; IP addresses,
locations, zip codes, free-text comments and IDs are dropped.

Requires openpyxl (only this script; the analysis reads the CSVs).
"""
import csv
import os
import re

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "derived")


def sheet_rows(path, sheet):
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    return [str(c) for c in rows[0]], rows[1:]


def num(x):
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    return float(x)


def outlet_key(path, sheet):
    _, rows = sheet_rows(path, sheet)
    key = {}
    for i, r in enumerate(rows[:60]):
        key[i + 2] = str(r[1]).strip().lower()  # item numbers run 2..61
    return key


def read_study(name, fname, sheet, keysheet, party_col, party_split):
    path = os.path.join(RAW, "pennycook2019", fname)
    key = outlet_key(path, keysheet)
    head, rows = sheet_rows(path, sheet)
    col = {h.lower(): i for i, h in enumerate(head)}
    out = []
    rid = 0
    for r in rows:
        party = num(r[col[party_col]])
        if party is None:  # same exclusion as the original scripts
            continue
        rep = int(party > party_split) if party_split is not None else int(party == 2)
        ec, sc = num(r[col["economic_conserv"]]), num(r[col["social_conserv"]])
        cons = None if ec is None or sc is None else (ec + sc - 2) / 8
        rid += 1
        for item, dom in key.items():
            t = num(r[col[f"trust_{item}"]])
            if t is None:
                continue
            fam = num(r[col[f"familiarity_{item}"]])
            out.append(dict(study=name, rid=f"{name}_{rid}", domain=dom,
                            trust=(t - 1) / 4, familiar=None if fam is None else int(fam),
                            rep=rep, cons=cons))
    return out, rid


def factchecker():
    """Mean of the 8 professional fact-checkers' trust ratings (Study 2 outlet list)."""
    path = os.path.join(RAW, "pennycook2019", "factchecker.xlsx")
    key = outlet_key(path, "Key")
    head, rows = sheet_rows(path, "Experts data")
    col = {h.lower(): i for i, h in enumerate(head)}
    fc = {}
    for item, dom in key.items():
        v = [num(r[col[f"trust_{item}"]]) for r in rows]
        v = [x for x in v if x is not None]
        fc[dom] = (sum(v) / len(v) - 1) / 4 if v else None
    return fc


def mbfc_slant():
    """Political slant (1-7, MBFC) hard-coded in the authors' Study 2 script."""
    path = os.path.join(RAW, "pennycook2019", "study2.do")
    key = outlet_key(os.path.join(RAW, "pennycook2019", "study2.xlsx"), "Key")
    sl = {}
    for m in re.finditer(r"replace slant=(\d) if source==(\d+)", open(path).read()):
        sl[key[int(m.group(2)) + 1]] = (int(m.group(1)) - 1) / 6
    return sl


def load_table(path, val, keycol="domain"):
    with open(path) as f:
        return {r[keycol].strip().lower(): float(r[val]) for r in csv.DictReader(f) if r[val]}


def lookup(table, dom):
    """Exact match, then without path or 'www.' (no registered-domain fallback:
    it maps abc.go.com to Disney's go.com)."""
    cands = [dom, dom.split("/")[0], dom.removeprefix("www.")]
    for c in cands:
        if c in table:
            return table[c], c
    return None, None


def main():
    os.makedirs(OUT, exist_ok=True)
    s1, n1 = read_study("s1", "study1.xlsx", "data", "key", "demrep", None)
    s2, n2 = read_study("s2", "study2.xlsx", "Lucid data", "Key", "demrep_c", 3)
    ratings = s1 + s2
    with open(os.path.join(OUT, "ratings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratings[0]))
        w.writeheader()
        w.writerows(ratings)

    lin = load_table(os.path.join(RAW, "lin2023", "domain_pc1.csv"), "pc1")
    rob = load_table(os.path.join(RAW, "robertson2018", "bias_scores", "bias_scores.csv"), "score")
    fc, sl = factchecker(), mbfc_slant()
    doms = sorted({r["domain"] for r in ratings})
    rows, report = [], []
    for d in doms:
        q, qm = lookup(lin, d)
        b, bm = lookup(rob, d)
        rows.append(dict(domain=d, quality=q, lean=None if b is None else (b + 1) / 2,
                         factcheck=fc.get(d), mbfc_slant=sl.get(d),
                         in_s1=int(any(r["domain"] == d for r in s1)),
                         in_s2=int(any(r["domain"] == d for r in s2))))
        if qm not in (None, d) or bm not in (None, d):
            report.append(f"  {d}: quality<-{qm}  lean<-{bm}")
    with open(os.path.join(OUT, "outlets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"respondents: study1={n1} study2={n2}; ratings={len(ratings)}")
    print(f"outlets={len(doms)}  with quality={sum(r['quality'] is not None for r in rows)}"
          f"  with lean={sum(r['lean'] is not None for r in rows)}"
          f"  with both={sum(r['quality'] is not None and r['lean'] is not None for r in rows)}")
    print("fuzzy matches:\n" + "\n".join(report))
    print("missing quality:", [r["domain"] for r in rows if r["quality"] is None])
    print("missing lean:", [r["domain"] for r in rows if r["lean"] is None])


if __name__ == "__main__":
    main()
