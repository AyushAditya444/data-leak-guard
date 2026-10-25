# Bundled data

| File | What | Source |
|---|---|---|
| `upi_handles.txt` | Known UPI handles (the part after `@`), grouped by app | Compiled by hand from public UPI app handles (Phase 2) |
| `pin_directory.tsv.gz` | Every PIN code with its post office, district and state (derived) | India Post, *All India Pincode Directory*, see below |
| `place_stopwords.txt` | Post office names that are everyday words, not used as place signals | Written for this project; extended on the dev split |

## All India Pincode Directory

- Publisher: Department of Posts, Ministry of Communications, Government of India.
- Catalog: <https://www.data.gov.in/catalog/all-india-pincode-directory>
- Resource id: `5c2f62fe-5afa-4119-a499-fec9d604d5bd` (CSV, 165,627 rows).
- Downloaded: 2026-10-25.
- Licence: [Government Open Data License – India](https://www.data.gov.in/Godl), the
  licence under which data.gov.in publishes its catalogues. Attribution: *"Contains data
  from the All India Pincode Directory, Department of Posts, Government of India,
  published on data.gov.in under the Government Open Data License – India."*

The raw CSV is **not** committed (it is 23 MB and only the derived file is needed). To
rebuild the derived file, download the CSV from the catalog into `data/raw/` and run:

```bash
python -m leakguard.detect.gazetteer build --csv data/raw/<file>.csv
```

What the build keeps: PIN, post office name with its office type removed ("Kothrud S.O"
→ "Kothrud"), district and state, deduplicated and sorted. Latitude, longitude, circle,
region, division and delivery status are dropped. A district or state given as `NA` is
kept as empty, so the PIN still counts as valid.
