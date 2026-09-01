# Change pairs

Eight before/after pairs mined from the reBEN LMDB by `scripts/find_change_pairs.py`,
ranked by measured change in water extent. Real Sentinel-2, five bands, 10 m GSD.
Regenerate or widen the search with `--scan` and `--top`.

Upload the two files of one pair, earlier date first, and ask about water.

| cell | place | water t1 → t2 | months | ice risk |
|---|---|---|---|---|
| `T35VNJ_44_63` | Finland | 8.8% → 81.8% | Nov → May | **yes** |
| `T34VDN_82_75` | Finland | 26.7% → 99.6% | Sep → Apr | **yes** |
| `T35VPK_22_25` | Finland | 75.3% → 23.3% | Jul → Sep | no |
| `T35VNK_14_01` | Finland | 16.6% → 66.3% | Nov → May | **yes** |
| `T29SNC_69_83` | Portugal | 25.9% → 67.8% | Aug → May | no |
| `T34TEQ_08_01` | Serbia | 26.0% → 65.2% | Aug → Apr | **yes** |
| `T29SNB_82_87` | Portugal | 86.9% → 52.3% | Oct → Mar | yes |
| `T34VDR_79_47` | Finland | 100.0% → 68.7% | Aug → May | no |

Verified end to end: `T29SNC_69_83` returns *"water increased by 60.32 ha (161.4% of its
earlier extent)"*.

## Read the ice column before demoing

NDWI reads a **frozen** lake as not-water — ice is bright in near-infrared where open water
is dark. A boreal winter/summer pair therefore shows water appearing from nothing, and the
honest description is "the lake thawed", not "the lake filled".

The first version of this search returned eight Finnish cells going from ~0% to ~90%
water, all neighbours in one tile. They were lake ice. Two filters now apply: water must be
present at **both** dates, which selects a change in extent rather than a change in phase,
and only one pair per tile is kept, so eight results are eight places rather than one lake
reported eight times. `possible_ice` flags any pair touching Nov–Apr; it is a flag, not a
rejection, since a winter date can carry partial ice even with open water present.

**For a demo, prefer the `ice risk = no` rows.** `T29SNC_69_83` (Portugal, Aug → May) is
the safest: a Mediterranean reservoir refilling, no freezing involved, and the same
physical story as the main demo pair.

## What this is not

BigEarthNet is a land-cover classification dataset, not a change-detection one. These pairs
exist because a patch id encodes tile, cell and acquisition date, and the same 1.2 km cell
is often imaged two to four times. The change is real but **seasonal** — reservoirs,
wetlands, crops. It is not the urban-expansion before/after that SECOND provides, and a
pair from here should never be presented as one.
