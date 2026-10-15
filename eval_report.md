# Eval report: dev set (rules only)

- Field-level accuracy: **66/66 = 100%**
- Auto-accepted (conf >= 0.75): 49/66 fields; precision **49/49 = 100%**

## Per-field accuracy

| field | correct | mean conf |
|---|---|---|
| co_number | 6/6 | 0.85 |
| project_name | 6/6 | 0.84 |
| date | 6/6 | 0.78 |
| requested_by | 6/6 | 0.81 |
| submitted_to | 6/6 | 0.85 |
| description | 6/6 | 0.78 |
| total_amount | 6/6 | 0.88 |
| currency | 6/6 | 0.79 |
| schedule_impact_days | 6/6 | 0.75 |
| status | 6/6 | 0.77 |
| line_items | 6/6 | 0.75 |

## Calibration (fields with a value)

| confidence | fields | accuracy |
|---|---|---|
| 0.00-0.50 | 3 | 100% |
| 0.50-0.75 | 13 | 100% |
| 0.75-0.90 | 7 | 100% |
| 0.90-1.00 | 42 | 100% |

## Silent errors (wrong but auto-accepted): 0


## Errors caught by review: 0


## Document-level checks

- `co_006_multi_co_log` expected `multiple_change_orders` -> PASS
- `co_008_scanned_image` expected `no_text_layer` -> PASS

---

# Eval report: holdout set (rules only)

- Field-level accuracy: **46/66 = 70%**
- Auto-accepted (conf >= 0.75): 40/66 fields; precision **38/40 = 95%**
- Error catch rate (wrong fields sent to review): **18/20 = 90%**

## Per-field accuracy

| field | correct | mean conf |
|---|---|---|
| co_number | 5/6 | 0.76 |
| project_name | 4/6 | 0.54 |
| date | 6/6 | 0.68 |
| requested_by | 4/6 | 0.83 |
| submitted_to | 3/6 | 0.55 |
| description | 5/6 | 0.67 |
| total_amount | 3/6 | 0.67 |
| currency | 6/6 | 0.86 |
| schedule_impact_days | 3/6 | 0.30 |
| status | 3/6 | 0.45 |
| line_items | 4/6 | 0.47 |

## Calibration (fields with a value)

| confidence | fields | accuracy |
|---|---|---|
| 0.00-0.50 | 6 | 67% |
| 0.50-0.75 | 7 | 43% |
| 0.75-0.90 | 5 | 100% |
| 0.90-1.00 | 35 | 94% |

## Silent errors (wrong but auto-accepted): 2

- `h03_pipe_table` **requested_by**: got `Ironwood Steel Erectors | Owner: Oak Hollow HOA`, expected `Ironwood Steel Erectors` (conf 0.90)
- `h05_tm_tag` **line_items**: got `1`, expected `3` (conf 0.90)

## Errors caught by review: 18

- `h01_pco_markup` **submitted_to**: got `None`, expected `Brightline Homes Inc.` (conf 0.00; not_found)
- `h01_pco_markup` **schedule_impact_days**: got `None`, expected `1` (conf 0.00; not_found)
- `h02_email_approval_cad` **requested_by**: got `Priya Nair`, expected `Castillo Roofing` (conf 0.50; email_sender_person)
- `h02_email_approval_cad` **submitted_to**: got `Ben Castillo <ben@castilloroofing.com>`, expected `Northgate Properties` (conf 0.62; from_email_header)
- `h02_email_approval_cad` **total_amount**: got `None`, expected `-4200.0` (conf 0.00; not_found)
- `h02_email_approval_cad` **schedule_impact_days**: got `None`, expected `0` (conf 0.00; not_found)
- `h02_email_approval_cad` **status**: got `None`, expected `approved` (conf 0.00; not_found)
- `h03_pipe_table` **project_name**: got `None`, expected `Oak Hollow Clubhouse` (conf 0.00; not_found)
- `h03_pipe_table` **submitted_to**: got `None`, expected `Oak Hollow HOA` (conf 0.00; not_found)
- `h03_pipe_table` **status**: got `None`, expected `approved` (conf 0.00; not_found)
- `h03_pipe_table` **line_items**: got `0`, expected `3` (conf 0.00; not_found)
- `h04_aia_deduct` **schedule_impact_days**: got `None`, expected `0` (conf 0.00; not_found)
- `h05_tm_tag` **co_number**: got `None`, expected `4471` (conf 0.00; not_found)
- `h05_tm_tag` **project_name**: got `None`, expected `Cimarron Ridge Park` (conf 0.00; not_found)
- `h05_tm_tag` **description**: got `None`, expected `Repair irrigation main break caused by others (utility locate miss)` (conf 0.00; not_found)
- `h05_tm_tag` **total_amount**: got `1170.0`, expected `1624.6` (conf 0.55; computed_from_line_items)
- `h05_tm_tag` **status**: got `None`, expected `pending` (conf 0.00; not_found)
- `h06_revised_total` **total_amount**: got `15636.0`, expected `14900.0` (conf 0.70; conflicting_candidate:14900.0, matches_line_item_sum)

