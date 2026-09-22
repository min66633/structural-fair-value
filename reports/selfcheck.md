# 데이터 품질 점검 (selfcheck)

> 실행 2026-09-17 11:50. 검사 37개 — 통과 37, 실패 0, 경고 0, 건너뜀 0. 실패 없음.

FAIL은 산출물을 믿을 수 없다는 뜻이고, WARN은 입력이 오래됐거나 값이 경계에 있다는 뜻이다. SKIP은 해당 단계를 아직 돌리지 않은 것이다.


## 키 유일성

| status   | check                                             | detail                                                                               |
|:---------|:--------------------------------------------------|:-------------------------------------------------------------------------------------|
| PASS     | fund_quarterly.parquet unique on cik+end          | 443,336 rows, 0 duplicate keys                                                       |
| PASS     | intangibles_quarterly.parquet unique on cik+end   | 443,336 rows, 0 duplicate keys                                                       |
| PASS     | universe_monthly.parquet unique on cik+month      | 802,712 rows, 0 duplicate keys                                                       |
| PASS     | omega_firm_year.parquet unique on cik+asof_year   | 21,763 rows, 0 duplicate keys                                                        |
| PASS     | rim_monthly.parquet unique on cik+month           | 248,557 rows, 0 duplicate keys                                                       |
| PASS     | demand_monthly.parquet unique on cik+month        | 325,283 rows, 0 duplicate keys                                                       |
| PASS     | returns_quarterly.parquet unique on cik+qend      | 172,672 rows, 0 duplicate keys                                                       |
| PASS     | decomp_monthly.parquet unique on cik+month        | 248,557 rows, 0 duplicate keys                                                       |
| PASS     | decomp_monthly_momexp.parquet unique on cik+month | 248,557 rows, 0 duplicate keys                                                       |
| PASS     | validation_panel_mom.parquet unique on cik+qend   | 83,943 rows, 0 duplicate keys                                                        |
| PASS     | rim_monthly_grbl.parquet unique on cik+month      | 248,557 rows, 0 duplicate keys                                                       |
| PASS     | expectations_monthly.parquet unique on cik+month  | 248,557 rows, 0 duplicate keys                                                       |
| PASS     | backlog_quarterly.parquet unique on cik+end       | 52,143 rows, 0 duplicate keys                                                        |
| PASS     | xbrl_raw extract is complete                      | 69 part files, manifest says 69 parts / 31,071,791 rows, written 2026-09-12T02:35:40 |
| PASS     | xbrl_rpo extract is complete                      | 15 part files, manifest says 15 parts / 137,431 rows, written 2026-09-13T12:44:32    |

## 시점 규칙 (point in time)

| status   | check                                                       | detail                                                                               |
|:---------|:------------------------------------------------------------|:-------------------------------------------------------------------------------------|
| PASS     | V_F uses only filings available at the month                | 0 rows use fundamentals filed after the month; fundamentals age median 91d, max 397d |
| PASS     | persistence omega usable from 30 June of the following year | 0 rows use a persistence model fitted on data not yet public                         |
| PASS     | 13F holdings used from quarter end + 46 days                | 0 rows use 13F holdings before the 46-day deadline; lag median 92d                   |
| PASS     | quarterly returns are forward and one quarter long          | gap min 90d max 92d                                                                  |

## 항등식

| status   | check                                                      | detail                                                                                                        |
|:---------|:-----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| PASS     | decomposition components + residual reconstruct log(P/V_F) | components 7, max within-month sd 3.1e-16, max |mean eps| 1.1e-14                                             |
| PASS     | premium splits into fundamental expectation and the rest   | max |log(P/V_F) - log(V_gr/V_F) - log(P/V_gr)| = 8.9e-16 on 248,228 rows                                      |
| PASS     | panel returns are the winsorised 13F returns               | max |r1 - winsorised ret_tot| = 0.0e+00 on 73,605 rows                                                        |
| PASS     | 13F implied returns agree with yfinance where both exist   | corr 0.9949 on 121,717 overlapping quarters                                                                   |
| PASS     | downstream tables are subsets of the valuation table       | decomp_monthly_momexp.parquet: subset; expectations_monthly.parquet: subset; rim_monthly_grbl.parquet: subset |

## 커버리지

| status   | check                                             | detail                                                                |
|:---------|:--------------------------------------------------|:----------------------------------------------------------------------|
| PASS     | in-universe firm count in the latest month        | latest month 2026-09-30: 2,015 firms (min over sample 188)            |
| PASS     | valuations in the latest month                    | latest month 2026-09-30: 1,878 valued firms, log(P/V_F) present 99.9% |
| PASS     | required-expectation coverage in the latest month | latest month 2026-09-30: g* 71%, omega* 22%, T* 32%                   |

## 값 범위

| status   | check                                            | detail                                                                                                        |
|:---------|:-------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| PASS     | valuation outputs inside their documented bounds | V_F finite True, >0 99.9%, max omega (raw) 0.950, max omega applied in the fade 0.900, r range [0.032, 0.138] |
| PASS     | no valuation used a stale macro input            | 0 rows valued with a carried-forward interest rate or ERP                                                     |
| PASS     | aggregate premium in a plausible range           | latest value-weighted log(P/V_F) = 0.78 (outside [-1, 2.5] means a broken input, not a market call)           |
| PASS     | no market cap implies a share-count typo         | latest month: 0 firms above $10tn market cap, largest 5.55tn                                                  |

## 입력 신선도

| status   | check                                         | detail                                                                                                                          |
|:---------|:----------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------|
| PASS     | raw input companyfacts.zip                    | last modified 11 days ago (refresh after 120)                                                                                   |
| PASS     | raw input submissions.zip                     | last modified 11 days ago (refresh after 120)                                                                                   |
| PASS     | raw input DGS10.csv                           | last modified 11 days ago (refresh after 45)                                                                                    |
| PASS     | raw input histimpl.html                       | last modified 11 days ago (refresh after 45)                                                                                    |
| PASS     | macro series cover the latest valuation month | ERP through 2025 (need 2025), 10-year yield through 2026-09-30 (need 2026-09-30)                                                |
| PASS     | fundamentals records keep up with filings     | 9% of the latest month's universe is valued on a fundamentals record older than 150 days (SEC companyfacts lag); median age 92d |