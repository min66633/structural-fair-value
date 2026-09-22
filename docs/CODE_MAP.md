# SFV — 구조적 적정가치 모형 (Structural Fair Value)

문서 위계: **소개 `README.md`** / **입문 `TUTORIAL.md`**(기초 개념과 메커니즘, 처음 읽을 것) / **현재 사양 `FRAMEWORK.md`**(모형·데이터 규칙·검증 위계·원안 대비 변경 이력) / 설계 명세 `docs/design_spec_20260905.md`(원안, 사전 등록 기준 2.5절, 단계별 진행·판정 기록 5-1절) / 결과 색인 `reports/INDEX.md` / 이 파일은 코드 → 산출물 → 내용의 대응. 경로는 저장소 루트 기준이다.

```
log P = log V_F + δ_S + δ_T + ε            (원안)
log P = log V_F + δ_S + δ_T + δ_F [+ δ_M + δ_E] + 산업FE + ε   (구현: δ_F 모형보정 특성; 사후 스펙에 δ_M 모멘텀, δ_E 펀더멘털 기대)
```

### 기호 정의
- **P**: 시점 t의 시장가격. 기업 단위 **시가총액**(주가 × 유통주식수, 복수 클래스는 합산)으로 계산한다. 같은 주식수로 나누면 주가/주당가치와 동일하므로 클래스 문제를 피하려고 기업 단위로 통일.
- **V_F**: 같은 시점의 잔여이익모형 자기자본 가치 총액 = 무형자산 조정 장부가 B_adj + 미래 초과이익의 현재가치. P와 같은 단위.
- **시점 규칙**: P는 월말 t 가격. V_F는 t 이전 공시분만 사용(`avail_date ≤ t`). 13F 보유는 분기말 + 46일부터, N-PORT는 제출일부터, 지속성 ω는 회계연도 T의 추정치를 T+1년 6월 30일부터 사용.
- **δ_S, δ_T, ε**: log(P/V_F)를 횡단면 회귀로 분해한 무단위 성분. 0.1이면 가격이 펀더멘털 가치보다 약 10% 높다는 뜻.

### V_F 계산식 (잔여이익모형, 단계 1에서 구현)

```
V_F,t = B_adj,t + Σ_{τ=1..T} (ROE_adj,t+τ − r) · B_adj,t+τ−1 / (1+r)^τ + TV_t
```

| 구성 요소 | 정의 | 출처·추정 |
|---|---|---|
| B_adj | GAAP 자기자본 B + K_int. 유형자산·인수 무형자산·영업권은 B에 이미 포함, K_int는 내부창출 무형자산만 추가 | K_int: R&D 전액·판관비 일부를 영구재고법으로 자본화. 세 파라미터(R&D 상각, 조직자본 상각, 판관비 자본화 몫)는 Ewens·Peters·Wang(Management Science 2025)이 인수 가격으로 식별한 SIC별 추정치 |
| E_adj, ROE_adj | E_adj = E + 당기 자본화 지출 − K_int 상각. ROE_adj = E_adj / B_adj,전기 | XBRL 패널 |
| 초과 ROE 경로 | x_τ = x_∞ + ω^τ · (x_0 − x_∞), x = ROE_adj − ROE*, x_∞ = a/(1−ω)(±10%p 절단), x_0는 ±50%p 절단. ROE*는 FF12 산업 중앙값 조정 ROE (대안: r) | **ω와 a는 가정하지 않고 패널 회귀로 추정**: x_{t+1} = a + (ω_0 + ω'z)·x_t + γ'z + e, z = 무형집약도, R&D 집약도, 매출총이익률 안정성, 로그 규모, 로그 연령, 산업집중도(HHI), 음의 초과 ROE 더미, 시총 상위 20%·5% 더미. 결과 연도 ≤ T 자료로 적합한 확장 윈도우, 추정치는 T+1년 6월 30일부터 사용. 적용 상한 0.90 |
| 장부가 경로 | B_adj,t+τ = B_adj,t+τ−1 · (1 + ROE_adj,t+τ · (1 − 총주주환원율)), 연 성장 ≤ 25% | 청산잉여관계. 환원율 = (배당 + 자사주매입 − 발행) / E_adj, 최근 4분기, [0, 1]로 절단 |
| r | r_f + β · ERP | r_f: 10년 국채 월말, ERP: Damodaran 전년 말 내재 ERP(연 단위), β: 60개월 회귀 (0.5~1.5로 절단) |
| TV_t | T년 이후 초과이익. 기본은 ω 감쇠가 계속된다고 보고 닫힌 형태로 합산: x_T · B_adj,T · ω / (1 + r − ω(1+g)) / (1+r)^T, x_T는 T년의 초과 ROE, g는 장부가 성장률. 보수적 대안: T 이후 초과이익 0 | T=10년 기본. 두 터미널 규칙의 차이를 민감도로 보고 |
| ICC | 위 식에서 V_F 대신 실제 P를 놓고 r에 대해 격자(0~40%, 0.5%p 간격) 역산한 내재 자본비용 | 보조 지표. 약 40%가 0% 하한에 걸린다. 별도의 이익 예측 모형은 쓰지 않는다 |

주의: 청산잉여관계가 성립하면 무형자산 조정은 이론값 V를 바꾸지 않는다. 조정이 바꾸는 것은 가치 중 관측된 장부가에 실리는 몫과 감쇠 가정에 실리는 몫의 비율, 그리고 ω를 추정하는 초과 ROE의 의미다.

| 항 | 뜻 | 데이터 |
|---|---|---|
| P | 시가총액 (주가 × 유통주식수) | yfinance 월별 가격, 13F 내재가격, XBRL dei 주식수 |
| V_F | 무형자산 조정·지속성 추정 잔여이익모형 가치 | SEC XBRL (companyfacts) |
| δ_S | 구조적 수요 프리미엄 (패시브 지분·벤치마킹 강도·순발행) | 13F, N-PORT |
| δ_T | 일시적 수급 압력 (펀드 흐름 유발 매매) | N-PORT 월별 유출입 |
| ε | 잔차 미스프라이싱 | — |

## 실행 — 월별 갱신

```bash
python run_sfv.py --check              # 무엇이 오래됐는지, 원본 입력은 며칠 됐는지
python run_sfv.py                      # core 체인 재계산 + g* 구간 + 점검 + 종목별 표 (약 11분)
python run_sfv.py --group ingest       # 원본 다운로드를 새로 받은 뒤 (중단 금지: 아래 주의)
python run_sfv.py --only rim,decompose # 특정 단계만
python run_sfv.py --from expectations  # 그 단계부터 끝까지
python -m app                          # 데스크톱 계산기 (PySide6 창). app/data가 있으면 그것을, 없으면 data/parquet을 읽음
python app/selftest.py [--live]        # 계산기 자체 점검 (화면 없이 실행; --live는 주가·금리 조회까지)
```

`run_sfv.py`가 단계 순서와 파일 의존성을 알고 있어, 출력이 모든 입력(모듈 소스 파일 포함)보다 새로우면 건너뛴다.
각 단계는 별도 프로세스로 돌린다(이 머신은 가용 메모리가 약 3GB라 한 번에 하나여야 한다).
그룹은 `ingest`(원본 파싱, 수동 갱신 필요), `core`(펀더멘털 → 종목별 표), `test`(사전 등록 예측 검정),
`extra`(분석 리포트)다. **예측 검정은 2014~2026년에 끝난 결과이지 월별 산출물이 아니므로 기본 실행에서 빠져 있다.**
월별 실행은 뒤 단계가 필요로 하는 패널만 만들고 멈추며, 검정 리포트는 기록으로 디스크에 남아 있다.
다시 만들려면 `python run_sfv.py --group test`. 반대로 요구 성장의 파라미터 구간(`intangible_sensitivity`, 약 4분)은
종목별 표의 g* 하한·상한이 거기서 나오므로 core에 들어 있다.

**`ingest` 그룹은 중간에 끊지 말 것.** `sfv.xbrl_extract`의 전체 통과는 약 25분이고, 이 머신은 포그라운드 10분,
백그라운드는 메모리 감시에 걸린다. 그래서 배치로 나눠 돌린다.

```bash
python -m sfv.xbrl_extract --start 0     --limit 5000
python -m sfv.xbrl_extract --start 5000  --limit 5000
python -m sfv.xbrl_extract --start 10000 --limit 5000
python -m sfv.xbrl_extract --start 15000           # 마지막 배치가 매니페스트를 쓰고 디렉터리를 교체한다
```

스테이징 디렉터리에 쓰고 완료 시에만 교체하므로 중단돼도 기존 추출은 그대로 남는다.
`sfv.selfcheck`가 매니페스트와 실제 파트 수를 대조해 잘린 추출을 FAIL로 잡는다.

| 산출물 | 내용 |
|---|---|
| `reports/sfv_report.html` | 종목별 결정표. 정렬·검색·산업 필터, 외부 의존 없음(파일로 열면 됨) |
| `reports/sfv_latest.csv` | 같은 표의 CSV. 요구 성장 g*와 그 상하한(자본화 파라미터 범위) 포함 |
| `reports/selfcheck.md` | 데이터 품질 점검 결과. FAIL이 있으면 산출물을 믿지 말 것 |
| `reports/code_review.md` | 전체 코드 리뷰 기록과 수정 내역 |
| `app/data/` | 데스크톱 계산기가 읽는 축약 테이블(약 19MB, 최신 월). `app_export` 단계가 만들며 파이프라인을 돌리면 같이 갱신된다 |

**수동으로 갱신해야 하는 원본**: `data/raw/companyfacts.zip`, `submissions.zip`, `data/raw/nport/*.zip`,
`data/raw/ftd/*.zip`, 13F 벌크(`data/raw/13f_bulk`), `data/raw/macro/DGS10.csv`(FRED 10년 국채),
`data/raw/macro/histimpl.html`(Damodaran 내재 ERP). 거시 파일이 오래되면 최신 월이 직전 값으로 평가되며
`macro_stale` 열과 selfcheck 경고로 표시된다.

## 단계 0 — 데이터 파이프라인 (전부 무료 소스)

| 모듈 | 입력 | 출력 (data/parquet/) |
|---|---|---|
| `sfv.xbrl_extract` | `data/raw/companyfacts.zip` (1.4GB) | `xbrl_raw/part-*.parquet`, `xbrl_entities.parquet` |
| `sfv.xbrl_panel` | xbrl_raw | `fund_quarterly.parquet` (point-in-time 분기 패널), `shares_dei.parquet` |
| `sfv.f13` | `data/raw/13f_bulk/*.zip` (2013Q2~) | `f13_filings.parquet`, `inst_own_quarterly.parquet` |
| `sfv.nport` | `data/raw/nport/*_nport.zip` (2019Q4~) | `nport_funds.parquet`, `nport_holdings/*.parquet`, `nport_identifiers.parquet` |
| `sfv.ftd` | `data/raw/ftd/cnsfails*.zip` (2009-07~) | `ftd_cusip_symbol.parquet`, `ftd_prices_monthly.parquet` |
| `sfv.submissions` | `data/raw/submissions.zip` | `sec_entities.parquet` (SIC·구명칭·티커) |
| `sfv.idmap` | 위 산출물 | `security_master.parquet` (CUSIP↔CIK↔티커), `ticker_cik.parquet` |
| `sfv.prices` | security_master + yfinance | `prices_monthly.parquet` (현재 상장 종목만, 생존편향) |
| `scripts/phase0_coverage.py` | 전부 | `reports/phase0_coverage.md` |

실행 순서: extract → panel; f13; nport; ftd; submissions → idmap → prices → coverage.
모든 모듈은 `python -m sfv.<module>` 로 실행하며 로그는 `reports/logs/`.

## 단계 1 — 모듈 F (펀더멘털 가치)

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.universe` | `universe_monthly.parquet`, `universe_ciks.parquet` | 기업-월 유니버스: 대표 보통주(최근 120일 안에 거래된 증권 중 거래일 수 최다), 가격(yf/FTD), 분할 기준 환산 주식수, 시총, 재무 연결, 오매칭 검사. 행은 마지막 공시일 + 400일까지 만들어 SEC 기록 지연에도 기업이 사라지지 않게 함. 금융(SIC 6000~6999) 제외하되 SIC 6199로 분류된 채굴·데이터센터 운영사 13곳은 `NONFINANCIAL_OVERRIDES`로 포함 |
| `sfv.intangibles [--params --d-rnd --d-org --sga-share --out]` | `intangibles_quarterly.parquet` | R&D·판관비 자본화. 파라미터 3종(R&D 상각 33~50%, 조직자본 상각 20%, 판관비 자본화 몫 20~51%)은 Ewens·Peters·Wang 산업별 추정치. `--params compustat`은 Li·Hall R&D 상각률, `--params flat`은 단일값 |
| `sfv.persistence` | `omega_firm_year.parquet`, `reports/phase1_persistence.md` | 초과 ROE 지속성 ω: 연도별·FM·시대 풀링, 확장 윈도우 기업별 예측 |
| `sfv.rim` | `rim_monthly.parquet` | V_F(10년 명시 + 감쇠 터미널), ICC, log(P/V_F). 경계: ω≤0.9, 장기 초과 ROE ±10%p, 초기 ±50%p, 장부 성장 ≤25%, 터미널 성장 ≤4% |
| `scripts/phase1_report.py` | `reports/phase1_module_f.md` | 체크포인트 리포트 |

## 단계 2 — 모듈 D (수요·수급)

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.demand` | `demand_monthly.parquet`, `fit_quarterly.parquet` | 13F 기관·big3·확장 패시브 비중(분기말+46일), 순발행 12개월, N-PORT 지수펀드 비중·S&P500 근사(펀드별 최신 보고 월별 이월), FIT(후행 3개월 흐름 창) |
| `scripts/phase2_report.py` | `reports/phase2_module_d.md` | 기술통계, FIT 승수(영향·3개월·12개월), log(P/V_F) 횡단면 관계 |
| `scripts/fit_by_float.py` | `reports/phase2_fit_by_float.md` | 흐름 승수를 패시브·기관·지수펀드 지분, 유통량 대용치(1 − big3 − 지수펀드), 규모의 월별 3분위와 상호작용으로 재추정. 유통량이 적을수록 승수가 크다는 패턴은 없음 |

## 단계 3 — 모듈 S (분해·검증)

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.returns` | `returns_quarterly.parquet` | 13F 내재가격 분기 수익률(상장폐지 포함, yfinance 대비 상관 0.996), 분할 보정, 배당 근사, 퇴출 플래그 |
| `sfv.decompose` | `decomp_monthly.parquet`, `reports/phase3_decomposition.md` | log(P/V_F) = a_t + δ_S + δ_T + δ_F + 산업FE + ε, 월별 횡단면, 분산 몫 |
| `sfv.validate` | `reports/phase3_validation.md` | Fama-MacBeth 예측 회귀(1·2·4분기), 벤치마크(B/M, E/P, Bartram·Grinblatt, ICC), 10분위 롱숏(시총가중 + 동일가중 보충), 표본외 복합점수, 상장폐지 민감도, 사전 등록 판정 |
| `sfv.decompose --momentum` | `decomp_monthly_mom.parquet`, `reports/phase3_decomposition_mom.md` | **사후 스펙(2026-09-06)**: 모멘텀 성분 δ_M(12-1개월·1개월 수익률, yfinance → 13F 내재가격 대체) 추가. `eps_A` = 모멘텀 제거 잔차, `eps_noM_A` = 제거 전 |
| `sfv.validate --decomp-file decomp_monthly_mom.parquet --label mom` | `validation_panel_mom.parquet`, `reports/phase3_validation_mom.md` | 모멘텀 스펙에 같은 검정 적용(탐색적 표시). 결과 요약은 `reports/phase3_momentum.md` |
| `scripts/verdict_vs_momentum.py --t0 --t1 [--decomp-file]` | 콘솔 → `reports/phase3_snapshot_verdict_2025.txt`, `..._2026ytd.txt` | 특정 연말 판정의 이후 수익률을 모멘텀·성장 특성과 분리해 회귀·5분위로 점검 |
| `scripts/ytd2026_check.py --t0 --t1` | 콘솔 → `reports/phase3_snapshot_ytd2026_check.txt` | 연말 판정 → 이듬해 수익률 5분위, V_F 변화의 선행·동행 검사, 대형주 판정표 |
| `scripts/phase3_vw_reversal.py` | `reports/phase3_vw_reversal.md` | 시총가중 역전의 정체: 가중 변형, 종목별 기여 귀속(엔비디아 등), 규모 3분위, 규모 중립 순위, 상위 50·100 부분집합 |

## 단계 4 — 시장 수준 재진단

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.implied` | `implied_monthly.parquet`, `reports/phase4_implied.md` | 시장 내재 지속성 ω*(ΣP = ΣV_F가 되는 공통 ω), 내재 경쟁우위 기간 T*, 할인율-지속성 프론티어, 2014년 고정 반사실 분해, 규모별 실현 지속성과 비교 |

## 단계 5 — 백로그(잔여이행의무) 확장

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.xbrl_extract --tags ... --cik-file` | `xbrl_rpo/part-*.parquet` | 유니버스 기업의 RPO·계약부채 태그 추출 |
| `sfv.backlog` | `backlog_quarterly.parquet`, `reports/phase5_backlog.md` | RPO 성장·두께로 잔차 설명, 미래 매출 예측, 잔차 분할(백로그 설명분 vs 나머지)의 수익률 예측 |
| `scripts/phase5_robustness.py [--label mom]` | `backlog_panel[_mom].parquet`, `reports/phase5_backlog_robustness[_mom].md` | 지평·벤치마크·하위 기간·분위 단조성·시총가중 vs 동일가중. `--label mom`은 모멘텀 제거 잔차 패널로 같은 분석 |

## 단계 6 — 백로그를 가치에 넣기

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.rim --roe-star coe` | `rim_monthly_coe.parquet` | 감쇠 목표 = 자본비용, 장기 초과이익 0인 고전적 대안(하한). `sfv.implied --rim-file ... --label coe`로 내재 지속성도 산출 |
| `sfv.backlog_value` + `sfv.rim --omega-file omega_firm_year_bl.parquet` | `rim_monthly_bl.parquet` | 백로그를 다음 해 초과 ROE 항으로 넣은 기대 모형(효과 미미 — ROE 채널 아님) |
| `scripts/phase6_channel.py` | `reports/phase6_channel.csv` | 백로그는 1~3년 매출·이익 성장을 예측하고 ROE 변화는 예측하지 않음 |
| `sfv.growth_rim [--backlog]` | `rim_monthly_gr.parquet`, `rim_monthly_grbl.parquet` | **2단계 모형**: 1~3년 명시적 이익 성장(과거 성장·규모·산업·백로그로 확장 윈도우 전망) + 4년째부터 감쇠. 최종 백로그 반영 모형 |
| `scripts/phase6_report.py --alt-file ... --label ...` | `reports/phase6_value_*.md` | 대안 모형 vs 기존: 집계·분산, 백로그 두께별, RPO 기업 수익률 예측, 대표 기업 |

## 단계 7 — 가격이 요구하는 기대 (종목별 역산, 도구 출력)

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.decompose --momentum --expectation` | `decomp_monthly_momexp.parquet`, `reports/phase3_decomposition_momexp.md` | 기대 성분 δ_E = 성장 단계 모형이 정당화하는 프리미엄 log(V_gr/V_F)(진단용). 전망치 자체는 특성·산업의 선형결합이라 계수가 폭발해 가치 채널로 넣음 |
| `sfv.expectations` | `expectations_monthly.parquet`, `expectations_calibration.parquet`, `expectations_calibration_cond.parquet`, `reports/phase7_expectations.md` | 종목·월별로 다른 입력을 고정하고 V = P가 되는 값을 역산: ω*(내재 지속성, 반감기), T*(현재 초과 ROE 유지 연수), g*_10y(10년 연간 이익성장, 헤드라인), g*_3y(단계 6 지평), g_model(펀더멘털 전망)과 비교. **수익률 축** `r_star_gXX`(성장 전망 5~30%별로 가격이 주는 연 수익률), **g*_10y_erpavg**(표본 평균 ERP에서의 g*), **p_achieve_3y**(요구 성장·규모·직전 성장·ROE·산업 로짓의 3년 달성 확률), **g*_10y_norm**과 `cycle_flag`(정규화 이익 = 5년 평균 조정 이익률 × 현재 매출 기준의 g*, 후행과 30% 넘게 다르면 이익 국면 표시). 조건부 기저율 표(`expectations_calibration_cond.parquet`)는 규모별과 직전 1년 성장별 두 차원. 검증: g*가 실현 성장을 예측하는지(FM), 요구 성장 달성 기저율(구간별·규모별), g*−g_model의 수익률 예측(무의미) |
| `scripts/phase7_sensitivity.py [--tickers] [--month]` | `reports/phase7_sensitivity.md` | 지속성 ω 하나에 V/P가 얼마나 흔들리는지(엔비디아 0.24→1.26배), 종목별 ω*·반감기 — 초대형주 가치평가가 어려운 이유 |

## 도구 — 점검과 출력

| 모듈 | 출력 | 내용 |
|---|---|---|
| `sfv.selfcheck [--expect-month]` | `reports/selfcheck.md`, `selfcheck_status.json` | 37개 불변식: 키 유일성, 시점 규칙(공시일·13F 46일·ω 사용 시점), 항등식(성분 합, 프리미엄 분할, 절단 수익률), 커버리지, 값 범위, 입력 신선도, 원본 추출 완결성(매니페스트 대조), SEC 기록 지연 비율(10% 초과 시 WARN). FAIL이면 종료 코드 1 |
| `sfv.report [--month]` | `reports/sfv_latest.csv`, `reports/sfv_report.html` | 종목별 결정표. 요구 기대(g*와 구간·ω*·T*) + 수익률 축(성장 10·15·20·25%에서 가격이 주는 수익률) + 모형 할인율과 g* 평균ERP + 3년 달성 확률 + 펀더멘털 전망 + 동료 대비 잔차 + 표시(초대형·적자·지속불가·백로그) + 시장 수준 요약 + 기저율(구간별·규모별) |
| `run_sfv.py` | `reports/logs/run_sfv_*.log` | 의존성 순서 실행기. staleness 기반 건너뛰기, 단계별 별도 프로세스, 실행 요약 |
| `scripts/make_structure_diagram.py [--dpi]` | `reports/sfv_structure.png` | 도구의 구조도(수식 없음): 자료 → 정리 → 가치 엔진 → 분해 → 역산 → 정방향 → 출력의 흐름과 각 단계의 설명, 핵심 생각·한계·읽는 순서. 수치는 산출물에서 읽음 |
| `scripts/make_framework_poster.py [--dpi]` | `reports/sfv_framework.png` | 프레임워크 한 장 요약 이미지: 항등식, V_F와 산업별 자본화 파라미터, 지속성 추정, 분해, 성장 단계, 역산, 최신 수치(g* 구간 포함), 읽는 법, 데이터·시점 규칙. 수치는 파이프라인 산출물에서 읽어오므로 데이터와 어긋나지 않는다 |
| `scripts/what_if.py --ticker T [--omega/--growth/--roe/--discount] [--rev-growth 40,30,20 --margin 55,52,50 --e0 B] [--consensus] [--since D] [--history N]` | 콘솔 | **정방향 시나리오(목표가)**: 연도별 매출 성장과 조정 이익률 경로로 이익 경로를 만들고 그 뒤 추정 감쇠·터미널을 붙여 가치, 주당 목표가, 그 경로에서 가격이 주는 수익률, 가치의 구성(장부·명시 구간·감쇠 구간·터미널), 그 매출 속도가 역사적으로 실현된 비율을 출력. `--e0`는 정규화 출발 이익(십억 달러), `--consensus`는 yfinance 컨센서스 매출 성장(올해·내년)을 경로로 사용. 적자 기업도 매출에서 출발하므로 평가 가능. **적자 기업 역산**: `--solve-growth --margin 경로 --years H`(이익률 경로를 주면 요구 매출 성장), `--solve-margin --rev-growth 경로`(매출 경로를 주면 요구 이익률). 역산 블록에 정규화 이익 기준 요구 성장과 이익 국면(피크·저점·정상)이, 경로의 기저율에는 그 회사의 직전 성장 무리 안 비율이 함께 나온다. 그 밖에 모형 가정을 직접 바꿔 가치를 다시 계산하고, 가격이 요구하는 ω*·g*·T*와 자본화 파라미터 범위, 수익률 축(성장 전망별 가격이 주는 수익률, `--growth`를 넣으면 그 전망의 수익률), 평균 ERP에서의 g*, 요구 성장의 분기별 추이(직전 1년 매출성장과 나란히; `--history 0`이면 2014-06 이후 전체)를 본다. `--since 매수일`은 그날의 전제(요구 성장)와 이후 실현 매출·이익 성장을 누적·연환산으로 대조하고 전제가 어떻게 움직였는지 읽어 준다. `--universe --omega-shift`로 시장 전체 효과 |
| `python -m app [--data DIR]` (`app/main.py`, `app/tabs/*.py`) | 창 | **데스크톱 계산기 (PySide6, 2026-09-13).** 왼쪽 종목 검색 목록, 헤더에 데이터 기준일(패널 월·실적 기준일·주가·국채·ERP·ω 연도)과 표시 칩(초대형·적자·이익국면·재무지연·수기 입력). 탭 여섯: **전제**(맨 위에 기호 없는 요약 서너 문장 — 이 가격이 전제하는 성장과 구간, 그만큼을 요구받은 기업이 실제로 해낸 비율, 두 성장 전망에서의 수익률, 이익 국면·재무지연 경고 — 그 아래 핵심 두 줄과 수익률 축, 나머지(모형 가치·ω*·T*·정규화 기준·평균 ERP 기준·할인율 구성·지속성 격자·기저율 상세)는 '자세히 보기'로 접힘. 각 줄에 마우스를 올리면 용어 풀이), **목표주가**(매출 성장·이익률 경로, 할인율, ω, 환원율, 출발 이익(후행·정규화·직접) → 주당 목표가, 그 경로의 수익률, 가치 구성 차트, 기저율; 적자 기업은 요구 성장·요구 이익률 풀기; 컨센서스 경로 채우기), **추이**(분기별 g*·정규화 g*·할인율·가격/가치 차트와 표), **매수일 대조**(scorecard), **수기 입력**(유니버스 밖 기업을 같은 엔진에; 무형자산 미자본화 편향과 미국 패널 기저율임을 표기), **도움말**(지금 고른 종목의 실제 숫자로 여섯 단계를 걸어가는 안내 + 헷갈리기 쉬운 것 넷 + 요구 성장을 목표주가 탭에 넣어 목표가가 현재가와 같아지는 것을 보여 주는 버튼). `주가·금리 갱신`은 yfinance 최신가와 FRED DGS10을 별도 스레드로 받아 시총(패널 주식수 × 가격비)과 할인율을 바꾸고 즉시 다시 역산한다(실적은 패널 그대로). 계산은 전부 `sfv.calc`라 명령줄과 숫자가 같다. 창을 닫으면 끝나는 프로그램이며 상시 프로세스가 아니다 |
| `sfv.store [--out DIR --month M]` | `app/data/*.parquet`, `macro.json` | 계산기와 명령줄이 공유하는 로더 `Store`. 전체 parquet(어느 월이든) 또는 축약 사본(최신 월)을 같은 속성(firms·hist·quarters·rev_panel·band·base_rates·industry·macro)으로 읽는다. 모듈로 실행하면 축약 사본을 쓴다(`app_export` 단계) |
| `sfv.calc` | 라이브러리 | 종목 단위 계산의 단일 원천: `inversion`(ω*·g*·T*·정규화 g*·구간·수익률 축·평균 ERP g*·격자), `scenario`(경로 → 가치·목표가·수익률·기저율), `solve_required`(적자 역산), `history`·`scorecard`, `with_price`(실시간 주가·금리 반영), `manual_row`(수기 입력 기업을 패널 행으로). 출력은 dict/DataFrame이며 인쇄는 호출자가 한다 |
| `sfv.live` | 라이브러리 | 네트워크 조회: `price`(yfinance, 클래스 주식은 HEIA→HEI-A 재시도), `rf`(FRED DGS10 CSV), `consensus`(yfinance 추정치). 실패해도 예외 없이 사유를 돌려준다 |
| `app/theme.py`, `app/assets/*.svg` | 라이브러리 | 어두운 테마 한 곳: 색 팔레트(`C`·`CHIP`·`LINE`), Qt 팔레트와 스타일시트, matplotlib 설정. Qt는 Fusion 스타일에 팔레트 우선으로 칠하고 스타일시트는 팔레트가 표현 못 하는 것만 담당한다. 스타일시트로 칠한 위젯은 서브컨트롤 그리기가 스타일시트로 넘어가므로 스핀박스·콤보·체크박스의 표시는 `app/assets`의 SVG로 공급한다 |
| `app/selftest.py [--live]` | 콘솔 | 계산기 자체 점검 38개(화면 없이): 종목 넷(흑자 초대형·적자·피크·재무지연)에서 탭마다 값이 나오는지, 탭의 시나리오가 라이브러리와 같은지, 기본값이 모형값 그대로인지, 수기 입력이 패널 종목의 g*를 재현하는지; `--live`는 실제 이벤트 루프에서 주가·금리를 받아 다시 역산하는 경로까지 |
| `scripts/model_sensitivity.py` | `reports/model_sensitivity.md` | 수요 측정 오차가 전파되는가, 경계값은 얼마나 자주 걸리는가 |
| `scripts/intangible_sensitivity.py [--keep]` | `reports/intangible_sensitivity.md`, `expectations_band.parquet`, `intangible_sensitivity_summary.parquet` | 자본화 파라미터를 바꿔(EPW, Li·Hall 두 열, 단일 상각률, 판관비 제외) 전 과정을 다시 계산하고 종목별 g*·ω*·T*·log(P/V)의 구간, 변형별 순위상관, 규모별 구간 폭을 저장. core 그룹(월별), 약 4분, 변형 산출물은 자동 삭제 |

### 설계 원칙
- **Point-in-time**: XBRL 값은 최초 보고치(first-reported)만 쓰고, 행마다 `avail_date`(최초 공시일)를 둔다. 정정치는 `*_last` 열에 따로 보관.
- **분기 흐름 복원**: 같은 시작일을 가진 누적값의 차분(Y−9M=Q4 등). 회계연도 추정 불필요.
- **생존편향**: 검증 수익률의 1차 소스는 13F 내재가격(상장폐지 포함). yfinance는 도구 출력과 보조 검정 전용.
- **단위 오류 방어**: 13F는 필링별 중앙값 내재가격으로 천달러/달러 단위 판별. N-PORT는 (CUSIP, 보고일) 중앙값 가격에서 벗어난 주식수를 가치÷중앙값으로 보정하고 플래그.
- **정정공시**: 13F는 (매니저 CIK, 분기)당 보유 행이 가장 많은 필링 하나만 사용. N-PORT는 /A가 원본을 대체.

### 알려진 한계
- 패시브 지분은 매니저 CIK 기준(big3 = BlackRock·Vanguard·State Street, ext = +Geode·Northern Trust·Schwab·Invesco·BNY) 하한 근사.
- N-PORT 유출입은 2019-08 이후만 공개.
- CUSIP↔CIK 연결 중 이름 매칭(name-fuzzy) 경로는 표본 검수 필요.
- 이 머신은 RAM 8GB라 모든 대용량 처리는 스트리밍·청크 방식.
