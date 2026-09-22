# 결과 문서 색인 (SFV 구조적 적정가치 모형)

네 곳에 기록이 나뉘어 있다.

| 무엇 | 어디 |
|---|---|
| **프레임워크 현재 사양** (모형식, 데이터·시점 규칙, 검증 위계, 읽는 법, 원안 대비 변경 이력, 한계) | `../FRAMEWORK.md` |
| **튜토리얼** (기초 개념부터 메커니즘까지: 현재가치·DCF·잔여이익모형·감쇠·무형자산·지속성 추정·역산·정방향·사용 순서·용어) | `../TUTORIAL.md` |
| 원안·사전 등록 판정 기준·단계별 진행 상태와 판정(결정 로그) | `../docs/design_spec_20260905.md` (2.5절, 5-1절) |
| 코드 → 산출물 → 내용의 대응, 기호·계산식, 설계 원칙, 알려진 한계 | `../docs/CODE_MAP.md` |
| 단계별 결과 리포트(숫자·표) | `reports/` (이 색인). 실행 로그 `reports/logs/`는 저장소에 넣지 않았다 |

작업 기간 2026-09-05 ~ 2026-09-12. 모든 리포트 상단에 생성일과 사용 데이터 기간이 있다.

## 단계별 리포트

| 단계 | 리포트 | 내용 | 판정·핵심 |
|---|---|---|---|
| 0 데이터 | `phase0_coverage.md` | XBRL·13F·N-PORT·FTD·가격 커버리지, 단위 오류 방어 | 유니버스 확정 |
| 1 모듈 F | `phase1_module_f.md`, `phase1_persistence.md` | 무형자산 자본화, 초과 ROE 지속성 추정(FM·확장 윈도우), RIM V_F, ICC | 지속성 EW 0.56 / 상위 5% 0.84; 밸류에이션 2014-06~ |
| 2 모듈 D | `phase2_module_d.md` | 13F·N-PORT 수요 변수, FIT 승수 | FIT 승수 0.38 (t 2.1) |
| 2 보충 | `phase2_fit_by_float.md` | 흐름 승수가 유통량이 적을수록 큰가 — 패시브·기관·지수펀드 지분, 유통량 대용치, 규모 3분위별 승수(2020-01~2026-09) | 승수 1~5로 유의(흐름은 가격을 움직임)하나 유통량·패시브 지분에 따른 단조 패턴 없음, 상호작용 |t| ≤ 1 |
| 3 모듈 S | `phase3_decomposition.md`, `phase3_validation.md` | log(P/V_F) 분해, 사전 등록 예측 검정 (2014-06~2026-09, 46분기) | **폐기 기준 충족** — 어느 성분도, 어느 벤치마크도 예측력 없음 |
| 3 보충 | `phase3_momentum.md` (요약), `phase3_decomposition_mom.md`, `phase3_validation_mom.md`, `phase5_backlog_robustness_mom.md` | 모멘텀 성분 δ_M 분리 (사후 스펙) | 잔차~모멘텀 0.30→0.02; 전체 표본 예측력 여전히 0; RPO 동일가중만 강화 |
| 3 보충 | `phase3_vw_reversal.md` | 시총가중 역전의 정체 | 초대형 승자 몇 종목(엔비디아·아마존·테슬라); 상위 50 안에서는 방향 없음; 규모 중립화로 안 고쳐짐 |
| 3 보충 | `phase3_snapshot_verdict_2025.txt`, `phase3_snapshot_verdict_2026ytd.txt`, `phase3_snapshot_ytd2026_check.txt` | 연말 판정 스냅샷과 모멘텀·특성 분리, V_F 선행/동행 | 2025 역전은 모멘텀·성장 성분; V_F는 주가를 뒤따름 |
| 3 보충 | `phase3_decomposition_momexp.md`, `phase3_validation_momexp.md` | 기대 성분 δ_E(log(V_gr/V_F)) 추가 스펙 | δ_E 분산 몫 15%, 계수 1.64; 판정 여전히 폐기 |
| 4 시장 수준 | `phase4_implied.md`, `phase4_implied_coe.md` | 시장 내재 지속성 ω*, T*, 프론티어, 반사실; 감쇠 목표 = 자본비용 대안 | ω* 0.92→0.960 (산업 모드, 반감기 17년), 자본비용 모드는 평평; 금리로 설명 안 됨 |
| 5 백로그 | `phase5_backlog.md`, `phase5_backlog_robustness.md` | RPO가 잔차·미래 매출을 설명하는지, 백로그 제거 잔차의 예측력 | 백로그 프리미엄은 실체 있는 기대; 잔차는 계약형 중소형주 동일가중에서만 예측(t −5.4), 시총가중 반대 |
| 6 백로그 가치 반영 | `phase6_backlog_fit.md`, `phase6_backlog_value.md`, `phase6_channel.csv`, `phase6_value_gr.md`, `phase6_value_grbl.md` | ROE 채널(효과 미미) vs 성장 채널(2단계 성장 RIM) | 종목별 판정은 합리화되나 집계 프리미엄은 불변(0.73→0.77) |
| 7 가격이 요구하는 것 | `phase7_expectations.md`, `phase7_sensitivity.md` | 종목별 ω*·T*·g*_10y 역산, 정규화 이익 기준 g*와 이익 국면 표시(1절), 수익률 축 r*(g)(1-1절), 가격이 요구한 성장의 실현 검증과 기저율(4-1절 구간별·기간별, 4-2절 조건부 로짓·규모별·직전 성장별), 지속성 민감도 | g*_10y는 실현 성장 예측(t 17)하나 수익률 예측 없음; 요구 20~30% 구간 3년 달성 9%; 성장 15% 전망에서 가격이 주는 수익률 중앙값 7.5%(상위 50은 6.2%); ERP 평균이면 g* +2.4%p. 2026-09 시총가중 g* 20.3%, 내재 ω* 0.960 |
| 도구 | `sfv_report.html`, `sfv_latest.csv`, `selfcheck.md`, `code_review.md` | 종목별 결정표(정렬·검색 가능한 단일 HTML): g* 하한·상한, 수익률 축(성장 10~25%), 모형 할인율, g* 평균ERP, 3년 달성 확률; 데이터 품질 점검 37개; 전체 코드 리뷰 기록, 2026-09-12 추출 중단 사고 기록, 문서·HTML 결함 기록, 2026-09-13 유니버스 누락 결함(SEC 기록 지연·대표 증권 규칙) 기록 | 리뷰: 계산 오류 없음, 운영 결함 4개 + 문서 오류 6건 + HTML 필터 결함 1건 + 유니버스 결함 2건 수정 |
| 계산기 | `app/` (`python -m app`), `app/data/` | 데스크톱 창(PySide6, 2026-09-13): 종목 검색 → 전제(역산 표·수익률 축·기저율·격자), 목표주가(내 경로 → 주당 목표가·수익률·가치 구성, 적자 기업 풀기, 컨센서스), 추이, 매수일 대조, 수기 입력(유니버스 밖 기업), 도움말. 주가·금리 실시간 갱신(실적은 패널). 계산은 `sfv/calc.py`로 명령줄과 공유 | `app/selftest.py` 38개 점검 통과(`--live` 포함); 명령줄 출력 리팩터링 전후 바이트 동일 |
| 민감도 | `model_sensitivity.md` | 수요 측정 오차가 전파되는가, 경계값은 얼마나 자주 걸리는가 | 수요 제외 시 잔차 상관 0.98; 경계 걸림 시총가중 68%(최신 월 84%) |
| 민감도 | `intangible_sensitivity.md` | 자본화 파라미터(EPW, Li·Hall 두 열, 단일 상각률, 판관비 제외)를 바꿔 전 과정 재계산; 종목별 구간(`expectations_band.parquet`)과 규모별 구간 폭 | 시장 내재 ω* 0.952~0.959 불변; 순위상관 0.92~0.98; 초대형주 g* 폭 1.5~4.1%p이나 유니버스 전체 중앙값 5.5%p(201위 이하 6.1%p, p90 24%p) |
| 구조도 | `sfv_structure.png` | 도구의 구조 한 장: 원본 자료 → 정리 → 가치 엔진 → 분해(진단) → 역산(핵심) → 정방향(목표가) → 출력, 오른쪽에 핵심 생각(가격 하나·미지수 둘), 감쇠의 뜻, 믿을 것과 참고할 것, 적자·전환 기업 처리, 한계, 아래에 읽는 순서 | `scripts/make_structure_diagram.py`로 재생성 |
| 요약 | `sfv_framework.png` | 프레임워크 한 장 요약 이미지: 항등식, V_F 수식과 산업별 자본화 파라미터, 지속성 추정, 분해, 성장 단계, 역산, 최신 수치(g* 구간), 읽는 법, 데이터·시점 규칙 | `scripts/make_framework_poster.py`로 재생성. 검정은 사용자 결정으로 절이 아니라 한계로만 표기 |

## 재현 방법

`python run_sfv.py --check`로 무엇이 오래됐는지 보고, `python run_sfv.py`로 core 체인을 다시 계산한다(약 7분).
실행기가 아래 순서와 파일 의존성을 알고 있으므로 손으로 순서를 맞출 필요는 없다. 단계 목록은 `python run_sfv.py --list`.

## 단계 순서 (참고, 각 모듈은 `python -m sfv.<모듈>`)

```
xbrl_extract → xbrl_panel → submissions/idmap → f13 → nport → ftd → prices → universe          (단계 0)
intangibles → persistence → rim [--roe-star coe --out ...]                                         (단계 1, 4 대안)
demand → returns                                                                                   (단계 2, 3)
decompose [--momentum] [--expectation] → validate [--decomp-file ... --label ...]                  (단계 3, 보충)
implied [--rim-file ... --label ...]                                                               (단계 4)
xbrl_extract --tags RPO... → backlog → scripts/phase5_robustness.py [--label mom]                  (단계 5)
backlog_value → growth_rim [--backlog] → scripts/phase6_channel.py, scripts/phase6_report.py       (단계 6)
expectations → scripts/phase7_sensitivity.py                                                       (단계 7)
scripts/phase3_vw_reversal.py, scripts/verdict_vs_momentum.py, scripts/ytd2026_check.py            (단계 3 보충 스냅샷)
scripts/intangible_sensitivity.py → selfcheck → report                                             (g* 구간, 점검과 출력)
```

## 판정의 위계

1. **사전 등록 판정(단계 3, `phase3_validation.md`)이 기준이다: 폐기.** 이후의 모멘텀·기대 성분 스펙, RPO 표본 결과, 포트폴리오 규칙은 전부 사후·탐색적이며 표본외 확인 전에는 채택 근거가 아니다.
2. 시장 수준 진단(단계 4)과 종목별 요구 기대(단계 7)는 예측 모형이 아니라 "가격이 전제하는 가정"을 읽는 도구다. 그 전제를 믿느냐는 사용자의 판단이다.
3. 초대형주(시총 상위 50, 2,000억 달러 초과)의 횡단면 판정은 검증된 효력이 없다(`phase3_vw_reversal.md`). 그 구간은 단계 7의 표로 읽는다.

## 미구현·후속 후보

- 폴리마켓 등 예측시장과의 비교: 가격 내재 성장률을 사건 확률로 바꾸는 브리지 필요. `phase7_expectations.md` 4-1절의 달성 기저율이 같은 단위의 출발점.
- 금융업 편입, ETF 기반 흐름 변수(둘 다 원안에서 2차로 미룬 항목).
- RPO 표본 잔차 신호와 초대형주 제외 규칙의 표본외 모니터링 — 사용자 결정(2026-09-12)으로 하지 않는다. 용도가 예측이 아니다.

## 종목 판단의 규칙 (2026-09-12 확정)

무형자산 자본화 파라미터를 문헌의 여섯 조합으로 바꾸면 종목 가치가 중앙값 46% 움직인다(초대형주 여섯 종목은 18~89%).
같은 변형에서 횡단면 순위상관은 0.92~0.98로 유지되고 시장 내재 ω*는 0.952~0.959로 불변이다. 요구 성장 g*의 구간 폭은
초대형주에서 1.5~4.1%p로 좁지만 유니버스 전체로는 중앙값 5.5%p이고, 시총 상위 200(중앙값 3.1%p, 5%p 안 70%)과
201위 이하(중앙값 6.1%p, p90 24%p, 5%p 안 41%)가 크게 다르다(`intangible_sensitivity.md`). 그래서:

1. **요구 기대를 읽는다.** 점 추정 가치가 아니라 g*·ω*·T*로 판단한다.
2. **상대 비교는 믿는다.** "어느 종목이 더 비싼가"는 파라미터에 좌우되지 않는다.
3. **구간을 함께 읽는다.** g* 하한·상한이 종목별 표에 실려 있다. 초대형주는 구간이 좁아 점으로 읽어도 되는 경우가 많지만 중소형주는 구간이 기본이다. 자기 기대가 그 구간 안이면 판단을 보류한다.
