# 데이터 — 저장소에 없는 것과 구하는 방법

이 폴더의 내용물은 저장소에 넣지 않았다(약 17GB, 전부 무료 공개 데이터). 저장소만 클론해도 되는 것과 안 되는 것은 이렇다.

| 클론 직후 되는 것 | 필요한 것 |
|---|---|
| 데스크톱 계산기 `python -m app`, 명령줄 `python scripts/what_if.py --ticker NVDA` | `app/data/`(최신 월 축약 테이블, 저장소에 포함) |
| 단위 테스트 `python -m pytest tests -q` | 없음(합성 데이터) |
| 파이프라인 재계산 `python run_sfv.py`, 리포트 재생성, 예측 검정 | 아래 원본 파일 전부 + 약 1시간 |

## 원본 파일과 출처

모두 무료이고 API 키가 필요 없다. SEC 다운로드는 User-Agent에 연락처를 요구하므로 먼저 환경변수를 둔다.

```bash
export SEC_CONTACT="이름 이메일@example.com"
```

| 놓을 곳 | 무엇 | 어디서 |
|---|---|---|
| `data/raw/companyfacts.zip` | 전 상장사 XBRL 재무 데이터(약 1.4GB) | SEC EDGAR 벌크: `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` |
| `data/raw/submissions.zip` | 제출 이력·SIC·티커(약 1.5GB) | SEC EDGAR 벌크: `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip` |
| `data/raw/company_tickers.json`, `company_tickers_exchange.json` | CIK↔티커 | `https://www.sec.gov/files/company_tickers.json`, `https://www.sec.gov/files/company_tickers_exchange.json` |
| `data/raw/13f_bulk/*.zip` | 13F 구조화 데이터, 분기별 zip, 2013Q2~ | SEC Form 13F data sets: `https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets` |
| `data/raw/nport/*_nport.zip` | N-PORT 구조화 데이터, 분기별 zip, 2019Q4~(약 11GB) | SEC Form N-PORT data sets: `https://www.sec.gov/data-research/sec-markets-data/form-n-port-data-sets` |
| `data/raw/ftd/cnsfails*.zip` | 결제 불이행(FTD) 파일, 반월별, 2009-07~ | SEC Fails-to-deliver data: `https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data` |
| `data/raw/macro/DGS10.csv` | 10년 국채 수익률(일별) | FRED: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` |
| `data/raw/macro/histimpl.html` | 내재 주식위험프리미엄(연도별) | Damodaran, NYU Stern: `https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html` |
| `data/raw/intangibles/ewens_peters_wang_2023.csv` | 산업별 무형자산 자본화 파라미터 | Ewens·Peters·Wang, `https://github.com/michaelewens/Intangible-capital-stocks` |
| (자동) | 월말 주가·분할·컨센서스 | yfinance. `sfv.prices`가 받는다 |

13F 벌크를 다른 곳에 두면 환경변수 `SFV_MACRO_DATA`로 그 상위 폴더를 가리킨다(`<SFV_MACRO_DATA>/13f_bulk/*.zip`).

## 재계산 순서

```bash
python run_sfv.py --check          # 무엇이 없거나 오래됐는지
python run_sfv.py --group ingest   # 원본 파싱 (xbrl_extract는 배치로, docs/CODE_MAP.md 참고)
python run_sfv.py                  # core: 펀더멘털 → 종목별 표 (약 11분)
python run_sfv.py --group test     # 사전 등록 예측 검정 (기록 재생성)
python run_sfv.py --group extra    # 분석 리포트
```

`data/parquet/`에 중간 산출물이 쌓이고, `reports/`의 리포트와 `app/data/`의 축약 테이블이 갱신된다. `reports/selfcheck.md`에 FAIL이 있으면 산출물을 믿지 말 것.

## 주의

- 이 머신 기준으로 가용 메모리 3GB에서 돌도록 짜여 있어 단계마다 별도 프로세스로 돈다. 메모리가 넉넉하면 더 빠르다.
- `xbrl_extract`는 약 25분이 걸리고 중간에 끊으면 안 된다. 스테이징 후 원자적 교체를 하므로 끊겨도 기존 추출은 남지만, 다시 돌려야 한다.
- 거시 파일(DGS10, histimpl)이 오래되면 최신 월이 직전 값으로 평가되고 `macro_stale` 열과 selfcheck 경고로 표시된다.
