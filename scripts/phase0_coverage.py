# -*- coding: utf-8 -*-
"""Phase 0 checkpoint: coverage report across every data source.

Writes reports/phase0_coverage.md. Every number here is measured from the
parquet outputs, with the data period stated next to it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sfv import config as C  # noqa: E402

pd.set_option("display.width", 200)
L = []


def h(t):
    L.append(f"\n## {t}\n")


def p(t=""):
    L.append(t)


def table(df: pd.DataFrame, floatfmt="{:,.3f}"):
    L.append(df.to_markdown(floatfmt=",.3f") if hasattr(df, "to_markdown") else df.to_string())


def main():
    p("# 단계 0 데이터 파이프라인 — 커버리지 리포트\n")
    p(f"> 생성 {pd.Timestamp.today().date()}. 모든 수치는 parquet 산출물에서 실측.")

    # ---------------- XBRL panel
    h("1. XBRL 재무 패널 (companyfacts.zip, 10-K/10-Q, first-reported)")
    E = pd.read_parquet(C.PQ / "xbrl_entities.parquet")
    P = pd.read_parquet(C.PQ / "fund_quarterly.parquet")
    p(f"- companyfacts 엔티티 {len(E):,}개, us-gaap 태그 보유 {int((E.n_tags_usgaap > 0).sum()):,}개")
    p(f"- 패널 행 {len(P):,} (cik×분기말), CIK {P.cik.nunique():,}개, 기간 {P.end.min().date()} ~ {P.end.max().date()}")
    lag = (P["avail_date"] - P["end"]).dt.days
    p(f"- 공시 가용 지연(분기말→최초 공시일): 중앙값 {lag.median():.0f}일, p75 {lag.quantile(.75):.0f}일, p90 {lag.quantile(.9):.0f}일")
    cov = P[["q_revenue", "q_net_income", "q_op_income", "equity", "assets", "q_rnd", "q_sga", "q_cfo", "q_capex",
             "q_buybacks", "q_dividends_paid", "shares_out", "ttm_net_income", "ttm_revenue", "q_interest_expense",
             "goodwill", "intangibles", "debt_lt"]].notna().mean().rename("coverage").to_frame()
    cov["coverage"] = (cov["coverage"] * 100).round(1)
    p("\n항목별 커버리지 (행 기준, %):\n")
    table(cov)
    yr = P.assign(y=P["qend"].dt.year).groupby("y").agg(ciks=("cik", "nunique"), rows=("cik", "size"),
                                                         ni=("q_net_income", lambda s: s.notna().mean() * 100))
    p("\n연도별 CIK 수와 순이익 커버리지(%):\n")
    table(yr.round(1))
    rest = ((P["q_net_income_last"] / P["q_net_income"]) - 1).abs()
    rest = rest[np.isfinite(rest) & (P["q_net_income"].abs() > 1e6)]
    p(f"\n- 순이익 정정 크기 |최종/최초−1| (|NI|>100만$): 중앙값 {rest.median():.4f}, p90 {rest.quantile(.9):.4f}, "
      f"p99 {rest.quantile(.99):.3f}, 5% 초과 비율 {(rest > 0.05).mean()*100:.1f}%")

    # ---------------- 13F
    h("2. 13F 기관 보유 (구조화 데이터셋, 정정공시 중복 제거)")
    A = pd.read_parquet(C.PQ / "inst_own_quarterly.parquet")
    F = pd.read_parquet(C.PQ / "f13_filings.parquet")
    p(f"- 필링 {len(F):,}건 (CIK×분기 중복 제거 후), 매니저 CIK {F.cik.nunique():,}개, 기간 {F.period.min().date()} ~ {F.period.max().date()}")
    p(f"- (분기, CUSIP) 행 {len(A):,}, CUSIP {A.cusip.nunique():,}개, 분기 {A.period.nunique()}개")
    q = A.groupby("period").agg(cusips=("cusip", "size"), holders_med=("n_holders", "median"),
                                inst_value_tn=("inst_value_usd", lambda s: s.sum() / 1e12))
    top = A[A["n_holders"] >= 20]
    q["big3_share_med_top"] = top.groupby("period")["big3_share_of_inst"].median()
    q["ext_share_med_top"] = top.groupby("period")["ext_share_of_inst"].median()
    p("\n분기별 (보유기관 20곳 이상 종목의 big3·확장 패시브 비중 중앙값은 기관보유 대비):\n")
    table(q.iloc[::4].round(3))
    p(f"\n- 마지막 분기 {q.index.max().date()}: CUSIP {int(q.cusips.iloc[-1]):,}개, 기관보유 총액 {q.inst_value_tn.iloc[-1]:.2f}조$")

    # ---------------- N-PORT
    h("3. N-PORT 펀드 보유·유출입 (2019Q4~)")
    NF = pd.read_parquet(C.PQ / "nport_funds.parquet")
    hp = sorted((C.PQ / "nport_holdings").glob("*.parquet"))
    nh = 0
    ncus = set()
    for f in hp:
        d = pd.read_parquet(f, columns=["cusip"])
        nh += len(d); ncus |= set(d["cusip"].unique())
    p(f"- 보고서 {len(NF):,}건, 시리즈(펀드) {NF.series_id.nunique():,}개, 보고일 {NF.report_date.min().date()} ~ {NF.report_date.max().date()}")
    p(f"- 지수형 이름 플래그 {int(NF.is_index_name.sum()):,}건 ({NF.is_index_name.mean()*100:.1f}%)")
    p(f"- 보통주 보유 행 {nh:,}, CUSIP {len(ncus):,}개, 분기 파일 {len(hp)}개")
    flows = NF[["sales_m1", "sales_m2", "sales_m3", "redem_m1", "redem_m2", "redem_m3"]].notna().all(axis=1).mean()
    p(f"- 월별 유출입 3개월 모두 존재 비율 {flows*100:.1f}%")
    if (C.PQ / "nport_identifiers.parquet").exists():
        NI = pd.read_parquet(C.PQ / "nport_identifiers.parquet")
        p(f"- CUSIP-티커 쌍 {len(NI):,} (CUSIP {NI.cusip.nunique():,}개)")

    # ---------------- FTD
    h("4. FTD CUSIP-심볼 이력 (2009-07~)")
    FT = pd.read_parquet(C.PQ / "ftd_cusip_symbol.parquet")
    FM = pd.read_parquet(C.PQ / "ftd_prices_monthly.parquet")
    p(f"- CUSIP-심볼 쌍 {len(FT):,}, CUSIP {FT.cusip.nunique():,}개, 기간 {FT.first_date.min().date()} ~ {FT.last_date.max().date()}")
    p(f"- 월별 가격 셀 {len(FM):,}")

    # ---------------- entities / master
    h("5. 증권 마스터 (CUSIP↔CIK↔티커)")
    SE = pd.read_parquet(C.PQ / "sec_entities.parquet")
    SM = pd.read_parquet(C.PQ / "security_master.parquet")
    p(f"- SEC 엔티티 {len(SE):,} (SIC 보유 {int(SE.sic.notna().sum()):,}, 10-K 제출 {int(SE.first_10k.notna().sum()):,})")
    p(f"- 마스터 CUSIP {len(SM):,}개 중 CIK 연결 {int(SM.cik.notna().sum()):,} ({SM.cik.notna().mean()*100:.1f}%)")
    table(SM["method"].value_counts().rename("cusips").to_frame())
    # value-weighted link rate on the latest 13F quarter
    last = A[A.period == A.period.max()].merge(SM[["cusip", "cik", "method"]], on="cusip", how="left")
    vw = last.groupby(last["cik"].notna())["inst_value_usd"].sum()
    p(f"\n- 최신 13F 분기 기관보유 가치 기준 CIK 연결률: {vw.get(True, 0) / vw.sum() * 100:.1f}%")
    # panel join: how many CIKs with fundamentals also have 13F ownership
    pk = set(P.cik.unique()); lk = set(SM.dropna(subset=["cik"]).cik.astype(int).unique())
    p(f"- 재무 패널 CIK {len(pk):,} 중 13F/FTD 증권과 연결된 CIK {len(pk & lk):,}")

    # ---------------- prices
    if (C.PQ / "prices_monthly.parquet").exists():
        h("6. 월별 가격 (yfinance, 현재 상장 종목만 — 생존편향 있음)")
        PR = pd.read_parquet(C.PQ / "prices_monthly.parquet")
        p(f"- 티커 {PR.ticker.nunique():,}개, 행 {len(PR):,}, 기간 {PR.month.min().date()} ~ {PR.month.max().date()}")

    h("7. 한계 요약")
    p("- 13F 내재가격은 분기 빈도이며 13F 보고 대상(기관 보유) 종목만 존재한다.")
    p("- N-PORT 유출입은 2019-08 이후만 공개되어 일시적 수급 성분의 표본이 7년이다.")
    p("- 패시브 지분은 매니저 CIK 기준 근사치(big3, 확장)로, 실제 패시브 비중의 하한이다.")
    p("- yfinance 월별 가격은 현재 상장 종목만 있어 월별 검정은 생존편향을 가진다.")
    p("- CUSIP↔CIK 연결의 이름 매칭 경로(name-fuzzy)는 오매칭 가능성이 있어 검증 표본을 수작업 점검해야 한다.")

    (C.REPORTS / "phase0_coverage.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
