# -*- coding: utf-8 -*-
"""Canonical fundamental concepts and the us-gaap tags that carry them.

Each concept lists tags in PRIORITY order: the panel builder takes the first
tag that has a value for the period, so the order encodes accounting judgement
(e.g. Revenues over the ASC 606 tag over the pre-2018 SalesRevenueNet).

Sign conventions are the taxonomy's own: expenses positive, payments
positive, income negative when a loss. Derived quantities (net income to
common, total payout, intangible-adjusted book) are built in xbrl_panel, not
here.

`kind` says whether the concept is a duration (flow) or an instant (stock);
that decides how quarterly values are recovered from year-to-date filings.
"""
from __future__ import annotations

CONCEPTS: dict[str, dict] = {
    # ---------------- income statement (duration) ----------------
    "revenue": dict(kind="duration", tags=[
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        "RevenuesNetOfInterestExpense",
        "OperatingLeasesIncomeStatementLeaseRevenue",
    ]),
    "cogs": dict(kind="duration", tags=[
        "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
        "CostOfServices", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ]),
    "gross_profit": dict(kind="duration", tags=["GrossProfit"]),
    "rnd": dict(kind="duration", tags=[
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        "ResearchAndDevelopmentInProcess",
    ]),
    "sga": dict(kind="duration", tags=[
        "SellingGeneralAndAdministrativeExpense",
    ]),
    "selling_marketing": dict(kind="duration", tags=[
        "SellingAndMarketingExpense", "SellingExpense", "MarketingExpense",
        "AdvertisingExpense",
    ]),
    "general_admin": dict(kind="duration", tags=[
        "GeneralAndAdministrativeExpense",
    ]),
    "opex": dict(kind="duration", tags=["OperatingExpenses", "CostsAndExpenses"]),
    "op_income": dict(kind="duration", tags=["OperatingIncomeLoss"]),
    "pretax_income": dict(kind="duration", tags=[
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ]),
    "tax": dict(kind="duration", tags=["IncomeTaxExpenseBenefit"]),
    "net_income": dict(kind="duration", tags=[
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLossAttributableToParentDiluted",
    ]),
    "net_income_incl_nci": dict(kind="duration", tags=["ProfitLoss"]),
    "nci_income": dict(kind="duration", tags=[
        "NetIncomeLossAttributableToNoncontrollingInterest",
    ]),
    "preferred_dividends": dict(kind="duration", tags=[
        "PreferredStockDividendsIncomeStatementImpact", "DividendsPreferredStock",
        "PreferredStockDividendsAndOtherAdjustments",
    ]),
    "interest_expense": dict(kind="duration", tags=[
        "InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt",
        "InterestAndDebtExpense", "InterestExpenseBorrowings",
    ]),
    "da": dict(kind="duration", tags=[
        "DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
        "DepreciationAmortizationAndAccretionNet", "Depreciation",
        "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"
    ]),
    "amort_intangibles": dict(kind="duration", tags=[
        "AmortizationOfIntangibleAssets", "AmortizationOfAcquiredIntangibleAssets",
    ]),
    "sbc": dict(kind="duration", tags=[
        "ShareBasedCompensation", "AllocatedShareBasedCompensationExpense",
        "ShareBasedCompensationExpense",
    ]),
    "comprehensive_income": dict(kind="duration", tags=["ComprehensiveIncomeNetOfTax"]),
    "eps_diluted": dict(kind="duration", tags=["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"]),
    "eps_basic": dict(kind="duration", tags=["EarningsPerShareBasic", "EarningsPerShareBasicAndDiluted"]),
    "shares_diluted_wavg": dict(kind="duration", tags=[
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ]),
    "shares_basic_wavg": dict(kind="duration", tags=[
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
    ]),
    "dps_declared": dict(kind="duration", tags=[
        "CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid",
    ]),
    # ---------------- cash flow (duration) ----------------
    "cfo": dict(kind="duration", tags=[
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    "capex": dict(kind="duration", tags=[
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
    ]),
    "capitalized_software": dict(kind="duration", tags=[
        "PaymentsToDevelopSoftware", "PaymentsForSoftware",
    ]),
    "acquisitions": dict(kind="duration", tags=[
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
    ]),
    "buybacks": dict(kind="duration", tags=[
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ]),
    "dividends_paid": dict(kind="duration", tags=[
        "PaymentsOfDividendsCommonStock", "PaymentsOfDividends",
        "PaymentsOfDividendsCash", "PaymentsOfOrdinaryDividends",
    ]),
    "dividends_paid_total": dict(kind="duration", tags=["PaymentsOfDividends"]),
    "equity_issuance": dict(kind="duration", tags=[
        "ProceedsFromIssuanceOfCommonStock", "ProceedsFromIssuanceOrSaleOfEquity",
        "ProceedsFromIssuanceInitialPublicOffering",
    ]),
    "options_exercised": dict(kind="duration", tags=[
        "ProceedsFromStockOptionsExercised",
        "ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions",
    ]),
    "debt_issued": dict(kind="duration", tags=[
        "ProceedsFromIssuanceOfLongTermDebt", "ProceedsFromIssuanceOfDebt",
    ]),
    "debt_repaid": dict(kind="duration", tags=[
        "RepaymentsOfLongTermDebt", "RepaymentsOfDebt",
    ]),
    # ---------------- balance sheet (instant) ----------------
    "assets": dict(kind="instant", tags=["Assets"]),
    "assets_current": dict(kind="instant", tags=["AssetsCurrent"]),
    "liabilities": dict(kind="instant", tags=["Liabilities"]),
    "liabilities_current": dict(kind="instant", tags=["LiabilitiesCurrent"]),
    "liab_and_equity": dict(kind="instant", tags=["LiabilitiesAndStockholdersEquity"]),
    "equity": dict(kind="instant", tags=[
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "equity_incl_nci": dict(kind="instant", tags=[
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "nci_equity": dict(kind="instant", tags=["MinorityInterest"]),
    "preferred_equity": dict(kind="instant", tags=[
        "PreferredStockValue", "PreferredStockValueOutstanding",
        "TemporaryEquityCarryingAmountAttributableToParent",
    ]),
    "retained_earnings": dict(kind="instant", tags=["RetainedEarningsAccumulatedDeficit"]),
    "aoci": dict(kind="instant", tags=["AccumulatedOtherComprehensiveIncomeLossNetOfTax"]),
    "treasury_stock": dict(kind="instant", tags=["TreasuryStockValue", "TreasuryStockCommonValue"]),
    "cash": dict(kind="instant", tags=[
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash", "CashAndDueFromBanks",
    ]),
    "st_investments": dict(kind="instant", tags=[
        "ShortTermInvestments", "MarketableSecuritiesCurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent", "AvailableForSaleSecuritiesCurrent",
    ]),
    "cash_and_st_inv": dict(kind="instant", tags=["CashCashEquivalentsAndShortTermInvestments"]),
    "lt_investments": dict(kind="instant", tags=[
        "LongTermInvestments", "MarketableSecuritiesNoncurrent",
        "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent",
    ]),
    "receivables": dict(kind="instant", tags=["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"]),
    "inventory": dict(kind="instant", tags=["InventoryNet"]),
    "ppe_net": dict(kind="instant", tags=["PropertyPlantAndEquipmentNet"]),
    "goodwill": dict(kind="instant", tags=["Goodwill"]),
    "intangibles": dict(kind="instant", tags=[
        "IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet",
        "IntangibleAssetsNetIncludingGoodwill",
    ]),
    "debt_lt": dict(kind="instant", tags=[
        "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligationsNoncurrent", "LongTermDebt",
        "LongTermNotesPayable", "SeniorLongTermNotes", "ConvertibleNotesPayableNoncurrent",
    ]),
    "debt_lt_total": dict(kind="instant", tags=["LongTermDebt", "DebtInstrumentCarryingAmount"]),
    "debt_st": dict(kind="instant", tags=[
        "LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
        "LongTermDebtAndCapitalLeaseObligationsCurrent", "CommercialPaper",
        "NotesPayableCurrent", "ShortTermDebtCurrent"
    ]),
    "lease_liab": dict(kind="instant", tags=[
        "OperatingLeaseLiability", "OperatingLeaseLiabilityNoncurrent",
    ]),
    "deferred_revenue": dict(kind="instant", tags=[
        "ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent",
    ]),
    # Outstanding only. CommonStockSharesIssued includes treasury shares
    # (JPMorgan: 4,105m issued vs 2,800m outstanding) and must not be a fallback.
    "shares_out": dict(kind="instant", tags=["CommonStockSharesOutstanding"]),
    "shares_issued": dict(kind="instant", tags=["CommonStockSharesIssued"]),
    "treasury_shares": dict(kind="instant", tags=["TreasuryStockShares", "TreasuryStockCommonShares"]),
    "dei_shares_out": dict(kind="instant", tags=["EntityCommonStockSharesOutstanding"]),
    "dei_public_float": dict(kind="instant", tags=["EntityPublicFloat"]),
}

# Map every listed tag to its concept(s); one tag can feed several concepts
# (ProfitLoss is a fallback for net_income and the primary for net_income_incl_nci).
TAG_TO_CONCEPTS: dict[str, list[str]] = {}
for _name, _spec in CONCEPTS.items():
    for _t in _spec["tags"]:
        TAG_TO_CONCEPTS.setdefault(_t, []).append(_name)

ALL_TAGS: frozenset[str] = frozenset(TAG_TO_CONCEPTS)
DURATION_CONCEPTS = [k for k, v in CONCEPTS.items() if v["kind"] == "duration"]
INSTANT_CONCEPTS = [k for k, v in CONCEPTS.items() if v["kind"] == "instant"]
