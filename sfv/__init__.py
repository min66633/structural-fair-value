"""SFV - Structural Fair Value: cross-sectional decomposition of equity prices.

    log P = log V_F + delta_S + delta_T + eps

Package layout (phase 0 = data pipeline):
    config        paths, SEC user agent, constants
    xbrl_extract  companyfacts.zip -> long fact table (whitelisted tags)
    xbrl_panel    long facts -> point-in-time quarterly fundamentals panel
    f13           13F INFOTABLE -> institutional / passive ownership per CUSIP-quarter
    nport         N-PORT -> fund holdings, monthly flows, CUSIP-ticker identifiers
    idmap         CIK <-> CUSIP <-> ticker security master
    prices        monthly prices (yfinance, survivorship-biased secondary route)
"""
