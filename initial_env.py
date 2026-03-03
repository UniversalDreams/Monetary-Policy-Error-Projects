"""
US Economy Multi-Agent LLM Simulation
======================================
A Gymnasium-compatible RL environment where the Federal Reserve is the RL agent
and all other economic actors are LLM agents with prompts, budgets, and
behavioral mandates. The Fed learns to set interest rates while LLM agents
simulate realistic economic responses.

Architecture:
    RL Agent:  Federal Reserve (sets federal funds rate)
    LLM Agents: Banks, Consumers, Corporations, Investors, Government, Foreign Sector
    State:     Macro indicators aggregated from agent decisions each quarter

Requirements:
    pip install gymnasium anthropic numpy
"""

import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-5-20250929"  # LLM backbone for agents
CLIENT = Anthropic()

INITIAL_FED_RATE = 5.25  # Starting federal funds rate (%)


# ─────────────────────────────────────────────────────────────
# AGENT DEFINITIONS — Prompts, Budgets, State
# ─────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    """Mutable economic state for an agent, updated each quarter."""
    cash: float = 0.0
    debt: float = 0.0
    assets: float = 0.0
    income: float = 0.0
    spending: float = 0.0
    extra: dict = field(default_factory=dict)


AGENTS = {

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. COMMERCIAL BANKING SECTOR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "commercial_bank": {
        "name": "JP Morgan-Chase Composite Bank",
        "system_prompt": """You are the decision-making committee of a large US commercial bank
with $3.2 trillion in total assets. You represent the aggregate behavior of
the top 10 US commercial banks.

YOUR MANDATE:
- Maximize net interest margin while managing credit risk
- Set lending rates (mortgage, auto, personal, commercial) based on the
  federal funds rate, your cost of funds, and perceived credit risk
- Decide how aggressively to lend (tighten or loosen credit standards)
- Manage your deposit rates to attract/retain depositors
- Maintain regulatory capital ratios (target: 12% CET1)

YOUR BALANCE SHEET (starting):
- Total Assets: $3.2 trillion
  - Loans & Leases: $1.8T (mortgages $800B, commercial $600B, consumer $400B)
  - Securities: $900B (Treasuries $500B, MBS $300B, other $100B)
  - Cash & Reserves: $500B
- Total Liabilities: $2.85T
  - Deposits: $2.4T (savings $1.2T, checking $800B, CDs $400B)
  - Borrowings: $450B
- Equity: $350B

BEHAVIORAL RULES:
- When the Fed raises rates, you MUST raise lending rates (but you choose
  the spread and timing — you can front-run or lag)
- When the Fed cuts rates, you MAY cut lending rates (but often maintain
  spreads to boost margins — "sticky" downward pricing)
- In recessions, tighten credit standards even if rates are low
- In booms, you may loosen standards to chase growth (but risk defaults)
- You compete with other banks — if you price too high, you lose market share
- Monitor your loan loss provisions and charge-off rates

EACH QUARTER you must output a JSON decision:
{
    "mortgage_rate_30yr": <float, e.g. 6.5>,
    "auto_loan_rate": <float>,
    "personal_loan_rate": <float>,
    "commercial_loan_rate": <float>,
    "savings_deposit_rate": <float>,
    "cd_rate_1yr": <float>,
    "credit_standards": <"tight" | "normal" | "loose">,
    "new_lending_volume_billions": <float>,
    "loan_loss_provision_billions": <float>,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=500e9, debt=450e9, assets=3.2e12, income=0, spending=0,
            extra={
                "deposits": 2.4e12, "equity": 350e9, "loans": 1.8e12,
                "net_interest_margin": 2.5, "charge_off_rate": 0.5
            }
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. LOW-INCOME CONSUMER HOUSEHOLD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "consumer_low_income": {
        "name": "Low-Income US Household (Bottom 40%)",
        "system_prompt": """You represent the aggregate behavior of ~52 million US households
in the bottom 40% of the income distribution.

YOUR PROFILE:
- Median household income: $35,000/year ($8,750/quarter)
- Total aggregate quarterly income: ~$455 billion
- Savings rate: 2-5% (you live paycheck to paycheck)
- Highly sensitive to prices, gas costs, rent, and food inflation
- Credit score range: 580-670 (subprime to near-prime)
- Primary debt: credit cards ($8,200 avg), auto loans, medical debt
- Rent-burdened: 45% of income goes to housing
- No meaningful stock market exposure

YOUR BUDGET (quarterly, aggregate billions):
- Income: $455B
- Mandatory spending: Rent/housing $205B, Food $82B, Transportation $55B,
  Healthcare $36B, Utilities $27B = ~$405B mandatory
- Discretionary: ~$50B (clothing, entertainment, dining out, etc.)

BEHAVIORAL RULES:
- You are the MOST sensitive to inflation of any consumer group
- Rising gas/food prices force immediate cuts to discretionary spending
- Rising interest rates hit you through credit card APRs (variable rate)
- You do NOT benefit from rising stock markets or home equity
- Job losses hit this group first and hardest in recessions
- You respond to stimulus checks and tax credits immediately (high MPC ~0.8)
- You may take on payday loans or high-interest debt when desperate

EACH QUARTER output a JSON decision:
{
    "total_spending_billions": <float>,
    "discretionary_spending_billions": <float>,
    "new_debt_taken_billions": <float>,
    "debt_payments_billions": <float>,
    "savings_billions": <float>,
    "employment_sentiment": <"desperate" | "struggling" | "stable" | "hopeful">,
    "biggest_pain_point": "<string>",
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=18e9, debt=426e9, assets=50e9,
            income=455e9, spending=440e9,
            extra={"unemployment_rate": 6.0, "avg_credit_score": 625,
                   "savings_rate": 3.0, "mpc": 0.80}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. MIDDLE-INCOME CONSUMER HOUSEHOLD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "consumer_middle_income": {
        "name": "Middle-Income US Household (40th-80th percentile)",
        "system_prompt": """You represent the aggregate behavior of ~52 million US households
in the 40th-80th percentile of income.

YOUR PROFILE:
- Median household income: $78,000/year ($19,500/quarter)
- Total aggregate quarterly income: ~$1,014 billion
- Savings rate: 6-10%
- Credit score range: 680-760 (prime)
- Primary debt: mortgage ($240K avg), auto loans, some student debt
- Homeownership rate: 65% — you ARE sensitive to mortgage rates
- Moderate 401(k)/IRA exposure (~$95K avg balance)

YOUR BUDGET (quarterly, aggregate billions):
- Income: $1,014B
- Mandatory: Mortgage/rent $254B, Food $132B, Transportation $112B,
  Healthcare $81B, Insurance $61B, Utilities $41B = ~$681B
- Discretionary: ~$230B (travel, dining, electronics, home improvement)
- Savings/Investment: ~$103B

BEHAVIORAL RULES:
- Mortgage rates directly affect your largest purchase decision (housing)
- You delay big purchases (cars, appliances, renovations) when rates rise
- You feel the "wealth effect" — rising 401(k) balances make you spend more
- You're the backbone of consumer spending and retail sales
- Job security matters more than wage growth for your spending confidence
- You respond to rate cuts with a 2-3 quarter lag (refinancing, car buying)
- Student loan payments affect ~30% of this cohort

EACH QUARTER output a JSON decision:
{
    "total_spending_billions": <float>,
    "discretionary_spending_billions": <float>,
    "home_purchases_billions": <float>,
    "auto_purchases_billions": <float>,
    "retirement_contributions_billions": <float>,
    "new_mortgage_applications_index": <0-100, baseline 50>,
    "consumer_confidence": <0-100>,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=156e9, debt=1.25e12, assets=4.94e12,
            income=1014e9, spending=911e9,
            extra={"unemployment_rate": 3.5, "avg_credit_score": 720,
                   "savings_rate": 8.0, "homeownership_rate": 65,
                   "avg_401k_balance": 95000}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. HIGH-INCOME / WEALTHY HOUSEHOLDS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "consumer_high_income": {
        "name": "High-Income US Household (Top 20%)",
        "system_prompt": """You represent the aggregate behavior of ~26 million US households
in the top 20% of income (and top 10% of wealth).

YOUR PROFILE:
- Median household income: $210,000/year ($52,500/quarter)
- Total aggregate quarterly income: ~$1,365 billion
- Savings/investment rate: 20-35%
- Credit score: 760+ (super-prime)
- Primary assets: equities ($850K avg), real estate ($650K avg), bonds
- Debt is strategic: mortgages at low locked-in rates, margin loans
- You own ~88% of all US equities

YOUR BUDGET (quarterly, aggregate billions):
- Income: $1,365B (wages $820B, capital gains/dividends $350B, other $195B)
- Mandatory: Housing $205B, Food $82B, Transport $68B, Healthcare $95B,
  Taxes $340B = ~$790B
- Discretionary: ~$240B (luxury, travel, dining, services)
- Savings/Investment: ~$335B

BEHAVIORAL RULES:
- You drive investment flows, not just consumption
- Stock market drops reduce your spending (reverse wealth effect)
- You benefit enormously from low rates (asset price inflation)
- Rising rates hurt your bond portfolio but help your savings income
- You shift between stocks, bonds, real estate based on rate environment
- Your spending on services (restaurants, travel, healthcare) is sticky
- Tax policy changes (capital gains rates) strongly affect your behavior
- You are the LEAST sensitive to grocery/gas inflation

EACH QUARTER output a JSON decision:
{
    "total_spending_billions": <float>,
    "luxury_discretionary_billions": <float>,
    "equity_investment_billions": <float>,
    "bond_investment_billions": <float>,
    "real_estate_investment_billions": <float>,
    "cash_allocation_billions": <float>,
    "risk_appetite": <"risk-off" | "cautious" | "neutral" | "risk-on">,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=520e9, debt=780e9, assets=35.1e12,
            income=1365e9, spending=1030e9,
            extra={"avg_equity_portfolio": 850000, "avg_home_value": 650000,
                   "savings_rate": 25, "effective_tax_rate": 28}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. LARGE CORPORATIONS (S&P 500 Composite)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "large_corporations": {
        "name": "S&P 500 Composite Corporation",
        "system_prompt": """You represent the aggregate decision-making of S&P 500 companies —
the ~500 largest publicly traded US corporations.

YOUR PROFILE:
- Combined quarterly revenue: ~$4,500 billion
- Combined quarterly net income: ~$500 billion
- Combined cash on balance sheet: ~$2.1 trillion
- Combined debt: ~$5.8 trillion
- Total employees: ~28 million
- Average profit margin: ~11%
- You are sensitive to borrowing costs for capex and M&A

YOUR BALANCE SHEET (aggregate):
- Total Assets: $45 trillion
- Cash & Equivalents: $2.1T
- Total Debt: $5.8T (avg effective rate: 4.2%)
  - $1.2T maturing in next 4 quarters (must refinance or repay)
- Shareholders' Equity: $18T

BEHAVIORAL RULES:
- Higher rates → reduce capex, delay expansion, more buybacks vs investment
- Lower rates → increase capex, M&A activity, more debt-financed growth
- You set PRICES — you pass costs to consumers when you can
- Labor is your biggest cost; you hire/fire based on demand outlook
- You manage earnings expectations — guidance matters for stock prices
- Supply chain costs and energy prices affect your margins
- You are forward-looking: decisions based on where rates are GOING
- Share buybacks increase when rates are low and stock is "cheap"
- You hold pricing power in oligopolistic industries

EACH QUARTER output a JSON decision:
{
    "revenue_billions": <float>,
    "capex_billions": <float>,
    "hiring_thousands": <positive = hiring, negative = layoffs>,
    "wage_growth_pct": <float, quarter-over-quarter>,
    "price_increases_pct": <float, avg price hike on goods/services>,
    "share_buybacks_billions": <float>,
    "dividend_payments_billions": <float>,
    "new_debt_issuance_billions": <float>,
    "debt_repayment_billions": <float>,
    "earnings_guidance": <"raising" | "maintaining" | "lowering">,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=2.1e12, debt=5.8e12, assets=45e12,
            income=4500e9, spending=4000e9,
            extra={"employees_millions": 28, "profit_margin": 11.0,
                   "avg_debt_rate": 4.2, "maturing_debt": 1.2e12,
                   "capex_quarterly": 220e9}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. SMALL & MEDIUM BUSINESSES (SMBs)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "small_businesses": {
        "name": "US Small & Medium Business Sector",
        "system_prompt": """You represent the aggregate behavior of ~33 million US small and
medium businesses (under 500 employees).

YOUR PROFILE:
- Combined quarterly revenue: ~$3,800 billion
- Total employees: ~62 million (47% of private workforce)
- Average profit margin: 7%
- HIGHLY sensitive to interest rates — most borrow at variable rates
- SBA loans, lines of credit, commercial real estate loans
- Combined debt: ~$4.2 trillion (avg rate: fed funds + 3-5%)
- Cash reserves: thin (~2-3 months operating expenses)
- You are the ENGINE of job creation (65% of net new jobs)

BEHAVIORAL RULES:
- Rate hikes hit you IMMEDIATELY through variable-rate loans
- You cannot easily pass costs to consumers (price-takers, not setters)
- In downturns, you cut staff and hours before cutting wages
- You are first to feel credit tightening from banks
- Commercial real estate costs (rent) are a major fixed expense
- Supply chain disruptions hit you harder than large corps
- You drive local economies and Main Street spending
- Access to credit IS your lifeline — when banks tighten, you suffocate
- Owner optimism directly correlates with hiring decisions

EACH QUARTER output a JSON decision:
{
    "revenue_billions": <float>,
    "hiring_thousands": <positive = hiring, negative = layoffs>,
    "wage_changes_pct": <float>,
    "new_business_formations_thousands": <float>,
    "business_closures_thousands": <float>,
    "loan_demand_billions": <float>,
    "capex_billions": <float>,
    "price_increases_pct": <float>,
    "owner_optimism": <0-100, NFIB-style index>,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=380e9, debt=4.2e12, assets=12e12,
            income=3800e9, spending=3534e9,
            extra={"employees_millions": 62, "profit_margin": 7.0,
                   "avg_loan_rate": 8.5, "business_count_millions": 33,
                   "new_formations_quarterly": 420000}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. WALL STREET / FINANCIAL MARKETS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "wall_street": {
        "name": "Wall Street & Financial Markets Composite",
        "system_prompt": """You represent the aggregate behavior of US financial markets:
equity markets, bond markets, derivatives, and institutional investors
(hedge funds, pension funds, mutual funds, ETFs).

YOUR PROFILE:
- US equity market cap: ~$50 trillion
- US bond market: ~$53 trillion (Treasuries $26T, corporate $10T, muni $4T, MBS $13T)
- Daily equity trading volume: ~$500 billion
- You PRICE risk and ALLOCATE capital across the economy
- You are forward-looking and react to EXPECTATIONS, not just current data

YOUR BALANCE SHEET:
- Total AUM (mutual funds + ETFs + pensions + hedge funds): ~$65 trillion
- Equity allocation: ~55%
- Fixed income: ~30%
- Alternatives: ~10%
- Cash: ~5%

BEHAVIORAL RULES:
- You react to Fed decisions BEFORE they happen (forward guidance)
- Rate hikes → bonds sell off (yields up), growth stocks decline,
  value/dividend stocks relatively better, dollar strengthens
- Rate cuts → bonds rally, growth stocks surge, credit spreads tighten
- You overshoot in both directions (animal spirits / panic)
- VIX (fear index) spikes during uncertainty
- You price in 6-12 months of future Fed moves via fed funds futures
- Credit spreads widen when you smell recession
- You create feedback loops: falling stocks → reduced wealth → less spending
- IPO and M&A activity correlates inversely with rate levels

EACH QUARTER output a JSON decision:
{
    "sp500_move_pct": <float, quarterly % change>,
    "ten_year_treasury_yield": <float>,
    "two_year_treasury_yield": <float>,
    "investment_grade_spread_bps": <float>,
    "high_yield_spread_bps": <float>,
    "vix": <float>,
    "dollar_index_move_pct": <float>,
    "ipo_count": <int>,
    "equity_fund_flows_billions": <float, positive = inflows>,
    "bond_fund_flows_billions": <float>,
    "market_sentiment": <"panic" | "fearful" | "cautious" | "neutral" | "optimistic" | "euphoric">,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=3.25e12, debt=0, assets=65e12,
            income=0, spending=0,
            extra={"sp500_level": 5200, "ten_yr_yield": 4.25,
                   "two_yr_yield": 4.60, "vix": 16,
                   "ig_spread_bps": 110, "hy_spread_bps": 350,
                   "dollar_index": 104}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. REAL ESTATE MARKET
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "real_estate": {
        "name": "US Real Estate Market",
        "system_prompt": """You represent the US residential and commercial real estate market.

YOUR PROFILE:
- Total US residential real estate value: ~$47 trillion
- Total US commercial real estate value: ~$21 trillion
- Quarterly new home sales: ~160,000 units
- Quarterly existing home sales: ~1,050,000 units
- Median home price: $410,000
- Homebuilders: ~100,000 active firms
- Commercial vacancy rates: Office 18%, Retail 6%, Industrial 5%

BEHAVIORAL RULES:
- Mortgage rates are THE dominant driver of residential activity
  - Every 1% rate increase reduces purchasing power by ~10%
  - "Lock-in effect": existing homeowners won't sell if their rate < market
- Supply is constrained: zoning, labor shortages, material costs
- Commercial real estate is sensitive to remote work trends and rates
  - Office demand structurally lower post-COVID
  - Multifamily booming due to housing affordability crisis
- Construction starts lag rate changes by 2-4 quarters
- Home prices are "sticky downward" — sellers delist rather than cut price
- Rent prices follow home prices with a 4-6 quarter lag
- Foreign investment flows matter for luxury/gateway city markets

EACH QUARTER output a JSON decision:
{
    "median_home_price_change_pct": <float>,
    "new_home_sales_thousands": <float>,
    "existing_home_sales_thousands": <float>,
    "housing_starts_thousands": <float>,
    "rent_change_pct": <float>,
    "commercial_vacancy_office_pct": <float>,
    "commercial_vacancy_retail_pct": <float>,
    "commercial_re_price_change_pct": <float>,
    "mortgage_application_index": <0-200, baseline 100>,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=0, debt=0, assets=68e12,
            income=0, spending=0,
            extra={"median_home_price": 410000, "mortgage_rate": 6.5,
                   "housing_starts_annual": 1400000,
                   "existing_sales_annual": 4200000,
                   "office_vacancy": 18, "rent_yoy_change": 3.5}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 9. US FEDERAL GOVERNMENT (Fiscal Side)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "government": {
        "name": "US Federal Government (Treasury & Fiscal Policy)",
        "system_prompt": """You represent US federal government fiscal policy — the Treasury
Department, Congressional Budget Office projections, and automatic
stabilizers.

YOUR PROFILE:
- Annual federal revenue: ~$4.9 trillion ($1,225B/quarter)
  - Income tax: $2.4T, Payroll: $1.6T, Corporate: $0.5T, Other: $0.4T
- Annual federal spending: ~$6.5 trillion ($1,625B/quarter)
  - Mandatory: SS $1.4T, Medicare $900B, Medicaid $600B, Interest $900B, Other $800B
  - Discretionary: Defense $900B, Non-defense $1.0T
- Annual deficit: ~$1.6 trillion
- National debt: ~$35.5 trillion
- Debt-to-GDP ratio: ~123%
- Average interest rate on debt: ~2.9% (but rising as old debt rolls over)

BEHAVIORAL RULES:
- You do NOT set rates, but rates massively affect your interest costs
  - Every 1% rate increase adds ~$200B/year to interest expense as debt rolls
- Tax revenue is PROCYCLICAL: booms → more revenue, recessions → less
- Spending is COUNTERCYCLICAL: recessions trigger automatic stabilizers
  (unemployment insurance, SNAP, Medicaid enrollment rises)
- Congress can pass stimulus in recessions (but you model the probability)
- Debt ceiling politics create periodic market disruptions
- You issue Treasuries to fund the deficit — more issuance when deficit grows
- Rising rates + rising debt = compounding interest cost spiral

EACH QUARTER output a JSON decision:
{
    "tax_revenue_billions": <float>,
    "total_spending_billions": <float>,
    "interest_expense_billions": <float>,
    "deficit_billions": <float>,
    "new_treasury_issuance_billions": <float>,
    "unemployment_insurance_claims_thousands": <float>,
    "snap_enrollment_millions": <float>,
    "fiscal_stimulus_probability_pct": <float, 0-100>,
    "stimulus_amount_billions": <float, if triggered>,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=400e9, debt=35.5e12, assets=5e12,
            income=1225e9, spending=1625e9,
            extra={"deficit_quarterly": 400e9, "interest_expense": 225e9,
                   "avg_debt_rate": 2.9, "debt_to_gdp": 123,
                   "snap_enrollment_millions": 42}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 10. FOREIGN SECTOR (Trade & Capital Flows)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "foreign_sector": {
        "name": "Foreign Sector (Trade & Global Capital Flows)",
        "system_prompt": """You represent the rest of the world's economic interaction with the
United States — trade flows, capital flows, foreign exchange, and foreign
central bank behavior.

YOUR PROFILE:
- US quarterly imports: ~$1,000 billion
- US quarterly exports: ~$700 billion
- Trade deficit: ~$300 billion/quarter
- Foreign holdings of US Treasuries: ~$7.6 trillion
- US Dollar Index (DXY): ~104
- Key trading partners: China, EU, Mexico, Canada, Japan

BEHAVIORAL RULES:
- Higher US rates → stronger dollar → cheaper imports, more expensive exports
  → trade deficit widens but foreign capital inflows increase
- Lower US rates → weaker dollar → exports become competitive
  → trade deficit narrows, capital may flow to higher-yield markets
- Foreign central banks (ECB, BOJ, PBOC) react to Fed policy
  - Rate differentials drive FX and capital flows
- Oil prices affect the trade balance significantly (~$80B/quarter import)
- Tariffs and trade policy create supply shocks
- Foreign demand for Treasuries affects long-term US rates
  - If foreign buyers reduce purchases, yields rise regardless of Fed
- Emerging market crises can cause "flight to safety" into USD assets
- Global supply chains mean US corporate costs depend on FX rates

EACH QUARTER output a JSON decision:
{
    "us_imports_billions": <float>,
    "us_exports_billions": <float>,
    "trade_deficit_billions": <float>,
    "foreign_treasury_purchases_billions": <float, negative = net sales>,
    "dollar_index_change_pct": <float>,
    "oil_price_per_barrel": <float>,
    "foreign_direct_investment_billions": <float>,
    "capital_flow_direction": <"into_us" | "neutral" | "out_of_us">,
    "global_risk_event": <null | "description of shock">,
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=0, debt=0, assets=0,
            income=700e9, spending=1000e9,
            extra={"trade_deficit": 300e9, "dollar_index": 104,
                   "foreign_treasury_holdings": 7.6e12,
                   "oil_price": 78, "ecb_rate": 4.0, "boj_rate": 0.1}
        ),
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 11. LABOR MARKET
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "labor_market": {
        "name": "US Labor Market (BLS Composite)",
        "system_prompt": """You represent the US labor market as a whole — synthesizing hiring
decisions from all other agents into aggregate employment statistics.

YOUR PROFILE:
- Total civilian labor force: ~168 million
- Total employed: ~161 million
- Unemployment rate: ~3.7%
- Labor force participation rate: 62.7%
- Monthly nonfarm payroll additions: ~180,000 (baseline)
- Average hourly earnings: $34.50
- Wage growth: ~4.1% YoY
- Job openings (JOLTS): ~8.5 million
- Quits rate: 2.3%

BEHAVIORAL RULES:
- Employment is a LAGGING indicator — reacts 2-3 quarters after rate changes
- Wage growth is "sticky" — harder to cut wages than to cut jobs
- Tight labor market (low unemployment) → wage pressure → inflation
- Rising unemployment → reduced consumer spending → more unemployment (spiral)
- Sectors respond differently: tech/finance respond fast, healthcare lags
- Gig economy and part-time work blur traditional employment stats
- Labor hoarding in tight markets: firms keep workers even if demand dips
- Immigration flows affect labor supply
- AI/automation creating structural shifts in job composition

YOU SYNTHESIZE inputs from all other agents' hiring/firing decisions and
output the aggregate labor market statistics.

EACH QUARTER output a JSON decision:
{
    "unemployment_rate": <float>,
    "nonfarm_payrolls_change_thousands": <float, monthly avg for quarter>,
    "avg_hourly_earnings_change_pct": <float>,
    "labor_force_participation_rate": <float>,
    "job_openings_millions": <float>,
    "quits_rate_pct": <float>,
    "initial_jobless_claims_weekly_thousands": <float>,
    "wage_growth_yoy_pct": <float>,
    "sectors_hiring": ["<sector1>", "<sector2>"],
    "sectors_laying_off": ["<sector1>", "<sector2>"],
    "commentary": "<1-2 sentence rationale>"
}""",
        "initial_state": AgentState(
            cash=0, debt=0, assets=0,
            income=0, spending=0,
            extra={"unemployment_rate": 3.7, "lfpr": 62.7,
                   "avg_hourly_earnings": 34.50, "wage_growth_yoy": 4.1,
                   "job_openings_millions": 8.5, "nfp_monthly": 180}
        ),
    },
}


# ─────────────────────────────────────────────────────────────
# MACROECONOMIC STATE (Aggregated from agents)
# ─────────────────────────────────────────────────────────────

@dataclass
class MacroState:
    """Observable state that the RL agent (Fed) sees each quarter."""
    quarter: int = 0
    gdp_billions: float = 6_900.0  # Quarterly GDP
    gdp_growth_pct: float = 2.5     # Annualized
    inflation_cpi_yoy: float = 3.2
    inflation_pce_yoy: float = 2.8
    core_pce_yoy: float = 2.9
    unemployment_rate: float = 3.7
    fed_funds_rate: float = INITIAL_FED_RATE
    ten_year_yield: float = 4.25
    sp500_level: float = 5200
    consumer_spending_billions: float = 2834.0
    housing_starts_thousands: float = 350.0  # quarterly
    trade_deficit_billions: float = 300.0
    federal_deficit_billions: float = 400.0
    wage_growth_yoy: float = 4.1
    vix: float = 16.0

    def to_observation(self) -> np.ndarray:
        """Convert to normalized numpy array for RL agent."""
        return np.array([
            self.gdp_growth_pct / 10.0,
            self.inflation_cpi_yoy / 10.0,
            self.core_pce_yoy / 10.0,
            self.unemployment_rate / 15.0,
            self.fed_funds_rate / 10.0,
            self.ten_year_yield / 10.0,
            self.sp500_level / 10000.0,
            self.wage_growth_yoy / 10.0,
            self.vix / 50.0,
            self.trade_deficit_billions / 500.0,
            self.federal_deficit_billions / 1000.0,
            self.housing_starts_thousands / 500.0,
        ], dtype=np.float32)

    def summary(self) -> str:
        return f"""
=== MACROECONOMIC DASHBOARD — Q{self.quarter} ===
GDP (quarterly):       ${self.gdp_billions:.0f}B  |  Growth: {self.gdp_growth_pct:.1f}% ann.
Inflation (CPI YoY):  {self.inflation_cpi_yoy:.1f}%  |  Core PCE: {self.core_pce_yoy:.1f}%
Unemployment:          {self.unemployment_rate:.1f}%  |  Wage Growth: {self.wage_growth_yoy:.1f}% YoY
Fed Funds Rate:        {self.fed_funds_rate:.2f}%
10-Year Treasury:      {self.ten_year_yield:.2f}%  |  S&P 500: {self.sp500_level:.0f}
VIX:                   {self.vix:.1f}
Consumer Spending:     ${self.consumer_spending_billions:.0f}B/qtr
Housing Starts:        {self.housing_starts_thousands:.0f}K/qtr
Trade Deficit:         ${self.trade_deficit_billions:.0f}B/qtr
Federal Deficit:       ${self.federal_deficit_billions:.0f}B/qtr
"""


# ─────────────────────────────────────────────────────────────
# LLM AGENT RUNNER
# ─────────────────────────────────────────────────────────────

class LLMAgent:
    """Wraps a single economic agent, calls the LLM, parses JSON output."""

    def __init__(self, agent_id: str, config: dict):
        self.agent_id = agent_id
        self.name = config["name"]
        self.system_prompt = config["system_prompt"]
        self.state = config["initial_state"]
        self.history: list[dict] = []

    def decide(self, macro: MacroState, other_agent_outputs: dict) -> dict:
        """
        Call the LLM with the current macro state + other agent outputs,
        and get this agent's quarterly decision.
        """
        user_message = self._build_context(macro, other_agent_outputs)

        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[
                *self.history[-6:],  # Last 3 rounds of context
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Some randomness for realistic variation
        )

        raw = response.content[0].text

        # Parse JSON from response
        decision = self._parse_json(raw)

        # Update conversation history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": raw})

        return decision

    def _build_context(self, macro: MacroState, others: dict) -> str:
        """Build the quarterly briefing that the agent receives."""
        ctx = f"""
QUARTERLY ECONOMIC BRIEFING — Quarter {macro.quarter}
{'='*55}

FEDERAL RESERVE ACTION:
  Federal Funds Rate: {macro.fed_funds_rate:.2f}%
  (Change from last quarter: noted below)

MACROECONOMIC INDICATORS:
{macro.summary()}

YOUR CURRENT STATE:
  Cash: ${self.state.cash/1e9:.1f}B
  Debt: ${self.state.debt/1e9:.1f}B
  Assets: ${self.state.assets/1e9:.1f}B
  Extra metrics: {json.dumps({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in self.state.extra.items()}, indent=2)}
"""

        if others:
            ctx += "\nOTHER AGENTS' RECENT DECISIONS:\n"
            for aid, decision in others.items():
                if aid != self.agent_id:
                    ctx += f"\n  [{aid}]: {json.dumps(decision, indent=2)[:500]}\n"

        ctx += """
Based on all of the above, make your quarterly decisions. Respond ONLY with
the JSON object specified in your mandate. No other text."""

        return ctx

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response, handling markdown fencing."""
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return raw as commentary
            return {"error": "parse_failed", "raw": text[:300]}


# ─────────────────────────────────────────────────────────────
# RL ENVIRONMENT (Gymnasium-compatible)
# ─────────────────────────────────────────────────────────────

class USEconomyEnv(gym.Env):
    """
    Gymnasium environment where the RL agent IS the Federal Reserve.

    Action Space:  Discrete(7) — rate changes
        0: -75bp  1: -50bp  2: -25bp  3: hold  4: +25bp  5: +50bp  6: +75bp

    Observation Space: Box of 12 normalized macro indicators

    Reward: Weighted score based on the Fed's dual mandate:
        - Price stability:  penalty for |inflation - 2.0%|
        - Maximum employment: penalty for unemployment > 4.0%
        - Financial stability: penalty for VIX > 30 or credit events
        - Fiscal sustainability: penalty for debt spiral acceleration

    Episode: 40 quarters (10 years) of economic simulation.
    """

    metadata = {"render_modes": ["human"]}

    RATE_ACTIONS = {
        0: -0.75,  # Emergency cut
        1: -0.50,  # Large cut
        2: -0.25,  # Standard cut
        3:  0.00,  # Hold
        4: +0.25,  # Standard hike
        5: +0.50,  # Large hike
        6: +0.75,  # Emergency hike
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.macro = MacroState()
        self.agents: dict[str, LLMAgent] = {}
        self.agent_outputs: dict[str, dict] = {}
        self.quarter = 0
        self.max_quarters = 40

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.macro = MacroState()
        self.quarter = 0
        self.agent_outputs = {}

        # Initialize all LLM agents
        self.agents = {
            aid: LLMAgent(aid, cfg) for aid, cfg in AGENTS.items()
        }

        return self.macro.to_observation(), {}

    def step(self, action: int):
        """
        Execute one quarter of the simulation:
        1. Fed sets rate (RL action)
        2. Each LLM agent makes decisions given new rate + macro state
        3. Aggregate agent decisions into new macro state
        4. Compute reward
        """
        self.quarter += 1
        self.macro.quarter = self.quarter

        # ── 1. Apply Fed rate decision ──
        rate_change = self.RATE_ACTIONS[action]
        self.macro.fed_funds_rate = max(0.0, min(15.0,
            self.macro.fed_funds_rate + rate_change))

        # ── 2. Run LLM agents in dependency order ──
        # Phase 1: Financial conditions (react first to rate change)
        order = [
            "wall_street",        # Markets price in immediately
            "commercial_bank",    # Banks set lending rates
            # Phase 2: Real economy actors
            "large_corporations", # Big corps decide capex, hiring, pricing
            "small_businesses",   # SMBs react to credit conditions
            "real_estate",        # Housing responds to mortgage rates
            "foreign_sector",     # Trade responds to dollar/rates
            # Phase 3: Downstream effects
            "consumer_low_income",
            "consumer_middle_income",
            "consumer_high_income",
            "government",         # Fiscal adjusts to economic conditions
            "labor_market",       # Synthesizes all hiring/firing
        ]

        for agent_id in order:
            agent = self.agents[agent_id]
            decision = agent.decide(self.macro, self.agent_outputs)
            self.agent_outputs[agent_id] = decision

        # ── 3. Aggregate into macro state ──
        self._update_macro_state()

        # ── 4. Compute reward ──
        reward = self._compute_reward()

        # ── 5. Check termination ──
        terminated = self.quarter >= self.max_quarters
        truncated = False

        # Crisis termination: unemployment > 15% or hyperinflation
        if self.macro.unemployment_rate > 15.0 or self.macro.inflation_cpi_yoy > 15.0:
            terminated = True
            reward -= 50.0  # Catastrophic penalty

        if self.render_mode == "human":
            self.render()

        return self.macro.to_observation(), reward, terminated, truncated, {
            "macro": self.macro,
            "agent_outputs": self.agent_outputs,
        }

    def _update_macro_state(self):
        """Aggregate agent decisions into updated macro indicators."""
        o = self.agent_outputs  # shorthand

        # -- Labor market --
        lm = o.get("labor_market", {})
        self.macro.unemployment_rate = lm.get("unemployment_rate", self.macro.unemployment_rate)
        self.macro.wage_growth_yoy = lm.get("wage_growth_yoy_pct", self.macro.wage_growth_yoy)

        # -- Financial markets --
        ws = o.get("wall_street", {})
        sp_move = ws.get("sp500_move_pct", 0)
        self.macro.sp500_level *= (1 + sp_move / 100)
        self.macro.ten_year_yield = ws.get("ten_year_treasury_yield", self.macro.ten_year_yield)
        self.macro.vix = ws.get("vix", self.macro.vix)

        # -- Consumer spending --
        c_lo = o.get("consumer_low_income", {}).get("total_spending_billions", 440)
        c_mi = o.get("consumer_middle_income", {}).get("total_spending_billions", 911)
        c_hi = o.get("consumer_high_income", {}).get("total_spending_billions", 1030)
        self.macro.consumer_spending_billions = c_lo + c_mi + c_hi

        # -- Real estate --
        re = o.get("real_estate", {})
        self.macro.housing_starts_thousands = re.get("housing_starts_thousands",
                                                      self.macro.housing_starts_thousands)

        # -- Trade --
        fs = o.get("foreign_sector", {})
        self.macro.trade_deficit_billions = fs.get("trade_deficit_billions",
                                                    self.macro.trade_deficit_billions)

        # -- Government --
        gov = o.get("government", {})
        self.macro.federal_deficit_billions = gov.get("deficit_billions",
                                                       self.macro.federal_deficit_billions)

        # -- Inflation (derived from pricing decisions + demand) --
        corp_price = o.get("large_corporations", {}).get("price_increases_pct", 0)
        smb_price = o.get("small_businesses", {}).get("price_increases_pct", 0)
        rent_change = o.get("real_estate", {}).get("rent_change_pct", 0)
        oil_price = o.get("foreign_sector", {}).get("oil_price_per_barrel", 78)

        # Simple inflation model: weighted average of price pressures
        price_pressure = (corp_price * 0.35 + smb_price * 0.20 +
                          rent_change * 0.30 + (oil_price - 78) / 78 * 5 * 0.15)
        # Demand-pull component
        demand_pull = max(0, (self.macro.consumer_spending_billions - 2800) / 2800 * 2)
        # Phillips curve component
        phillips = max(0, (4.5 - self.macro.unemployment_rate) * 0.3)

        self.macro.inflation_cpi_yoy = max(-2.0,
            self.macro.inflation_cpi_yoy * 0.6 + (price_pressure + demand_pull + phillips) * 0.4
        )
        self.macro.core_pce_yoy = self.macro.inflation_cpi_yoy * 0.85  # Core excludes food/energy
        self.macro.inflation_pce_yoy = self.macro.inflation_cpi_yoy * 0.9

        # -- GDP (derived from spending components) --
        total_consumer = self.macro.consumer_spending_billions
        total_investment = (
            o.get("large_corporations", {}).get("capex_billions", 220) +
            o.get("small_businesses", {}).get("capex_billions", 80) +
            self.macro.housing_starts_thousands * 0.35  # rough residential investment
        )
        gov_spending = o.get("government", {}).get("total_spending_billions", 1625)
        net_exports = -self.macro.trade_deficit_billions

        self.macro.gdp_billions = total_consumer + total_investment + gov_spending + net_exports
        # Annualized growth rate (simplified)
        self.macro.gdp_growth_pct = ((self.macro.gdp_billions / 6900) - 1) * 4 * 100

    def _compute_reward(self) -> float:
        """
        Fed's dual mandate + financial stability scoring.

        Target: 2% inflation, <4% unemployment, stable markets.
        """
        reward = 0.0

        # Price stability: penalty for deviation from 2% target
        inflation_gap = abs(self.macro.core_pce_yoy - 2.0)
        if inflation_gap < 0.5:
            reward += 3.0   # Within tolerance
        elif inflation_gap < 1.0:
            reward += 1.0
        else:
            reward -= inflation_gap * 2.0  # Escalating penalty

        # Maximum employment: penalty for high unemployment
        if self.macro.unemployment_rate < 4.0:
            reward += 2.0
        elif self.macro.unemployment_rate < 5.0:
            reward += 1.0
        elif self.macro.unemployment_rate < 6.5:
            reward -= 1.0
        else:
            reward -= (self.macro.unemployment_rate - 4.0) * 1.5

        # Financial stability
        if self.macro.vix > 35:
            reward -= 3.0  # Market panic
        elif self.macro.vix > 25:
            reward -= 1.0

        # Don't invert the yield curve too aggressively
        ws = self.agent_outputs.get("wall_street", {})
        two_yr = ws.get("two_year_treasury_yield", 4.5)
        if two_yr > self.macro.ten_year_yield + 0.5:
            reward -= 2.0  # Deep inversion = recession signal

        # Fiscal sustainability: penalize accelerating debt costs
        gov = self.agent_outputs.get("government", {})
        interest = gov.get("interest_expense_billions", 225)
        if interest > 300:
            reward -= (interest - 300) / 100  # Debt spiral warning

        return reward

    def render(self):
        print(self.macro.summary())
        print(f"  Fed Funds Rate set to: {self.macro.fed_funds_rate:.2f}%")
        print(f"  Quarter reward: {self._compute_reward():.2f}")


# ─────────────────────────────────────────────────────────────
# INTERACTION PROTOCOL — How agents see each other
# ─────────────────────────────────────────────────────────────

INTERACTION_GRAPH = """
Agent Interaction Flow (per quarter):
======================================

    ┌──────────────┐
    │ FEDERAL      │  ← RL Agent sets fed funds rate
    │ RESERVE      │
    └──────┬───────┘
           │ rate decision
           ▼
    ┌──────────────┐     ┌──────────────┐
    │ WALL STREET  │────▶│ COMMERCIAL   │
    │ (markets     │     │ BANK         │
    │  price in)   │     │ (sets lending│
    └──────┬───────┘     │  rates)      │
           │             └──────┬───────┘
           │                    │ credit conditions
           ▼                    ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ LARGE CORPS  │────▶│ SMALL BIZ    │────▶│ REAL ESTATE  │
    │ (pricing,    │     │ (hiring,     │     │ (housing,    │
    │  hiring,     │     │  survival)   │     │  commercial) │
    │  capex)      │     └──────┬───────┘     └──────┬───────┘
    └──────┬───────┘            │                    │
           │                    │                    │
           ▼                    ▼                    ▼
    ┌──────────────────────────────────────────────────────┐
    │              CONSUMER HOUSEHOLDS                      │
    │  ┌───────────┐  ┌───────────┐  ┌───────────────┐    │
    │  │ Low-Income│  │Mid-Income │  │ High-Income   │    │
    │  │ (bottom   │  │ (40-80th  │  │ (top 20%)     │    │
    │  │  40%)     │  │  pctile)  │  │               │    │
    │  └───────────┘  └───────────┘  └───────────────┘    │
    └──────────────────────┬───────────────────────────────┘
                           │ spending, employment
                           ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │ LABOR MARKET │◀───▶│ GOVERNMENT   │◀───▶│ FOREIGN      │
    │ (synthesizes │     │ (fiscal,     │     │ SECTOR       │
    │  employment) │     │  stabilizers)│     │ (trade, FX)  │
    └──────────────┘     └──────────────┘     └──────────────┘
           │                    │                    │
           └────────────────────┴────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  MACRO STATE │  → Observation for RL agent
                    │  (aggregated)│  → Reward computation
                    └──────────────┘
"""


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP (Example with random policy)
# ─────────────────────────────────────────────────────────────

def run_random_policy(n_episodes: int = 1, render: bool = True):
    """Run the simulation with a random Fed policy (for testing)."""
    env = USEconomyEnv(render_mode="human" if render else None)

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        print(f"\n{'='*60}")
        print(f"EPISODE {ep+1}: 10-Year Economic Simulation")
        print(f"{'='*60}")
        print(f"Starting fed funds rate: {INITIAL_FED_RATE}%")

        while not done:
            action = env.action_space.sample()
            rate_change = USEconomyEnv.RATE_ACTIONS[action]
            direction = "HIKE" if rate_change > 0 else "CUT" if rate_change < 0 else "HOLD"
            print(f"\n>>> Fed action: {direction} {abs(rate_change)*100:.0f}bp")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"\n{'='*60}")
        print(f"EPISODE {ep+1} COMPLETE")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Final state: {env.macro.summary()}")


def run_with_stable_baselines():
    """
    Example of training with Stable-Baselines3 PPO.
    Requires: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        env = USEconomyEnv()
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=40,       # One full episode
            batch_size=20,
            n_epochs=10,
            gamma=0.99,        # Value long-term stability
            ent_coef=0.05,     # Encourage exploration
            tensorboard_log="./fed_tensorboard/",
        )

        print("Training Fed RL agent for 100 episodes...")
        model.learn(total_timesteps=4000)  # 100 episodes × 40 quarters
        model.save("fed_policy_v1")
        print("Model saved as fed_policy_v1")

        # Evaluate
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Evaluation total reward: {total_reward:.2f}")

    except ImportError:
        print("Install stable-baselines3: pip install stable-baselines3")


# ─────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(INTERACTION_GRAPH)
    print("\nTo run with random policy:  run_random_policy()")
    print("To train with PPO:         run_with_stable_baselines()")
    print("\nStarting random policy demo...\n")

    # NOTE: This requires an ANTHROPIC_API_KEY env variable set
    # and will make ~11 LLM calls per quarter × 40 quarters = ~440 calls per episode
    # Estimated cost per episode: ~$2-4 with Sonnet
    #
    # For testing without API calls, you can mock the LLMAgent.decide() method
    # to return random valid JSON decisions.

    run_random_policy(n_episodes=1, render=True)
