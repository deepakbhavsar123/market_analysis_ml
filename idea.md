# Forex Portfolio Market Value Analysis - Agentic Solution

## Problem Statement

Users with forex/spot trading portfolios experience daily market value fluctuations. These fluctuations can be attributed to multiple factors:
- **New deals** being added to the portfolio
- **Spot rate fluctuations** affecting existing positions
- **Forward rate (FwdRate) fluctuations** impacting forward contracts

## Objective

Develop an **agentic solution** that autonomously analyzes forex trading data to provide detailed breakdown of market value fluctuations by identifying the contribution of each factor:
1. New deals
2. Forward rate changes
3. Spot rate changes

This will give users comprehensive insights into **why** their market value changed and **what specific factors** drove the fluctuation.

## Data Sources

- Dataset and column descriptions are provided in the `dataset/` folder
- `Cashflows_FX_v3(Data_updated).csv` - Main trading data
- `FX_Field_Descriptions.csv` - Data dictionary

## Dataset Column Descriptions

### Identification & System Fields
- **queryid**: Internal id of the query
- **rptno**: Unique report number associated with each report run
- **DealNo**: The unique Quantum deal number
- **DealId**: The unique Analytics deal identifier
- **DealSide**: Used to identify the separate sides of a deal (e.g., for FX there are two sides, the Sell (Pay) of one currency is one side and Buy (Receive) of the other currency is the other side)

### Transaction Classification
- **TransactionType**: Transaction Type is a system defined categorization of treasury products (E.g. FX, IS, CS, MM etc)
- **ForwardType**: Defines the type of FX Deal (e.g. Spot/Forward, Back to Back or Swap)
- **SettlementType**: Defines whether the currency amounts are to be delivered or non delivered
- **PayorReceive**: Used to identify whether the individual deal side is Paying (selling) or Receiving (Buying) each currency

### Date Fields
- **PositionDate**: Date the report was run for
- **CashFlowDate**: The date of the Cashflow is to be settled in the market
- **DealDate**: The date the transaction was undertaken
- **MaturityDate**: The Maturity date of the transaction

### Business Structure
- **Entity**: Your business structure (portfolio) at a transaction-level against which the deal is booked, this is the lowest level in the business structure
- **EntityGroup1**: The first level parent of the Entity, part of your business structure at the grouping level
- **EntityGroup2**: The second level parent of the Entity, part of your business structure at the grouping level

### Counterparty Information
- **Counterparty**: The counterparty in the market to whom the deal was transacted with
- **CounterpartyGroup1**: The first level parent of the Counterparty
- **CounterpartyGroup2**: The second level parent of the Counterparty

### Instrument & Currency Information
- **Instrument**: The instrument the deal is based on
- **InstrumentGroup1**: The first level parent of the Instrument
- **CcyPair**: The two currencies traded in the cross currency deal. The first currency in the Pair is the Commodity Currency (1 unit), the second is the term currency (per units)
- **Currency**: The transaction Currency, this is reflected for each of the two sides of an FX Deal
- **BaseCcy**: The "Base Currency" as defined in Valuation Control or the "Currency Conversion" currency selected in Assumptions. Used to convert all other currencies to this currency

### Valuation & Market Value Fields
- **ValuationModel**: The FX Valuation Model used to revalue the deal, for FX this will be either Forward, Forward NPV or NPV Spot
- **MarketValue**: The Market Value reflected in Transaction Currency of the deal using the revaluation methodology specified in the Valuation Model Column
- **BaseMV**: **[KEY FIELD]** The Market Value reflected in Base Currency of the deal using the revaluation methodology specified in the Valuation Model Column
- **FaceValue**: The face value (amount) of a deal reflected in transaction currency for each deal side
- **PrincipalOutstanding**: Total future principal cashflows for a transaction

### Rate Information
- **SpotRate**: **[KEY FIELD]** The spot rate to convert the transaction currency to base currency
- **SpotFactor**: The spot rate to convert the transaction currency to base currency expressed as a factor (multiplier)
- **FwdRate**: **[KEY FIELD]** The forward rate to convert the transaction currency to base currency
- **FwdFactor**: The forward rate to convert the transaction currency to base currency expressed as a multiplier
- **ZeroRate**: The Zero rate from Position Date to Maturity Date, also known as a zero-coupon rate. Calculation varies by Valuation Method:
  - If Valuation Method = FORWARD: Zero Rate is 0.00 as not needed
  - If Valuation Method = FORWARD NPV: Zero Rate is in the Base Currency
  - If Valuation Method = NPV SPOT: Zero Rate is in the Transaction Currency

### Risk Metrics & Sensitivities
- **BPDelta**: Value change in transaction currency for a user-defined shift in interest rates, calculated for the FORWARD NPV and NPV Spot valuation models
- **BPGamma**: The rate of change of delta, calculated for the FORWARD NPV and NPV Spot valuation models
- **IRR**: The internal rate of return for a transaction, calculated for the FORWARD NPV and NPV Spot valuation models
- **MacaulayDuration**: The weighted average time until cash flows are received, calculated for the FORWARD NPV and NPV Spot valuation models
- **ModifiedDuration**: A measure used to express the sensitivity of a financial instrument's price to changes in interest rates, calculated for the FORWARD NPV and NPV Spot valuation models
- **Theta**: The rate of change of value with respect to time, calculated for the FORWARD NPV and NPV Spot valuation models
- **Convexity**: Measures curvature of price-yield curve. Defined as the second derivative of the price with respect to yield, calculated for the FORWARD NPV and NPV Spot valuation models

### Discount Factors & Curves
- **CurrencyDF**: The transaction currency discount factor for the cashflow date of a transaction, calculated for the FORWARD NPV and NPV Spot valuation models
- **BaseDF**: The Base currency discount factor for the cashflow date of a transaction, calculated for the FORWARD NPV and NPV Spot valuation models
- **DiscountYieldCurve**: The discount curve used to calculate the present value of future cashflows, not required for the FORWARD valuation model

### Time Profile & Maturity
- **CashflowType**: The type of the Cashflow
- **TimeProfile**: The Days to Maturity grouped into Time Profile Monthly Buckets (Monthly1, Monthly2, Monthly3, etc) reflecting the time period when the deal will mature
- **TermofDeal**: The full term (in days) from deal inception until Maturity of the deal, reflected as the Maturity Date - Deal Date
- **DaystoMaturity**: The time remaining (in days) until Maturity of the deal, reflected as the Maturity Date - Position Date
- **TimeProfileStart**: The start date of the relevant Time Profile bucket
- **TimeProfileEnd**: The end date of the relevant Time Profile bucket

## Important Data Structure

**Deal Structure**: Each deal in the dataset contains **two rows**:
- **Pay leg**: Negative BaseMV value (amount deducted from account)
- **Receive leg**: Positive BaseMV value (amount received)

This dual-row structure must be considered in all calculations to ensure accurate attribution analysis. When calculating market value changes for deals, both legs must be processed together to get the net impact.

## Manual Process Breakdown

(This is just using the newDeals, FwdRate & SpotRate there can be other features which might have also contributed)

The current manual analysis follows these steps:

### Step 1: New Deals Analysis
**Objective**: Identify contribution of new deals to market value fluctuation

**Process**:
- Compare two consecutive dates (e.g., 2012-March-06 vs 2012-March-07)
- Calculate total market value difference: `Date2_MarketValue - Date1_MarketValue`
- Identify new deals: deals that exist on Date2 but not on Date1
- Sum the market value of these new deals to determine their contribution to the overall fluctuation

### Step 2: Forward Rate Impact Analysis
**Objective**: Quantify impact of forward rate fluctuations on existing deals

**Process**:
- Exclude new deals from the analysis
- Filter deals with valuation models: `"FORWARD"` and `"FORWARD NPV"`
- **Important**: Group by DealId to handle pay/receive pairs correctly
- Identify deals with FwdRate fluctuations between the two dates
- Calculate the net market value change for each deal (sum of pay + receive legs)
- Determine contribution: `Sum of (Date2_NetBaseMV - Date1_NetBaseMV)` for deals with forward rate fluctuations

### Step 3: Spot Rate Impact Analysis
**Objective**: Measure impact of spot rate changes on spot deals

**Process**:
- Filter deals with valuation model: `"NPV SPOT"`
- **Important**: Group by DealId to handle pay/receive pairs correctly
- Identify deals with Spot Rate fluctuations between the two dates
- Calculate the net market value change for each deal (sum of pay + receive legs)
- Determine contribution: `Sum of (Date2_NetBaseMV - Date1_NetBaseMV)` for deals with spot rate fluctuations

## Expected Outcome

The Machine Learning based solution which can give the factors which contributed to the BaseMV rise/fall.