# Forex Analytics Dashboard - Wireframe Specification

## Overview
Interactive dashboard featuring a linear BaseMV graph with clickable date range boxes for significant changes (>5% jump/dip).

---

## Main Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FOREX ANALYTICS DASHBOARD                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  🏠 Dashboard  📊 Analysis  📈 Reports  ⚙️ Settings            👤 User Menu │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                HEADER METRICS                               │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   📊 Total BaseMV  │  📈 YTD Change    │  🔄 Active Deals │  ⚡ Last Update     │
│   $2.48B          │  +$450M (+15.2%) │  1,247 deals    │  2 mins ago         │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            BASEMV TREND ANALYSIS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  📅 Date Range: [Jan 2012 ▼] to [Dec 2012 ▼]     🔍 Zoom: [1M] [3M] [6M] [1Y] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  $3B ┌─────────────────────────────────────────────────────────────────┐   │
│      │                                                                 │   │
│  $2B │     ●                    ●──●                                   │   │
│      │    ╱ ╲                  ╱    ╲              [📈+12.5%]         │   │
│  $1B │   ╱   ╲                ╱      ╲            ┌─────────────┐     │   │
│      │  ╱     ╲              ╱        ╲           │  Mar 9-12   │     │   │
│  $0  │ ●       ●──●──●──●──●            ●──●──●──●│  2012       │●──● │   │
│      │                                             └─────────────┘     │   │
│ -$1B │                [📉-8.7%]                                        │   │
│      │               ┌─────────────┐                                   │   │
│ -$2B │               │  Feb 15-20  │                                   │   │
│      │               │  2012       │                                   │   │
│ -$3B └───────────────└─────────────┘───────────────────────────────────┘   │
│       Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct   Nov   Dec │
│                                                                             │
│  🔍 Significant Changes Detected: 3 periods with >5% variance              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dashboard Layout with Collapsible User Filter Sidebar

### Expanded Sidebar View:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FOREX ANALYTICS DASHBOARD                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  🏠 Dashboard  📊 Analysis  📈 Reports  ⚙️ Settings            👤 User Menu │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┬──────────────────────────────────────────────────────────┐
│                  │                    HEADER METRICS                       │
│   👥 USER FILTER │ ┌─────────────┬─────────────┬─────────────┬─────────────┐ │
│   [◀] Collapse   │ │📊 Total BaseMV│📈 YTD Change │🔄 Active   │⚡ Last     │ │
│                  │ │  $1.8B       │+$320M (+21%)│  947 deals │ Update     │ │
│  ┌─────────────┐ │ │              │             │            │ 2 mins ago │ │
│  │ User Type   │ │ └─────────────┴─────────────┴─────────────┴─────────────┘ │
│  │ ●All Users  │ │                                                          │
│  │ ○Entity     │ │ ┌────────────────────────────────────────────────────────┐ │
│  │ ○Counterparty│ │ │                 BASEMV TREND ANALYSIS                  │ │
│  └─────────────┘ │ ├────────────────────────────────────────────────────────┤ │
│                  │ │📅 Date: [Jan 2012▼] to [Dec 2012▼] 🔍[1M][3M][6M][1Y]│ │
│  ✅ FILTER ACTIVE│ │                                                        │ │
│  🎯 Entity:      │ │  $3B ┌──────────────────────────────────────────────┐ │ │
│     Goldman Sachs│ │      │                                              │ │ │
│                  │ │  $2B │     ●                    ●──●                │ │ │
│  📊 Filtered:    │ │      │    ╱ ╲                  ╱    ╲   [📈+12.5%]  │ │ │
│     947/1,247    │ │  $1B │   ╱   ╲                ╱      ╲ ┌───────────┐│ │ │
│     (76%)        │ │      │  ╱     ╲              ╱        ╲│  Mar 9-12 ││ │ │
│                  │ │  $0  │ ●       ●──●──●──●──●          ●│  2012     ││ │ │
│  📈 Data Overview│ │      │                                 └───────────┘│ │ │
│  • Records: 947  │ │ -$1B │                [📉-8.7%]                     │ │ │
│  • Currencies: 8 │ │      │               ┌─────────────┐                │ │ │
│  • Deal Types: 5 │ │ -$2B │               │  Feb 15-20  │                │ │ │
│                  │ │      │               │  2012       │                │ │ │
│  🔄 RESET FILTER │ │ -$3B └───────────────└─────────────┘────────────────┘ │ │
│                  │ │       Jan  Feb  Mar  Apr  May  Jun  Jul  Aug  Sep    │ │
│  ┌─────────────┐ │ │                                                        │ │
│  │Recent Files │ │ │  🔍 Significant Changes: 3 periods with >5% variance   │ │
│  │• Analysis_  │ │ └────────────────────────────────────────────────────────┘ │
│  │  20250123   │ │                                                          │
│  │• Report_GS  │ │ ┌────────────────────────────────────────────────────────┐ │
│  │  20250122   │ │ │              SIGNIFICANT CHANGE PERIODS                │ │
│  │• Summary_   │ │ ├────────────────────────────────────────────────────────┤ │
│  │  All_Users  │ │ │ [📉 Feb 15-20] [📈 Mar 9-12] [📈 Sep 5-8] [View All] │ │
│  └─────────────┘ │ │     (-8.7%)      (+12.5%)     (+7.3%)                 │ │
└──────────────────┴──┴────────────────────────────────────────────────────────┘
```

### Collapsed Sidebar View:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FOREX ANALYTICS DASHBOARD                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  🏠 Dashboard  📊 Analysis  📈 Reports  ⚙️ Settings            👤 User Menu │
└─────────────────────────────────────────────────────────────────────────────┘

┌──┬─────────────────────────────────────────────────────────────────────────┐
│  │                        HEADER METRICS                                   │
│👥│ ┌─────────────┬─────────────┬─────────────┬─────────────────────────────┐ │
│▶ │ │📊 Total BaseMV│📈 YTD Change │🔄 Active   │⚡ Last Update              │ │
│  │ │  $1.8B       │+$320M (+21%)│  947 deals │ 2 mins ago                 │ │
│  │ └─────────────┴─────────────┴─────────────┴─────────────────────────────┘ │
│🎯│                                                                         │
│✅│ ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ │🔍 Active Filter: Entity - Goldman Sachs (947/1,247 deals)           │ │
│📊│ └───────────────────────────────────────────────────────────────────────┘ │
│  │                                                                         │
│🔄│ ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ │                    BASEMV TREND ANALYSIS                              │ │
│📁│ ├───────────────────────────────────────────────────────────────────────┤ │
│  │ │📅 Date: [Jan 2012▼] to [Dec 2012▼] 🔍 Zoom: [1M][3M][6M][1Y]       │ │
│  │ │                                                                       │ │
│  │ │ $3B ┌─────────────────────────────────────────────────────────────┐   │ │
│  │ │     │                                                             │   │ │
│  │ │ $2B │     ●                    ●──●                               │   │ │
│  │ │     │    ╱ ╲                  ╱    ╲              [📈+12.5%]      │   │ │
│  │ │ $1B │   ╱   ╲                ╱      ╲            ┌─────────────┐  │   │ │
│  │ │     │  ╱     ╲              ╱        ╲           │  Mar 9-12   │  │   │ │
│  │ │ $0  │ ●       ●──●──●──●──●            ●──●──●──●│  2012       │  │   │ │
│  │ │     │                                             └─────────────┘  │   │ │
│  │ │-$1B │                [📉-8.7%]                                     │   │ │
│  │ │     │               ┌─────────────┐                               │   │ │
│  │ │-$2B │               │  Feb 15-20  │                               │   │ │
│  │ │     │               │  2012       │                               │   │ │
│  │ │-$3B └───────────────└─────────────┘───────────────────────────────┘   │ │
│  │ │      Jan   Feb   Mar   Apr   May   Jun   Jul   Aug   Sep   Oct      │ │
│  │ │                                                                       │ │
│  │ │ 🔍 Significant Changes Detected: 3 periods with >5% variance         │ │
│  │ └───────────────────────────────────────────────────────────────────────┘ │
└──┴─────────────────────────────────────────────────────────────────────────┘
```

### Sidebar Component Details:

#### User Selection Section:
```
┌─────────────────┐
│ 👥 USER FILTER  │
│ [◀] Collapse    │
│                 │
│ ┌─────────────┐ │
│ │ User Type   │ │
│ │ ●All Users  │ │
│ │ ○Entity     │ │
│ │ ○Counterparty│ │
│ └─────────────┘ │
│                 │
│ 📋 Select Entity│
│ ┌─────────────┐ │
│ │Goldman Sachs▼│ │
│ └─────────────┘ │
│                 │
│ OR              │
│                 │
│ 🏢 Counterparty │
│ ┌─────────────┐ │
│ │Select...   ▼│ │
│ └─────────────┘ │
└─────────────────┘
```

#### Active Filter Status:
```
┌─────────────────┐
│ ✅ FILTER ACTIVE│
│ 🎯 Entity:      │
│   Goldman Sachs │
│                 │
│ 📊 Filtered:    │
│   947/1,247     │
│   (76%)         │
│                 │
│ 📈 Data Stats:  │
│ • Records: 947  │
│ • Currencies: 8 │
│ • Deal Types: 5 │
│ • Date Range:   │
│   Jan-Dec 2012  │
└─────────────────┘
```

#### Recent Analysis Files:
```
┌─────────────────┐
│ 📁 RECENT FILES │
├─────────────────┤
│ • Analysis_     │
│   20250123_1415 │
│                 │
│ • Report_GS_    │
│   20250122_0930 │
│                 │
│ • Summary_All_  │
│   Users_20250121│
│                 │
│ • Export_Entity │
│   _20250120     │
│                 │
│ [📂 View All]   │
└─────────────────┘
```

┌─────────────────────────────────────────────────────────────────────────────┐
│                          SIGNIFICANT CHANGE PERIODS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┬─────────────────┬─────────────────┐                   │
│  │ 📉 Feb 15-20    │ 📈 Mar 9-12     │ 📈 Sep 5-8      │                  │
│  │ 2012 (-8.7%)    │ 2012 (+12.5%)   │ 2012 (+7.3%)    │                   │
│  └─────────────────┴─────────────────┴─────────────────┘                   │
│  │ ████████████████ ACTIVE TAB ████████████████████████████████████████     │
│                                                                             │
│  ┌─ PERIOD SUMMARY: Mar 9-12, 2012 ──────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  💰 **BaseMV Change:** +$615M (+12.5%)  │  📅 **Duration:** 4 days     │ │
│  │  📊 **From:** -$2.48B → **To:** -$1.86B │  🎯 **Impact:** HIGH         │ │
│  │                                                                        │ │
│  │  ┌─ KEY DRIVERS ──────────────────────────────────────────────────────┐ │ │
│  │  │ 🆕 New Deals: 19 (-$356M)    📤 Matured: 86 (+$1.83B)             │ │ │
│  │  │ 🔄 Existing: 128 (-$11.6M)   💱 Rates: EUR↓0.5%, AUD↓1.8%         │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  🔝 **Top Impact Deals:**                                              │ │
│  │  • FX-12847 (Matured): +$456M in JPY/EUR                              │ │
│  │  • FX-11234 (Matured): +$387M in JPY                                  │ │
│  │  • FX-15692 (New): -$125M in EUR                                      │ │
│  │                                                                        │ │
│  │  � **Key Insight:** The significant improvement was driven by 86      │ │
│  │  deals maturing and releasing $1.83B in negative BaseMV burden,       │ │
│  │  outweighing the negative impact of 19 new deals.                     │ │
│  │                                                                        │ │
│  │  [📊 Detailed Analysis] [📈 Advanced Charts] [📤 Export Report]        │ │
│  │                                                                         | |
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

### Alternative Tab Views:

#### When Feb 15-20, 2012 Tab is Selected:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┬─────────────────┬─────────────────┐                   │
│  │ 📉 Feb 15-20    │ 📈 Mar 9-12     │ 📈 Sep 5-8      │  [🔄 View All]    │
│  │ 2012 (-8.7%)    │ 2012 (+12.5%)   │ 2012 (+7.3%)    │                   │
│  └─────────────────┴─────────────────┴─────────────────┘                   │
│  │ ████████████████ ACTIVE TAB ████████████████████████████████████████     │
│                                                                             │
│  ┌─ PERIOD SUMMARY: Feb 15-20, 2012 ─────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  💰 **BaseMV Change:** -$425M (-8.7%)   │  📅 **Duration:** 6 days     │ │
│  │  📊 **From:** -$2.05B → **To:** -$2.48B │  🎯 **Impact:** HIGH         │ │
│  │                                                                        │ │
│  │  ┌─ KEY DRIVERS ──────────────────────────────────────────────────────┐ │ │
│  │  │ 🆕 New Deals: 35 (-$680M)    📤 Matured: 12 (+$156M)              │ │ │
│  │  │ 🔄 Existing: 89 (+$99M)      💱 Rates: USD↑2.1%, GBP↓1.3%         │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  🔝 **Top Impact Deals:**                                              │ │
│  │  • FX-10234 (New): -$245M in USD/GBP                                  │ │
│  │  • FX-10567 (New): -$189M in USD                                      │ │
│  │  • FX-09876 (Existing): +$67M in EUR                                  │ │
│  │                                                                        │ │
│  │  💡 **Key Insight:** The decline was primarily driven by 35 new       │ │
│  │  deals with significant negative impact, particularly in USD/GBP      │ │
│  │  pairs during market volatility.                                      │ │
│  │                                                                        │ │
│  │  [📊 Detailed Analysis] [📈 Advanced Charts] [📤 Export Report]        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### When Sep 5-8, 2012 Tab is Selected:
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┬─────────────────┬─────────────────┐                   │
│  │ 📉 Feb 15-20    │ 📈 Mar 9-12     │ 📈 Sep 5-8      │  [🔄 View All]    │
│  │ 2012 (-8.7%)    │ 2012 (+12.5%)   │ 2012 (+7.3%)    │                   │
│  └─────────────────┴─────────────────┴─────────────────┘                   │
│  │                                     ████████████████ ACTIVE TAB ███████ │
│                                                                             │
│  ┌─ PERIOD SUMMARY: Sep 5-8, 2012 ───────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  💰 **BaseMV Change:** +$380M (+7.3%)   │  📅 **Duration:** 4 days     │ │
│  │  📊 **From:** -$1.25B → **To:** -$870M  │  🎯 **Impact:** MEDIUM       │ │
│  │                                                                        │ │
│  │  ┌─ KEY DRIVERS ──────────────────────────────────────────────────────┐ │ │
│  │  │ 🆕 New Deals: 8 (-$125M)     📤 Matured: 23 (+$345M)              │ │ │
│  │  │ 🔄 Existing: 156 (+$160M)    💱 Rates: JPY↑1.8%, CHF↓0.9%         │ │ │
│  │  └────────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                        │ │
│  │  🔝 **Top Impact Deals:**                                              │ │
│  │  • FX-18234 (Matured): +$123M in JPY                                  │ │
│  │  • FX-17891 (Existing): +$89M in CHF/JPY                              │ │
│  │  • FX-18456 (Matured): +$78M in EUR                                   │ │
│  │                                                                        │ │
│  │  💡 **Key Insight:** Balanced improvement from both deal maturity      │ │
│  │  and favorable JPY rate movements, with minimal negative impact       │ │
│  │  from new deal additions.                                             │ │
│  │                                                                        │ │
│  │  [📊 Detailed Analysis] [📈 Advanced Charts] [📤 Export Report]        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed View Modal (When Clicking on a Period Box)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📈 PERIOD ANALYSIS: March 9-12, 2012                              [✕ Close] │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ OVERVIEW ─────────────────────────────────────────────────────────────┐ │
│  │  📅 Period: Mar 9, 2012 → Mar 12, 2012 (4 days)                      │ │
│  │  📊 BaseMV Change: +$615M (+12.5%)                                    │ │
│  │  💰 From: -$2.48B → To: -$1.86B                                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ KEY METRICS ──────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────────┬─────────────────┬─────────────────┬─────────────┐ │ │
│  │  │ 🆕 New Deals    │ 🔄 Existing     │ 📤 Matured      │ 💱 Rate     │ │ │
│  │  │                 │    Deals        │    Deals        │    Impact   │ │ │
│  │  │ 19 deals        │ 128 deals       │ 86 deals        │ Mixed       │ │ │
│  │  │ -$356M impact   │ -$11.6M change  │ +$1.83B relief  │ EUR ↓0.5%   │ │ │
│  │  └─────────────────┴─────────────────┴─────────────────┴─────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ DETAILED BREAKDOWN ───────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  📈 CONTRIBUTION ANALYSIS                                              │ │
│  │  ├─ Matured Deals: +$1.83B (297% of total change) ████████████████    │ │
│  │  ├─ New Deals: -$356M (-58% of total change) ██████                   │ │
│  │  └─ Existing Deals: -$11.6M (-2% of total change) █                   │ │
│  │                                                                        │ │
│  │  🔝 TOP IMPACT DEALS                                                   │ │
│  │  ┌─────────────┬─────────────┬─────────────┬─────────────────────────┐ │ │
│  │  │ Deal ID     │ Type        │ Impact      │ Currencies              │ │ │
│  │  ├─────────────┼─────────────┼─────────────┼─────────────────────────┤ │ │
│  │  │ FX-12847    │ Matured     │ +$456M      │ JPY, EUR                │ │ │
│  │  │ FX-11234    │ Matured     │ +$387M      │ JPY                     │ │ │
│  │  │ FX-15692    │ New         │ -$125M      │ EUR                     │ │ │
│  │  │ FX-13457    │ New         │ -$89M       │ JPY                     │ │ │
│  │  │ FX-14785    │ Existing    │ -$45M       │ AUD                     │ │ │
│  │  └─────────────┴─────────────┴─────────────┴─────────────────────────┘ │ │
│  │                                                                        │ │
│  │  💱 CURRENCY RATE CHANGES                                              │ │
│  │  ├─ EUR Forward: 0.749 → 0.745 (-0.5%)                                │ │
│  │  ├─ AUD Forward: 0.719 → 0.706 (-1.8%)                                │ │
│  │  └─ JPY Spot: 105.985 → 106.015 (+0.03%)                              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─ AI BUSINESS SUMMARY ──────────────────────────────────────────────────┐ │
│  │  💡 **Executive Summary**                                              │ │
│  │                                                                        │ │
│  │  Between March 9 and March 12, 2012, the portfolio experienced a      │ │
│  │  significant improvement in BaseMV of approximately **$1.46 billion**, │ │
│  │  rising from a negative **$2.48 billion** to **$1.02 billion**.       │ │
│  │                                                                        │ │
│  │  **Key Drivers:**                                                      │ │
│  │  • **Matured Deals:** 86 deals released **$1.83B** negative burden    │ │
│  │  • **New Deals:** 19 deals contributed **-$356M** to portfolio        │ │
│  │  • **Rate Fluctuations:** Minor EUR/AUD rate decreases               │ │
│  │                                                                        │ │
│  │  The improvement was predominantly driven by deal lifecycle            │ │
│  │  management rather than market rate movements.                         │ │
│  │                                                                        │ │
│  │  [📊 Generate Full Report] [📤 Export Analysis] [🔄 Refresh Data]      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                          [📋 View Raw Data] [📈 Advanced Charts]         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. **Collapsible Sidebar Component**
- **Width:** 250px expanded, 50px collapsed
- **Toggle:** Click button to expand/collapse
- **Sections:**
  - User type selection (radio buttons)
  - Entity/Counterparty dropdown
  - Filter status indicator
  - Data overview statistics
  - Recent analysis files
- **Persistence:** Remember collapsed/expanded state
- **Responsive:** Auto-collapse on mobile devices

### 2. **User Filter Interface**
- **User Type Selection:** Radio buttons for All Users, Entity, Counterparty
- **Dynamic Dropdowns:** Entity list or Counterparty list based on selection
- **Real-time Filtering:** Data updates immediately on selection
- **Filter Status:** Visual indicator showing active filters and coverage percentage
- **Reset Option:** One-click button to clear all filters

### 3. **Main Graph Component**
- **Type:** Interactive line chart with zoom controls
- **Features:**
  - Hover tooltips showing exact values
  - Zoom controls (1M, 3M, 6M, 1Y)
  - Date range selectors
  - Highlight significant change periods

### 2. **Significant Change Detection**
- **Threshold:** >5% change in BaseMV over consecutive periods
- **Visual Indicators:**
  - 📈 Green for positive changes
  - 📉 Red for negative changes
  - Percentage and absolute value display

### 3. **Period Analysis Interface**
- **Layout:** Tabbed interface with summary view below
- **Tab Headers:** Show period, change percentage, and visual indicators
- **Active Tab Indicator:** Visual highlight showing selected period
- **Summary Content:** 
  - Key metrics in grid format
  - Top impact deals list
  - Key insights summary
  - Action buttons for detailed analysis
- **View All Option:** Comparison table and trend analysis across all periods
- **Interaction:** Click tabs to switch periods, access detailed modal for deep-dive analysis

### 4. **Detailed Analysis Modal**
- **Sections:**
  - Overview metrics
  - Key metrics grid
  - Detailed breakdown with contribution analysis
  - Top impact deals table
  - Currency rate changes
  - AI-generated business summary
- **Actions:** Export, generate reports, view raw data

---

## Responsive Design Considerations

### Desktop (1200px+)
```
┌──────────────┬──────────────────────────────────────────────┐
│   SIDEBAR    │                MAIN CONTENT                  │
│   (250px)    │              (950px)                         │
│              │                                              │
│ - User Filter│ - Header metrics (4-column)                  │
│ - Filter Status│ - Full graph area                          │
│ - Data Stats │ - 3-column period tabs                       │
│ - Recent Files│ - Large modal dialogs                       │
└──────────────┴──────────────────────────────────────────────┘
```

### Tablet (768px - 1199px)
```
┌──────────┬────────────────────────────────────────────────┐
│ SIDEBAR  │              MAIN CONTENT                      │
│ (200px)  │            (568px - 968px)                     │
│          │                                                │
│- Compact │ - Header metrics (2x2 grid)                    │
│  Filters │ - Graph slightly compressed                    │
│- Status  │ - Period tabs (2-column on smaller screens)    │
│- Files   │ - Modal with scrollable content                │
└──────────┴────────────────────────────────────────────────┘
```

### Mobile (< 768px)
```
┌─────────────────────────────────────┐
│           MOBILE LAYOUT             │
├─────────────────────────────────────┤
│ [☰] Menu  [🔍] Filter    [👤] User  │
├─────────────────────────────────────┤
│      Header metrics (stacked)       │
├─────────────────────────────────────┤
│         Graph (pan/zoom)            │
├─────────────────────────────────────┤
│     Period tabs (stacked)           │
├─────────────────────────────────────┤
│      Full-screen modals             │
└─────────────────────────────────────┘

Sidebar becomes:
- Slide-out drawer (triggered by ☰ button)
- Overlay on top of main content
- Full-height overlay
- Touch-friendly controls
```

---

## Color Scheme & Styling

### Primary Colors
- **Background:** #F8F9FA (Light gray)
- **Primary:** #007BFF (Blue)
- **Success:** #28A745 (Green) - for positive changes
- **Danger:** #DC3545 (Red) - for negative changes
- **Warning:** #FFC107 (Amber) - for alerts

### Typography
- **Headers:** Inter, 24px, Bold
- **Body:** Inter, 14px, Regular
- **Metrics:** Inter, 18px, Semi-bold
- **Labels:** Inter, 12px, Medium

### Interactive Elements
- **Buttons:** Rounded corners (8px), hover effects
- **Cards:** Box shadow, hover lift effect
- **Modal:** Backdrop blur, smooth transitions

---

## Technical Implementation Notes

### Data Structure Required
```javascript
{
  "significantPeriods": [
    {
      "startDate": "2012-03-09",
      "endDate": "2012-03-12",
      "changePercent": 12.5,
      "changeAmount": 615000000,
      "changeType": "increase",
      "analysis": { /* detailed analysis data */ }
    }
  ],
  "dailyBaseMV": [
    { "date": "2012-01-01", "value": -2480000000 },
    // ... more daily data
  ]
}
```

### Key Interactions
1. **Sidebar toggle:** Expand/collapse sidebar to maximize content area
2. **User filter selection:** Choose All Users, Entity, or Counterparty filtering
3. **Dynamic filtering:** Real-time data updates based on user selection
4. **Filter status:** Visual feedback showing active filters and data coverage
5. **Graph hover:** Show tooltip with date and value
6. **Period tab selection:** Switch between different significant change periods
7. **View All toggle:** Compare all periods in summary table format
8. **Detailed Analysis button:** Open comprehensive modal for selected period
9. **Date range selection:** Update graph and recalculate periods
10. **Zoom controls:** Focus on specific time periods
11. **Export functions:** Generate PDF/Excel reports for individual periods or all periods
12. **Recent files access:** Quick access to previously generated analysis reports

This wireframe provides a comprehensive view of the dashboard with clear user flow and detailed specifications for implementation. Would you like me to elaborate on any specific component or create additional views?
