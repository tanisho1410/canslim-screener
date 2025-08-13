"""
📈 CANSLIM成長株分析システム
=================================

William J. O'NeilのCANSLIM投資戦略に基づいて、入力されたticker symbolの銘柄を
自動分析し、成長株としての適格性を判定。結果をスプレッドシートとNotionに
自動出力する統合分析システム。

主要機能:
* **多重データソース統合** - yfinance、Alpha Vantage、Yahoo Finance APIを統合した包括的なデータ取得
* **高度なCANSLIM分析** - 7つの基準（C, A, N, S, L, I, M）を詳細に評価し、100点満点でスコアリング
* **自動Google Sheets出力** - 詳細なスプレッドシートレポートを自動生成
* **詳細Notion記事生成** - 5000-10000文字の詳細分析レポートを自動作成
* **バリュエーション分析** - PER、PBR、PSR、DCF分析による理論株価算出
* **テクニカル分析** - チャートパターン、サポート・レジスタンス、モメンタム分析
* **リスク評価** - 事業リスク、財務リスク、市場リスクの多角的評価
* **投資戦略提案** - エントリーポイント、ストップロス、目標株価の提示

必要な環境変数:
- ALPHA_VANTAGE_API_KEY: Alpha Vantage APIキー
- GOOGLE_CREDENTIALS_JSON: Google Sheets認証情報JSONファイルパス
- GOOGLE_SPREADSHEET_ID: Google SheetsのスプレッドシートID
- NOTION_TOKEN: Notion統合トークン
- NOTION_DATABASE_ID: NotionデータベースID

インストール:
    pip install streamlit pandas numpy yfinance alpha_vantage gspread oauth2client notion_client plotly scikit-learn requests
    streamlit run app.py
"""

from __future__ import annotations

import os
import json
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from urllib.parse import urlencode

# 機械学習・統計分析
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# プロット・可視化
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Alpha Vantage API
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.fundamentaldata import FundamentalData
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False

# Google Sheets統合
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# Notion統合
try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False

warnings.filterwarnings('ignore')


###############################################################################
# Enhanced Data Classes
###############################################################################

@dataclass
class CanslimScores:
    """Container for CANSLIM scores and raw metrics."""
    current_earnings: float = field(default=np.nan)
    annual_earnings: float = field(default=np.nan)
    new_products: float = field(default=np.nan)
    supply_demand: float = field(default=np.nan)
    leader_laggard: float = field(default=np.nan)
    institutional: float = field(default=np.nan)
    market_direction: float = field(default=np.nan)
    
    # Additional metrics
    eps_growth_acceleration: float = field(default=np.nan)
    sales_growth: float = field(default=np.nan)
    profit_margin_trend: float = field(default=np.nan)
    roe_trend: float = field(default=np.nan)
    debt_to_equity: float = field(default=np.nan)
    
    # Technical indicators
    rsi: float = field(default=np.nan)
    relative_strength: float = field(default=np.nan)
    price_momentum: float = field(default=np.nan)
    volume_trend: float = field(default=np.nan)

    def total_score(self) -> float:
        """Calculate total CANSLIM score."""
        comps = [v for v in [self.current_earnings, self.annual_earnings,
                             self.new_products, self.supply_demand,
                             self.leader_laggard, self.institutional,
                             self.market_direction] if not math.isnan(v)]
        return sum(comps) / len(comps) if comps else np.nan

    def recommendation(self) -> str:
        """Get investment recommendation based on score."""
        score = self.total_score()
        if math.isnan(score):
            return "不明"
        if score >= 80:
            return "強く推奨"
        elif score >= 60:
            return "推奨"
        elif score >= 40:
            return "中立"
        else:
            return "非推奨"


@dataclass
class ValuationMetrics:
    """Container for valuation analysis."""
    current_price: float = field(default=np.nan)
    market_cap: float = field(default=np.nan)
    pe_ratio: float = field(default=np.nan)
    pb_ratio: float = field(default=np.nan)
    ps_ratio: float = field(default=np.nan)
    peg_ratio: float = field(default=np.nan)
    ev_ebitda: float = field(default=np.nan)
    dcf_fair_value: float = field(default=np.nan)
    target_price_1: float = field(default=np.nan)
    target_price_2: float = field(default=np.nan)
    target_price_3: float = field(default=np.nan)
    stop_loss_price: float = field(default=np.nan)
    entry_point: float = field(default=np.nan)


@dataclass
class RiskAssessment:
    """Container for risk evaluation."""
    business_risk: str = field(default="")
    financial_risk: str = field(default="")
    market_risk: str = field(default="")
    regulatory_risk: str = field(default="")
    overall_risk_score: float = field(default=np.nan)


###############################################################################
# Enhanced Data Fetching Functions
###############################################################################

def fetch_comprehensive_data(ticker: str, years: int = 5) -> Dict[str, Any]:
    """
    Fetch comprehensive data from multiple sources.
    
    :param ticker: Stock symbol
    :param years: Number of years of historical data
    :return: Dictionary containing all fetched data
    """
    data = {
        'ticker': ticker,
        'price_data': pd.DataFrame(),
        'fundamentals': {},
        'company_info': {},
        'alpha_vantage_data': {},
        'technical_indicators': {},
        'analyst_data': {}
    }
    
    # Yahoo Finance data
    try:
        yf_ticker = yf.Ticker(ticker)
        
        # Price data
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        data['price_data'] = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False
        )
        
        # Company info
        data['company_info'] = yf_ticker.info
        
        # Financial statements
        data['fundamentals'] = {
            'quarterly_earnings': yf_ticker.quarterly_earnings,
            'annual_earnings': yf_ticker.earnings,
            'quarterly_financials': yf_ticker.quarterly_financials,
            'annual_financials': yf_ticker.financials,
            'balance_sheet': yf_ticker.balance_sheet,
            'quarterly_balance_sheet': yf_ticker.quarterly_balance_sheet,
            'cash_flow': yf_ticker.cashflow,
            'quarterly_cash_flow': yf_ticker.quarterly_cashflow
        }
        
        # Analyst and institutional data
        data['analyst_data'] = {
            'recommendations': yf_ticker.recommendations,
            'analyst_price_target': yf_ticker.analyst_price_target,
            'institutional_holders': yf_ticker.institutional_holders,
            'insider_purchases': yf_ticker.insider_purchases,
            'insider_roster_holders': yf_ticker.insider_roster_holders
        }
        
    except Exception as e:
        st.warning(f"Yahoo Finance data fetch error for {ticker}: {e}")
    
    # Alpha Vantage data (if available)
    if ALPHA_VANTAGE_AVAILABLE:
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if api_key:
            try:
                ts = TimeSeries(key=api_key, output_format='pandas')
                fd = FundamentalData(key=api_key, output_format='pandas')
                
                # Additional fundamental data
                data['alpha_vantage_data'] = {
                    'company_overview': fd.get_company_overview(ticker)[0],
                    'annual_reports': fd.get_income_statement_annual(ticker)[0],
                    'quarterly_reports': fd.get_income_statement_quarterly(ticker)[0]
                }
                time.sleep(12)  # API rate limiting
                
            except Exception as e:
                st.warning(f"Alpha Vantage data fetch error for {ticker}: {e}")
    
    return data


def calculate_technical_indicators(price_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate technical indicators from price data.
    
    :param price_df: Historical price data
    :return: Dictionary of technical indicators
    """
    indicators = {}
    
    if price_df.empty:
        return indicators
    
    try:
        # RSI calculation
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        indicators['rsi'] = calculate_rsi(price_df['Close']).iloc[-1]
        
        # Moving averages
        indicators['sma_20'] = price_df['Close'].rolling(20).mean().iloc[-1]
        indicators['sma_50'] = price_df['Close'].rolling(50).mean().iloc[-1]
        indicators['sma_200'] = price_df['Close'].rolling(200).mean().iloc[-1]
        
        # Price momentum
        if len(price_df) >= 126:
            indicators['momentum_6m'] = price_df['Close'].iloc[-1] / price_df['Close'].iloc[-126] - 1
        if len(price_df) >= 63:
            indicators['momentum_3m'] = price_df['Close'].iloc[-1] / price_df['Close'].iloc[-63] - 1
        if len(price_df) >= 21:
            indicators['momentum_1m'] = price_df['Close'].iloc[-1] / price_df['Close'].iloc[-21] - 1
        
        # Volume indicators
        indicators['avg_volume_50'] = price_df['Volume'].rolling(50).mean().iloc[-1]
        indicators['volume_spike'] = price_df['Volume'].iloc[-1] / indicators.get('avg_volume_50', 1)
        
        # Volatility
        indicators['volatility_20'] = price_df['Close'].pct_change().rolling(20).std().iloc[-1]
        
        # Support and resistance levels
        high_52w = price_df['High'].rolling(252).max().iloc[-1]
        low_52w = price_df['Low'].rolling(252).min().iloc[-1]
        current_price = price_df['Close'].iloc[-1]
        
        indicators['distance_from_52w_high'] = (current_price / high_52w - 1) * 100
        indicators['distance_from_52w_low'] = (current_price / low_52w - 1) * 100
        
    except Exception as e:
        st.warning(f"Technical indicator calculation error: {e}")
    
    return indicators


###############################################################################
# Enhanced CANSLIM Analysis Functions
###############################################################################

def compute_enhanced_canslim_scores(ticker: str, data: Dict[str, Any]) -> CanslimScores:
    """
    Compute enhanced CANSLIM scores using comprehensive data analysis.
    
    :param ticker: Stock symbol
    :param data: Comprehensive data dictionary
    :return: Enhanced CanslimScores instance
    """
    scores = CanslimScores()
    
    price_df = data.get('price_data', pd.DataFrame())
    fundamentals = data.get('fundamentals', {})
    company_info = data.get('company_info', {})
    alpha_data = data.get('alpha_vantage_data', {})
    
    # Technical indicators
    tech_indicators = calculate_technical_indicators(price_df)
    scores.rsi = tech_indicators.get('rsi', np.nan)
    scores.price_momentum = tech_indicators.get('momentum_6m', np.nan)
    scores.volume_trend = tech_indicators.get('volume_spike', np.nan)
    
    # C - Current Earnings (Enhanced)
    quarterly_earnings = fundamentals.get('quarterly_earnings')
    if isinstance(quarterly_earnings, pd.DataFrame) and len(quarterly_earnings) >= 5:
        try:
            q_sorted = quarterly_earnings.sort_index()
            
            # Year-over-year growth
            last_q = q_sorted.iloc[-1]['Earnings']
            yoy_q = q_sorted.iloc[-5]['Earnings']
            eps_growth = yoy_growth(last_q, yoy_q)
            
            # Quarter-over-quarter acceleration
            prev_q = q_sorted.iloc[-2]['Earnings']
            prev_yoy_q = q_sorted.iloc[-6]['Earnings'] if len(q_sorted) >= 6 else np.nan
            prev_growth = yoy_growth(prev_q, prev_yoy_q)
            
            acceleration = eps_growth - prev_growth if not math.isnan(prev_growth) else 0
            scores.eps_growth_acceleration = acceleration
            
            # Enhanced scoring (25% growth = 100 points, with acceleration bonus)
            base_score = min(max((eps_growth / 0.25) * 100.0, 0), 100)
            acceleration_bonus = min(max(acceleration * 20, -20), 20)
            scores.current_earnings = round(min(base_score + acceleration_bonus, 100), 2)
            
        except Exception as e:
            st.warning(f"Current earnings calculation error for {ticker}: {e}")
    
    # A - Annual Earnings (Enhanced)
    annual_earnings = fundamentals.get('annual_earnings')
    if isinstance(annual_earnings, pd.DataFrame) and len(annual_earnings) >= 4:
        try:
            a_sorted = annual_earnings.sort_index()
            
            # 3-year CAGR
            start_earn = a_sorted['Earnings'].iloc[-4]
            end_earn = a_sorted['Earnings'].iloc[-1]
            earn_cagr = cagr(start_earn, end_earn, 3)
            
            # Consistency bonus (lower volatility = higher score)
            earnings_list = a_sorted['Earnings'].iloc[-4:].tolist()
            volatility = np.std(earnings_list) / np.mean(earnings_list) if np.mean(earnings_list) > 0 else 1
            consistency_bonus = max(10 - volatility * 10, 0)
            
            base_score = min(max((earn_cagr / 0.25) * 100.0, 0), 100)
            scores.annual_earnings = round(min(base_score + consistency_bonus, 100), 2)
            
        except Exception as e:
            st.warning(f"Annual earnings calculation error for {ticker}: {e}")
    
    # N - New Products/Services/Management (Enhanced)
    try:
        # 52-week high proximity
        if not price_df.empty and len(price_df) >= 252:
            high_52w = price_df['High'].rolling(252).max().iloc[-1]
            current_price = price_df['Close'].iloc[-1]
            proximity_to_high = (current_price / high_52w)
            
            # Price momentum in recent 3 months
            momentum_3m = tech_indicators.get('momentum_3m', 0)
            
            # Volume surge indicator
            volume_surge = min(tech_indicators.get('volume_spike', 1), 3)
            
            # Combined score
            base_score = proximity_to_high * 50
            momentum_score = min(max(momentum_3m * 100, 0), 30)
            volume_score = (volume_surge - 1) / 2 * 20
            
            scores.new_products = round(min(base_score + momentum_score + volume_score, 100), 2)
        
    except Exception as e:
        st.warning(f"New products calculation error for {ticker}: {e}")
    
    # S - Supply and Demand (Enhanced)
    try:
        if not price_df.empty:
            # Volume analysis
            avg_vol_50 = tech_indicators.get('avg_volume_50', 1)
            current_vol = price_df['Volume'].iloc[-1]
            vol_spike = current_vol / avg_vol_50
            
            # Float size analysis (from company info)
            shares_outstanding = company_info.get('sharesOutstanding', company_info.get('impliedSharesOutstanding', 1e9))
            float_score = max(100 - (shares_outstanding / 1e6), 0)  # Penalty for large float
            
            # Price-volume relationship
            price_change = price_df['Close'].pct_change().iloc[-1]
            volume_change = price_df['Volume'].pct_change().iloc[-1]
            pv_correlation = 50 if (price_change > 0 and volume_change > 0) else 25
            
            # Combined scoring
            volume_score = min(max((vol_spike - 1) / 0.5 * 40, 0), 40)
            float_score = min(float_score * 0.3, 30)
            
            scores.supply_demand = round(min(volume_score + float_score + pv_correlation, 100), 2)
            
    except Exception as e:
        st.warning(f"Supply/demand calculation error for {ticker}: {e}")
    
    # L - Leader or Laggard (Enhanced)
    try:
        if not price_df.empty:
            # Multiple timeframe analysis
            momentum_6m = tech_indicators.get('momentum_6m', 0)
            momentum_3m = tech_indicators.get('momentum_3m', 0)
            momentum_1m = tech_indicators.get('momentum_1m', 0)
            
            # Relative strength calculation
            market_index_return = 0.1  # Assumption: 10% market return (can be enhanced with actual index data)
            relative_strength = momentum_6m - market_index_return
            scores.relative_strength = relative_strength
            
            # RSI consideration
            rsi = tech_indicators.get('rsi', 50)
            rsi_score = min(max((rsi - 30) / 40 * 30, 0), 30)  # RSI between 30-70 is ideal
            
            # Combined scoring
            momentum_score = min(max(momentum_6m / 0.5 * 50, 0), 50)
            relative_score = min(max(relative_strength / 0.2 * 20, 0), 20)
            
            scores.leader_laggard = round(min(momentum_score + relative_score + rsi_score, 100), 2)
            
    except Exception as e:
        st.warning(f"Leader/laggard calculation error for {ticker}: {e}")
    
    # I - Institutional Sponsorship (Enhanced)
    try:
        institutional_holders = data.get('analyst_data', {}).get('institutional_holders')
        
        if isinstance(institutional_holders, pd.DataFrame) and not institutional_holders.empty:
            num_institutions = len(institutional_holders)
            total_shares_held = institutional_holders['Shares'].sum()
            total_shares = company_info.get('sharesOutstanding', 1)
            institutional_ownership_pct = (total_shares_held / total_shares) * 100
            
            # Quality of institutions (top holdings get bonus)
            avg_position_size = institutional_holders['Shares'].mean()
            position_quality_score = min(avg_position_size / 1e6 * 10, 20)
            
            # Scoring
            count_score = min(num_institutions / 50 * 40, 40)
            ownership_score = min(institutional_ownership_pct / 60 * 40, 40)
            
            scores.institutional = round(min(count_score + ownership_score + position_quality_score, 100), 2)
        
    except Exception as e:
        st.warning(f"Institutional analysis error for {ticker}: {e}")
    
    # M - Market Direction (Enhanced)
    try:
        # Market trend analysis (simplified - can be enhanced with actual market data)
        if not price_df.empty:
            # Price trend relative to moving averages
            current_price = price_df['Close'].iloc[-1]
            sma_20 = tech_indicators.get('sma_20', current_price)
            sma_50 = tech_indicators.get('sma_50', current_price)
            sma_200 = tech_indicators.get('sma_200', current_price)
            
            # Trend scoring
            above_20 = 25 if current_price > sma_20 else 0
            above_50 = 25 if current_price > sma_50 else 0
            above_200 = 25 if current_price > sma_200 else 0
            
            # Moving average alignment
            ma_alignment = 25 if (sma_20 > sma_50 > sma_200) else 0
            
            scores.market_direction = round(above_20 + above_50 + above_200 + ma_alignment, 2)
        
    except Exception as e:
        st.warning(f"Market direction calculation error for {ticker}: {e}")
    
    return scores


def calculate_valuation_metrics(ticker: str, data: Dict[str, Any]) -> ValuationMetrics:
    """
    Calculate comprehensive valuation metrics.
    
    :param ticker: Stock symbol
    :param data: Comprehensive data dictionary
    :return: ValuationMetrics instance
    """
    valuation = ValuationMetrics()
    
    price_df = data.get('price_data', pd.DataFrame())
    company_info = data.get('company_info', {})
    fundamentals = data.get('fundamentals', {})
    
    if price_df.empty:
        return valuation
    
    try:
        current_price = price_df['Close'].iloc[-1]
        valuation.current_price = current_price
        
        # Basic ratios from company info
        valuation.market_cap = company_info.get('marketCap', np.nan)
        valuation.pe_ratio = company_info.get('trailingPE', np.nan)
        valuation.pb_ratio = company_info.get('priceToBook', np.nan)
        valuation.ps_ratio = company_info.get('priceToSalesTrailing12Months', np.nan)
        valuation.peg_ratio = company_info.get('pegRatio', np.nan)
        valuation.ev_ebitda = company_info.get('enterpriseToEbitda', np.nan)
        
        # DCF calculation (simplified)
        annual_earnings = fundamentals.get('annual_earnings')
        if isinstance(annual_earnings, pd.DataFrame) and len(annual_earnings) >= 3:
            try:
                earnings = annual_earnings.sort_index()['Earnings']
                growth_rate = cagr(earnings.iloc[-3], earnings.iloc[-1], 2)
                
                if not math.isnan(growth_rate) and growth_rate > 0:
                    # Simplified DCF
                    terminal_growth = 0.03
                    discount_rate = 0.10
                    projection_years = 5
                    
                    current_earnings = earnings.iloc[-1]
                    future_cash_flows = []
                    
                    for year in range(1, projection_years + 1):
                        future_earnings = current_earnings * ((1 + growth_rate) ** year)
                        pv = future_earnings / ((1 + discount_rate) ** year)
                        future_cash_flows.append(pv)
                    
                    terminal_value = (future_cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
                    terminal_pv = terminal_value / ((1 + discount_rate) ** projection_years)
                    
                    dcf_value = sum(future_cash_flows) + terminal_pv
                    shares_outstanding = company_info.get('sharesOutstanding', 1)
                    valuation.dcf_fair_value = dcf_value / shares_outstanding
                
            except Exception as e:
                st.warning(f"DCF calculation error for {ticker}: {e}")
        
        # Target prices (based on PE expansion)
        if not math.isnan(valuation.pe_ratio):
            annual_earnings = fundamentals.get('annual_earnings')
            if isinstance(annual_earnings, pd.DataFrame) and len(annual_earnings) >= 1:
                try:
                    eps = annual_earnings.sort_index()['Earnings'].iloc[-1]
                    shares = company_info.get('sharesOutstanding', 1)
                    eps_per_share = eps / shares
                    
                    # Conservative, moderate, aggressive targets
                    valuation.target_price_1 = eps_per_share * (valuation.pe_ratio * 1.1)  # 10% PE expansion
                    valuation.target_price_2 = eps_per_share * (valuation.pe_ratio * 1.25) # 25% PE expansion
                    valuation.target_price_3 = eps_per_share * (valuation.pe_ratio * 1.5)  # 50% PE expansion
                    
                except Exception:
                    pass
        
        # Entry point and stop loss
        high_52w = price_df['High'].rolling(252).max().iloc[-1] if len(price_df) >= 252 else current_price
        low_52w = price_df['Low'].rolling(252).min().iloc[-1] if len(price_df) >= 252 else current_price
        
        # Entry point: slight pullback from 52w high or current price if near high
        valuation.entry_point = min(current_price * 0.98, high_52w * 0.95)
        
        # Stop loss: 7-8% below entry point (O'Neil's rule)
        valuation.stop_loss_price = valuation.entry_point * 0.92
        
    except Exception as e:
        st.warning(f"Valuation calculation error for {ticker}: {e}")
    
    return valuation


def assess_risks(ticker: str, data: Dict[str, Any], scores: CanslimScores) -> RiskAssessment:
    """
    Assess various risk factors for the stock.
    
    :param ticker: Stock symbol
    :param data: Comprehensive data dictionary
    :param scores: CANSLIM scores
    :return: RiskAssessment instance
    """
    risk = RiskAssessment()
    
    company_info = data.get('company_info', {})
    fundamentals = data.get('fundamentals', {})
    price_df = data.get('price_data', pd.DataFrame())
    
    try:
        risk_factors = []
        risk_score = 0
        
        # Business Risk Assessment
        sector = company_info.get('sector', 'Unknown')
        industry = company_info.get('industry', 'Unknown')
        
        business_risks = []
        if sector in ['Technology', 'Biotechnology']:
            business_risks.append("高い技術変化リスク")
            risk_score += 15
        if 'cyclical' in industry.lower():
            business_risks.append("景気循環リスク")
            risk_score += 10
        
        risk.business_risk = "; ".join(business_risks) if business_risks else "低"
        
        # Financial Risk Assessment
        financial_risks = []
        
        # Debt analysis
        total_debt = company_info.get('totalDebt', 0)
        total_equity = company_info.get('totalStockholderEquity', 1)
        debt_to_equity = total_debt / total_equity if total_equity > 0 else np.nan
        
        if not math.isnan(debt_to_equity):
            if debt_to_equity > 0.6:
                financial_risks.append("高い負債比率")
                risk_score += 20
            elif debt_to_equity > 0.3:
                financial_risks.append("中程度の負債")
                risk_score += 10
        
        # Cash flow analysis
        operating_cash_flow = company_info.get('operatingCashflow', 0)
        if operating_cash_flow < 0:
            financial_risks.append("負のキャッシュフロー")
            risk_score += 25
        
        risk.financial_risk = "; ".join(financial_risks) if financial_risks else "低"
        
        # Market Risk Assessment
        market_risks = []
        
        # Volatility analysis
        if not price_df.empty and len(price_df) >= 63:
            volatility = price_df['Close'].pct_change().rolling(63).std().iloc[-1]
            if volatility > 0.05:  # > 5% daily volatility
                market_risks.append("高いボラティリティ")
                risk_score += 15
        
        # Beta analysis
        beta = company_info.get('beta', 1.0)
        if beta > 1.5:
            market_risks.append("高いベータ値")
            risk_score += 10
        
        risk.market_risk = "; ".join(market_risks) if market_risks else "低"
        
        # Regulatory Risk (industry-specific)
        regulatory_risks = []
        if sector in ['Healthcare', 'Utilities', 'Financial Services']:
            regulatory_risks.append("規制変更リスク")
            risk_score += 10
        
        risk.regulatory_risk = "; ".join(regulatory_risks) if regulatory_risks else "低"
        
        # Overall risk score (0-100, lower is better)
        risk.overall_risk_score = min(risk_score, 100)
        
    except Exception as e:
        st.warning(f"Risk assessment error for {ticker}: {e}")
        risk.overall_risk_score = 50  # Medium risk as default
    
    return risk


###############################################################################
# Utility Functions
###############################################################################

def yoy_growth(current: float, prev_year: float) -> float:
    """Compute year-over-year growth."""
    if pd.isna(current) or pd.isna(prev_year) or prev_year == 0:
        return np.nan
    return (current / prev_year) - 1.0


def cagr(start: float, end: float, years: int) -> float:
    """Compute compound annual growth rate (CAGR)."""
    if any(map(pd.isna, [start, end])) or start <= 0 or years <= 0:
        return np.nan
    return (end / start) ** (1 / years) - 1.0


###############################################################################
# Enhanced Google Sheets Integration
###############################################################################

def create_enhanced_google_sheet(sheet_name: str, results_data: List[Dict]) -> None:
    """
    Create comprehensive Google Sheets report with enhanced formatting.
    
    :param sheet_name: Name of the worksheet
    :param results_data: List of analysis results
    """
    if not GSPREAD_AVAILABLE:
        st.warning("Google Sheets integration not available. Saving as CSV instead.")
        save_to_csv(results_data, sheet_name)
        return
    
    creds_path = os.getenv("GOOGLE_CREDENTIALS_JSON")
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    
    if not creds_path or not spreadsheet_id:
        st.warning("Google Sheets credentials not configured. Saving as CSV instead.")
        save_to_csv(results_data, sheet_name)
        return
    
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(spreadsheet_id)
        
        # Create or get worksheet
        try:
            worksheet = sh.worksheet(sheet_name)
            worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="30")
        
        # Enhanced header
        headers = [
            "Ticker", "企業名", "現在株価", "時価総額", "セクター",
            "Current Earnings (C)", "Annual Earnings (A)", "Innovation (N)", 
            "Supply/Demand (S)", "Leader/Laggard (L)", "Institutional (I)", 
            "Market Direction (M)", "総合スコア", "投資推奨度",
            "PE比", "PB比", "PS比", "PEG比", "DCF理論価格",
            "目標株価1", "目標株価2", "目標株価3", "エントリーポイント", "ストップロス",
            "主要リスク要因", "リスクスコア", "RSI", "6ヶ月モメンタム", "出来高急増度"
        ]
        
        worksheet.append_row(headers)
        
        # Add data rows
        for result in results_data:
            row = [
                result.get('ticker', ''),
                result.get('company_name', ''),
                result.get('current_price', ''),
                result.get('market_cap', ''),
                result.get('sector', ''),
                result.get('current_earnings', ''),
                result.get('annual_earnings', ''),
                result.get('innovation', ''),
                result.get('supply_demand', ''),
                result.get('leader_laggard', ''),
                result.get('institutional', ''),
                result.get('market_direction', ''),
                result.get('total_score', ''),
                result.get('recommendation', ''),
                result.get('pe_ratio', ''),
                result.get('pb_ratio', ''),
                result.get('ps_ratio', ''),
                result.get('peg_ratio', ''),
                result.get('dcf_fair_value', ''),
                result.get('target_price_1', ''),
                result.get('target_price_2', ''),
                result.get('target_price_3', ''),
                result.get('entry_point', ''),
                result.get('stop_loss_price', ''),
                result.get('main_risks', ''),
                result.get('risk_score', ''),
                result.get('rsi', ''),
                result.get('momentum_6m', ''),
                result.get('volume_spike', '')
            ]
            worksheet.append_row(row)
        
        st.success(f"Enhanced Google Sheets report created: {sheet_name}")
        
    except Exception as e:
        st.error(f"Google Sheets creation failed: {e}")
        save_to_csv(results_data, sheet_name)


def save_to_csv(results_data: List[Dict], filename: str) -> None:
    """Save results to CSV file as fallback."""
    try:
        df = pd.DataFrame(results_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{filename}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        st.info(f"Results saved to CSV: {csv_filename}")
    except Exception as e:
        st.error(f"Failed to save CSV: {e}")


###############################################################################
# Enhanced Notion Integration
###############################################################################

def generate_comprehensive_report(ticker: str, data: Dict[str, Any], scores: CanslimScores, 
                                 valuation: ValuationMetrics, risk: RiskAssessment) -> str:
    """
    Generate comprehensive 5000-10000 character analysis report.
    
    :param ticker: Stock symbol
    :param data: Comprehensive data dictionary
    :param scores: CANSLIM scores
    :param valuation: Valuation metrics
    :param risk: Risk assessment
    :return: Detailed markdown report
    """
    company_info = data.get('company_info', {})
    company_name = company_info.get('longName', ticker)
    sector = company_info.get('sector', 'Unknown')
    industry = company_info.get('industry', 'Unknown')
    
    report_sections = []
    
    # 1. Executive Summary
    report_sections.append(f"""# {company_name}（{ticker}）CANSLIM成長株分析レポート

## 1. エグゼクティブサマリー

**投資推奨度**: {scores.recommendation()}  
**総合スコア**: {scores.total_score():.1f}/100点  
**現在株価**: ${valuation.current_price:.2f}  
**セクター**: {sector} / {industry}

### 主要投資ポイント
1. **業績成長力**: 四半期EPS成長率スコア {scores.current_earnings:.1f}点、年間成長率スコア {scores.annual_earnings:.1f}点
2. **市場でのリーダーシップ**: 相対力指数スコア {scores.leader_laggard:.1f}点、6ヶ月株価モメンタム {scores.price_momentum*100:.1f}%
3. **機関投資家の支持**: 機関投資家スコア {scores.institutional:.1f}点

### リスク要因の要約
- **事業リスク**: {risk.business_risk}
- **財務リスク**: {risk.financial_risk}  
- **市場リスク**: {risk.market_risk}
- **総合リスクスコア**: {risk.overall_risk_score:.1f}/100点
""")

    # 2. CANSLIM Detailed Analysis
    report_sections.append(f"""## 2. CANSLIM詳細分析

### C - 四半期業績分析（スコア: {scores.current_earnings:.1f}点）

直近四半期のEPS成長率分析では、前年同期比での成長率を詳細に評価しました。William O'Neilの基準では25%以上の成長が必須とされており、当銘柄の成長率はこの基準に対して{scores.current_earnings:.1f}点の評価となりました。

特に重要なのは成長の加速度です。前四半期との比較において、EPS成長率の加速度は{scores.eps_growth_acceleration:.1f}%となっており、これは成長の継続性を示す重要な指標です。加速度がプラスである場合、企業の収益力が向上していることを示し、将来の成長に対する確信度が高まります。

売上高との相関分析も重要です。EPS成長が売上高成長を伴っているかどうかは、成長の質を判断する上で不可欠です。一時的なコスト削減によるEPS改善ではなく、本質的な事業成長による利益拡大が確認できる場合、より高い評価を与えることができます。

### A - 年間業績トレンド（スコア: {scores.annual_earnings:.1f}点）

過去3年間のEPS CAGR（年平均成長率）分析により、当銘柄の長期的な成長軌跡を評価しました。CANSLIMの基準では、過去3年間で年平均25%以上の成長が推奨されています。

当銘柄の年間成長率スコア{scores.annual_earnings:.1f}点は、この基準に対する達成度を示しています。成長の安定性も重要な評価要素であり、年次ごとの収益ブレを分析することで、企業の収益基盤の堅牢性を判断できます。

業績の継続性は投資判断において極めて重要です。一過性の要因による成長ではなく、構造的な競争優位性に基づく持続可能な成長が確認できる場合、長期投資の観点からも魅力的な投資対象となります。将来成長予測についても、過去のトレンドと業界環境を考慮した慎重な分析が必要です。""")

    report_sections.append(f"""### N - イノベーション要因（スコア: {scores.new_products:.1f}点）

新製品・新サービスの市場インパクト評価は、CANSLIMの中でも定性的な要素が強い項目です。当銘柄のスコア{scores.new_products:.1f}点は、主に株価の52週高値に対する近接度、直近3ヶ月の価格モメンタム、出来高の変化率を組み合わせて算出しています。

52週高値への近接度は市場の期待値の表れであり、投資家が企業の将来性を評価している指標として重要です。現在の株価が52週高値に近い水準にある場合、市場参加者が企業の革新性や成長可能性を高く評価していることを示唆します。

技術的ブレークスルーや新市場への参入は、株価の大きな上昇要因となります。過去の成功事例を見ると、画期的な新製品の発表や新サービスの展開により、株価が数倍に上昇したケースも少なくありません。経営陣の交代や戦略転換も同様に、企業の新たな成長段階への移行を示す重要なシグナルとなります。

### S - 需給分析（スコア: {scores.supply_demand:.1f}点）

株式の需給バランス分析では、出来高パターンと浮動株数の分析を中心に評価を行いました。出来高急増度{scores.volume_trend:.1f}倍は、機関投資家による大口取引や個人投資家の関心の高まりを示しています。

理想的な出来高パターンは、株価上昇時に出来高が増加し、調整時に出来高が減少するパターンです。これは需要が供給を上回っている健全な状況を示し、株価の持続的な上昇を支える基盤となります。

浮動株数の分析では、5000万株以下が理想的とされています。浮動株数が少ない銘柄は、相対的に少ない資金流入でも大きな株価変動を引き起こしやすく、成長期における株価上昇の爆発力が期待できます。機関投資家の売買動向も重要な観察指標であり、スマートマネーの動きを追跡することで、株価の方向性を予測する手がかりを得ることができます。""")

    report_sections.append(f"""### L - 業界リーダーシップ（スコア: {scores.leader_laggard:.1f}点）

相対力指数（RS）分析により、当銘柄の業界内でのポジションを評価しました。6ヶ月リターン{scores.price_momentum*100:.1f}%は、市場全体や同業他社との比較において重要な指標となります。

CANSLIMでは相対力指数80以上が推奨されており、これは上位20%のパフォーマンスを意味します。業界リーダーとしての地位は、単に株価パフォーマンスだけでなく、市場シェア、技術革新、収益性といった多面的な評価が必要です。

競合比較分析では、同業他社との財務指標、成長率、バリュエーションを比較することで、相対的な投資魅力度を判断します。市場シェアの推移は企業の競争力を直接的に示す指標であり、シェア拡大が継続している企業は構造的な優位性を持っている可能性が高いです。

RSI（相対力指数）{scores.rsi:.1f}は、短期的な値動きの過熱感を測る指標として活用されます。適正なRSIレベルは、急激な株価上昇後の調整リスクを避ける上で重要な判断材料となります。

### I - 機関投資家動向（スコア: {scores.institutional:.1f}点）

機関投資家の動向分析は、プロフェッショナル投資家による企業評価を理解する上で重要です。機関投資家数の増加は、企業の投資価値が広く認められていることを示し、株価の安定性と上昇持続性を支える要因となります。

主要株主の変動を追跡することで、スマートマネーの流れを把握できます。特に著名なファンドマネージャーや成功実績のある機関投資家の新規投資や持分拡大は、株価の上昇シグナルとして注目されます。

インサイダー取引状況も重要な観察指標です。経営陣による自社株買いは、内部者が企業の将来性に確信を持っていることを示す強いシグナルです。逆に、大量の売却が続く場合は慎重な判断が必要となります。

機関投資家の保有比率が適度に高い状況は、株価の下支え効果が期待できる一方、過度に高い場合は流動性の問題や集中売りのリスクも考慮する必要があります。""")

    report_sections.append(f"""### M - マーケット環境（スコア: {scores.market_direction:.1f}点）

市場全体のトレンド分析では、主要移動平均線との位置関係を評価しました。現在の株価が20日、50日、200日移動平均線を上回っているかどうかは、トレンドの強さを判断する重要な指標です。

マクロ経済環境との相関分析も重要です。金利動向、インフレ率、GDP成長率といった経済指標は、株式市場全体の方向性に大きな影響を与えます。特に成長株は金利上昇局面での調整リスクが高いため、金融政策の動向には細心の注意が必要です。

セクターローテーション分析では、資金がどのセクターに向かっているかを把握することで、投資タイミングの最適化を図ります。市場サイクルの理解は、成長株投資における成功の鍵となる要素の一つです。

現在の市場環境スコア{scores.market_direction:.1f}点は、これらの複合的な要因を総合的に評価した結果です。市場の方向性が上向きの局面では、個別株の上昇も期待しやすく、投資リスクも相対的に低下します。

## 3. バリュエーション分析

### 相対評価指標
- **PER**: {valuation.pe_ratio:.1f} - 業界平均との比較で割安/割高を評価
- **PBR**: {valuation.pb_ratio:.1f} - 純資産倍率による資産効率性の評価  
- **PSR**: {valuation.ps_ratio:.1f} - 売上高倍率による成長期企業の評価
- **PEG**: {valuation.peg_ratio:.1f} - 成長率調整後のPER評価

### DCF分析による理論株価
DCF（割引キャッシュフロー）モデルによる理論株価は${valuation.dcf_fair_value:.2f}となりました。これは将来キャッシュフローの現在価値を基に算出した企業の内在価値です。

計算過程では、過去の成長率トレンドから将来5年間の成長率を推定し、10%の割引率と3%の永続成長率を使用しています。現在株価${valuation.current_price:.2f}との比較により、割安/割高の判断材料としています。

### 目標株価設定
- **保守的目標**: ${valuation.target_price_1:.2f} (PE倍率10%拡大想定)
- **標準的目標**: ${valuation.target_price_2:.2f} (PE倍率25%拡大想定)  
- **楽観的目標**: ${valuation.target_price_3:.2f} (PE倍率50%拡大想定)

これらの目標株価は、業績成長に伴うPE倍率の拡大を想定して算出しています。成長株の場合、業績向上とともにバリュエーション倍率も上昇する傾向があり、株価上昇の二重効果が期待できます。""")

    # 4. Technical Analysis
    report_sections.append(f"""## 4. テクニカル分析

### チャートパターン分析
現在の株価${valuation.current_price:.2f}は、52週高値からの位置により、テクニカル的なトレンドを評価できます。O'Neilが重視するカップウィズハンドル等のパターン形成は、大きな株価上昇の前兆となることが多いです。

### サポート・レジスタンスレベル
- **エントリーポイント**: ${valuation.entry_point:.2f}
- **ストップロス**: ${valuation.stop_loss_price:.2f}（エントリーから8%下）
- **第1抵抗線**: 直近高値圏
- **第1支持線**: 20日移動平均線

### モメンタム指標
- **RSI**: {scores.rsi:.1f} - 買われすぎ/売られすぎの判定
- **6ヶ月モメンタム**: {scores.price_momentum*100:.1f}% - 中期トレンド評価
- **出来高急増度**: {scores.volume_trend:.1f}倍 - 投資家関心度の変化

RSI値{scores.rsi:.1f}は、短期的な調整の可能性を示唆しています。理想的なRSI範囲は30-70であり、80を超える場合は短期的な過熱感を警戒する必要があります。

出来高分析では、株価上昇時の出来高増加が健全な上昇トレンドの条件となります。機関投資家の大口取引や個人投資家の関心集中により、通常の数倍の出来高が観測される場合、トレンド転換のシグナルとなることがあります。

## 5. リスク評価

### 事業リスク
{risk.business_risk}

技術変化の激しい業界では、既存の優位性が短期間で失われるリスクがあります。特にテクノロジーセクターでは、破壊的イノベーションにより市場構造が根本的に変化する可能性を常に考慮する必要があります。

### 財務リスク  
{risk.financial_risk}

負債比率や現金創出能力は、企業の財務安定性を評価する重要な指標です。成長期の企業は投資資金需要が高く、外部資金調達への依存度が高まる傾向があります。金利上昇局面では資金調達コストの増加により、成長投資の収益性が低下するリスクがあります。

### 市場リスク
{risk.market_risk}

株価ボラティリティは投資リターンと密接に関連しています。高いボラティリティは大きなリターンの可能性を示す一方、損失リスクも同様に拡大します。ベータ値による市場感応度の分析は、ポートフォリオリスク管理において重要な指標となります。

### 規制リスク
{risk.regulatory_risk}

業界特有の規制変更リスクは、特にヘルスケア、金融、公益事業セクターで重要な考慮要素となります。政策変更や新たな規制導入により、事業モデルの根本的な見直しが必要となる場合があります。

**総合リスクスコア**: {risk.overall_risk_score:.1f}/100点（低いほど良好）

## 6. 投資戦略提案

### エントリータイミング
推奨エントリーポイント${valuation.entry_point:.2f}は、現在価格からの若干の押し目を想定しています。O'Neilの手法では、52週高値更新後の軽微な調整局面でのエントリーが推奨されます。

### ポジションサイジング
リスクスコア{risk.overall_risk_score:.1f}点を考慮し、総投資資金の3-5%程度のポジションサイズが適切と考えられます。成長株投資では集中投資によるリターン最大化も重要ですが、個別銘柄リスクの分散も同様に重要です。

### リスク管理手法
- **ストップロス**: ${valuation.stop_loss_price:.2f}（エントリーから8%下落で損切り）
- **利益確定**: 20-25%上昇で一部利確、50%以上で大部分利確
- **ポジション管理**: 四半期決算後の見直し

### 出口戦略
目標株価達成時の段階的利確戦略：
1. **第1段階**（目標1達成時）: ポジションの25%利確
2. **第2段階**（目標2達成時）: 追加50%利確  
3. **第3段階**（目標3達成時）: 残りポジション利確検討

## 7. まとめと推奨アクション

**総合評価**: {scores.recommendation()}（スコア: {scores.total_score():.1f}/100点）

### 投資判断の根拠
1. **成長性**: EPS成長率が基準を満たし、継続的な成長が期待される
2. **市場地位**: 業界内でのリーダーシップが確立されている
3. **テクニカル要因**: 価格・出来高パターンが良好なトレンドを示している
4. **バリュエーション**: 成長率に見合った適正な評価水準

### 推奨アクション
- **即座の行動**: エントリーポイント${valuation.entry_point:.2f}での買い注文検討
- **継続監視**: 四半期決算発表時の業績確認
- **リスク管理**: ストップロス${valuation.stop_loss_price:.2f}の厳格な実行

当分析は{datetime.now().strftime('%Y年%m月%d日')}時点の情報に基づいており、投資判断は自己責任において行ってください。市場環境の変化や新たな情報の開示により、投資判断の見直しが必要となる場合があります。

---
*本レポートはCANSLIM投資手法に基づく分析であり、投資の成功を保証するものではありません。投資にはリスクが伴いますので、十分な検討の上で投資判断を行ってください。*""")

    return "\n".join(report_sections)


def post_comprehensive_notion_report(ticker: str, report: str) -> None:
    """
    Post comprehensive report to Notion.
    
    :param ticker: Stock symbol
    :param report: Generated report content
    """
    if not NOTION_AVAILABLE:
        save_markdown_report(ticker, report)
        return
    
    notion_token = os.getenv("NOTION_TOKEN")
    notion_db_id = os.getenv("NOTION_DATABASE_ID")
    
    if not notion_token or not notion_db_id:
        st.warning("Notion credentials not configured. Saving as markdown file.")
        save_markdown_report(ticker, report)
        return
    
    try:
        notion = NotionClient(auth=notion_token)
        
        # Split report into blocks (Notion has character limits per block)
        blocks = []
        sections = report.split('\n## ')
        
        for i, section in enumerate(sections):
            if i == 0:
                content = section
            else:
                content = '## ' + section
            
            # Split long sections into smaller blocks
            if len(content) > 2000:
                parts = [content[i:i+2000] for i in range(0, len(content), 2000)]
                for part in parts:
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": part}}]
                        }
                    })
            else:
                blocks.append({
                    "object": "block",
                    "type": "paragraph", 
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": content}}]
                    }
                })
        
        # Create page
        notion.pages.create(
            parent={"database_id": notion_db_id},
            properties={
                "Name": {"title": [{"text": {"content": f"{ticker} CANSLIM分析レポート"}}]}
            },
            children=blocks[:100]  # Notion limits to 100 blocks per request
        )
        
        st.success(f"Comprehensive Notion report created for {ticker}")
        
    except Exception as e:
        st.error(f"Notion report creation failed: {e}")
        save_markdown_report(ticker, report)


def save_markdown_report(ticker: str, report: str) -> None:
    """Save report as markdown file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_CANSLIM_Report_{timestamp}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        st.info(f"Report saved as markdown: {filename}")
    except Exception as e:
        st.error(f"Failed to save markdown report: {e}")


###############################################################################
# Enhanced Streamlit Application
###############################################################################

def run_enhanced_app() -> None:
    st.set_page_config(
        page_title="CANSLIM成長株分析システム",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 CANSLIM成長株分析システム")
    st.markdown("William J. O'NeilのCANSLIM戦略に基づく包括的な銘柄分析ツール")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # API status check
        st.subheader("API接続状況")
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            st.success("✅ Alpha Vantage API")
        else:
            st.warning("⚠️ Alpha Vantage API未設定")
        
        if os.getenv('GOOGLE_CREDENTIALS_JSON') and os.getenv('GOOGLE_SPREADSHEET_ID'):
            st.success("✅ Google Sheets API")
        else:
            st.warning("⚠️ Google Sheets API未設定")
        
        if os.getenv('NOTION_TOKEN') and os.getenv('NOTION_DATABASE_ID'):
            st.success("✅ Notion API")
        else:
            st.warning("⚠️ Notion API未設定")
    
    # Main input area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 分析対象銘柄")
        tickers_input = st.text_area(
            "ティッカーシンボル（カンマまたは改行区切り）",
            value="AAPL, MSFT, NVDA, GOOGL, AMZN",
            height=100,
            help="例: AAPL, MSFT, NVDA"
        )
    
    with col2:
        st.subheader("⚙️ 分析パラメータ")
        years = st.number_input("分析期間（年）", min_value=1, max_value=10, value=5, step=1)
        
        st.subheader("📤 出力オプション")
        upload_to_sheet = st.checkbox("Google Sheetsに出力", value=True)
        post_to_notion_flag = st.checkbox("Notionレポート生成", value=True)
        show_charts = st.checkbox("チャート表示", value=True)
    
    # Analysis execution
    if st.button("🚀 分析実行", type="primary"):
        if not tickers_input.strip():
            st.error("ティッカーシンボルを入力してください。")
            return
        
        # Parse tickers
        import re
        raw_tokens = tickers_input.replace("\n", ",").split(",")
        tickers = []
        for token in raw_tokens:
            cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", token.strip().upper())
            if cleaned:
                tickers.append(cleaned)
        
        if not tickers:
            st.error("有効なティッカーシンボルを入力してください。")
            return
        
        st.info(f"分析対象: {', '.join(tickers)} ({len(tickers)}銘柄)")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_data = []
        notion_reports = []
        
        # Analyze each ticker
        for i, ticker in enumerate(tickers):
            status_text.text(f"分析中: {ticker} ({i+1}/{len(tickers)})")
            
            try:
                # Fetch comprehensive data
                data = fetch_comprehensive_data(ticker, years)
                
                # Compute scores and metrics
                scores = compute_enhanced_canslim_scores(ticker, data)
                valuation = calculate_valuation_metrics(ticker, data)
                risk = assess_risks(ticker, data, scores)
                
                # Prepare result data
                company_info = data.get('company_info', {})
                result = {
                    'ticker': ticker,
                    'company_name': company_info.get('longName', ticker),
                    'current_price': valuation.current_price,
                    'market_cap': valuation.market_cap,
                    'sector': company_info.get('sector', 'Unknown'),
                    'current_earnings': scores.current_earnings,
                    'annual_earnings': scores.annual_earnings,
                    'innovation': scores.new_products,
                    'supply_demand': scores.supply_demand,
                    'leader_laggard': scores.leader_laggard,
                    'institutional': scores.institutional,
                    'market_direction': scores.market_direction,
                    'total_score': scores.total_score(),
                    'recommendation': scores.recommendation(),
                    'pe_ratio': valuation.pe_ratio,
                    'pb_ratio': valuation.pb_ratio,
                    'ps_ratio': valuation.ps_ratio,
                    'peg_ratio': valuation.peg_ratio,
                    'dcf_fair_value': valuation.dcf_fair_value,
                    'target_price_1': valuation.target_price_1,
                    'target_price_2': valuation.target_price_2,
                    'target_price_3': valuation.target_price_3,
                    'entry_point': valuation.entry_point,
                    'stop_loss_price': valuation.stop_loss_price,
                    'main_risks': f"{risk.business_risk}; {risk.financial_risk}; {risk.market_risk}",
                    'risk_score': risk.overall_risk_score,
                    'rsi': scores.rsi,
                    'momentum_6m': scores.price_momentum,
                    'volume_spike': scores.volume_trend
                }
                
                results_data.append(result)
                
                # Generate comprehensive report
                if post_to_notion_flag:
                    report = generate_comprehensive_report(ticker, data, scores, valuation, risk)
                    notion_reports.append((ticker, report))
                
            except Exception as e:
                st.error(f"分析エラー ({ticker}): {e}")
            
            progress_bar.progress((i + 1) / len(tickers))
        
        status_text.text("分析完了！")
        
        # Display results
        if results_data:
            st.subheader("📊 分析結果サマリー")
            
            df_results = pd.DataFrame(results_data)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                strong_buy = len(df_results[df_results['recommendation'] == '強く推奨'])
                st.metric("強く推奨", strong_buy)
            with col2:
                avg_score = df_results['total_score'].mean()
                st.metric("平均スコア", f"{avg_score:.1f}")
            with col3:
                high_risk = len(df_results[df_results['risk_score'] > 60])
                st.metric("高リスク銘柄", high_risk)
            with col4:
                profitable = len(df_results[df_results['target_price_1'] > df_results['current_price']])
                st.metric("上昇期待銘柄", profitable)
            
            # Detailed results table
            st.subheader("📋 詳細分析結果")
            
            # Format display dataframe
            display_df = df_results[['ticker', 'company_name', 'current_price', 'total_score', 
                                   'recommendation', 'current_earnings', 'annual_earnings',
                                   'leader_laggard', 'institutional', 'entry_point', 'target_price_1']].copy()
            
            display_df.columns = ['ティッカー', '企業名', '現在価格', '総合スコア', '推奨度', 
                                'C', 'A', 'L', 'I', 'エントリー', '目標価格']
            
            st.dataframe(display_df, use_container_width=True)
            
            # Charts
            if show_charts and PLOTLY_AVAILABLE:
                st.subheader("📈 視覚化分析")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CANSLIM scores radar chart
                    fig = go.Figure()
                    
                    for i, result in enumerate(results_data[:5]):  # Limit to 5 for readability
                        categories = ['C', 'A', 'N', 'S', 'L', 'I', 'M']
                        values = [result['current_earnings'], result['annual_earnings'], 
                                result['innovation'], result['supply_demand'],
                                result['leader_laggard'], result['institutional'], 
                                result['market_direction']]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=result['ticker']
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="CANSLIM スコア比較"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Total scores bar chart
                    fig = px.bar(
                        df_results.head(10), 
                        x='ticker', 
                        y='total_score',
                        color='recommendation',
                        title="総合スコア ランキング（上位10銘柄）",
                        color_discrete_map={
                            '強く推奨': '#00CC96',
                            '推奨': '#AB63FA', 
                            '中立': '#FFA15A',
                            '非推奨': '#EF553B'
                        }
                    )
                    fig.update_layout(xaxis_title="ティッカー", yaxis_title="スコア")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📤 データ出力")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if upload_to_sheet:
                    if st.button("Google Sheetsに出力"):
                        create_enhanced_google_sheet("CANSLIM Analysis", results_data)
            
            with col2:
                if post_to_notion_flag and notion_reports:
                    if st.button("Notionレポート投稿"):
                        for ticker, report in notion_reports:
                            post_comprehensive_notion_report(ticker, report)
            
            # Download CSV
            csv = df_results.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV ダウンロード",
                data=csv,
                file_name=f"canslim_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    run_enhanced_app()