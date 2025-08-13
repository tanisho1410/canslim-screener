# 📈 CANSLIM成長株分析システム

William J. O'NeilのCANSLIM投資戦略に基づく包括的な銘柄分析ツール

![CANSLIM Analysis](https://img.shields.io/badge/Analysis-CANSLIM-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8+-yellow)

## 🚀 主要機能

### 📊 CANSLIM分析エンジン
- **C** - 四半期EPS成長率分析（加速度ボーナス含む）
- **A** - 年間EPS成長率分析（一貫性評価含む）
- **N** - イノベーション評価（52週高値近接度、モメンタム、出来高分析）
- **S** - 需給分析（出来高急増、浮動株数、価格-出来高相関）
- **L** - リーダーシップ評価（相対力指数、RSI、モメンタム）
- **I** - 機関投資家分析（保有機関数、保有比率、ポジション品質）
- **M** - 市場環境分析（移動平均線との位置関係、トレンド評価）

### 🔗 多重データソース統合
- **yfinance**: 基本的な株価・財務データ
- **Alpha Vantage**: 詳細な財務指標（API設定時）
- **Yahoo Finance**: リアルタイム価格データ

### 📈 バリュエーション分析
- PER、PBR、PSR、PEG比
- DCF（割引キャッシュフロー）による理論株価算出
- 3段階目標株価設定（保守・標準・楽観）
- エントリーポイント・ストップロス価格提案

### 📋 Google Sheets出力
- 詳細スプレッドシート（29列のデータ）
- ティッカー、企業名、CANSLIM各スコア、バリュエーション指標
- リスク評価、テクニカル指標を含む包括的レポート

### 📝 Notion記事生成
- **5000-10000文字の詳細分析レポート**
- エグゼクティブサマリー
- CANSLIM各項目の詳細分析
- バリュエーション・テクニカル・リスク分析
- 投資戦略提案とアクションプラン

### 🎛️ ユーザーインターフェース
- Streamlitベースの直感的なUI
- 複数銘柄の一括分析対応
- リアルタイム進捗表示
- 視覚化チャート（レーダーチャート、バーチャート）
- API接続状況表示

## 🛠️ インストール

### 必要な環境
- Python 3.8以上
- インターネット接続

### 依存関係のインストール
```bash
pip install streamlit pandas numpy yfinance alpha_vantage gspread oauth2client notion_client plotly scikit-learn requests
```

### 環境変数設定（オプション）
高度な機能を利用するには以下の環境変数を設定してください：

```bash
# Alpha Vantage API（詳細財務データ用）
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"

# Google Sheets連携用
export GOOGLE_CREDENTIALS_JSON="/path/to/your/credentials.json"
export GOOGLE_SPREADSHEET_ID="your_spreadsheet_id"

# Notion連携用
export NOTION_TOKEN="your_notion_integration_token"
export NOTION_DATABASE_ID="your_notion_database_id"
```

## 🚀 実行方法

```bash
streamlit run app.py
```

アプリケーションが起動したら、ブラウザで以下のURLにアクセス：
- **ローカル**: http://localhost:8501
- **ネットワーク**: http://[your-ip]:8501

## 📱 使用方法

### 基本的な使い方
1. **銘柄入力**: ティッカーシンボルをカンマまたは改行区切りで入力
   - 例: `AAPL, MSFT, NVDA, GOOGL, AMZN`
2. **分析パラメータ設定**: 分析期間（年数）を選択
3. **出力オプション選択**: Google Sheets出力、Notionレポート生成、チャート表示
4. **分析実行**: 🚀分析実行ボタンをクリック

### 高度な機能
- **一括分析**: 複数銘柄を同時に分析
- **視覚化**: CANSLIMスコアのレーダーチャート表示
- **データ出力**: CSV、Google Sheets、Notionへの自動出力
- **詳細レポート**: 各銘柄の包括的な投資分析レポート

## 📊 出力例

### 分析結果テーブル
| ティッカー | 企業名 | 現在価格 | 総合スコア | 推奨度 | C | A | L | I |
|-----------|--------|----------|------------|---------|---|---|---|---|
| AAPL | Apple Inc. | $150.00 | 85.2 | 強く推奨 | 90 | 85 | 88 | 82 |

### Notionレポート（抜粋）
```markdown
# Apple Inc.（AAPL）CANSLIM成長株分析レポート

## エグゼクティブサマリー
**投資推奨度**: 強く推奨
**総合スコア**: 85.2/100点
**現在株価**: $150.00

### 主要投資ポイント
1. **業績成長力**: 四半期EPS成長率90点、年間成長率85点
2. **市場リーダーシップ**: 相対力指数88点
3. **機関投資家支持**: 機関投資家スコア82点
```

## 🔧 API設定

### Alpha Vantage
1. [Alpha Vantage](https://www.alphavantage.co/)でAPIキーを取得
2. 環境変数`ALPHA_VANTAGE_API_KEY`に設定

### Google Sheets
1. Google Cloud Consoleでサービスアカウント作成
2. 認証情報JSONファイルをダウンロード
3. 環境変数で設定

### Notion
1. Notion統合を作成
2. データベースを準備
3. トークンとデータベースIDを環境変数に設定

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## 📄 ライセンス

MIT License

## ⚠️ 免責事項

本ツールは投資判断の参考情報を提供するものであり、投資の成功を保証するものではありません。投資はリスクを伴いますので、最終的な投資判断は自己責任で行ってください。

## 📞 サポート

質問やサポートが必要な場合は、GitHubのIssuesページをご利用ください。

---

**開発者**: [tanisho1410](https://github.com/tanisho1410)  
**プロジェクト**: [CANSLIM Screener](https://github.com/tanisho1410/canslim-screener)
