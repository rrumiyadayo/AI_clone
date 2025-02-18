# AI_clone
Cloning AI from NAS
## プロジェクト概要 
AI-generating AI (AIによるAI生成) の研究開発に挑戦中！！特に、NAS（ニューラルアーキテクチャ探索）に注目して、AI自身が一番かしこいニューラルネットワークの形を自動でデザインしちゃう技術の探求＆実装に全力で頑張ってるんです！
## チームメンバー
* rrumiyadayo
* azimsofi
* arif-sofi
## 疑問
1. NASとは何ものなのか
2. NASの使い方？【Neural architecture search (NAS)[1][2] is a technique for automating the design of artificial neural networks (ANN)】BEGS the question, how on earth do we use this tool.
3. NASのAI強化学習での役割
## 説明
1. 一言で言うと、ニューラルアーキテクチャ探索（NAS)とはAIが自分に最適な脳の構造を自分で作る技術です。強化学習や進化的アルゴリズムで、最適なネットワーク構造を自動的に探索します。
   * (https://ai-compass.weeybrid.co.jp/algorizm/exploring-optimal-structures-neural-architecture-search/)
2. 一般的な手順は以下の通りです：
   1. 準備:
      1. データ: 機械学習で解きたい問題のデータを用意します。（例：画像とラベルのセット、文章とカテゴリのセットなど）
      1. ツール: NASが使えるツールやライブラリを選び、インストールします。（例：AutoGluon, Keras Tunerなど）
   1. 設定:
      1. 「やりたいこと」を指示: ツールに、解きたい問題の種類（分類、回帰など）を伝えます。
      1. データをセット: 用意したデータをツールに読み込ませます。
      1. (必要に応じて)探索範囲を調整: どんなモデル構造を探すか、大まかな範囲を指定することもできます。（例：層の数は最大10個まで、など。デフォルトでもOK）
   1. 実行:
      1. 「自動設計スタート！」: ツールの実行ボタンを押して、NASに最適なモデル構造を探させます。
   1. 結果確認:
      1. ベストモデルをゲット: NASが見つけた最も良いモデル構造を確認します。
      2. 学習＆評価: そのモデル構造を使って、実際にデータを学習させ、性能を評価します。
