# TraceGraph — Temporal Graph Intelligence for AML Detection

TraceGraph is a temporal graph-based AML investigation system that models financial transactions as directed temporal networks.

The system combines transaction-level, account-level, graph, and temporal features to prioritize potentially suspicious transaction activity for investigation.

## Key Components

- Temporal transaction analysis
- Directed transaction graphs
- Multi-scale temporal analysis
- Multi-channel graph representations
- Logistic Regression
- Random Forest
- XGBoost
- CNN / ResNet experiments
- Hybrid graph and transaction features
- Risk-aware reverse and forward tracing
- Fan-in and rapid-hop analysis
- Suspicious temporal cycle detection
- Streamlit investigation dashboard

## Dataset

The project was evaluated using PublicAMLSimData / AMLSim.

The selected experiment contains 17,144 transactions, 100 accounts, 18 alerts and 18 fraud-labelled transactions.

The raw dataset is not included because of its size.

See `data/README.md`.

## Experimental Result

In the evaluated AMLSim dataset, TraceGraph identified all 18 fraud-labelled transactions as high-risk.

This is a dataset-specific experimental result and should not be interpreted as proof of real-world AML detection performance.

The final investigation stage also identified two suspicious temporal cycles.

## Repository Structure

```text
TraceGraph/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
├── notebooks/
├── src/
├── dashboard/
├── models/
├── results/
├── docs/
└── screenshots/
```

## Notebook

The complete experimental pipeline is available at:

`notebooks/TraceGraph_AML.ipynb`

## Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

## Research Scope

TraceGraph is an AML investigation-support and risk-prioritization framework.

Risk scores and detected patterns are intended to support further investigation and human/compliance review rather than independently determine whether a transaction constitutes financial crime.

## License

MIT License
