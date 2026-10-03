# TraceGraph Architecture

TraceGraph is a temporal graph intelligence pipeline for AML investigation support.

## Pipeline

Transaction Sources
-> Preprocessing and Validation
-> Temporal Transaction Store
-> Multi-Scale Graph Construction
-> Graph and Temporal Features
-> Multi-Channel Graph Representation
-> Machine Learning / Deep Learning
-> Risk Scoring
-> Suspicious Network Detection
-> Reverse / Forward Tracing
-> Pattern and Cycle Detection
-> Investigation Report
-> Streamlit Dashboard

## Multi-channel graph representation

Three graph channels are used:

1. Transaction amount
2. Transaction count
3. Transaction velocity

## Investigation patterns

- Risk-aware fan-in
- Suspicious reverse funding
- Suspicious forward flow
- Risk-aware rapid-hop activity
- Suspicious temporal cycles

TraceGraph is designed as an investigation-support and risk-prioritization framework rather than an autonomous fraud determination system.
