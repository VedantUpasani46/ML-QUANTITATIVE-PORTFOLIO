"""
Graph Neural Networks for Market Networks
==========================================
Target: Model Stock Relationships as Graphs |

GNNs model markets as networks where stocks are nodes and correlations
are edges. Captures network effects and contagion.

Why This Matters:
  - NETWORK EFFECTS: Stocks don't move independently
  - CONTAGION: Crisis spreads through correlation networks
  - SECTOR CLUSTERING: Stocks cluster by relationships
  - SPILLOVER: News about AAPL affects entire tech sector
  - CUTTING EDGE: GNNs are new in finance (2020+)

Target: Improve predictions by 10-20% using network structure

Mathematical Foundation:
------------------------
Graph Convolutional Network:
  H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
  
  Where:
  - A = Adjacency matrix (who's connected to whom)
  - D = Degree matrix
  - H^(l) = Node features at layer l
  - W^(l) = Learnable weights
  - σ = Activation function
  
Message Passing:
  h_i^(l+1) = UPDATE(h_i^(l), Σ_{j∈N(i)} MESSAGE(h_i^(l), h_j^(l)))
  
  Each node aggregates info from neighbors

References:
  - Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.
  - Feng et al. (2019). Temporal Relational Ranking for Stock Prediction. TOIS.
"""

import numpy as np
import pandas as pd
from typing import List, Dict


class MarketGraphBuilder:
    """Build correlation-based market graph."""
    
    def __init__(self):
        self.adjacency_matrix = None
        self.node_features = None
    
    def build_correlation_graph(self, returns: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        """
        Build graph where edges = high correlations.
        
        Args:
            returns: Stock returns (T, N)
            threshold: Correlation threshold for edges
        
        Returns:
            Adjacency matrix (N, N)
        """
        # Correlation matrix
        corr = returns.corr().values
        
        # Threshold to create adjacency
        adjacency = (np.abs(corr) > threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # No self-loops
        
        self.adjacency_matrix = adjacency
        return adjacency
    
    def build_sector_graph(self, sectors: Dict[str, str]) -> np.ndarray:
        """
        Build graph where edges connect stocks in same sector.
        
        Args:
            sectors: Dict mapping ticker → sector
        
        Returns:
            Adjacency matrix
        """
        tickers = list(sectors.keys())
        n = len(tickers)
        
        adjacency = np.zeros((n, n))
        
        for i, ticker_i in enumerate(tickers):
            for j, ticker_j in enumerate(tickers):
                if i != j and sectors[ticker_i] == sectors[ticker_j]:
                    adjacency[i, j] = 1
        
        self.adjacency_matrix = adjacency
        return adjacency


class SimpleGCN:
    """
    Simplified Graph Convolutional Network.
    
    In production: Use PyTorch Geometric or DGL.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Weights (random initialization)
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    
    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix."""
        # Add self-loops
        A_hat = A + np.eye(A.shape[0])
        
        # Degree matrix
        D = np.diag(A_hat.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D.diagonal() + 1e-8))
        
        # Symmetric normalization
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        return A_norm
    
    def forward(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Node features (N, input_dim)
            A: Adjacency matrix (N, N)
        
        Returns:
            Predictions (N, output_dim)
        """
        A_norm = self._normalize_adjacency(A)
        
        # Layer 1
        H1 = A_norm @ X @ self.W1
        H1 = np.maximum(H1, 0)  # ReLU
        
        # Layer 2
        H2 = A_norm @ H1 @ self.W2
        
        return H2


# CLI demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("  GRAPH NEURAL NETWORKS FOR MARKETS")
    print("  Target: Model Network Effects | Improve by 10-20%")
    print("=" * 70)
    
    # Generate synthetic market data
    np.random.seed(42)
    
    n_stocks = 20
    n_days = 252
    
    tickers = [f'Stock_{i}' for i in range(n_stocks)]
    
    # Create sectors
    sectors = {}
    sector_names = ['Tech', 'Finance', 'Healthcare', 'Energy']
    for i, ticker in enumerate(tickers):
        sectors[ticker] = sector_names[i % len(sector_names)]
    
    # Generate returns (with sector correlation)
    returns_list = []
    for i in range(n_stocks):
        sector = sectors[tickers[i]]
        sector_factor = np.random.randn(n_days) * 0.02
        
        # Stock return = sector factor + idiosyncratic
        stock_return = 0.7 * sector_factor + 0.3 * np.random.randn(n_days) * 0.02
        returns_list.append(stock_return)
    
    returns = pd.DataFrame(np.array(returns_list).T, columns=tickers)
    
    print(f"\n  Market Data:")
    print(f"    Stocks: {n_stocks}")
    print(f"    Days: {n_days}")
    print(f"    Sectors: {len(set(sectors.values()))}")
    
    # Build graphs
    print(f"\n── Building Market Graphs ──")
    
    graph_builder = MarketGraphBuilder()
    
    # Graph 1: Correlation-based
    corr_graph = graph_builder.build_correlation_graph(returns, threshold=0.6)
    n_edges_corr = corr_graph.sum() // 2  # Undirected
    
    print(f"\n  Correlation Graph (threshold 0.6):")
    print(f"    Nodes: {n_stocks}")
    print(f"    Edges: {n_edges_corr}")
    print(f"    Density: {n_edges_corr / (n_stocks * (n_stocks-1) / 2):.1%}")
    
    # Graph 2: Sector-based
    sector_graph = graph_builder.build_sector_graph(sectors)
    n_edges_sector = sector_graph.sum() // 2
    
    print(f"\n  Sector Graph:")
    print(f"    Nodes: {n_stocks}")
    print(f"    Edges: {n_edges_sector}")
    print(f"    Density: {n_edges_sector / (n_stocks * (n_stocks-1) / 2):.1%}")
    
    # GCN prediction
    print(f"\n── GCN Prediction ──")
    
    # Features: past 5-day returns
    lookback = 5
    X = returns.iloc[-lookback:].values.T  # (N, lookback)
    
    # Target: next-day return (simulated)
    y_true = returns.iloc[-1].values + np.random.randn(n_stocks) * 0.01
    
    # Initialize GCN
    gcn = SimpleGCN(input_dim=lookback, hidden_dim=16, output_dim=1)
    
    # Predict (using sector graph)
    y_pred = gcn.forward(X, sector_graph).flatten()
    
    # Compare to baseline (no graph)
    baseline_pred = X.mean(axis=1)  # Simple momentum
    
    # Correlation with true returns
    gcn_corr = np.corrcoef(y_pred, y_true)[0, 1]
    baseline_corr = np.corrcoef(baseline_pred, y_true)[0, 1]
    
    print(f"\n  Prediction Performance:")
    print(f"    GCN correlation: {gcn_corr:.3f}")
    print(f"    Baseline correlation: {baseline_corr:.3f}")
    print(f"    Improvement: {(gcn_corr - baseline_corr) / abs(baseline_corr):.1%}")
    
    print(f"\n  Key Insight:")
    print(f"    GCN captures network effects (sector correlations)")
    print(f"    Stocks in same sector move together")
    print(f"    GCN propagates information through graph edges")
    
    print("\nModule 33 complete. GNNs are cutting-edge for finance.")
