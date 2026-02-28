"""
Blockchain Analytics & DeFi Protocol Analysis
==============================================
Target: Detect Smart Contract Vulnerabilities | DeFi Arbitrage |

This module implements blockchain data analysis, smart contract auditing,
and DeFi protocol evaluation for crypto trading and risk management.

Why This Matters for Crypto/Fintech:
  - SMART CONTRACT AUDITING: $300K-$500K per audit
  - DEFI ARBITRAGE: Flash loans enable risk-free profits
  - ON-CHAIN ANALYTICS: Whale tracking, MEV detection
  - REGULATORY: SEC/CFTC need blockchain forensics
  - FINTECH: Banks exploring blockchain for settlements

Target: Identify profitable DeFi arbitrage, audit smart contracts

Interview insight (Jump Crypto Managing Director):
Q: "Your DeFi arbitrage bot captures $2M/month. How?"
A: "Three strategies: (1) **DEX arbitrage**—Same token trades at different prices
    on Uniswap vs SushiSwap. We detect 1% spreads, execute flash loan arbitrage
    (borrow $10M, buy cheap, sell expensive, repay, keep profit). Zero capital
    risk. Capture $50K-$200K per trade, 100 trades/month = $2M. (2) **Liquidation
    arbitrage**—When Aave/Compound borrowers get liquidated, we buy collateral at
    discount (5-10% below market). Flash loan: Repay debt → claim collateral →
    sell at market. Risk-free $100K-$500K per liquidation. (3) **MEV extraction**—
    Front-run large trades on Uniswap (we see mempool, submit tx with higher gas,
    execute first). Profit from price impact. $20K-$50K per block. Total: $2M/month
    on $0 capital (all flash loans). But: Competition fierce, need <50ms latency."

Mathematical Foundation:
------------------------
Automated Market Maker (AMM) Pricing:
  Uniswap constant product: x·y = k
  Price: p = y/x

  After trade Δx: y' = k/(x + Δx)
  Price impact: Δp/p ≈ Δx/x

Flash Loan Arbitrage:
  Profit = (P_2 - P_1)·Q - gas_cost
  where P_1 = buy price, P_2 = sell price, Q = quantity

  No capital needed (flash loan repaid in same transaction)

Smart Contract Vulnerability Detection:
  Reentrancy: Function calls back before state update
  Integer overflow: uint256 wraps at 2^256
  Access control: Missing onlyOwner modifier

References:
  - Buterin (2014). Ethereum White Paper.
  - Adams et al. (2021). Uniswap v3 Core. Uniswap Labs.
  - Daian et al. (2020). Flash Boys 2.0: Frontrunning in DEXs. IEEE S&P.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# AMM (Automated Market Maker) Pricing
# ---------------------------------------------------------------------------

@dataclass
class AMMPool:
    """Constant product AMM pool (Uniswap-style)."""
    token0_reserve: float
    token1_reserve: float
    fee: float = 0.003  # 0.3% fee

    @property
    def k(self) -> float:
        """Constant product k = x·y"""
        return self.token0_reserve * self.token1_reserve

    def get_price(self, token: str = 'token0') -> float:
        """Get current price of token."""
        if token == 'token0':
            return self.token1_reserve / self.token0_reserve
        else:
            return self.token0_reserve / self.token1_reserve

    def get_amount_out(self, amount_in: float, token_in: str = 'token0') -> float:
        """
        Calculate output amount for given input (including fee).

        Formula: Δy = y·Δx·(1-fee) / (x + Δx·(1-fee))
        """
        if token_in == 'token0':
            x = self.token0_reserve
            y = self.token1_reserve
        else:
            x = self.token1_reserve
            y = self.token0_reserve

        amount_in_with_fee = amount_in * (1 - self.fee)
        amount_out = (y * amount_in_with_fee) / (x + amount_in_with_fee)

        return amount_out

    def execute_swap(self, amount_in: float, token_in: str = 'token0'):
        """Execute swap (updates reserves)."""
        amount_out = self.get_amount_out(amount_in, token_in)

        if token_in == 'token0':
            self.token0_reserve += amount_in
            self.token1_reserve -= amount_out
        else:
            self.token1_reserve += amount_in
            self.token0_reserve -= amount_out

        return amount_out


# ---------------------------------------------------------------------------
# DEX Arbitrage Detector
# ---------------------------------------------------------------------------

class DEXArbitrageDetector:
    """
    Detect arbitrage opportunities across DEXs.

    Identifies price discrepancies and calculates profitable trades.
    """

    def __init__(self, min_profit_bps: float = 10):
        self.min_profit_bps = min_profit_bps  # Minimum profit (basis points)
        self.opportunities = []

    def detect_arbitrage(self,
                        dex1_pool: AMMPool,
                        dex2_pool: AMMPool,
                        token_name: str = 'ETH',
                        max_trade_size: float = 100) -> Optional[Dict]:
        """
        Detect arbitrage opportunity between two DEXs.

        Args:
            dex1_pool: Pool on first DEX
            dex2_pool: Pool on second DEX
            token_name: Token to arbitrage
            max_trade_size: Maximum trade size

        Returns:
            Arbitrage opportunity if exists
        """
        # Get prices
        price1 = dex1_pool.get_price('token0')
        price2 = dex2_pool.get_price('token0')

        # Check for arbitrage
        if abs(price1 - price2) / min(price1, price2) < self.min_profit_bps / 10000:
            return None  # No profitable arbitrage

        # Determine direction
        if price1 < price2:
            # Buy on DEX1, sell on DEX2
            buy_dex = dex1_pool
            sell_dex = dex2_pool
            buy_price = price1
            sell_price = price2
        else:
            # Buy on DEX2, sell on DEX1
            buy_dex = dex2_pool
            sell_dex = dex1_pool
            buy_price = price2
            sell_price = price1

        # Optimal trade size (simplified - assumes linear price impact)
        # In practice: use calculus to maximize profit
        optimal_size = min(max_trade_size, buy_dex.token0_reserve * 0.05)

        # Calculate profit
        amount_out_buy = buy_dex.get_amount_out(optimal_size, 'token0')
        amount_out_sell = sell_dex.get_amount_out(amount_out_buy, 'token1')

        profit = amount_out_sell - optimal_size
        profit_pct = profit / optimal_size

        if profit_pct * 10000 > self.min_profit_bps:
            opportunity = {
                'token': token_name,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'trade_size': optimal_size,
                'profit': profit,
                'profit_pct': profit_pct,
                'roi_bps': profit_pct * 10000
            }

            self.opportunities.append(opportunity)
            return opportunity

        return None


# ---------------------------------------------------------------------------
# Flash Loan Arbitrage Simulator
# ---------------------------------------------------------------------------

class FlashLoanArbitrage:
    """
    Simulate flash loan arbitrage.

    Borrow → Trade → Repay in single transaction (no capital needed).
    """

    def __init__(self, flash_loan_fee: float = 0.0009):  # 0.09% (Aave)
        self.flash_loan_fee = flash_loan_fee
        self.trades = []

    def execute_arbitrage(self,
                         dex1_pool: AMMPool,
                         dex2_pool: AMMPool,
                         loan_amount: float,
                         gas_cost_usd: float = 50) -> Dict:
        """
        Execute flash loan arbitrage.

        Steps:
        1. Borrow loan_amount (no collateral)
        2. Buy on cheaper DEX
        3. Sell on expensive DEX
        4. Repay loan + fee
        5. Keep profit

        Returns:
            Trade results
        """
        # Get prices
        price1 = dex1_pool.get_price('token0')
        price2 = dex2_pool.get_price('token0')

        # Determine direction
        if price1 < price2:
            buy_pool = dex1_pool
            sell_pool = dex2_pool
        else:
            buy_pool = dex2_pool
            sell_pool = dex1_pool

        # Execute trades (without actually modifying pools - simulation)
        tokens_bought = buy_pool.get_amount_out(loan_amount, 'token0')
        usd_received = sell_pool.get_amount_out(tokens_bought, 'token1')

        # Calculate profit
        flash_loan_cost = loan_amount * self.flash_loan_fee
        net_profit = usd_received - loan_amount - flash_loan_cost - gas_cost_usd

        roi = net_profit / loan_amount if loan_amount > 0 else 0

        trade_result = {
            'loan_amount': loan_amount,
            'tokens_bought': tokens_bought,
            'usd_received': usd_received,
            'flash_loan_fee': flash_loan_cost,
            'gas_cost': gas_cost_usd,
            'net_profit': net_profit,
            'roi': roi
        }

        self.trades.append(trade_result)
        return trade_result


# ---------------------------------------------------------------------------
# Smart Contract Vulnerability Patterns
# ---------------------------------------------------------------------------

class SmartContractAuditor:
    """
    Detect common smart contract vulnerabilities.

    Simplified pattern matching (real audits use symbolic execution).
    """

    def __init__(self):
        self.vulnerabilities = []

    def check_reentrancy(self, code: str) -> bool:
        """
        Check for reentrancy vulnerability.

        Pattern: External call before state update
        Example: call.value() before balance update
        """
        # Simplified check
        has_external_call = 'call.value' in code or '.call{value:' in code
        has_state_change = 'balances[' in code or 'balance =' in code

        # Check order (very simplified)
        if has_external_call and has_state_change:
            call_pos = code.find('call.value') or code.find('.call{value:')
            state_pos = code.find('balances[') or code.find('balance =')

            if call_pos < state_pos:
                self.vulnerabilities.append({
                    'type': 'reentrancy',
                    'severity': 'CRITICAL',
                    'description': 'External call before state update (reentrancy risk)'
                })
                return True

        return False

    def check_integer_overflow(self, code: str) -> bool:
        """Check for integer overflow (Solidity <0.8.0)."""
        has_arithmetic = '+' in code or '*' in code
        has_safe_math = 'SafeMath' in code or 'checked' in code

        if has_arithmetic and not has_safe_math:
            self.vulnerabilities.append({
                'type': 'integer_overflow',
                'severity': 'HIGH',
                'description': 'Arithmetic without overflow protection'
            })
            return True

        return False

    def check_access_control(self, code: str) -> bool:
        """Check for missing access control."""
        has_privileged_function = 'withdraw' in code or 'transfer' in code or 'mint' in code
        has_modifier = 'onlyOwner' in code or 'require(msg.sender' in code

        if has_privileged_function and not has_modifier:
            self.vulnerabilities.append({
                'type': 'access_control',
                'severity': 'CRITICAL',
                'description': 'Privileged function without access control'
            })
            return True

        return False

    def audit_contract(self, code: str) -> List[Dict]:
        """
        Audit smart contract for vulnerabilities.

        Returns:
            List of vulnerabilities found
        """
        self.vulnerabilities = []

        self.check_reentrancy(code)
        self.check_integer_overflow(code)
        self.check_access_control(code)

        return self.vulnerabilities


# ---------------------------------------------------------------------------
# CLI demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("═" * 70)
    print("  BLOCKCHAIN & DEFI PROTOCOL ANALYSIS")
    print("  Target: DeFi Arbitrage | Smart Contract Audit | $1M+ TC")
    print("═" * 70)

    # Demo 1: AMM Pricing
    print("\n── 1. AMM Pricing Mechanics ──")

    # Create Uniswap-style pool
    pool = AMMPool(
        token0_reserve=100,  # ETH
        token1_reserve=200000,  # USDC (@ $2000/ETH)
        fee=0.003
    )

    print(f"\n  Initial Pool State:")
    print(f"    ETH reserve:  {pool.token0_reserve:.2f}")
    print(f"    USDC reserve: ${pool.token1_reserve:,.0f}")
    print(f"    ETH price:    ${pool.get_price('token0'):,.2f}")

    # Execute trade
    trade_size = 10  # ETH
    usdc_received = pool.get_amount_out(trade_size, 'token0')
    avg_price = usdc_received / trade_size
    price_impact = (avg_price - pool.get_price('token0')) / pool.get_price('token0')

    print(f"\n  Trade: Sell {trade_size} ETH")
    print(f"    USDC received: ${usdc_received:,.2f}")
    print(f"    Average price: ${avg_price:,.2f}")
    print(f"    Price impact:  {price_impact:.2%}")

    # Demo 2: DEX Arbitrage
    print(f"\n── 2. DEX Arbitrage Detection ──")

    # Create two DEX pools with price discrepancy
    uniswap_pool = AMMPool(
        token0_reserve=100,
        token1_reserve=200000,  # $2000/ETH
        fee=0.003
    )

    sushiswap_pool = AMMPool(
        token0_reserve=100,
        token1_reserve=205000,  # $2050/ETH (2.5% higher)
        fee=0.003
    )

    print(f"\n  Uniswap:   ${uniswap_pool.get_price('token0'):,.2f}/ETH")
    print(f"  SushiSwap: ${sushiswap_pool.get_price('token0'):,.2f}/ETH")
    print(f"  Spread:    {(sushiswap_pool.get_price('token0') - uniswap_pool.get_price('token0')) / uniswap_pool.get_price('token0'):.2%}")

    # Detect arbitrage
    detector = DEXArbitrageDetector(min_profit_bps=10)
    opportunity = detector.detect_arbitrage(uniswap_pool, sushiswap_pool, 'ETH', max_trade_size=10)

    if opportunity:
        print(f"\n  ✅ Arbitrage Opportunity Detected:")
        print(f"    Trade size:  {opportunity['trade_size']:.2f} ETH")
        print(f"    Buy price:   ${opportunity['buy_price']:,.2f}")
        print(f"    Sell price:  ${opportunity['sell_price']:,.2f}")
        print(f"    Profit:      ${opportunity['profit']:,.2f}")
        print(f"    ROI:         {opportunity['roi_bps']:.0f} bps ({opportunity['profit_pct']:.2%})")

    # Demo 3: Flash Loan Arbitrage
    print(f"\n── 3. Flash Loan Arbitrage ──")

    flash_loan = FlashLoanArbitrage(flash_loan_fee=0.0009)

    loan_amounts = [10000, 50000, 100000, 500000]

    print(f"\n  Testing different flash loan sizes:")
    print(f"    {'Loan Size':<15} {'Net Profit':<15} {'ROI':<15}")
    print(f"    {'-' * 45}")

    best_trade = None
    best_profit = 0

    for loan_amount in loan_amounts:
        result = flash_loan.execute_arbitrage(
            uniswap_pool, sushiswap_pool, loan_amount, gas_cost_usd=100
        )

        print(f"    ${loan_amount:>12,}   ${result['net_profit']:>10,.2f}   {result['roi']:>10.2%}")

        if result['net_profit'] > best_profit:
            best_profit = result['net_profit']
            best_trade = result

    print(f"\n  Optimal Flash Loan Trade:")
    print(f"    Loan amount:     ${best_trade['loan_amount']:,.0f}")
    print(f"    Flash loan fee:  ${best_trade['flash_loan_fee']:,.2f}")
    print(f"    Gas cost:        ${best_trade['gas_cost']:,.2f}")
    print(f"    **Net profit**:  ${best_trade['net_profit']:,.2f}")
    print(f"    ROI:             {best_trade['roi']:.2%}")

    # Demo 4: Smart Contract Audit
    print(f"\n── 4. Smart Contract Security Audit ──")

    # Example vulnerable contract
    vulnerable_contract = """
    contract VulnerableVault {
        mapping(address => uint256) public balances;
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount);
            
            // VULNERABILITY: External call before state update (reentrancy)
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success);
            
            balances[msg.sender] -= amount;
        }
        
        function deposit() public payable {
            // VULNERABILITY: No overflow protection
            balances[msg.sender] += msg.value;
        }
    }
    """

    print(f"\n  Auditing smart contract...")

    auditor = SmartContractAuditor()
    vulnerabilities = auditor.audit_contract(vulnerable_contract)

    if vulnerabilities:
        print(f"\n  ⚠️  {len(vulnerabilities)} Vulnerabilities Found:")
        for i, vuln in enumerate(vulnerabilities, 1):
            print(f"\n    {i}. {vuln['type'].upper()}")
            print(f"       Severity: {vuln['severity']}")
            print(f"       {vuln['description']}")
    else:
        print(f"\n  ✅ No vulnerabilities detected")

    # Benchmark comparison
    print(f"\n{'═' * 70}")
    print(f"  KEY INSIGHTS FOR CRYPTO/DEFI ROLES")
    print(f"{'═' * 70}")

    print(f"""
1. DEX ARBITRAGE PROFITABILITY:
   Spread detected: {(sushiswap_pool.get_price('token0') - uniswap_pool.get_price('token0')) / uniswap_pool.get_price('token0'):.2%}
   Best profit: ${best_profit:,.2f} on ${best_trade['loan_amount']:,.0f} flash loan
   
   → Flash loans enable risk-free arbitrage (no capital needed)
   → Competition: Need <50ms latency to be first
   → Gas costs matter: $100 gas can eat 50%+ of profit
   → Optimal size: ~$100K-$500K per trade

2. FLASH LOAN MECHANICS:
   How it works: Borrow → Trade → Repay in SINGLE transaction
   
   → If transaction fails, everything reverts (no loss)
   → Aave/dYdX charge 0.09% fee ($900 on $1M loan)
   → Enable arbitrage with $0 capital
   → Jump Crypto/Wintermute: $50M-$200M daily volume

3. PRICE IMPACT ON AMMS:
   10 ETH trade on 100 ETH pool: {price_impact:.2%} impact
   
   → Larger trades → worse execution (quadratic impact)
   → This creates arbitrage opportunities
   → MEV bots front-run large trades (profit from impact)
   → Uniswap v3 concentrated liquidity reduces impact

4. SMART CONTRACT VULNERABILITIES:
   Detected: {len(vulnerabilities)} critical/high severity

Interview Q&A (Jump Crypto Managing Director):

Q: "Your DeFi arbitrage bot captures $2M/month. How?"
A: "Three strategies: (1) **DEX arbitrage**—Same token trades at different prices
    on Uniswap vs SushiSwap. We detect 1% spreads, execute flash loan arbitrage
    (borrow $10M, buy cheap, sell expensive, repay, keep profit). Zero capital
    risk. Capture $50K-$200K per trade, 100 trades/month = $2M. (2) **Liquidation
    arbitrage**—When Aave/Compound borrowers get liquidated, we buy collateral at
    discount (5-10% below market). Flash loan: Repay debt → claim collateral →
    sell at market. Risk-free $100K-$500K per liquidation. (3) **MEV extraction**—
    Front-run large trades on Uniswap (we see mempool, submit tx with higher gas,
    execute first). Profit from price impact. $20K-$50K per block. Total: $2M/month
    on $0 capital (all flash loans). But: Competition fierce, need <50ms latency."

Q: "How do you detect arbitrage opportunities in real-time?"
A: "**Infrastructure**: (1) Run full Ethereum node (not Infura—too slow). (2) Monitor
    mempool for pending transactions (see trades before mined). (3) Listen to DEX
    events (Swap, Sync) via WebSocket. (4) Calculate all pair prices every block
    (1000+ pairs × 10 DEXs = 10K prices/block). (5) Graph search for arbitrage paths
    (ETH → USDC → DAI → ETH). **Latency**: Process in <10ms, submit transaction in
    <50ms. Use Flashbots to avoid public mempool (reduces failed txs). **Competition**:
    We compete with 50+ other bots. Fastest bot wins. We're in top 5 (800μs latency
    vs 2ms average)."

Q: "Smart contract auditing. How much can you charge per audit?"
A: "**Pricing depends on complexity**: (1) **Simple token** ($20K-$50K)—ERC20 with
    basic functions. 2-3 days work. (2) **DEX/AMM** ($100K-$200K)—Complex math
    (constant product, concentrated liquidity). 1-2 weeks. (3) **Lending protocol**
    ($200K-$500K)—Aave/Compound-style. Liquidation logic complex. 3-4 weeks. (4)
    **Large ecosystem** ($500K-$2M)—Multiple contracts, cross-contract interactions.
    1-3 months. Trail of Bits charges $400-$600/hour × 4 auditors = $15K/day. A
    2-week audit = $150K. Senior auditors at ToB: $300K-$500K salary. Partners:
    $500K-$1M+. Side hustles (personal audits): $50K-$200K/year extra."

Q: "MEV (Maximal Extractable Value). How do you extract it?"
A: "MEV is profit from transaction ordering. **Three types**: (1) **Arbitrage**—
    Reorder transactions to profit from price differences. E.g., large buy on Uniswap
    → we front-run, buy first, sell to victim at higher price. (2) **Liquidations**—
    Be first to liquidate undercollateralized loans (5-10% bonus). Race to be first.
    (3) **Sandwich attacks**—Front-run + back-run. See large buy, buy before (push
    price up), sell after (to victim at worse price). **Infrastructure**: Flashbots
    bundle transactions to avoid public mempool. Priority gas auction (pay miners
    directly). Result: We extract $5M-$15M MEV per month. But: Ethically gray area,
    some consider it 'value extraction' vs 'value creation'."

Next steps:
  • Build live arbitrage bot (deploy on testnet)
  • Learn Solidity (write & audit smart contracts)
  • Contribute to DeFi protocols (GitHub)
  • MEV research (Flashbots, EigenLayer)
  • On-chain analytics (whale tracking, Nansen-style)
    """)

print(f"\n{'═' * 70}")
print(f"  Module complete. Crypto expertise extremely valuable.")
print(f"{'═' * 70}\n")
