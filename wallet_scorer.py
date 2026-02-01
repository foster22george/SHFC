"""
Polymarket Wallet Scoring System
=================================
This script analyzes wallet performance from daily leaderboard snapshots over the past 2 weeks.
It generates performance scores for each wallet by category (politics, sports, crypto, etc.)
and outputs ranked CSV files.

Author: Portfolio Analysis Tool
Date: February 2026
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class WalletScorer:
    """
    Analyzes wallet performance across Polymarket categories using historical snapshots.
    
    Scoring Methodology:
    - Tracks rank changes over 2-week period
    - Measures P&L growth rate
    - Considers consistency (how often they appear in top 100)
    - Volume-weighted performance
    - Recency bias (more recent performance weighted higher)
    """
    
    def __init__(self, snapshots_dir='snapshots', lookback_days=14):
        """
        Initialize the wallet scorer.
        
        Args:
            snapshots_dir (str): Path to the snapshots folder containing daily CSVs
                                 EDIT THIS if your snapshots folder is in a different location
            lookback_days (int): Number of days to analyze (default: 14 for 2 weeks)
        """
        self.snapshots_dir = Path(snapshots_dir)
        self.lookback_days = lookback_days
        self.today = datetime.now()
        self.cutoff_date = self.today - timedelta(days=lookback_days)
        
        # Data containers
        self.snapshot_files = []
        self.all_data = {}  # {category: DataFrame}
        self.scores = {}    # {category: DataFrame with scores}
        
        print(f"Initialized WalletScorer")
        print(f"Snapshots directory: {self.snapshots_dir.absolute()}")
        print(f"Lookback period: {lookback_days} days")
        print(f"Analyzing from {self.cutoff_date.date()} to {self.today.date()}")
    
    def discover_snapshot_files(self):
        """
        Scan the snapshots directory and identify all relevant CSV files.
        
        Expected filename formats:
        - Format 1: YYYY-MM-DD_<category>_<timeframe>.csv (e.g., 2026-01-15_politics_day.csv)
        - Format 2: leaderboard_YYYYMMDD.csv (e.g., leaderboard_20260129.csv)
        
        For Format 2, categories are determined from a 'category' column in the CSV itself.
        
        Returns:
            dict: Organized snapshot files by category
        """
        if not self.snapshots_dir.exists():
            raise FileNotFoundError(
                f"Snapshots directory not found: {self.snapshots_dir}\n"
                f"Please update the 'snapshots_dir' parameter in __init__ to point to your snapshots folder."
            )
        
        print(f"\nScanning for snapshot files in: {self.snapshots_dir}")
        
        # Find all CSV files
        csv_files = list(self.snapshots_dir.glob('*.csv'))
        print(f"Found {len(csv_files)} total CSV files")
        
        # Organize by category
        categorized_files = {}
        
        for filepath in csv_files:
            try:
                filename = filepath.stem  # Remove .csv extension
                parts = filename.split('_')
                
                # Try Format 1: YYYY-MM-DD_category_timeframe.csv
                if len(parts) >= 3 and '-' in parts[0]:
                    date_str = parts[0]
                    category = parts[1]
                    timeframe = parts[2]
                    
                    # Parse date
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    # Only include files within lookback period
                    if file_date < self.cutoff_date:
                        continue
                    
                    # Only use 'day' timeframe to avoid double-counting
                    if timeframe != 'day':
                        continue
                    
                    # Store file info
                    if category not in categorized_files:
                        categorized_files[category] = []
                    
                    categorized_files[category].append({
                        'path': filepath,
                        'date': file_date,
                        'category': category,
                        'timeframe': 'day'
                    })
                
                # Try Format 2: leaderboard_YYYYMMDD.csv
                elif len(parts) == 2 and parts[0] == 'leaderboard':
                    date_str = parts[1]
                    
                    # Parse date (YYYYMMDD format)
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    
                    # Only include files within lookback period
                    if file_date < self.cutoff_date:
                        print(f"Skipping {filepath.name}: Outside date range ({file_date.date()} < {self.cutoff_date.date()})")
                        continue
                    
                    # For leaderboard files, we need to read the CSV to get categories
                    # Mark this file for processing (we'll handle categories during load)
                    if 'combined' not in categorized_files:
                        categorized_files['combined'] = []
                    
                    categorized_files['combined'].append({
                        'path': filepath,
                        'date': file_date,
                        'category': 'combined',  # Will be split by category column
                        'timeframe': 'day'
                    })
                    
                else:
                    print(f"Skipping {filepath.name}: Unrecognized format")
                    continue
                
            except Exception as e:
                print(f"Error parsing {filepath.name}: {e}")
                continue
        
        # Sort files by date for each category
        for category in categorized_files:
            categorized_files[category].sort(key=lambda x: x['date'])
        
        self.snapshot_files = categorized_files
        
        print(f"\nCategorized snapshots within {self.lookback_days}-day window:")
        for category, files in categorized_files.items():
            print(f"  {category}: {len(files)} snapshots")
        
        return categorized_files
    
    def load_snapshots(self):
        """
        Load all relevant snapshot CSV files into memory.
        
        Expected CSV columns:
        - rank: Position on leaderboard (1-100)
        - address: Wallet address
        - pnl: Profit and loss
        - volume: Trading volume
        - userName: Display name
        - category: (Optional) Trading category - if present, file will be split by category
        - timeframe: (Optional) Timeframe indicator
        
        Returns:
            dict: {category: DataFrame}
        """
        print(f"\nLoading snapshot data...")
        
        all_snapshots = {}
        
        for file_category, files in self.snapshot_files.items():
            for file_info in files:
                try:
                    df = pd.read_csv(file_info['path'])
                    
                    # Add metadata
                    df['snapshot_date'] = file_info['date']
                    
                    # Check if this is a combined file with category column
                    if 'category' in df.columns and file_category == 'combined':
                        # Split by category
                        for category in df['category'].unique():
                            category_df = df[df['category'] == category].copy()
                            category_df['category'] = category
                            
                            # Ensure required columns exist
                            required_cols = ['rank', 'address', 'pnl', 'volume', 'userName']
                            missing_cols = [col for col in required_cols if col not in category_df.columns]
                            
                            if missing_cols:
                                print(f"Warning: {file_info['path'].name} missing columns: {missing_cols}")
                                continue
                            
                            # Convert numeric columns
                            category_df['rank'] = pd.to_numeric(category_df['rank'], errors='coerce')
                            category_df['pnl'] = pd.to_numeric(category_df['pnl'], errors='coerce')
                            category_df['volume'] = pd.to_numeric(category_df['volume'], errors='coerce')
                            
                            # Add to category snapshots
                            if category not in all_snapshots:
                                all_snapshots[category] = []
                            all_snapshots[category].append(category_df)
                    
                    else:
                        # Regular categorized file
                        category = file_info['category']
                        df['category'] = category
                        
                        # Ensure required columns exist
                        required_cols = ['rank', 'address', 'pnl', 'volume', 'userName']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            print(f"Warning: {file_info['path'].name} missing columns: {missing_cols}")
                            continue
                        
                        # Convert numeric columns
                        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                        
                        # Add to category snapshots
                        if category not in all_snapshots:
                            all_snapshots[category] = []
                        all_snapshots[category].append(df)
                    
                except Exception as e:
                    print(f"Error loading {file_info['path'].name}: {e}")
                    continue
        
        # Combine snapshots for each category
        combined_snapshots = {}
        for category, snapshot_list in all_snapshots.items():
            if snapshot_list:
                combined_snapshots[category] = pd.concat(snapshot_list, ignore_index=True)
                print(f"  {category}: {len(snapshot_list)} snapshots, {len(combined_snapshots[category])} total records")
        
        self.all_data = combined_snapshots
        return combined_snapshots
    
    def calculate_scores(self):
        """
        Calculate comprehensive performance scores for each wallet by category.
        
        Scoring Components (0-100 scale):
        1. Rank Improvement Score (30%): How much their rank improved
        2. P&L Growth Score (25%): Rate of P&L increase
        3. Consistency Score (20%): How often they appear in top 100
        4. Volume Score (15%): Trading volume relative to peers
        5. Recency Score (10%): Better recent performance weighted higher
        
        Returns:
            dict: {category: DataFrame with scores}
        """
        print(f"\nCalculating performance scores...")
        
        scored_categories = {}
        
        for category, df in self.all_data.items():
            print(f"\n  Analyzing {category}...")
            
            # Get unique wallets
            unique_wallets = df['address'].unique()
            print(f"    Found {len(unique_wallets)} unique wallets")
            
            wallet_scores = []
            
            for wallet in unique_wallets:
                wallet_data = df[df['address'] == wallet].sort_values('snapshot_date')
                
                if len(wallet_data) < 2:
                    # Need at least 2 data points for meaningful scoring
                    continue
                
                # Get first and last snapshot
                first_snap = wallet_data.iloc[0]
                last_snap = wallet_data.iloc[-1]
                
                # === 1. RANK IMPROVEMENT SCORE (30 points) ===
                rank_change = first_snap['rank'] - last_snap['rank']  # Positive = improvement
                # Normalize: moving from 100 to 1 = 100 points, scale to 30
                rank_score = min(30, max(0, (rank_change / 99) * 30))
                
                # === 2. P&L GROWTH SCORE (25 points) ===
                pnl_change = last_snap['pnl'] - first_snap['pnl']
                pnl_growth_rate = (pnl_change / abs(first_snap['pnl'])) if first_snap['pnl'] != 0 else 0
                # Cap at 200% growth = max score
                pnl_score = min(25, max(0, (pnl_growth_rate / 2.0) * 25))
                
                # === 3. CONSISTENCY SCORE (20 points) ===
                appearances = len(wallet_data)
                max_possible_appearances = len(df['snapshot_date'].unique())
                consistency_rate = appearances / max_possible_appearances
                consistency_score = consistency_rate * 20
                
                # === 4. VOLUME SCORE (15 points) ===
                avg_volume = wallet_data['volume'].mean()
                # Compare to category average
                category_avg_volume = df['volume'].mean()
                volume_ratio = avg_volume / category_avg_volume if category_avg_volume > 0 else 1
                volume_score = min(15, max(0, volume_ratio * 7.5))  # 2x avg = max score
                
                # === 5. RECENCY SCORE (10 points) ===
                # Check if they're improving in recent snapshots (last 3 vs first 3)
                if len(wallet_data) >= 6:
                    recent_avg_rank = wallet_data.tail(3)['rank'].mean()
                    early_avg_rank = wallet_data.head(3)['rank'].mean()
                    recency_improvement = early_avg_rank - recent_avg_rank
                    recency_score = min(10, max(0, (recency_improvement / 50) * 10))
                else:
                    recency_score = 5  # Neutral score if not enough data
                
                # === TOTAL SCORE (0-100) ===
                total_score = rank_score + pnl_score + consistency_score + volume_score + recency_score
                
                wallet_scores.append({
                    'address': wallet,
                    'userName': last_snap['userName'],
                    'category': category,
                    'total_score': round(total_score, 2),
                    'rank_score': round(rank_score, 2),
                    'pnl_score': round(pnl_score, 2),
                    'consistency_score': round(consistency_score, 2),
                    'volume_score': round(volume_score, 2),
                    'recency_score': round(recency_score, 2),
                    'first_rank': first_snap['rank'],
                    'last_rank': last_snap['rank'],
                    'rank_change': rank_change,
                    'first_pnl': first_snap['pnl'],
                    'last_pnl': last_snap['pnl'],
                    'pnl_change': pnl_change,
                    'pnl_growth_pct': round(pnl_growth_rate * 100, 2),
                    'appearances': appearances,
                    'avg_volume': round(avg_volume, 2),
                    'days_tracked': (last_snap['snapshot_date'] - first_snap['snapshot_date']).days
                })
            
            # Convert to DataFrame and rank by total score
            scores_df = pd.DataFrame(wallet_scores)
            scores_df = scores_df.sort_values('total_score', ascending=False).reset_index(drop=True)
            scores_df['score_rank'] = range(1, len(scores_df) + 1)
            
            scored_categories[category] = scores_df
            
            print(f"    Scored {len(scores_df)} wallets")
            print(f"    Top score: {scores_df['total_score'].max():.2f}")
            print(f"    Avg score: {scores_df['total_score'].mean():.2f}")
        
        self.scores = scored_categories
        return scored_categories
    
    def get_top_wallets(self, category, n=5):
        """
        Get the top N wallets for a specific category.
        
        Args:
            category (str): Category name (e.g., 'politics', 'sports')
            n (int): Number of top wallets to return
        
        Returns:
            DataFrame: Top N wallets with scores
        """
        if category not in self.scores:
            print(f"Category '{category}' not found. Available: {list(self.scores.keys())}")
            return None
        
        return self.scores[category].head(n)
    
    def save_scores(self, output_dir='scores'):
        """
        Save all scored wallets to CSV files in the output directory.
        
        Args:
            output_dir (str): Directory to save score files
                             EDIT THIS if you want scores saved elsewhere
        
        File naming: <category>_wallet_scores_MMDDYYYY.csv
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        today_str = datetime.now().strftime('%m%d%Y')
        
        print(f"\nSaving scores to: {output_path.absolute()}")
        
        saved_files = []
        
        for category, scores_df in self.scores.items():
            filename = f"{category}_wallet_scores_{today_str}.csv"
            filepath = output_path / filename
            
            scores_df.to_csv(filepath, index=False)
            saved_files.append(filename)
            
            print(f"  ✓ {filename} ({len(scores_df)} wallets)")
        
        # Create a summary index file
        self._create_summary_index(output_path, today_str, saved_files)
        
        print(f"\n✓ Saved {len(saved_files)} category score files")
        return saved_files
    
    def _create_summary_index(self, output_path, date_str, saved_files):
        """
        Create a summary CSV with top 5 wallets from each category.
        
        Args:
            output_path (Path): Output directory
            date_str (str): Date string for filename
            saved_files (list): List of saved filenames
        """
        summary_data = []
        
        for category, scores_df in self.scores.items():
            top_5 = scores_df.head(5)
            
            for idx, row in top_5.iterrows():
                summary_data.append({
                    'category': category,
                    'rank': row['score_rank'],
                    'address': row['address'],
                    'userName': row['userName'],
                    'total_score': row['total_score'],
                    'last_rank': row['last_rank'],
                    'last_pnl': row['last_pnl'],
                    'pnl_growth_pct': row['pnl_growth_pct'],
                    'recommendation': 'FOLLOW' if row['score_rank'] <= 5 else 'WATCH'
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_path / f"TOP_WALLETS_SUMMARY_{date_str}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"  ✓ TOP_WALLETS_SUMMARY_{date_str}.csv (top 5 per category)")
    
    def print_summary(self):
        """Print a summary of the scoring results."""
        print("\n" + "=" * 80)
        print("WALLET SCORING SUMMARY")
        print("=" * 80)
        
        for category, scores_df in self.scores.items():
            print(f"\n{category.upper()} - Top 5 Wallets:")
            print("-" * 80)
            
            top_5 = scores_df.head(5)
            
            for idx, row in top_5.iterrows():
                print(f"\n  #{row['score_rank']}: {row['userName']} ({row['address'][:10]}...)")
                print(f"    Total Score: {row['total_score']:.2f}/100")
                print(f"    Rank: {int(row['first_rank'])} → {int(row['last_rank'])} ({'+' if row['rank_change'] > 0 else ''}{int(row['rank_change'])})")
                print(f"    P&L: ${row['first_pnl']:,.0f} → ${row['last_pnl']:,.0f} ({'+' if row['pnl_change'] > 0 else ''}${row['pnl_change']:,.0f}, {row['pnl_growth_pct']:.1f}%)")
                print(f"    Consistency: {row['appearances']} appearances over {row['days_tracked']} days")
        
        print("\n" + "=" * 80)
    
    def run_full_analysis(self, save=True):
        """
        Execute the complete scoring pipeline.
        
        Args:
            save (bool): Whether to save results to CSV
        
        Returns:
            dict: Scores by category
        """
        print("=" * 80)
        print("STARTING WALLET SCORING ANALYSIS")
        print("=" * 80)
        
        # Step 1: Discover files
        self.discover_snapshot_files()
        
        if not self.snapshot_files:
            print("\nNo snapshot files found in date range. Exiting.")
            return None
        
        # Step 2: Load data
        self.load_snapshots()
        
        if not self.all_data:
            print("\nNo data loaded. Exiting.")
            return None
        
        # Step 3: Calculate scores
        self.calculate_scores()
        
        # Step 4: Display summary
        self.print_summary()
        
        # Step 5: Save results
        if save:
            self.save_scores()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        return self.scores


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block - run this script to score wallets.
    
    TO USE THIS SCRIPT:
    1. Ensure your 'snapshots' folder is in the same directory as this script
       OR update the 'snapshots_dir' parameter below to point to your folder
    
    2. Run the script: python wallet_scorer.py
    
    3. Results will be saved to the 'scores' folder
    """
    
    # ========================================================================
    # CONFIGURATION - EDIT THESE PARAMETERS AS NEEDED
    # ========================================================================
    
    SNAPSHOTS_DIR = 'snapshots'  # Path to your snapshots folder
    LOOKBACK_DAYS = 14           # Number of days to analyze (2 weeks)
    OUTPUT_DIR = 'scores'        # Where to save score files
    
    # ========================================================================
    
    # Initialize scorer
    scorer = WalletScorer(
        snapshots_dir=SNAPSHOTS_DIR,
        lookback_days=LOOKBACK_DAYS
    )
    
    # Run complete analysis
    scores = scorer.run_full_analysis(save=True)
    
    # Optional: Access specific category scores
    if scores:
        # Example: Get top 5 politics traders
        if 'politics' in scores:
            print("\n" + "=" * 80)
            print("POLITICS - TOP 5 WALLETS TO FOLLOW:")
            print("=" * 80)
            top_politics = scorer.get_top_wallets('politics', n=5)
            print(top_politics[['score_rank', 'userName', 'address', 'total_score', 'last_pnl', 'pnl_growth_pct']])
        
        # You can access all scores via: scorer.scores
        # Individual category: scorer.scores['politics']
        # Top N wallets: scorer.get_top_wallets('sports', n=10)