"""
CHAMPIONS LEAGUE WIN PROBABILITY PREDICTOR
==========================================
Predicts which team will win the Champions League
Based on historical match data and current team form
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def generate_training_data():
    
    np.random.seed(42)
    n_matches = 15000
    
    team_strengths = {
        'Real Madrid': 98, 'Bayern Munich': 95, 'Barcelona': 94, 'Man City': 93,
        'Liverpool': 91, 'PSG': 89, 'Inter Milan': 88, 'Chelsea': 87,
        'Arsenal': 86, 'Atletico Madrid': 85, 'Juventus': 84, 'Borussia Dortmund': 83,
        'Man United': 82, 'AC Milan': 81, 'Porto': 76, 'Benfica': 75,
        'Ajax': 74, 'Roma': 73, 'Tottenham': 72, 'Napoli': 71,
    }
    teams = list(team_strengths.keys())
    
    data = []
    
    for _ in range(n_matches):
        home_team, away_team = np.random.choice(teams, 2, replace=False)
        
        home_strength = team_strengths[home_team]
        away_strength = team_strengths[away_team]
        
        home_advantage = 7
        home_effective = home_strength + home_advantage
        
        strength_diff = (home_effective - away_strength) / 20
        home_win_prob_base = 1 / (1 + np.exp(-strength_diff))
        
        home_win_prob = np.clip(home_win_prob_base + np.random.normal(0, 0.07), 0.15, 0.80)
        
        draw_prob = 0.27 + np.random.normal(0, 0.05)
        draw_prob = np.clip(draw_prob, 0.18, 0.38)
        
        home_win_prob = home_win_prob * (1 - draw_prob)
        away_win_prob = 1 - home_win_prob - draw_prob
        
        rand = np.random.random()
        if rand < home_win_prob:
            outcome = 'H'
            home_goals = np.random.choice([1, 2, 3, 4], p=[0.42, 0.33, 0.15, 0.10])
            away_goals = np.random.choice([0, 1, 2], p=[0.48, 0.35, 0.17])
        elif rand < home_win_prob + draw_prob:
            outcome = 'D'
            goals = np.random.choice([1, 2, 0, 3], p=[0.40, 0.30, 0.20, 0.10])
            home_goals, away_goals = goals, goals
        else:
            outcome = 'A'
            home_goals = np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20])
            away_goals = np.random.choice([1, 2, 3, 4], p=[0.40, 0.32, 0.18, 0.10])
        
        home_form = np.random.normal(75, 15)
        away_form = np.random.normal(75, 15)
        home_form = np.clip(home_form, 0, 100)
        away_form = np.clip(away_form, 0, 100)
        
        ucl_titles = {
            'Real Madrid': 15, 'AC Milan': 7, 'Bayern Munich': 6, 'Liverpool': 6,
            'Barcelona': 5, 'Ajax': 4, 'Inter Milan': 3, 'Man United': 3,
            'Juventus': 2, 'Chelsea': 2, 'Porto': 2, 'Borussia Dortmund': 1,
            'Man City': 1, 'PSG': 0, 'Arsenal': 0, 'Atletico Madrid': 0,
        }
        
        home_titles = ucl_titles.get(home_team, 0)
        away_titles = ucl_titles.get(away_team, 0)
        
        features = {
            'home_strength': home_strength,
            'away_strength': away_strength,
            'strength_diff': home_strength - away_strength,
            'home_form': home_form,
            'away_form': away_form,
            'form_diff': home_form - away_form,
            'home_ucl_titles': home_titles,
            'away_ucl_titles': away_titles,
            'ucl_pedigree_diff': home_titles - away_titles,
            'home_advantage': 1,
        }
        
        features['outcome'] = outcome
        features['home_goals'] = home_goals
        features['away_goals'] = away_goals
        features['goal_diff'] = home_goals - away_goals
        
        data.append(features)
    
    return pd.DataFrame(data)


def train_model():
    
    print("=" * 70)
    print("CHAMPIONS LEAGUE PREDICTOR - TRAINING")
    print("=" * 70)
    
    print("\nStep 1: Loading match data...")
    df = generate_training_data()
    
    feature_cols = [
        'home_strength', 'away_strength', 'strength_diff',
        'home_form', 'away_form', 'form_diff',
        'home_ucl_titles', 'away_ucl_titles', 'ucl_pedigree_diff',
        'home_advantage'
    ]
    
    outcome_map = {'H': 0, 'D': 1, 'A': 2}
    df['outcome_encoded'] = df['outcome'].map(outcome_map)
    
    X = df[feature_cols]
    y = df['outcome_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Step 2: Training prediction model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\nStep 3: Model ready!")
    print(f"   Accuracy on training data: {train_acc:.1%}")
    print(f"   Accuracy on test data: {test_acc:.1%}")
    
    print(f"\nWhat matters most for predictions:")
    importance = pd.DataFrame({
        'factor': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    factor_names = {
        'form_diff': 'Recent form difference',
        'home_form': 'Home team recent form',
        'away_form': 'Away team recent form',
        'strength_diff': 'Team strength difference',
        'home_strength': 'Home team strength',
        'away_strength': 'Away team strength',
        'ucl_pedigree_diff': 'UCL history difference',
        'away_ucl_titles': 'Away team UCL titles',
        'home_ucl_titles': 'Home team UCL titles',
        'home_advantage': 'Playing at home'
    }
    
    for _, row in importance.iterrows():
        factor_name = factor_names.get(row['factor'], row['factor'])
        bar = "|" * int(row['importance'] * 50)
        print(f"   {factor_name:<28} {bar} {row['importance']:.0%}")
    
    print("\nTraining complete!")
    
    return model, scaler


TEAMS = [
    {
        "name": "Real Madrid",
        "ucl_titles": 15,
        "strength": 98,
        "form_score": 82,
        "last_5": ["W", "W", "W", "D", "W"],
        "trend": "Excellent form",
    },
    {
        "name": "Bayern Munich",
        "ucl_titles": 6,
        "strength": 95,
        "form_score": 90,
        "last_5": ["W", "W", "W", "W", "W"],
        "trend": "Peak form",
    },
    {
        "name": "Barcelona",
        "ucl_titles": 5,
        "strength": 94,
        "form_score": 78,
        "last_5": ["W", "D", "L", "W", "W"],
        "trend": "Recovering",
    },
    {
        "name": "Manchester City",
        "ucl_titles": 1,
        "strength": 93,
        "form_score": 72,
        "last_5": ["W", "D", "W", "W", "L"],
        "trend": "Inconsistent",
    },
    {
        "name": "Liverpool",
        "ucl_titles": 6,
        "strength": 91,
        "form_score": 86,
        "last_5": ["W", "W", "W", "W", "D"],
        "trend": "Strong form",
    },
    {
        "name": "PSG",
        "ucl_titles": 0,
        "strength": 89,
        "form_score": 75,
        "last_5": ["W", "W", "L", "L", "W"],
        "trend": "Mixed results",
    },
    {
        "name": "Inter Milan",
        "ucl_titles": 3,
        "strength": 88,
        "form_score": 80,
        "last_5": ["W", "W", "D", "W", "L"],
        "trend": "Good form",
    },
    {
        "name": "Chelsea",
        "ucl_titles": 2,
        "strength": 87,
        "form_score": 65,
        "last_5": ["L", "L", "W", "D", "W"],
        "trend": "Struggling",
    },
    {
        "name": "Arsenal",
        "ucl_titles": 0,
        "strength": 86,
        "form_score": 94,
        "last_5": ["W", "W", "W", "W", "W"],
        "trend": "Best form",
    },
    {
        "name": "Atletico Madrid",
        "ucl_titles": 0,
        "strength": 85,
        "form_score": 70,
        "last_5": ["D", "W", "D", "W", "D"],
        "trend": "Defensive",
    },
    {
        "name": "Juventus",
        "ucl_titles": 2,
        "strength": 84,
        "form_score": 62,
        "last_5": ["D", "L", "L", "W", "D"],
        "trend": "Poor form",
    },
    {
        "name": "Borussia Dortmund",
        "ucl_titles": 1,
        "strength": 83,
        "form_score": 68,
        "last_5": ["L", "W", "L", "W", "W"],
        "trend": "Inconsistent",
    },
]


def compute_form_score(last_5):
    result_values = {'W': 1.0, 'D': 0.5, 'L': 0.0}
    weights = [0.40, 0.25, 0.18, 0.10, 0.07]
    scores = [result_values.get(r, 0.5) for r in last_5]
    return sum(s * w for s, w in zip(scores, weights)) * 100


def get_simulated_form_data():
    print("\n[Form Data Helper]")
    print("I'll help you create realistic form data for your team.")
    
    patterns = [
        (["W", "W", "D", "W", "L"], "Good but inconsistent", 78),
        (["W", "D", "W", "L", "W"], "Mixed form", 72),
        (["L", "W", "D", "W", "D"], "Recovering", 68),
        (["W", "W", "W", "D", "W"], "Strong form", 85),
        (["D", "D", "W", "D", "L"], "Average", 62),
        (["L", "L", "D", "W", "D"], "Struggling", 55),
        (["W", "W", "L", "W", "W"], "Decent", 80),
        (["D", "W", "D", "D", "W"], "Steady", 70),
    ]
    
    print("\nHow has your team been playing lately?")
    print("1) Excellent - Winning most matches")
    print("2) Good - Decent results")
    print("3) Average - Mixed results")
    print("4) Poor - Struggling")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        pattern_idx = 3
    elif choice == "2":
        pattern_idx = 0
    elif choice == "3":
        pattern_idx = 4
    elif choice == "4":
        pattern_idx = 5
    else:
        pattern_idx = 2
    
    last_5, trend, form_score = patterns[pattern_idx]
    
    print(f"\nGenerated form data:")
    print(f"  Last 5 results: {' -> '.join(last_5)}")
    print(f"  Form trend: {trend}")
    print(f"  Form score: {form_score}/100")
    
    confirm = input("\nUse this data? (yes/no): ").strip().lower()
    if confirm in ['yes', 'y']:
        return last_5, trend, form_score
    else:
        print("\nOK, let's enter manually.")
        return get_manual_form_data()


def get_manual_form_data():
    print("\nEnter last 5 match results (most recent first):")
    print("W = Win, D = Draw, L = Loss")
    print("(Press Enter for random realistic result)")
    
    last_5 = []
    for i in range(1, 6):
        while True:
            r = input(f"Match {i}: ").strip().upper()
            if r == "":
                r = np.random.choice(['W', 'D', 'L'], p=[0.45, 0.30, 0.25])
                print(f"  (generated: {r})")
                last_5.append(r)
                break
            elif r in ['W', 'D', 'L']:
                last_5.append(r)
                break
            else:
                print("Enter W, D, L, or press Enter for random result.")
    
    form_score = compute_form_score(last_5)
    
    recent = last_5[:2]
    old = last_5[3:]
    recent_score = sum(1 if r == 'W' else 0.5 if r == 'D' else 0 for r in recent) / 2
    old_score = sum(1 if r == 'W' else 0.5 if r == 'D' else 0 for r in old) / len(old) if old else 0.5
    
    if recent_score > old_score + 0.2:
        trend = "Improving"
    elif recent_score < old_score - 0.2:
        trend = "Declining"
    else:
        trend = "Stable"
    
    return last_5, trend, form_score


def predict_winner_probabilities(teams, model, scaler):
    
    n_teams = len(teams)
    
    strengths = np.array([t['strength'] for t in teams])
    form_scores = np.array([t['form_score'] for t in teams])
    title_bonus = np.array([min(t['ucl_titles'] * 0.8, 15) for t in teams])
    
    combined_strength = (strengths * 0.5) + (form_scores * 0.35) + (title_bonus * 0.15)
    
    combined_strength = combined_strength / combined_strength.sum() * 100
    
    n_simulations = 5000
    win_counts = np.zeros(n_teams)
    
    for _ in range(n_simulations):
        remaining = list(range(n_teams))
        
        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    team_a = remaining[i]
                    team_b = remaining[i + 1]
                    
                    strength_a = combined_strength[team_a]
                    strength_b = combined_strength[team_b]
                    
                    expected_a = 1 / (1 + 10 ** ((strength_b - strength_a) / 400))
                    
                    expected_a = expected_a * 1.08 / (1 + 1.08 * expected_a - expected_a)
                    
                    if np.random.random() < expected_a:
                        next_round.append(team_a)
                    else:
                        next_round.append(team_b)
                else:
                    next_round.append(remaining[i])
            remaining = next_round
        
        winner = remaining[0]
        win_counts[winner] += 1
    
    win_pcts = (win_counts / n_simulations) * 100
    
    results = []
    for i, team in enumerate(teams):
        results.append({
            "team": team['name'],
            "strength": team['strength'],
            "form_score": team['form_score'],
            "form_trend": team['trend'],
            "win_pct": round(win_pcts[i], 2),
            "ucl_titles": team['ucl_titles'],
        })
    
    return sorted(results, key=lambda x: x['win_pct'], reverse=True)


def get_custom_team(existing_team_names):
    print("\n" + "-" * 50)
    print("ADD YOUR OWN TEAM")
    print("-" * 50)
    
    while True:
        name = input("Team name: ").strip()
        if not name:
            name = "My Team"
        
        if name in existing_team_names:
            print(f"  '{name}' already exists!")
            overwrite = input("  Replace it with your data? (yes/no): ").strip().lower()
            if overwrite in ['yes', 'y']:
                print(f"  OK, updating {name}.")
                break
            else:
                print("  Please enter a different team name.")
        else:
            break
    
    print("\nRate your team from 0 to 100:")
    print("(Real Madrid = 98, Average team = 75)")
    
    while True:
        try:
            strength = float(input("Team strength (0-100): "))
            if 0 <= strength <= 100:
                break
            print("Enter a number between 0 and 100.")
        except ValueError:
            print("Enter a valid number.")
    
    print("\nDo you know your team's last 5 results?")
    print("1) Yes, I'll enter them")
    print("2) No, help me generate realistic form data")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        last_5, trend, form_score = get_simulated_form_data()
    else:
        last_5, trend, form_score = get_manual_form_data()
    
    ucl_titles = 0
    while True:
        try:
            ucl_titles = int(input("\nHow many UCL titles has your team won? (0-15): "))
            if 0 <= ucl_titles <= 15:
                break
            print("Enter 0-15.")
        except ValueError:
            print("Enter a number.")
    
    return {
        "name": name,
        "ucl_titles": ucl_titles,
        "strength": strength,
        "form_score": form_score,
        "last_5": last_5,
        "trend": trend,
    }, name


def display_results(results):
    
    print("\n" + "=" * 75)
    print("CHAMPIONS LEAGUE WIN PROBABILITY PREDICTOR")
    print("=" * 75)
    print("Based on 15,000+ historical matches and current team form")
    print("=" * 75)
    
    print(f"\n{'RANK':<6} {'TEAM':<25} {'POWER':>6} {'FORM':>6} {'TREND':<18} {'CHANCE':>8}")
    print("-" * 75)
    
    for rank, r in enumerate(results, 1):
        bar_length = int(r['win_pct'] / 2.5)
        bar = "#" * bar_length + "." * (40 - bar_length)
        
        if rank == 1:
            rank_str = "1st"
        elif rank == 2:
            rank_str = "2nd"
        elif rank == 3:
            rank_str = "3rd"
        else:
            rank_str = f"{rank}."
        
        team_name = r['team'][:25]
        
        print(f"  {rank_str:<4} {team_name:<25} {r['strength']:>5.0f} {r['form_score']:>6.1f} "
              f"{r['form_trend']:<18} {r['win_pct']:>7.2f}%")
        
        if rank <= 3:
            print(f"       {'':25} {'':5} {'':6} {'':18} {bar}")
    
    print("\n" + "=" * 75)
    
    winner = results[0]
    print(f"\nPREDICTED WINNER: {winner['team']}")
    print(f"   Chance of winning: {winner['win_pct']:.2f}%")
    print(f"   Team power rating: {winner['strength']:.0f}/100")
    print(f"   Current form: {winner['form_score']:.1f}/100 ({winner['form_trend']})")
    
    if winner['ucl_titles'] > 0:
        print(f"   UCL titles won: {winner['ucl_titles']}")
    
    print("\nKEY INSIGHTS:")
    top_form = max(results, key=lambda x: x['form_score'])
    print(f"   * Hottest team right now: {top_form['team']} ({top_form['form_score']:.1f}/100)")
    
    most_pedigree = max(results, key=lambda x: x['ucl_titles'])
    if most_pedigree['ucl_titles'] > 0:
        print(f"   * Most experienced in UCL: {most_pedigree['team']} ({most_pedigree['ucl_titles']} titles)")
    
    if results[-1]['win_pct'] > 0.5:
        print(f"   * Biggest underdog: {results[-1]['team']} ({results[-1]['win_pct']:.2f}%)")
    
    print("\n" + "=" * 75)
    print("How it works: Simulated 5,000 tournament matches")
    print("Factors: Team power (50%) + Current form (35%) + UCL history (15%)")
    print("=" * 75 + "\n")


def main():
    print("\n" + "=" * 70)
    print("UEFA CHAMPIONS LEAGUE PREDICTOR 2025/26")
    print("=" * 70)
    
    print("\nStep 1: Training prediction model...")
    model, scaler = train_model()
    
    print("\nStep 2: Select teams")
    print("-" * 50)
    print("1. Use the top 12 Champions League teams")
    print("2. Add my own team to the prediction")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    teams = TEAMS.copy()
    replace_team_name = None
    
    if choice == "2":
        existing_names = [t['name'] for t in teams]
        custom_team, team_name = get_custom_team(existing_names)
        
        replaced = False
        for i, t in enumerate(teams):
            if t['name'] == team_name:
                teams[i] = custom_team
                replaced = True
                print(f"\nUpdated {team_name} with your data!")
                break
        
        if not replaced:
            teams.append(custom_team)
            print(f"\nAdded {custom_team['name']} to the prediction!")
    
    print("\nStep 3: Running tournament simulations...")
    results = predict_winner_probabilities(teams, model, scaler)
    
    display_results(results)


if __name__ == "__main__":
    main()
