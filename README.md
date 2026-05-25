# UCL Predictor — It predicted PSG. PSG won.

> ML-powered UEFA Champions League winner predictor using 
> Random Forest + Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Random%20Forest-orange)
![UCL 2025](https://img.shields.io/badge/UCL%202025-PSG%20✓-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## The prediction
Before the 2024/25 Champions League final, this model ranked 
PSG as the most likely winner. PSG won the UCL 2025.

## How it works
The model combines three signals to estimate each team's 
win probability:

| Factor | Weight |
|--------|--------|
| Team strength | 50% |
| Recent form | 35% |
| UCL pedigree | 15% |

These feed into a Random Forest classifier trained on 15,000+ 
simulated matches calibrated with real UEFA statistics. 
Tournament outcomes are then estimated via 5,000 Monte Carlo 
simulations across the full knockout bracket.

## Features
- Random Forest trained on large-scale match data
- 5,000 Monte Carlo simulations per prediction run
- Dynamic team strength & form evaluation
- Custom team input — define your own squad stats
- Clean CLI output with ranked win probabilities

## Quick start
```bash
git clone https://github.com/ria0304/ucl-predictor
cd ucl-predictor
pip install -r requirements.txt
python ucl_predictor.py
```

## Tech stack
`Python` `NumPy` `Pandas` `Scikit-learn`

## Methodology
Traditional UCL predictions rely on bookmaker odds or 
Elo ratings alone. This system simulates the full tournament 
bracket thousands of times, letting randomness and team 
quality interact the way real knockout football does — 
any team can lose on the night, but the best teams win 
more simulations.

## Future scope
- Train on real historical UCL match data (not simulated)
- Add player-level features (injuries, suspensions)
- Build a web UI for the 2025/26 season predictions
- Integrate live odds comparison


## License

This project is licensed under the [MIT License](LICENSE).
