Champions League Win Probability Predictor (ML-Based)

A machine learning-powered system that predicts UEFA Champions League winners using historical match data and probabilistic simulation techniques.

The model is trained on 15,000+ simulated matches calibrated with real-world UEFA statistics, incorporating key factors such as team strength, recent form, home advantage, and historical UCL performance. A Random Forest classifier is used to learn match outcome patterns, achieving strong predictive performance on test data.

To estimate tournament outcomes, the system runs 5,000 Monte Carlo simulations, modeling knockout-stage progression and calculating each team’s probability of winning the tournament. The prediction engine combines team strength (50%), current form (35%), and UCL pedigree (15%) to generate realistic win probabilities.

The application also supports custom team input, allowing users to define team strength, recent performance, and historical success, making it interactive and adaptable.

Key Features:

Machine Learning model (Random Forest) trained on large-scale data
Monte Carlo simulation for tournament prediction
Dynamic team strength and form evaluation
Custom team input with simulated or manual data
Clean CLI-based output with ranked probabilities

Tech Stack: Python, NumPy, Pandas, Scikit-learn
