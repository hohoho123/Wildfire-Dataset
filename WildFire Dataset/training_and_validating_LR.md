## Training and validation (evaluate) flowchart to get the model with best regularization
Training Set (184k)     Validation Set (46k)
       ↓                        ↓
   Train models            Evaluate models
   - L1 (Lasso)            - Calculate Val_F1
   - L2 (Ridge)            - Calculate Val_AUC
   - ElasticNet            - Calculate Val_Acc
       ↓                        ↓
                    Compare Val_F1 scores
                           ↓
                Pick model with highest Val_F1
                           ↓
                      BEST MODEL

LogisticRegresionCV (Cross-validation) is used to auto select the best c (regularization strength) for each regularization methods when training the LR model on the same dataset. 

## Trains 3 Different Regularization Methods:
1. L1 (Lasso) Regularization

Forces weak features to ZERO coefficients
Automatic feature selection
Best when some features are truly irrelevant

2. L2 (Ridge) Regularization

Shrinks all coefficients (but keeps them)
Handles correlated features well
Best when most features contribute something

3. ElasticNet (L1 + L2)

Combines both approaches
Most robust overall
Gets benefits of both methods

No regularization: Model tries to fit EVERY detail in training data → overfits
With regularization: Model is penalized for being too complex → generalizes better

## C parameter
Large C (e.g., 100): Weak regularization, model can be complex
Small C (e.g., 0.01): Strong regularization, model forced to be simple