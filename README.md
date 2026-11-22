# FinStratNNMC

This project provides a facility to simulate MC-paths and 
learn strategies to optimize a given objective.

## Usage

1. Create a Config with extends the [base config](configs/config.py).

2. Simulate the assets, build and train the neural network.

```python
# ------------ 1. Load simulated test asset prices ------------
asset_prices_sim = simulate_w_config(merton_config)
sim_asset_prices_tf = tf.convert_to_tensor(asset_prices_sim)
train_asset_prices, test_asset_prices = train_validation_split(sim_asset_prices_tf, 0.3)

# ------------ 2. Compile model & train model & evaluate ------------
nn = build_prop_model_from_config(merton_config)

# Compile
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
nn.compile(optimizer=optimizer, loss=merton_config.loss_func)

# Fit
nn.fit(train_asset_prices
       , tf.zeros([tf.shape(train_asset_prices)[0]
       , merton_config.n_time_steps, merton_config.n_assets])
       , epochs=merton_config.epochs
       , callbacks=[GetWeights()]
       , batch_size=2**6)

# Predict test
test_predictions = nn.predict(test_asset_prices)
```

## Example

For a demonstration, see [Merton fractions example](test/merton_fractions) and, in particular, the [output dashboard](test/merton_fractions/results.png).

## Limitations

The simulated processes are multidimensional GBMs. This will be extended in the future to accomodate other stochastic processes.

## Methodology

See [documentation](Documentation.pdf).
