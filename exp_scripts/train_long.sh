# Default
python run_clm.py num_token_mult=6 model.n_layer=18 

# Skip and VP-less (Fig 9)
python run_clm.py num_token_mult=6 model.n_layer=18 model=skipless

# Parallel skip-and-VP-less (Fig 10)
python run_clm.py num_token_mult=6 model.n_layer=18 model=skipless-parallel
