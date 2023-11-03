# Default Pre-LN
python run_clm.py num_token_mult=2 model.n_layer=18

# Value-SkipInit
python run_clm.py model=skipless num_token_mult=2 model.value_skip_gain=null model.proj_skip_gain=null model.val_init_type=orth model.n_layer=18

# Skip and VP-less (Fig 9)
python run_clm.py model=skipless num_token_mult=2 model.n_layer=18

# Parallel skip-and-VP-less (Fig 10)
python run_clm.py model=skipless-parallel num_token_mult=2 model.n_layer=18

# Fig 10 but without normalisation
python run_clm.py model=skipless-parallel num_token_mult=2 model.n_layer=18 model.norm_type=none

# Default Parallel block
python run_clm.py model=default-parallel num_token_mult=2 model.n_layer=18
