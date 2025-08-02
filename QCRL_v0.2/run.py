import logging, os, multiprocessing as mp

# 1) make every logger inherit this blanket rule
logging.disable(logging.CRITICAL)        # blocks everything ≤ CRITICAL

# 2) be sure child processes inherit the rule (spawn, not fork)
mp.set_start_method("spawn", force=True)

# 3) optional: stop the colour codes DI-engine adds
os.environ["DI_DISABLE_COLOR"] = "true"

# now import / run DI-engine
from ding.entry import serial_pipeline
from config import main_config, create_config

serial_pipeline([main_config, create_config],
                seed=42,
                max_env_step=100)     # whatever quick smoke-test length you need
