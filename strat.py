def ma_crossover_signal(short_ma, long_ma):
    return (short_ma > long_ma).astype(int)