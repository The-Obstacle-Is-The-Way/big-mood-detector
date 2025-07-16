# Why is the resolution of the periodograms so low?

The density of period samples plotted in the periodograms cannot be increased without increasing the sampling frequency of your measurements. In the examples in [`demo_independent.ipynb`](https://github.com/mmoskon/CosinorPy/blob/master/demo_dependent.ipynb) the periodograms are plotted on the data that were sampled every 2 hours (`1/fs`) for `test1` and `test2`, and every hour for `test3` and `test4`.  In the first two cases we have 25 samples (`N`), and in the latter two cases we have 49 samples. The frequency resolution (minimal frequency) is limited to 

`fs/N = 1/(2 * 25) = 0.02 h^{-1}`

in the first case and 

`1/(1 * 49) ≈ 0.02 h^{-1}`

in the second case. The highest detectable frequency is defined as half of the sampling frequency, which is `0.25 h^{-1}` for the first two tests and `0.5 h^{-1}` for the second two tests. This means that in the first two cases, detected frequencies are in the range from `0.02 h^{-1}`  to `0.25 h^{-1}`  with a step `0.02 h^{-1}`, and in the last two cases in the range from `0.01 h^{-1}`  to `0.5 h^{-1}`  with a step that equals approximately `0.02 h^{-1}`.

For the first two cases the power spectral density (PSD) can be evaluated for the following frequencies

`F1 = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2 , 0.22, 0.24]`

For the last two cases the PSD can be evaluated for the following frequencies

`F2 = [0.02040816 0.04081633 0.06122449 0.08163265 0.10204082 0.12244898 0.14285714 0.16326531 0.18367347 0.20408163 0.2244898  0.24489796 0.26530612 0.28571429 0.30612245 0.32653061 0.34693878 0.36734694 0.3877551  0.40816327 0.42857143 0.44897959 0.46938776 0.48979592]`

When transformed to periods, these have the following values:

`P1 = 1/F1 = [50.         25.         16.66666667 12.5        10.          8.33333333  7.14285714  6.25        5.55555556  5.          4.54545455  4.16666667]`

`P2 = 1/F2 = [49.         24.5        16.33333333 12.25        9.8         8.16666667  7.          6.125       5.44444444  4.9         4.45454545  4.08333333  3.76923077  3.5         3.26666667  3.0625      2.88235294  2.72222222  2.57894737  2.45        2.33333333  2.22727273  2.13043478  2.04166667]`

